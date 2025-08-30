"""DNS management for failover operations."""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
import structlog

logger = structlog.get_logger(__name__)


class DNSRecord:
    """DNS record configuration."""
    
    def __init__(
        self,
        domain: str,
        record_type: str,
        name: str,
        value: str,
        ttl: int = 300,
        priority: Optional[int] = None
    ):
        """Initialize DNS record.
        
        Args:
            domain: Domain name
            record_type: Record type (A, AAAA, CNAME, MX, TXT)
            name: Record name
            value: Record value
            ttl: Time to live in seconds
            priority: Priority for MX records
        """
        self.domain = domain
        self.record_type = record_type
        self.name = name
        self.value = value
        self.ttl = ttl
        self.priority = priority
        self.record_id: Optional[str] = None


class DNSManager:
    """Manages DNS records for failover."""
    
    def __init__(
        self,
        api_token: str,
        provider: str = "digitalocean"
    ):
        """Initialize DNS manager.
        
        Args:
            api_token: API token for DNS provider
            provider: DNS provider (digitalocean, cloudflare, route53)
        """
        self.api_token = api_token
        self.provider = provider
        
        # Provider-specific configuration
        self.api_base_url = self._get_api_base_url()
        self.headers = self._get_headers()
        
        # Cache DNS records
        self.cached_records: Dict[str, List[DNSRecord]] = {}
        self.last_cache_update: Optional[datetime] = None
    
    def _get_api_base_url(self) -> str:
        """Get API base URL for provider.
        
        Returns:
            API base URL
        """
        urls = {
            "digitalocean": "https://api.digitalocean.com/v2",
            "cloudflare": "https://api.cloudflare.com/client/v4",
            "route53": "https://route53.amazonaws.com/2013-04-01"
        }
        
        return urls.get(self.provider, urls["digitalocean"])
    
    def _get_headers(self) -> Dict[str, str]:
        """Get API headers for provider.
        
        Returns:
            Request headers
        """
        if self.provider == "digitalocean":
            return {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
        elif self.provider == "cloudflare":
            return {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
        else:
            return {}
    
    async def get_records(self, domain: str, force_refresh: bool = False) -> List[DNSRecord]:
        """Get DNS records for domain.
        
        Args:
            domain: Domain name
            force_refresh: Force cache refresh
            
        Returns:
            List of DNS records
        """
        # Check cache
        if not force_refresh and domain in self.cached_records:
            cache_age = (datetime.utcnow() - self.last_cache_update).total_seconds()
            if cache_age < 300:  # 5 minute cache
                return self.cached_records[domain]
        
        try:
            if self.provider == "digitalocean":
                records = await self._get_do_records(domain)
            elif self.provider == "cloudflare":
                records = await self._get_cf_records(domain)
            else:
                logger.warning(f"Unsupported provider: {self.provider}")
                return []
            
            # Update cache
            self.cached_records[domain] = records
            self.last_cache_update = datetime.utcnow()
            
            return records
            
        except Exception as e:
            logger.error(f"Failed to get DNS records", error=str(e))
            return []
    
    async def _get_do_records(self, domain: str) -> List[DNSRecord]:
        """Get DigitalOcean DNS records.
        
        Args:
            domain: Domain name
            
        Returns:
            List of DNS records
        """
        url = f"{self.api_base_url}/domains/{domain}/records"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status != 200:
                    raise Exception(f"API error: {response.status}")
                
                data = await response.json()
                
                records = []
                for record_data in data.get("domain_records", []):
                    record = DNSRecord(
                        domain=domain,
                        record_type=record_data["type"],
                        name=record_data["name"],
                        value=record_data["data"],
                        ttl=record_data.get("ttl", 300),
                        priority=record_data.get("priority")
                    )
                    record.record_id = str(record_data["id"])
                    records.append(record)
                
                return records
    
    async def _get_cf_records(self, domain: str) -> List[DNSRecord]:
        """Get Cloudflare DNS records.
        
        Args:
            domain: Domain name
            
        Returns:
            List of DNS records
        """
        # First get zone ID
        zone_id = await self._get_cf_zone_id(domain)
        if not zone_id:
            return []
        
        url = f"{self.api_base_url}/zones/{zone_id}/dns_records"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status != 200:
                    raise Exception(f"API error: {response.status}")
                
                data = await response.json()
                
                records = []
                for record_data in data.get("result", []):
                    record = DNSRecord(
                        domain=domain,
                        record_type=record_data["type"],
                        name=record_data["name"],
                        value=record_data["content"],
                        ttl=record_data.get("ttl", 300),
                        priority=record_data.get("priority")
                    )
                    record.record_id = record_data["id"]
                    records.append(record)
                
                return records
    
    async def _get_cf_zone_id(self, domain: str) -> Optional[str]:
        """Get Cloudflare zone ID for domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Zone ID or None
        """
        url = f"{self.api_base_url}/zones?name={domain}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                zones = data.get("result", [])
                
                if zones:
                    return zones[0]["id"]
                
                return None
    
    async def update_record(self, record: DNSRecord) -> bool:
        """Update DNS record.
        
        Args:
            record: DNS record to update
            
        Returns:
            True if successful
        """
        try:
            if self.provider == "digitalocean":
                return await self._update_do_record(record)
            elif self.provider == "cloudflare":
                return await self._update_cf_record(record)
            else:
                logger.warning(f"Unsupported provider: {self.provider}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update DNS record", error=str(e))
            return False
    
    async def _update_do_record(self, record: DNSRecord) -> bool:
        """Update DigitalOcean DNS record.
        
        Args:
            record: DNS record to update
            
        Returns:
            True if successful
        """
        if not record.record_id:
            logger.error("Record ID required for update")
            return False
        
        url = f"{self.api_base_url}/domains/{record.domain}/records/{record.record_id}"
        
        data = {
            "type": record.record_type,
            "name": record.name,
            "data": record.value,
            "ttl": record.ttl
        }
        
        if record.priority is not None:
            data["priority"] = record.priority
        
        async with aiohttp.ClientSession() as session:
            async with session.put(url, headers=self.headers, json=data) as response:
                success = response.status == 200
                
                if success:
                    logger.info(
                        f"DNS record updated",
                        domain=record.domain,
                        name=record.name,
                        value=record.value
                    )
                else:
                    logger.error(f"Failed to update DNS record: {response.status}")
                
                return success
    
    async def _update_cf_record(self, record: DNSRecord) -> bool:
        """Update Cloudflare DNS record.
        
        Args:
            record: DNS record to update
            
        Returns:
            True if successful
        """
        if not record.record_id:
            logger.error("Record ID required for update")
            return False
        
        zone_id = await self._get_cf_zone_id(record.domain)
        if not zone_id:
            return False
        
        url = f"{self.api_base_url}/zones/{zone_id}/dns_records/{record.record_id}"
        
        data = {
            "type": record.record_type,
            "name": record.name,
            "content": record.value,
            "ttl": record.ttl
        }
        
        if record.priority is not None:
            data["priority"] = record.priority
        
        async with aiohttp.ClientSession() as session:
            async with session.put(url, headers=self.headers, json=data) as response:
                success = response.status == 200
                
                if success:
                    logger.info(
                        f"DNS record updated",
                        domain=record.domain,
                        name=record.name,
                        value=record.value
                    )
                else:
                    logger.error(f"Failed to update DNS record: {response.status}")
                
                return success
    
    async def failover_dns(
        self,
        domain: str,
        from_ip: str,
        to_ip: str,
        record_type: str = "A"
    ) -> bool:
        """Perform DNS failover.
        
        Args:
            domain: Domain name
            from_ip: Current IP address
            to_ip: New IP address
            record_type: DNS record type
            
        Returns:
            True if successful
        """
        logger.info(
            f"Starting DNS failover",
            domain=domain,
            from_ip=from_ip,
            to_ip=to_ip
        )
        
        try:
            # Get current records
            records = await self.get_records(domain, force_refresh=True)
            
            # Find records to update
            updated_count = 0
            
            for record in records:
                if record.record_type == record_type and record.value == from_ip:
                    # Update record
                    record.value = to_ip
                    
                    if await self.update_record(record):
                        updated_count += 1
                    else:
                        logger.error(f"Failed to update record: {record.name}")
            
            # Clear cache after update
            if domain in self.cached_records:
                del self.cached_records[domain]
            
            logger.info(
                f"DNS failover completed",
                domain=domain,
                updated_records=updated_count
            )
            
            return updated_count > 0
            
        except Exception as e:
            logger.error(f"DNS failover failed", error=str(e))
            return False
    
    async def verify_dns_propagation(
        self,
        domain: str,
        expected_ip: str,
        nameservers: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """Verify DNS propagation.
        
        Args:
            domain: Domain to check
            expected_ip: Expected IP address
            nameservers: List of nameservers to check
            
        Returns:
            Dict of nameserver -> propagated status
        """
        if not nameservers:
            nameservers = [
                "8.8.8.8",  # Google
                "1.1.1.1",  # Cloudflare
                "9.9.9.9",  # Quad9
            ]
        
        results = {}
        
        for ns in nameservers:
            try:
                import socket
                
                # Resolve domain using specific nameserver
                # This is simplified - would use dnspython in production
                resolved_ip = socket.gethostbyname(domain)
                
                results[ns] = resolved_ip == expected_ip
                
            except Exception as e:
                logger.error(f"DNS check failed for {ns}", error=str(e))
                results[ns] = False
        
        return results