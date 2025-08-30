"""IP whitelist management for network segmentation.

Implements IP-based access control for different network zones
and security levels.
"""

import ipaddress
from typing import Set, List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class NetworkZone(Enum):
    """Network security zones."""
    
    PUBLIC = "public"      # Public API endpoints
    PRIVATE = "private"    # Internal services (database, cache)
    RESTRICTED = "restricted"  # Secrets and sensitive operations
    MANAGEMENT = "management"  # Admin and monitoring


@dataclass
class IPRule:
    """IP access rule."""
    
    ip_address: str  # Can be single IP or CIDR
    zone: NetworkZone
    description: str
    enabled: bool = True
    
    def matches(self, ip: str) -> bool:
        """Check if an IP matches this rule.
        
        Args:
            ip: IP address to check
            
        Returns:
            True if IP matches the rule
        """
        if not self.enabled:
            return False
        
        try:
            # Parse the rule IP (could be single IP or network)
            if '/' in self.ip_address:
                # CIDR notation
                network = ipaddress.ip_network(self.ip_address, strict=False)
                ip_obj = ipaddress.ip_address(ip)
                return ip_obj in network
            else:
                # Single IP
                return ipaddress.ip_address(ip) == ipaddress.ip_address(self.ip_address)
        except (ipaddress.AddressValueError, ValueError) as e:
            logger.error("Invalid IP address", 
                        rule_ip=self.ip_address, 
                        check_ip=ip, 
                        error=str(e))
            return False


class IPWhitelistManager:
    """Manages IP whitelist for network access control."""
    
    # Default rules for different zones
    DEFAULT_RULES = {
        NetworkZone.PUBLIC: [
            # Allow localhost for development
            IPRule("127.0.0.1", NetworkZone.PUBLIC, "Localhost"),
            IPRule("::1", NetworkZone.PUBLIC, "Localhost IPv6"),
        ],
        NetworkZone.PRIVATE: [
            # Private network ranges
            IPRule("10.0.0.0/8", NetworkZone.PRIVATE, "Private network class A"),
            IPRule("172.16.0.0/12", NetworkZone.PRIVATE, "Private network class B"),
            IPRule("192.168.0.0/16", NetworkZone.PRIVATE, "Private network class C"),
        ],
        NetworkZone.RESTRICTED: [
            # Only specific IPs for restricted zone
            IPRule("127.0.0.1", NetworkZone.RESTRICTED, "Localhost only"),
        ],
        NetworkZone.MANAGEMENT: [
            # Management network
            IPRule("10.0.1.0/24", NetworkZone.MANAGEMENT, "Management subnet"),
        ]
    }
    
    def __init__(self, custom_rules: Optional[List[IPRule]] = None):
        """Initialize IP whitelist manager.
        
        Args:
            custom_rules: Custom IP rules to add
        """
        self._rules: Dict[NetworkZone, List[IPRule]] = {
            zone: rules.copy() for zone, rules in self.DEFAULT_RULES.items()
        }
        
        # Add custom rules
        if custom_rules:
            for rule in custom_rules:
                self.add_rule(rule)
        
        # Cache for performance
        self._cache: Dict[str, Dict[NetworkZone, bool]] = {}
    
    def add_rule(self, rule: IPRule):
        """Add an IP rule.
        
        Args:
            rule: IP rule to add
        """
        if rule.zone not in self._rules:
            self._rules[rule.zone] = []
        
        self._rules[rule.zone].append(rule)
        self._clear_cache()
        
        logger.info("Added IP rule", 
                   ip=rule.ip_address, 
                   zone=rule.zone.value,
                   description=rule.description)
    
    def remove_rule(self, ip_address: str, zone: NetworkZone):
        """Remove an IP rule.
        
        Args:
            ip_address: IP address or CIDR
            zone: Network zone
        """
        if zone in self._rules:
            self._rules[zone] = [
                r for r in self._rules[zone] 
                if r.ip_address != ip_address
            ]
            self._clear_cache()
            
            logger.info("Removed IP rule", 
                       ip=ip_address, 
                       zone=zone.value)
    
    def is_allowed(self, ip: str, zone: NetworkZone) -> bool:
        """Check if an IP is allowed in a zone.
        
        Args:
            ip: IP address to check
            zone: Network zone to access
            
        Returns:
            True if IP is allowed in the zone
        """
        # Check cache first
        cache_key = f"{ip}:{zone.value}"
        if ip in self._cache and zone in self._cache[ip]:
            return self._cache[ip][zone]
        
        # Check rules for the zone
        allowed = False
        if zone in self._rules:
            for rule in self._rules[zone]:
                if rule.matches(ip):
                    allowed = True
                    break
        
        # Cache result
        if ip not in self._cache:
            self._cache[ip] = {}
        self._cache[ip][zone] = allowed
        
        if not allowed:
            logger.debug("IP access denied", 
                        ip=ip, 
                        zone=zone.value)
        
        return allowed
    
    def check_access(self, ip: str, required_zones: List[NetworkZone]) -> bool:
        """Check if an IP has access to all required zones.
        
        Args:
            ip: IP address to check
            required_zones: List of zones that need access
            
        Returns:
            True if IP has access to all required zones
        """
        for zone in required_zones:
            if not self.is_allowed(ip, zone):
                return False
        return True
    
    def get_allowed_zones(self, ip: str) -> Set[NetworkZone]:
        """Get all zones an IP has access to.
        
        Args:
            ip: IP address to check
            
        Returns:
            Set of allowed zones
        """
        allowed_zones = set()
        for zone in NetworkZone:
            if self.is_allowed(ip, zone):
                allowed_zones.add(zone)
        return allowed_zones
    
    def enable_rule(self, ip_address: str, zone: NetworkZone):
        """Enable a disabled rule.
        
        Args:
            ip_address: IP address or CIDR
            zone: Network zone
        """
        if zone in self._rules:
            for rule in self._rules[zone]:
                if rule.ip_address == ip_address:
                    rule.enabled = True
                    self._clear_cache()
                    logger.info("Enabled IP rule", 
                               ip=ip_address, 
                               zone=zone.value)
                    break
    
    def disable_rule(self, ip_address: str, zone: NetworkZone):
        """Disable a rule without removing it.
        
        Args:
            ip_address: IP address or CIDR
            zone: Network zone
        """
        if zone in self._rules:
            for rule in self._rules[zone]:
                if rule.ip_address == ip_address:
                    rule.enabled = False
                    self._clear_cache()
                    logger.info("Disabled IP rule", 
                               ip=ip_address, 
                               zone=zone.value)
                    break
    
    def list_rules(self, zone: Optional[NetworkZone] = None) -> List[IPRule]:
        """List IP rules.
        
        Args:
            zone: Specific zone to list, or None for all
            
        Returns:
            List of IP rules
        """
        if zone:
            return self._rules.get(zone, []).copy()
        
        all_rules = []
        for zone_rules in self._rules.values():
            all_rules.extend(zone_rules)
        return all_rules
    
    def _clear_cache(self):
        """Clear the IP cache."""
        self._cache.clear()
    
    def validate_ip(self, ip: str) -> bool:
        """Validate if a string is a valid IP address.
        
        Args:
            ip: IP address string
            
        Returns:
            True if valid IP
        """
        try:
            ipaddress.ip_address(ip)
            return True
        except (ipaddress.AddressValueError, ValueError):
            return False
    
    def export_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Export rules as dictionary.
        
        Returns:
            Dictionary of rules by zone
        """
        export = {}
        for zone, rules in self._rules.items():
            export[zone.value] = [
                {
                    "ip_address": rule.ip_address,
                    "description": rule.description,
                    "enabled": rule.enabled
                }
                for rule in rules
            ]
        return export
    
    def import_rules(self, rules_dict: Dict[str, List[Dict[str, Any]]]):
        """Import rules from dictionary.
        
        Args:
            rules_dict: Dictionary of rules by zone
        """
        for zone_str, rules_list in rules_dict.items():
            try:
                zone = NetworkZone(zone_str)
                for rule_dict in rules_list:
                    rule = IPRule(
                        ip_address=rule_dict["ip_address"],
                        zone=zone,
                        description=rule_dict.get("description", ""),
                        enabled=rule_dict.get("enabled", True)
                    )
                    self.add_rule(rule)
            except (ValueError, KeyError) as e:
                logger.error("Failed to import rule", 
                            zone=zone_str, 
                            error=str(e))


class NetworkSegmentation:
    """Implements network segmentation policies."""
    
    def __init__(self, whitelist_manager: Optional[IPWhitelistManager] = None):
        """Initialize network segmentation.
        
        Args:
            whitelist_manager: IP whitelist manager
        """
        self.whitelist_manager = whitelist_manager or IPWhitelistManager()
        
        # Define service to zone mappings
        self.service_zones = {
            "api": [NetworkZone.PUBLIC],
            "database": [NetworkZone.PRIVATE, NetworkZone.RESTRICTED],
            "vault": [NetworkZone.RESTRICTED],
            "redis": [NetworkZone.PRIVATE],
            "monitoring": [NetworkZone.MANAGEMENT],
            "admin": [NetworkZone.MANAGEMENT, NetworkZone.RESTRICTED]
        }
    
    def can_access_service(self, ip: str, service: str) -> bool:
        """Check if an IP can access a service.
        
        Args:
            ip: Source IP address
            service: Service name
            
        Returns:
            True if access is allowed
        """
        if service not in self.service_zones:
            logger.warning("Unknown service", service=service)
            return False
        
        required_zones = self.service_zones[service]
        
        # Check if IP has access to any of the required zones
        for zone in required_zones:
            if self.whitelist_manager.is_allowed(ip, zone):
                return True
        
        logger.warning("Service access denied", 
                      ip=ip, 
                      service=service,
                      required_zones=[z.value for z in required_zones])
        return False
    
    def get_accessible_services(self, ip: str) -> List[str]:
        """Get list of services accessible from an IP.
        
        Args:
            ip: Source IP address
            
        Returns:
            List of accessible service names
        """
        allowed_zones = self.whitelist_manager.get_allowed_zones(ip)
        accessible_services = []
        
        for service, required_zones in self.service_zones.items():
            # Check if IP has access to any required zone
            if any(zone in allowed_zones for zone in required_zones):
                accessible_services.append(service)
        
        return accessible_services
    
    def enforce_ssh_tunnel(self, ip: str) -> bool:
        """Check if SSH tunnel is required for an IP.
        
        Args:
            ip: Source IP address
            
        Returns:
            True if SSH tunnel is required
        """
        # Require SSH tunnel for non-private IPs accessing restricted services
        if not self.whitelist_manager.is_allowed(ip, NetworkZone.PRIVATE):
            # External IP - requires SSH tunnel for any restricted access
            if self.whitelist_manager.is_allowed(ip, NetworkZone.RESTRICTED):
                return True
        
        return False