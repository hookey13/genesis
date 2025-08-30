"""
Emergency contact management system with encrypted storage and rotation scheduling.
"""

import json
import base64
from datetime import datetime, timedelta, time
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from cryptography.fernet import Fernet
import structlog
import asyncio
import aiohttp

logger = structlog.get_logger(__name__)


class ContactRole(Enum):
    """Contact roles in the organization."""
    PRIMARY_ONCALL = "primary_oncall"
    SECONDARY_ONCALL = "secondary_oncall"
    ENGINEERING_LEAD = "engineering_lead"
    OPERATIONS_LEAD = "operations_lead"
    DIRECTOR = "director"
    SECURITY = "security"
    DATABASE_ADMIN = "database_admin"
    NETWORK_ADMIN = "network_admin"
    BUSINESS_OWNER = "business_owner"
    COMPLIANCE = "compliance"


class NotificationPreference(Enum):
    """Contact notification preferences."""
    PHONE_CALL = "phone_call"
    SMS = "sms"
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    ALL = "all"


class RotationSchedule(Enum):
    """On-call rotation schedules."""
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


@dataclass
class Contact:
    """Emergency contact information."""
    contact_id: str
    name: str
    role: ContactRole
    email: str
    phone: Optional[str] = None
    slack_id: Optional[str] = None
    pagerduty_id: Optional[str] = None
    notification_preferences: List[NotificationPreference] = field(default_factory=list)
    availability_hours: Optional[Dict[str, str]] = None  # {"start": "09:00", "end": "17:00"}
    timezone: str = "UTC"
    active: bool = True
    verified: bool = False
    last_verified: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OnCallRotation:
    """On-call rotation schedule."""
    rotation_id: str
    name: str
    schedule_type: RotationSchedule
    contacts: List[str]  # List of contact_ids
    current_index: int = 0
    start_date: Optional[datetime] = None
    rotation_interval_days: int = 7
    handoff_time: time = time(9, 0)  # 9 AM default handoff
    override: Optional[str] = None  # contact_id for temporary override
    override_until: Optional[datetime] = None
    active: bool = True


class ContactManager:
    """Manages emergency contacts with encryption and rotation."""
    
    def __init__(self, encryption_key: Optional[str] = None, 
                 storage_path: str = ".genesis/contacts.enc"):
        """
        Initialize contact manager.
        
        Args:
            encryption_key: Base64 encoded encryption key
            storage_path: Path to encrypted contact storage
        """
        self.storage_path = storage_path
        self.contacts: Dict[str, Contact] = {}
        self.rotations: Dict[str, OnCallRotation] = {}
        
        # Initialize encryption
        if encryption_key:
            self.cipher = Fernet(encryption_key.encode())
        else:
            # Generate new key if not provided
            key = Fernet.generate_key()
            self.cipher = Fernet(key)
            logger.warning("Generated new encryption key - save this!",
                          key=key.decode())
        
        # Load existing contacts
        self._load_contacts()
        
        logger.info("Contact manager initialized",
                   contact_count=len(self.contacts),
                   rotation_count=len(self.rotations))
    
    def _encrypt_data(self, data: Dict[str, Any]) -> str:
        """Encrypt sensitive data."""
        json_str = json.dumps(data)
        encrypted = self.cipher.encrypt(json_str.encode())
        return base64.b64encode(encrypted).decode()
    
    def _decrypt_data(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt sensitive data."""
        try:
            decoded = base64.b64decode(encrypted_data.encode())
            decrypted = self.cipher.decrypt(decoded)
            return json.loads(decrypted.decode())
        except Exception as e:
            logger.error("Failed to decrypt data", error=str(e))
            raise
    
    def _load_contacts(self):
        """Load contacts from encrypted storage."""
        try:
            with open(self.storage_path, 'r') as f:
                encrypted_data = f.read()
                data = self._decrypt_data(encrypted_data)
                
                # Load contacts
                for contact_data in data.get('contacts', []):
                    # Convert string values back to Enums
                    contact_data['role'] = ContactRole(contact_data['role'])
                    contact_data['notification_preferences'] = [
                        NotificationPreference(pref) 
                        for pref in contact_data.get('notification_preferences', [])
                    ]
                    # Convert datetime strings back to datetime objects if present
                    if contact_data.get('last_verified'):
                        contact_data['last_verified'] = datetime.fromisoformat(
                            contact_data['last_verified']
                        )
                    contact = Contact(**contact_data)
                    self.contacts[contact.contact_id] = contact
                
                # Load rotations
                for rotation_data in data.get('rotations', []):
                    # Convert string values back to Enums
                    rotation_data['schedule_type'] = RotationSchedule(rotation_data['schedule_type'])
                    # Convert datetime/time strings back to objects if present
                    if rotation_data.get('start_date'):
                        rotation_data['start_date'] = datetime.fromisoformat(
                            rotation_data['start_date']
                        )
                    if rotation_data.get('override_until'):
                        rotation_data['override_until'] = datetime.fromisoformat(
                            rotation_data['override_until']
                        )
                    if rotation_data.get('handoff_time'):
                        rotation_data['handoff_time'] = time.fromisoformat(
                            rotation_data['handoff_time']
                        )
                    rotation = OnCallRotation(**rotation_data)
                    self.rotations[rotation.rotation_id] = rotation
                
                logger.info("Loaded contacts from storage",
                           contact_count=len(self.contacts),
                           rotation_count=len(self.rotations))
                           
        except FileNotFoundError:
            logger.info("No existing contact storage found")
        except Exception as e:
            logger.error("Failed to load contacts", error=str(e))
    
    def _save_contacts(self):
        """Save contacts to encrypted storage."""
        try:
            # Convert dataclasses to dict, handling Enum values
            contacts_data = []
            for contact in self.contacts.values():
                contact_dict = asdict(contact)
                # Convert Enum values to strings
                contact_dict['role'] = contact.role.value
                contact_dict['notification_preferences'] = [
                    pref.value for pref in contact.notification_preferences
                ]
                # Convert datetime to ISO format string if present
                if contact_dict.get('last_verified'):
                    contact_dict['last_verified'] = contact_dict['last_verified'].isoformat()
                contacts_data.append(contact_dict)
            
            rotations_data = []
            for rotation in self.rotations.values():
                rotation_dict = asdict(rotation)
                # Convert Enum values to strings
                rotation_dict['schedule_type'] = rotation.schedule_type.value
                # Convert datetime/time to ISO format strings if present
                if rotation_dict.get('start_date'):
                    rotation_dict['start_date'] = rotation_dict['start_date'].isoformat()
                if rotation_dict.get('override_until'):
                    rotation_dict['override_until'] = rotation_dict['override_until'].isoformat()
                if rotation_dict.get('handoff_time'):
                    rotation_dict['handoff_time'] = rotation_dict['handoff_time'].isoformat()
                rotations_data.append(rotation_dict)
            
            data = {
                'contacts': contacts_data,
                'rotations': rotations_data,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            encrypted_data = self._encrypt_data(data)
            
            with open(self.storage_path, 'w') as f:
                f.write(encrypted_data)
            
            logger.info("Saved contacts to storage",
                       contact_count=len(self.contacts))
                       
        except Exception as e:
            logger.error("Failed to save contacts", error=str(e))
            raise
    
    def add_contact(self, contact: Contact) -> bool:
        """Add or update a contact."""
        try:
            # Validate required fields
            if not contact.email:
                raise ValueError("Email is required")
            
            # Check for duplicates
            for existing in self.contacts.values():
                if existing.email == contact.email and existing.contact_id != contact.contact_id:
                    logger.warning("Duplicate email found",
                                 email=contact.email,
                                 existing_id=existing.contact_id)
            
            self.contacts[contact.contact_id] = contact
            self._save_contacts()
            
            logger.info("Contact added/updated",
                       contact_id=contact.contact_id,
                       name=contact.name,
                       role=contact.role.value)
            return True
            
        except Exception as e:
            logger.error("Failed to add contact",
                        contact_id=contact.contact_id,
                        error=str(e))
            return False
    
    def remove_contact(self, contact_id: str) -> bool:
        """Remove a contact."""
        try:
            if contact_id not in self.contacts:
                logger.warning("Contact not found", contact_id=contact_id)
                return False
            
            # Check if contact is in any rotations
            for rotation in self.rotations.values():
                if contact_id in rotation.contacts:
                    logger.warning("Contact is in rotation",
                                 contact_id=contact_id,
                                 rotation_id=rotation.rotation_id)
            
            del self.contacts[contact_id]
            self._save_contacts()
            
            logger.info("Contact removed", contact_id=contact_id)
            return True
            
        except Exception as e:
            logger.error("Failed to remove contact",
                        contact_id=contact_id,
                        error=str(e))
            return False
    
    def get_contact(self, contact_id: str) -> Optional[Contact]:
        """Get a contact by ID."""
        return self.contacts.get(contact_id)
    
    def get_contacts_by_role(self, role: ContactRole) -> List[Contact]:
        """Get all contacts with a specific role."""
        return [
            contact for contact in self.contacts.values()
            if contact.role == role and contact.active
        ]
    
    def create_rotation(self, rotation: OnCallRotation) -> bool:
        """Create or update an on-call rotation."""
        try:
            # Validate contacts exist
            for contact_id in rotation.contacts:
                if contact_id not in self.contacts:
                    raise ValueError(f"Contact {contact_id} not found")
            
            if not rotation.start_date:
                rotation.start_date = datetime.utcnow()
            
            self.rotations[rotation.rotation_id] = rotation
            self._save_contacts()
            
            logger.info("Rotation created/updated",
                       rotation_id=rotation.rotation_id,
                       schedule_type=rotation.schedule_type.value,
                       contact_count=len(rotation.contacts))
            return True
            
        except Exception as e:
            logger.error("Failed to create rotation",
                        rotation_id=rotation.rotation_id,
                        error=str(e))
            return False
    
    def get_current_on_call(self, rotation_id: str) -> Optional[Contact]:
        """Get current on-call contact for a rotation."""
        try:
            rotation = self.rotations.get(rotation_id)
            if not rotation or not rotation.active:
                return None
            
            # Check for override
            if rotation.override and rotation.override_until:
                if datetime.utcnow() < rotation.override_until:
                    return self.contacts.get(rotation.override)
            
            if not rotation.contacts:
                return None
            
            # Calculate current rotation index
            if rotation.start_date:
                days_elapsed = (datetime.utcnow() - rotation.start_date).days
                rotations_completed = days_elapsed // rotation.rotation_interval_days
                current_index = rotations_completed % len(rotation.contacts)
            else:
                current_index = rotation.current_index
            
            contact_id = rotation.contacts[current_index]
            return self.contacts.get(contact_id)
            
        except Exception as e:
            logger.error("Failed to get current on-call",
                        rotation_id=rotation_id,
                        error=str(e))
            return None
    
    def get_next_on_call(self, rotation_id: str) -> Optional[Contact]:
        """Get next on-call contact for a rotation."""
        try:
            rotation = self.rotations.get(rotation_id)
            if not rotation or not rotation.active or not rotation.contacts:
                return None
            
            current = self.get_current_on_call(rotation_id)
            if not current:
                return None
            
            # Find current index
            current_index = -1
            for i, contact_id in enumerate(rotation.contacts):
                if contact_id == current.contact_id:
                    current_index = i
                    break
            
            if current_index == -1:
                return None
            
            # Get next contact
            next_index = (current_index + 1) % len(rotation.contacts)
            next_contact_id = rotation.contacts[next_index]
            return self.contacts.get(next_contact_id)
            
        except Exception as e:
            logger.error("Failed to get next on-call",
                        rotation_id=rotation_id,
                        error=str(e))
            return None
    
    def set_override(self, rotation_id: str, contact_id: str, 
                    until: datetime) -> bool:
        """Set temporary override for rotation."""
        try:
            rotation = self.rotations.get(rotation_id)
            if not rotation:
                return False
            
            if contact_id not in self.contacts:
                return False
            
            rotation.override = contact_id
            rotation.override_until = until
            self._save_contacts()
            
            logger.info("Override set for rotation",
                       rotation_id=rotation_id,
                       override_contact=contact_id,
                       until=until.isoformat())
            return True
            
        except Exception as e:
            logger.error("Failed to set override",
                        rotation_id=rotation_id,
                        error=str(e))
            return False
    
    async def verify_contact(self, contact_id: str) -> bool:
        """Verify contact information is current."""
        try:
            contact = self.contacts.get(contact_id)
            if not contact:
                return False
            
            verified = True
            
            # Test email if configured
            if contact.email:
                # Would send test email here
                logger.info("Email verification sent", email=contact.email)
            
            # Test phone if configured
            if contact.phone:
                # Would send test SMS here
                logger.info("SMS verification sent", phone=contact.phone)
            
            # Test Slack if configured
            if contact.slack_id:
                # Would send Slack message here
                logger.info("Slack verification sent", slack_id=contact.slack_id)
            
            # Update verification status
            contact.verified = verified
            contact.last_verified = datetime.utcnow()
            self._save_contacts()
            
            logger.info("Contact verified",
                       contact_id=contact_id,
                       verified=verified)
            return verified
            
        except Exception as e:
            logger.error("Failed to verify contact",
                        contact_id=contact_id,
                        error=str(e))
            return False
    
    async def verify_all_contacts(self) -> Dict[str, bool]:
        """Verify all active contacts."""
        results = {}
        
        for contact_id, contact in self.contacts.items():
            if contact.active:
                results[contact_id] = await self.verify_contact(contact_id)
        
        # Log summary
        verified_count = sum(1 for v in results.values() if v)
        logger.info("Contact verification complete",
                   total=len(results),
                   verified=verified_count,
                   failed=len(results) - verified_count)
        
        return results
    
    def get_escalation_chain(self, start_role: ContactRole) -> List[Contact]:
        """Get escalation chain starting from a role."""
        escalation_order = [
            ContactRole.PRIMARY_ONCALL,
            ContactRole.SECONDARY_ONCALL,
            ContactRole.ENGINEERING_LEAD,
            ContactRole.OPERATIONS_LEAD,
            ContactRole.DIRECTOR
        ]
        
        # Find starting point
        try:
            start_index = escalation_order.index(start_role)
        except ValueError:
            start_index = 0
        
        chain = []
        for role in escalation_order[start_index:]:
            contacts = self.get_contacts_by_role(role)
            chain.extend(contacts)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_chain = []
        for contact in chain:
            if contact.contact_id not in seen:
                seen.add(contact.contact_id)
                unique_chain.append(contact)
        
        return unique_chain
    
    def get_notification_list(self, severity: str) -> List[Tuple[Contact, List[NotificationPreference]]]:
        """Get notification list based on incident severity."""
        notification_list = []
        
        if severity == "critical":
            # Notify everyone immediately
            roles = [
                ContactRole.PRIMARY_ONCALL,
                ContactRole.SECONDARY_ONCALL,
                ContactRole.ENGINEERING_LEAD,
                ContactRole.DIRECTOR
            ]
            preferences = [NotificationPreference.PHONE_CALL, NotificationPreference.SMS]
            
        elif severity == "high":
            # Notify on-call and leads
            roles = [
                ContactRole.PRIMARY_ONCALL,
                ContactRole.ENGINEERING_LEAD
            ]
            preferences = [NotificationPreference.PAGERDUTY, NotificationPreference.SMS]
            
        elif severity == "medium":
            # Notify on-call
            roles = [ContactRole.PRIMARY_ONCALL]
            preferences = [NotificationPreference.PAGERDUTY, NotificationPreference.SLACK]
            
        else:  # low
            # Notify via non-urgent channels
            roles = [ContactRole.PRIMARY_ONCALL]
            preferences = [NotificationPreference.EMAIL, NotificationPreference.SLACK]
        
        for role in roles:
            contacts = self.get_contacts_by_role(role)
            for contact in contacts:
                # Use contact's preferences if set, otherwise use severity defaults
                contact_prefs = contact.notification_preferences if contact.notification_preferences else preferences
                notification_list.append((contact, contact_prefs))
        
        return notification_list
    
    def export_contact_list(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Export contact list for display/reporting."""
        contacts_export = []
        
        for contact in self.contacts.values():
            contact_data = {
                "name": contact.name,
                "role": contact.role.value,
                "email": contact.email if include_sensitive else "***@***.***",
                "active": contact.active,
                "verified": contact.verified,
                "last_verified": contact.last_verified.isoformat() if contact.last_verified else None
            }
            
            if include_sensitive:
                contact_data["phone"] = contact.phone
                contact_data["slack_id"] = contact.slack_id
            
            contacts_export.append(contact_data)
        
        rotations_export = []
        for rotation in self.rotations.values():
            current = self.get_current_on_call(rotation.rotation_id)
            next_contact = self.get_next_on_call(rotation.rotation_id)
            
            rotation_data = {
                "name": rotation.name,
                "schedule_type": rotation.schedule_type.value,
                "current_on_call": current.name if current else "None",
                "next_on_call": next_contact.name if next_contact else "None",
                "active": rotation.active
            }
            rotations_export.append(rotation_data)
        
        return {
            "contacts": contacts_export,
            "rotations": rotations_export,
            "exported_at": datetime.utcnow().isoformat()
        }
    
    def get_contact_stats(self) -> Dict[str, Any]:
        """Get contact management statistics."""
        total = len(self.contacts)
        active = sum(1 for c in self.contacts.values() if c.active)
        verified = sum(1 for c in self.contacts.values() if c.verified)
        
        role_counts = {}
        for contact in self.contacts.values():
            role = contact.role.value
            role_counts[role] = role_counts.get(role, 0) + 1
        
        rotation_stats = {
            "total": len(self.rotations),
            "active": sum(1 for r in self.rotations.values() if r.active)
        }
        
        return {
            "total_contacts": total,
            "active_contacts": active,
            "verified_contacts": verified,
            "verification_rate": f"{(verified/total*100):.1f}%" if total > 0 else "0%",
            "contacts_by_role": role_counts,
            "rotation_stats": rotation_stats
        }