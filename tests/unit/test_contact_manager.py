"""
Unit tests for emergency contact management system.
"""

import pytest
from datetime import datetime, timedelta, time
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import os
from cryptography.fernet import Fernet

from genesis.operations.contact_manager import (
    ContactManager, Contact, OnCallRotation,
    ContactRole, NotificationPreference, RotationSchedule
)


@pytest.fixture
def temp_storage():
    """Create temporary storage file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.enc') as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    try:
        os.unlink(temp_path)
    except:
        pass


@pytest.fixture
def contact_manager(temp_storage):
    """Create contact manager with temporary storage."""
    key = Fernet.generate_key()
    return ContactManager(
        encryption_key=key.decode(),
        storage_path=temp_storage
    )


@pytest.fixture
def sample_contact():
    """Create sample contact."""
    return Contact(
        contact_id="CONT-001",
        name="John Doe",
        role=ContactRole.PRIMARY_ONCALL,
        email="john.doe@example.com",
        phone="+1234567890",
        slack_id="U12345",
        pagerduty_id="P12345",
        notification_preferences=[
            NotificationPreference.PHONE_CALL,
            NotificationPreference.SLACK
        ],
        availability_hours={"start": "09:00", "end": "17:00"},
        timezone="America/New_York",
        active=True,
        verified=True,
        last_verified=datetime.utcnow()
    )


@pytest.fixture
def sample_rotation():
    """Create sample rotation."""
    return OnCallRotation(
        rotation_id="ROT-001",
        name="Primary On-Call",
        schedule_type=RotationSchedule.WEEKLY,
        contacts=["CONT-001", "CONT-002", "CONT-003"],
        current_index=0,
        start_date=datetime.utcnow(),
        rotation_interval_days=7,
        handoff_time=time(9, 0),
        active=True
    )


class TestContact:
    """Test Contact class."""
    
    def test_contact_creation(self, sample_contact):
        """Test contact creation."""
        assert sample_contact.contact_id == "CONT-001"
        assert sample_contact.name == "John Doe"
        assert sample_contact.role == ContactRole.PRIMARY_ONCALL
        assert sample_contact.email == "john.doe@example.com"
        assert NotificationPreference.PHONE_CALL in sample_contact.notification_preferences
    
    def test_contact_defaults(self):
        """Test contact with defaults."""
        contact = Contact(
            contact_id="CONT-002",
            name="Jane Smith",
            role=ContactRole.ENGINEERING_LEAD,
            email="jane@example.com"
        )
        
        assert contact.phone is None
        assert contact.active is True
        assert contact.verified is False
        assert contact.notification_preferences == []


class TestOnCallRotation:
    """Test OnCallRotation class."""
    
    def test_rotation_creation(self, sample_rotation):
        """Test rotation creation."""
        assert sample_rotation.rotation_id == "ROT-001"
        assert sample_rotation.schedule_type == RotationSchedule.WEEKLY
        assert len(sample_rotation.contacts) == 3
        assert sample_rotation.rotation_interval_days == 7
    
    def test_rotation_defaults(self):
        """Test rotation with defaults."""
        rotation = OnCallRotation(
            rotation_id="ROT-002",
            name="Secondary",
            schedule_type=RotationSchedule.BIWEEKLY,
            contacts=["CONT-001"]
        )
        
        assert rotation.current_index == 0
        assert rotation.handoff_time == time(9, 0)
        assert rotation.override is None
        assert rotation.active is True


class TestContactManager:
    """Test ContactManager class."""
    
    def test_manager_initialization(self, contact_manager):
        """Test manager initialization."""
        assert contact_manager.contacts == {}
        assert contact_manager.rotations == {}
        assert contact_manager.cipher is not None
    
    def test_encryption_decryption(self, contact_manager):
        """Test data encryption and decryption."""
        test_data = {
            "sensitive": "secret_value",
            "number": 42,
            "list": [1, 2, 3]
        }
        
        encrypted = contact_manager._encrypt_data(test_data)
        assert encrypted != str(test_data)
        assert isinstance(encrypted, str)
        
        decrypted = contact_manager._decrypt_data(encrypted)
        assert decrypted == test_data
    
    def test_add_contact(self, contact_manager, sample_contact):
        """Test adding a contact."""
        success = contact_manager.add_contact(sample_contact)
        
        assert success
        assert sample_contact.contact_id in contact_manager.contacts
        assert contact_manager.contacts[sample_contact.contact_id] == sample_contact
    
    def test_add_contact_validation(self, contact_manager):
        """Test contact validation."""
        # Contact without email
        invalid_contact = Contact(
            contact_id="INVALID",
            name="Invalid",
            role=ContactRole.PRIMARY_ONCALL,
            email=""
        )
        
        success = contact_manager.add_contact(invalid_contact)
        assert not success
        assert "INVALID" not in contact_manager.contacts
    
    def test_remove_contact(self, contact_manager, sample_contact):
        """Test removing a contact."""
        contact_manager.add_contact(sample_contact)
        
        success = contact_manager.remove_contact(sample_contact.contact_id)
        assert success
        assert sample_contact.contact_id not in contact_manager.contacts
        
        # Try removing non-existent
        success = contact_manager.remove_contact("NONEXISTENT")
        assert not success
    
    def test_get_contact(self, contact_manager, sample_contact):
        """Test getting a contact."""
        contact_manager.add_contact(sample_contact)
        
        retrieved = contact_manager.get_contact(sample_contact.contact_id)
        assert retrieved == sample_contact
        
        none_contact = contact_manager.get_contact("NONEXISTENT")
        assert none_contact is None
    
    def test_get_contacts_by_role(self, contact_manager):
        """Test getting contacts by role."""
        # Add multiple contacts
        contact1 = Contact(
            contact_id="C1",
            name="Contact 1",
            role=ContactRole.PRIMARY_ONCALL,
            email="c1@example.com"
        )
        contact2 = Contact(
            contact_id="C2",
            name="Contact 2",
            role=ContactRole.PRIMARY_ONCALL,
            email="c2@example.com"
        )
        contact3 = Contact(
            contact_id="C3",
            name="Contact 3",
            role=ContactRole.ENGINEERING_LEAD,
            email="c3@example.com"
        )
        
        contact_manager.add_contact(contact1)
        contact_manager.add_contact(contact2)
        contact_manager.add_contact(contact3)
        
        oncall_contacts = contact_manager.get_contacts_by_role(ContactRole.PRIMARY_ONCALL)
        assert len(oncall_contacts) == 2
        assert contact1 in oncall_contacts
        assert contact2 in oncall_contacts
        
        lead_contacts = contact_manager.get_contacts_by_role(ContactRole.ENGINEERING_LEAD)
        assert len(lead_contacts) == 1
        assert contact3 in lead_contacts
    
    def test_create_rotation(self, contact_manager, sample_rotation):
        """Test creating a rotation."""
        # Add contacts first
        for contact_id in sample_rotation.contacts:
            contact = Contact(
                contact_id=contact_id,
                name=f"Contact {contact_id}",
                role=ContactRole.PRIMARY_ONCALL,
                email=f"{contact_id}@example.com"
            )
            contact_manager.add_contact(contact)
        
        success = contact_manager.create_rotation(sample_rotation)
        assert success
        assert sample_rotation.rotation_id in contact_manager.rotations
    
    def test_create_rotation_validation(self, contact_manager, sample_rotation):
        """Test rotation validation."""
        # Try creating rotation with non-existent contacts
        success = contact_manager.create_rotation(sample_rotation)
        assert not success
        assert sample_rotation.rotation_id not in contact_manager.rotations
    
    def test_get_current_on_call(self, contact_manager):
        """Test getting current on-call contact."""
        # Set up contacts and rotation
        contacts = []
        for i in range(3):
            contact = Contact(
                contact_id=f"C{i}",
                name=f"Contact {i}",
                role=ContactRole.PRIMARY_ONCALL,
                email=f"c{i}@example.com"
            )
            contacts.append(contact)
            contact_manager.add_contact(contact)
        
        rotation = OnCallRotation(
            rotation_id="ROT-TEST",
            name="Test Rotation",
            schedule_type=RotationSchedule.WEEKLY,
            contacts=[c.contact_id for c in contacts],
            start_date=datetime.utcnow() - timedelta(days=8),  # Started 8 days ago
            rotation_interval_days=7
        )
        contact_manager.create_rotation(rotation)
        
        current = contact_manager.get_current_on_call("ROT-TEST")
        assert current is not None
        # After 8 days with 7-day rotation, should be on second contact
        assert current.contact_id == "C1"
    
    def test_rotation_override(self, contact_manager):
        """Test rotation override."""
        # Set up contacts
        contact1 = Contact(
            contact_id="C1",
            name="Regular",
            role=ContactRole.PRIMARY_ONCALL,
            email="regular@example.com"
        )
        contact2 = Contact(
            contact_id="C2",
            name="Override",
            role=ContactRole.PRIMARY_ONCALL,
            email="override@example.com"
        )
        contact_manager.add_contact(contact1)
        contact_manager.add_contact(contact2)
        
        # Create rotation
        rotation = OnCallRotation(
            rotation_id="ROT-OVERRIDE",
            name="Override Test",
            schedule_type=RotationSchedule.WEEKLY,
            contacts=["C1"]
        )
        contact_manager.create_rotation(rotation)
        
        # Set override
        override_until = datetime.utcnow() + timedelta(hours=24)
        success = contact_manager.set_override("ROT-OVERRIDE", "C2", override_until)
        assert success
        
        # Check current on-call
        current = contact_manager.get_current_on_call("ROT-OVERRIDE")
        assert current.contact_id == "C2"
    
    def test_get_next_on_call(self, contact_manager):
        """Test getting next on-call contact."""
        # Set up contacts
        contacts = []
        for i in range(3):
            contact = Contact(
                contact_id=f"C{i}",
                name=f"Contact {i}",
                role=ContactRole.PRIMARY_ONCALL,
                email=f"c{i}@example.com"
            )
            contacts.append(contact)
            contact_manager.add_contact(contact)
        
        rotation = OnCallRotation(
            rotation_id="ROT-NEXT",
            name="Next Test",
            schedule_type=RotationSchedule.WEEKLY,
            contacts=[c.contact_id for c in contacts],
            start_date=datetime.utcnow()
        )
        contact_manager.create_rotation(rotation)
        
        current = contact_manager.get_current_on_call("ROT-NEXT")
        next_contact = contact_manager.get_next_on_call("ROT-NEXT")
        
        assert current.contact_id == "C0"
        assert next_contact.contact_id == "C1"
    
    @pytest.mark.asyncio
    async def test_verify_contact(self, contact_manager, sample_contact):
        """Test contact verification."""
        contact_manager.add_contact(sample_contact)
        
        verified = await contact_manager.verify_contact(sample_contact.contact_id)
        assert verified
        
        updated = contact_manager.get_contact(sample_contact.contact_id)
        assert updated.verified
        assert updated.last_verified is not None
    
    def test_get_escalation_chain(self, contact_manager):
        """Test getting escalation chain."""
        # Create contacts for different roles
        roles_contacts = [
            (ContactRole.PRIMARY_ONCALL, "Primary"),
            (ContactRole.SECONDARY_ONCALL, "Secondary"),
            (ContactRole.ENGINEERING_LEAD, "Lead"),
            (ContactRole.DIRECTOR, "Director")
        ]
        
        for i, (role, name) in enumerate(roles_contacts):
            contact = Contact(
                contact_id=f"ESC-{i}",
                name=name,
                role=role,
                email=f"{name.lower()}@example.com"
            )
            contact_manager.add_contact(contact)
        
        chain = contact_manager.get_escalation_chain(ContactRole.SECONDARY_ONCALL)
        
        assert len(chain) == 3  # Secondary, Lead, Director
        assert chain[0].name == "Secondary"
        assert chain[1].name == "Lead"
        assert chain[2].name == "Director"
    
    def test_get_notification_list(self, contact_manager):
        """Test getting notification list by severity."""
        # Add contacts
        oncall = Contact(
            contact_id="ON",
            name="OnCall",
            role=ContactRole.PRIMARY_ONCALL,
            email="oncall@example.com"
        )
        lead = Contact(
            contact_id="LEAD",
            name="Lead",
            role=ContactRole.ENGINEERING_LEAD,
            email="lead@example.com"
        )
        contact_manager.add_contact(oncall)
        contact_manager.add_contact(lead)
        
        # Test critical severity
        critical_list = contact_manager.get_notification_list("critical")
        assert len(critical_list) >= 2
        
        # Test low severity
        low_list = contact_manager.get_notification_list("low")
        assert len(low_list) == 1
    
    def test_export_contact_list(self, contact_manager, sample_contact):
        """Test exporting contact list."""
        contact_manager.add_contact(sample_contact)
        
        # Export without sensitive data
        export = contact_manager.export_contact_list(include_sensitive=False)
        assert len(export["contacts"]) == 1
        assert export["contacts"][0]["email"] == "***@***.***"
        assert "phone" not in export["contacts"][0]
        
        # Export with sensitive data
        export_sensitive = contact_manager.export_contact_list(include_sensitive=True)
        assert export_sensitive["contacts"][0]["email"] == sample_contact.email
        assert export_sensitive["contacts"][0]["phone"] == sample_contact.phone
    
    def test_get_contact_stats(self, contact_manager):
        """Test getting contact statistics."""
        # Add some contacts
        for i in range(5):
            contact = Contact(
                contact_id=f"STAT-{i}",
                name=f"Contact {i}",
                role=ContactRole.PRIMARY_ONCALL if i < 3 else ContactRole.ENGINEERING_LEAD,
                email=f"contact{i}@example.com",
                active=i < 4,
                verified=i < 2
            )
            contact_manager.add_contact(contact)
        
        stats = contact_manager.get_contact_stats()
        
        assert stats["total_contacts"] == 5
        assert stats["active_contacts"] == 4
        assert stats["verified_contacts"] == 2
        assert stats["contacts_by_role"]["primary_oncall"] == 3
        assert stats["contacts_by_role"]["engineering_lead"] == 2
    
    def test_persistence(self, temp_storage):
        """Test contact persistence across manager instances."""
        key = Fernet.generate_key().decode()
        
        # Create first manager and add data
        manager1 = ContactManager(encryption_key=key, storage_path=temp_storage)
        contact = Contact(
            contact_id="PERSIST",
            name="Persistent",
            role=ContactRole.PRIMARY_ONCALL,
            email="persist@example.com"
        )
        manager1.add_contact(contact)
        
        # Create second manager with same key and storage
        manager2 = ContactManager(encryption_key=key, storage_path=temp_storage)
        
        # Check data was loaded
        loaded_contact = manager2.get_contact("PERSIST")
        assert loaded_contact is not None
        assert loaded_contact.name == "Persistent"
        assert loaded_contact.email == "persist@example.com"