"""
Two-Factor Authentication API endpoints.
Provides endpoints for 2FA setup, verification, and management.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from genesis.api.auth import get_current_user, get_db_session
from genesis.data.models_db import TwoFAAttempt, User
from genesis.security.audit_logger import AuditLogger
from genesis.security.password_manager import SecurePasswordManager
from genesis.security.totp_manager import TOTPManager
from genesis.security.vault_client import VaultClient

logger = structlog.get_logger(__name__)
audit_logger = AuditLogger()

# Rate limiter for 2FA attempts
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/auth/2fa", tags=["2FA"])


class Setup2FARequest(BaseModel):
    """Request model for 2FA setup initiation."""
    pass


class Setup2FAResponse(BaseModel):
    """Response model for 2FA setup."""
    qr_code: str = Field(..., description="Base64 encoded QR code image")
    secret: str = Field(..., description="TOTP secret for manual entry")
    backup_codes: list[str] = Field(..., description="Backup recovery codes")


class Enable2FARequest(BaseModel):
    """Request model for enabling 2FA."""
    verification_code: str = Field(..., min_length=6, max_length=6, description="TOTP code from authenticator")


class Verify2FARequest(BaseModel):
    """Request model for 2FA verification."""
    code: str = Field(..., description="TOTP or backup code")


class Disable2FARequest(BaseModel):
    """Request model for disabling 2FA."""
    password: str = Field(..., description="User password for verification")
    code: str = Field(..., description="Current TOTP code")


class RegenerateBackupCodesRequest(BaseModel):
    """Request model for regenerating backup codes."""
    current_code: str = Field(..., description="Current TOTP code for verification")


async def get_totp_manager() -> TOTPManager:
    """Get TOTP manager instance with Vault integration."""
    vault = VaultClient()
    await vault.initialize()
    return TOTPManager(vault_manager=vault)


async def log_2fa_attempt(
    db: AsyncSession,
    user_id: int,
    ip_address: str,
    code_used: str,
    success: bool
) -> None:
    """
    Log 2FA attempt for rate limiting and auditing.
    
    Args:
        db: Database session
        user_id: User ID
        ip_address: Client IP address
        code_used: Code attempted (masked)
        success: Whether attempt succeeded
    """
    attempt = TwoFAAttempt(
        user_id=user_id,
        ip_address=ip_address,
        code_used=code_used[:3] + "***",  # Mask code for security
        success=success,
        attempted_at=datetime.now(UTC)
    )
    db.add(attempt)
    await db.commit()


async def check_rate_limit(
    db: AsyncSession,
    user_id: int,
    ip_address: str
) -> bool:
    """
    Check if user has exceeded 2FA attempt rate limit.
    
    Args:
        db: Database session
        user_id: User ID
        ip_address: Client IP address
        
    Returns:
        True if within limits, False if exceeded
    """
    # Check attempts in last minute
    one_minute_ago = datetime.now(UTC) - timedelta(minutes=1)

    result = await db.execute(
        select(TwoFAAttempt).where(
            TwoFAAttempt.user_id == user_id,
            TwoFAAttempt.attempted_at >= one_minute_ago,
            TwoFAAttempt.success == False
        )
    )

    failed_attempts = result.scalars().all()

    # Max 5 failed attempts per minute
    if len(failed_attempts) >= 5:
        logger.warning(
            "2FA rate limit exceeded",
            user_id=user_id,
            ip_address=ip_address,
            attempts=len(failed_attempts)
        )
        return False

    return True


@router.post("/setup", response_model=Setup2FAResponse)
async def setup_2fa(
    request: Setup2FARequest,
    current_user: User = Depends(get_current_user),
    totp_manager: TOTPManager = Depends(get_totp_manager)
) -> Setup2FAResponse:
    """
    Initiate 2FA setup for current user.
    
    Returns QR code and backup codes for 2FA setup.
    """
    try:
        # Check if already setup
        if current_user.two_fa_enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="2FA already enabled for this account"
            )

        # Generate 2FA setup
        setup_data = await totp_manager.setup_2fa(
            user_id=str(current_user.id),
            username=current_user.username
        )

        # Log audit event
        client_ip = request.client.host if request.client else "unknown"
        await audit_logger.log_event(
            event_type="2FA_SETUP_INITIATED",
            user_id=current_user.id,
            details={
                "username": current_user.username,
                "ip_address": client_ip
            }
        )

        logger.info(
            "2FA setup initiated",
            user_id=current_user.id,
            username=current_user.username
        )

        return Setup2FAResponse(
            qr_code=setup_data['qr_code'],
            secret=setup_data['secret'],
            backup_codes=setup_data['backup_codes']
        )

    except Exception as e:
        logger.error(
            "2FA setup failed",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to setup 2FA"
        )


@router.post("/enable")
async def enable_2fa(
    request: Enable2FARequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
    totp_manager: TOTPManager = Depends(get_totp_manager)
) -> dict[str, Any]:
    """
    Enable 2FA after successful verification.
    
    User must verify with a code from their authenticator app.
    """
    try:
        # Check if already enabled
        if current_user.two_fa_enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="2FA already enabled"
            )

        # Enable 2FA with verification
        success = await totp_manager.enable_2fa(
            user_id=str(current_user.id),
            verification_code=request.verification_code
        )

        if not success:
            await audit_logger.log_event(
                event_type="2FA_ENABLE_FAILED",
                user_id=current_user.id,
                details={"reason": "Invalid verification code"}
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid verification code"
            )

        # Update user record
        await db.execute(
            update(User).where(User.id == current_user.id).values(
                two_fa_enabled=True,
                two_fa_setup_at=datetime.now(UTC)
            )
        )
        await db.commit()

        # Log audit event
        await audit_logger.log_event(
            event_type="2FA_ENABLED",
            user_id=current_user.id,
            details={"username": current_user.username}
        )

        logger.info(
            "2FA enabled successfully",
            user_id=current_user.id
        )

        return {
            "status": "success",
            "message": "2FA has been enabled for your account"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to enable 2FA",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enable 2FA"
        )


@router.post("/verify")
@limiter.limit("10/minute")
async def verify_2fa(
    request: Request,
    verify_request: Verify2FARequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
    totp_manager: TOTPManager = Depends(get_totp_manager)
) -> dict[str, Any]:
    """
    Verify 2FA code during login or sensitive operations.
    
    Supports both TOTP codes and backup codes.
    """
    ip_address = get_remote_address(request)

    try:
        # Check rate limiting
        if not await check_rate_limit(db, current_user.id, ip_address):
            await audit_logger.log_event(
                event_type="2FA_RATE_LIMIT_EXCEEDED",
                user_id=current_user.id,
                details={"ip_address": ip_address}
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many attempts. Please try again later."
            )

        # Verify 2FA code
        is_valid, code_type = await totp_manager.validate_2fa(
            user_id=str(current_user.id),
            code=verify_request.code
        )

        # Log attempt
        await log_2fa_attempt(
            db=db,
            user_id=current_user.id,
            ip_address=ip_address,
            code_used=verify_request.code,
            success=is_valid
        )

        if not is_valid:
            await audit_logger.log_event(
                event_type="2FA_VERIFICATION_FAILED",
                user_id=current_user.id,
                details={
                    "ip_address": ip_address,
                    "code_type": code_type
                }
            )

            if code_type == "not_enabled":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="2FA not enabled for this account"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid 2FA code"
                )

        # Log successful verification
        await audit_logger.log_event(
            event_type="2FA_VERIFICATION_SUCCESS",
            user_id=current_user.id,
            details={
                "ip_address": ip_address,
                "code_type": code_type
            }
        )

        logger.info(
            "2FA verification successful",
            user_id=current_user.id,
            code_type=code_type
        )

        return {
            "status": "success",
            "code_type": code_type,
            "message": "2FA verification successful"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "2FA verification error",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="2FA verification failed"
        )


@router.post("/disable")
async def disable_2fa(
    request: Disable2FARequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
    totp_manager: TOTPManager = Depends(get_totp_manager)
) -> dict[str, Any]:
    """
    Disable 2FA for user account.
    
    Requires password and current TOTP code for security.
    """
    try:
        # Verify 2FA is enabled
        if not current_user.two_fa_enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="2FA not enabled for this account"
            )

        # Verify current TOTP code
        is_valid, _ = await totp_manager.validate_2fa(
            user_id=str(current_user.id),
            code=request.code
        )

        if not is_valid:
            await audit_logger.log_event(
                event_type="2FA_DISABLE_FAILED",
                user_id=current_user.id,
                details={"reason": "Invalid TOTP code"}
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid 2FA code"
            )

        # Verify password for additional security
        password_manager = SecurePasswordManager()
        if not password_manager.verify_password(data.password, current_user.password_hash):
            await audit_logger.log_event(
                event_type="2FA_DISABLE_FAILED",
                user_id=current_user.id,
                details={"reason": "Invalid password"}
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid password"
            )

        # Disable 2FA
        success = await totp_manager.disable_2fa(user_id=str(current_user.id))

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to disable 2FA"
            )

        # Update user record
        await db.execute(
            update(User).where(User.id == current_user.id).values(
                two_fa_enabled=False,
                two_fa_setup_at=None
            )
        )
        await db.commit()

        # Log audit event
        await audit_logger.log_event(
            event_type="2FA_DISABLED",
            user_id=current_user.id,
            details={"username": current_user.username}
        )

        logger.info(
            "2FA disabled successfully",
            user_id=current_user.id
        )

        return {
            "status": "success",
            "message": "2FA has been disabled for your account"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to disable 2FA",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to disable 2FA"
        )


@router.post("/regenerate-backup-codes")
async def regenerate_backup_codes(
    request: RegenerateBackupCodesRequest,
    current_user: User = Depends(get_current_user),
    totp_manager: TOTPManager = Depends(get_totp_manager)
) -> dict[str, Any]:
    """
    Generate new set of backup codes.
    
    Requires current TOTP code for verification.
    """
    try:
        # Verify 2FA is enabled
        if not current_user.two_fa_enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="2FA not enabled for this account"
            )

        # Verify current TOTP code
        is_valid, code_type = await totp_manager.validate_2fa(
            user_id=str(current_user.id),
            code=request.current_code
        )

        if not is_valid or code_type != "totp":
            await audit_logger.log_event(
                event_type="BACKUP_CODES_REGENERATION_FAILED",
                user_id=current_user.id,
                details={"reason": "Invalid TOTP code"}
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid TOTP code"
            )

        # Generate new backup codes
        new_codes = await totp_manager.regenerate_backup_codes(
            user_id=str(current_user.id)
        )

        if not new_codes:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate new backup codes"
            )

        # Log audit event
        await audit_logger.log_event(
            event_type="BACKUP_CODES_REGENERATED",
            user_id=current_user.id,
            details={
                "username": current_user.username,
                "codes_count": len(new_codes)
            }
        )

        logger.info(
            "Backup codes regenerated",
            user_id=current_user.id
        )

        return {
            "status": "success",
            "backup_codes": new_codes,
            "message": "New backup codes generated. Please save them securely."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to regenerate backup codes",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to regenerate backup codes"
        )


@router.get("/status")
async def get_2fa_status(
    current_user: User = Depends(get_current_user)
) -> dict[str, Any]:
    """
    Get 2FA status for current user.
    
    Returns whether 2FA is enabled and required.
    """
    return {
        "enabled": current_user.two_fa_enabled,
        "required": current_user.two_fa_required,
        "setup_at": current_user.two_fa_setup_at.isoformat() if current_user.two_fa_setup_at else None
    }


@router.post("/enforce-admin")
async def enforce_admin_2fa(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
) -> dict[str, Any]:
    """
    Enforce 2FA for all admin accounts.
    
    This endpoint should only be accessible to super admins.
    """
    try:
        # Check if user is super admin
        if current_user.role != "super_admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only super admins can enforce 2FA"
            )

        # Update all admin accounts to require 2FA
        result = await db.execute(
            update(User).where(User.role.in_(["admin", "super_admin"])).values(
                two_fa_required=True
            )
        )
        await db.commit()

        affected_users = result.rowcount

        # Log audit event
        await audit_logger.log_event(
            event_type="2FA_ENFORCED_FOR_ADMINS",
            user_id=current_user.id,
            details={
                "affected_users": affected_users,
                "enforced_by": current_user.username
            }
        )

        logger.info(
            "2FA enforced for admin accounts",
            enforced_by=current_user.id,
            affected_users=affected_users
        )

        return {
            "status": "success",
            "affected_users": affected_users,
            "message": f"2FA requirement enforced for {affected_users} admin accounts"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to enforce admin 2FA",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enforce admin 2FA"
        )
