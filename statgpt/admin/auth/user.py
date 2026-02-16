from dataclasses import dataclass

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jwt import InvalidTokenError

from statgpt.admin.audit.context import set_audit_context
from statgpt.admin.auth.oidc import JwtTokenVerifier, TokenValidationError, TokenValidator
from statgpt.admin.settings.oidc_auth import oidc_auth_settings
from statgpt.common.config import logger

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token", auto_error=oidc_auth_settings.oidc_auth_enabled
)


@dataclass
class User:
    name: str
    id: str | None = None


async def require_jwt_auth(token: str = Depends(oauth2_scheme)) -> User:

    if oidc_auth_settings.oidc_auth_enabled:
        try:
            payload = JwtTokenVerifier.create().verify(token)
            try:
                TokenValidator.from_config().validate(payload.raw)
            except TokenValidationError as e:
                logger.info(f"Unauthorized token: {str(e)}")
                raise HTTPException(status_code=403, detail=str(e))

            user = User(name=payload.username, id=payload.user_id)
            set_audit_context(
                performed_by=payload.user_id,
                performed_by_name=payload.performed_by_name,
            )
            return user
        except InvalidTokenError as e:
            logger.info(f"Invalid Bearer token: {str(e)}")
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    else:
        user = User("Anonymous")
        set_audit_context(performed_by=user.id, performed_by_name=user.name)
        return user
