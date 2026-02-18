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


@dataclass(frozen=True)
class User:
    id: str
    username: str
    name: str


async def _require_jwt_auth(token: str = Depends(oauth2_scheme)) -> User:
    if oidc_auth_settings.oidc_auth_enabled:
        try:
            payload = JwtTokenVerifier.create().verify(token)
            try:
                TokenValidator.from_config().validate(payload.raw)
            except TokenValidationError as e:
                logger.info(f"Unauthorized token: {str(e)}")
                raise HTTPException(status_code=403, detail=str(e))

            return User(
                id=payload.user_id, username=payload.username, name=payload.performed_by_name
            )
        except InvalidTokenError as e:
            logger.warning(f"Invalid Bearer token: {str(e)}")
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    else:
        return User(id="Anonymous", username="Anonymous", name="Anonymous")


async def require_jwt_auth(user: User = Depends(_require_jwt_auth)) -> User:
    set_audit_context(performed_by=user.id, performed_by_name=user.name)
    return user
