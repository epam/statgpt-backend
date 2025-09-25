from dataclasses import dataclass

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jwt import InvalidTokenError

from admin_portal.auth.oidc import JwtTokenVerifier, TokenValidationError, TokenValidator
from admin_portal.settings.oidc_auth import oidc_auth_settings
from common.config import logger

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token", auto_error=oidc_auth_settings.oidc_auth_enabled
)


@dataclass
class User:
    name: str


async def require_jwt_auth(token: str = Depends(oauth2_scheme)) -> User:

    if oidc_auth_settings.oidc_auth_enabled:
        try:
            payload = JwtTokenVerifier.create().verify(token)
            try:
                TokenValidator.from_config().validate(payload.raw)
            except TokenValidationError as e:
                logger.info(f"Unauthorized token: {str(e)}")
                raise HTTPException(status_code=403, detail=str(e))

            return User(payload.username)
        except InvalidTokenError as e:
            logger.info(f"Invalid Bearer token: {str(e)}")
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    else:
        return User("Anonymous")
