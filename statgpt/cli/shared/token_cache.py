"""Token cache for persisting authentication tokens."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from statgpt.cli.settings import cli_settings


@dataclass
class CachedToken:
    """Cached authentication token with metadata."""

    access_token: str
    expires_at: float  # Unix timestamp
    provider: str
    refresh_token: str | None = None

    def is_expired(self, buffer_seconds: int = 60) -> bool:
        """Check if token is expired or about to expire.

        Args:
            buffer_seconds: Consider expired if within this many seconds of expiry

        Returns:
            True if token is expired or about to expire
        """
        return time.time() >= (self.expires_at - buffer_seconds)

    def has_refresh_token(self) -> bool:
        """Check if a refresh token is available."""
        return self.refresh_token is not None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = {
            "access_token": self.access_token,
            "expires_at": self.expires_at,
            "provider": self.provider,
        }
        if self.refresh_token:
            data["refresh_token"] = self.refresh_token
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create from dictionary."""
        return cls(
            access_token=data["access_token"],
            expires_at=data["expires_at"],
            provider=data["provider"],
            refresh_token=data.get("refresh_token"),
        )


def _get_data_dir() -> Path:
    """Get CLI data directory from settings."""
    return Path(cli_settings.cli_data_dir)


class TokenCache:
    """File-based token cache stored in CLI data directory."""

    @property
    def _cache_dir(self) -> Path:
        """Get cache directory path."""
        return _get_data_dir()

    @property
    def _cache_file(self) -> Path:
        """Get cache file path."""
        return self._cache_dir / "token_cache.json"

    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists with proper permissions."""
        self._cache_dir.mkdir(mode=0o700, exist_ok=True)

    def get_token(self) -> CachedToken | None:
        """Get cached token if it exists and is valid.

        Returns:
            CachedToken if valid token exists, None otherwise
        """
        token = self.get_token_raw()
        if token is None:
            return None

        if token.is_expired():
            # Don't clear if there's a refresh token - caller may want to refresh
            if not token.has_refresh_token():
                self.clear()
            return None

        return token

    def get_token_raw(self) -> CachedToken | None:
        """Get cached token data regardless of expiration.

        Use this when you need to check for refresh tokens on expired tokens.

        Returns:
            CachedToken if exists (may be expired), None otherwise
        """
        if not self._cache_file.exists():
            return None

        try:
            with open(self._cache_file) as f:
                data = json.load(f)
            return CachedToken.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            # Invalid cache file, remove it
            self.clear()
            return None

    def save_token(
        self,
        access_token: str,
        expires_in: int,
        provider: str,
        refresh_token: str | None = None,
    ) -> CachedToken:
        """Save a token to cache.

        Args:
            access_token: The access token string
            expires_in: Token lifetime in seconds
            provider: Authentication provider name
            refresh_token: Optional refresh token for token renewal

        Returns:
            The cached token
        """
        self._ensure_cache_dir()

        expires_at = time.time() + expires_in
        token = CachedToken(
            access_token=access_token,
            expires_at=expires_at,
            provider=provider,
            refresh_token=refresh_token,
        )

        # Write with restricted permissions
        with open(self._cache_file, "w") as f:
            json.dump(token.to_dict(), f)

        # Ensure file has restricted permissions
        self._cache_file.chmod(0o600)

        return token

    def clear(self) -> None:
        """Clear the cached token."""
        if self._cache_file.exists():
            self._cache_file.unlink()

    def get_token_info(self) -> dict | None:
        """Get information about the cached token (without the token itself).

        Returns:
            Dictionary with token info or None if no valid token
        """
        token = self.get_token()
        if token is None:
            return None

        remaining = token.expires_at - time.time()
        return {
            "provider": token.provider,
            "expires_in_seconds": int(remaining),
            "expires_at": token.expires_at,
        }


# Global token cache instance
token_cache = TokenCache()
