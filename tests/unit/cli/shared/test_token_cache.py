"""Tests for CLI token cache module."""

import json
import os
import platform
from unittest.mock import patch

import pytest

from statgpt.cli.shared.token_cache import CachedToken, TokenCache


class TestCachedToken:
    """Tests for CachedToken dataclass."""

    def test_not_expired(self):
        """Token should not be expired when expires_at is in the future."""
        with patch("time.time", return_value=1000.0):
            token = CachedToken(
                access_token="test_token",
                expires_at=1120.0,  # 120 seconds in the future
                provider="test",
            )
            assert token.is_expired() is False

    def test_expired(self):
        """Token should be expired when expires_at is in the past."""
        with patch("time.time", return_value=1000.0):
            token = CachedToken(
                access_token="test_token",
                expires_at=990.0,  # 10 seconds in the past
                provider="test",
            )
            assert token.is_expired() is True

    def test_expired_within_buffer(self):
        """Token should be expired when within buffer period."""
        with patch("time.time", return_value=1000.0):
            token = CachedToken(
                access_token="test_token",
                expires_at=1030.0,  # 30 seconds in the future
                provider="test",
            )
            # Default buffer is 60 seconds, so token expiring in 30s is considered expired
            assert token.is_expired(buffer_seconds=60) is True
            # But with smaller buffer, it's not expired
            assert token.is_expired(buffer_seconds=20) is False

    def test_expired_custom_buffer(self):
        """Token expiration should respect custom buffer."""
        with patch("time.time", return_value=1000.0):
            token = CachedToken(
                access_token="test_token",
                expires_at=1010.0,  # 10 seconds in the future
                provider="test",
            )
            assert token.is_expired(buffer_seconds=5) is False
            assert token.is_expired(buffer_seconds=15) is True

    def test_has_refresh_token_true(self):
        """has_refresh_token should return True when refresh token exists."""
        token = CachedToken(
            access_token="test_token",
            expires_at=1000.0,
            provider="test",
            refresh_token="refresh_token",
        )
        assert token.has_refresh_token() is True

    def test_has_refresh_token_false(self):
        """has_refresh_token should return False when refresh token is None."""
        token = CachedToken(
            access_token="test_token",
            expires_at=1000.0,
            provider="test",
            refresh_token=None,
        )
        assert token.has_refresh_token() is False

    def test_to_dict_without_refresh_token(self):
        """to_dict should not include refresh_token if None."""
        token = CachedToken(
            access_token="test_token",
            expires_at=1000.0,
            provider="test",
        )
        data = token.to_dict()
        assert data == {
            "access_token": "test_token",
            "expires_at": 1000.0,
            "provider": "test",
        }
        assert "refresh_token" not in data

    def test_to_dict_with_refresh_token(self):
        """to_dict should include refresh_token if present."""
        token = CachedToken(
            access_token="test_token",
            expires_at=1000.0,
            provider="test",
            refresh_token="refresh_token",
        )
        data = token.to_dict()
        assert data == {
            "access_token": "test_token",
            "expires_at": 1000.0,
            "provider": "test",
            "refresh_token": "refresh_token",
        }

    def test_from_dict_without_refresh_token(self):
        """from_dict should create token without refresh_token."""
        data = {
            "access_token": "test_token",
            "expires_at": 1000.0,
            "provider": "test",
        }
        token = CachedToken.from_dict(data)
        assert token.access_token == "test_token"
        assert token.expires_at == 1000.0
        assert token.provider == "test"
        assert token.refresh_token is None

    def test_from_dict_with_refresh_token(self):
        """from_dict should create token with refresh_token."""
        data = {
            "access_token": "test_token",
            "expires_at": 1000.0,
            "provider": "test",
            "refresh_token": "refresh_token",
        }
        token = CachedToken.from_dict(data)
        assert token.refresh_token == "refresh_token"

    def test_serialization_roundtrip(self):
        """Token should survive to_dict/from_dict roundtrip."""
        original = CachedToken(
            access_token="test_token",
            expires_at=1000.0,
            provider="test",
            refresh_token="refresh_token",
        )
        restored = CachedToken.from_dict(original.to_dict())
        assert restored.access_token == original.access_token
        assert restored.expires_at == original.expires_at
        assert restored.provider == original.provider
        assert restored.refresh_token == original.refresh_token


class TestTokenCache:
    """Tests for TokenCache class."""

    @pytest.fixture
    def token_cache(self, tmp_path, monkeypatch):
        """TokenCache with temp directory."""
        monkeypatch.setattr("statgpt.cli.shared.token_cache._get_data_dir", lambda: tmp_path)
        return TokenCache()

    def test_get_token_none_cached(self, token_cache):
        """get_token should return None when no token is cached."""
        assert token_cache.get_token() is None

    def test_save_and_get_token(self, token_cache):
        """Saved token should be retrievable."""
        with patch("time.time", return_value=1000.0):
            token_cache.save_token(
                access_token="test_token",
                expires_in=3600,
                provider="test",
            )

        with patch("time.time", return_value=1000.0):
            token = token_cache.get_token()
            assert token is not None
            assert token.access_token == "test_token"
            assert token.provider == "test"
            assert token.expires_at == 4600.0  # 1000 + 3600

    def test_save_and_get_token_with_refresh(self, token_cache):
        """Saved token with refresh token should be retrievable."""
        with patch("time.time", return_value=1000.0):
            token_cache.save_token(
                access_token="test_token",
                expires_in=3600,
                provider="test",
                refresh_token="refresh_token",
            )

        with patch("time.time", return_value=1000.0):
            token = token_cache.get_token()
            assert token is not None
            assert token.refresh_token == "refresh_token"

    def test_get_expired_token_returns_none(self, token_cache):
        """get_token should return None for expired token without refresh token."""
        with patch("time.time", return_value=1000.0):
            token_cache.save_token(
                access_token="test_token",
                expires_in=100,
                provider="test",
            )

        # Move time forward past expiration
        with patch("time.time", return_value=2000.0):
            token = token_cache.get_token()
            assert token is None

    def test_get_expired_token_with_refresh_returns_none_but_keeps_file(
        self, token_cache, tmp_path
    ):
        """get_token returns None for expired token with refresh, but keeps file."""
        with patch("time.time", return_value=1000.0):
            token_cache.save_token(
                access_token="test_token",
                expires_in=100,
                provider="test",
                refresh_token="refresh_token",
            )

        # Move time forward past expiration
        with patch("time.time", return_value=2000.0):
            token = token_cache.get_token()
            assert token is None
            # File should still exist (for potential refresh)
            assert (tmp_path / "token_cache.json").exists()

    def test_get_token_raw_returns_expired_token(self, token_cache):
        """get_token_raw should return token even if expired."""
        with patch("time.time", return_value=1000.0):
            token_cache.save_token(
                access_token="test_token",
                expires_in=100,
                provider="test",
            )

        # Move time forward past expiration
        with patch("time.time", return_value=2000.0):
            token = token_cache.get_token_raw()
            assert token is not None
            assert token.access_token == "test_token"

    def test_clear_token(self, token_cache, tmp_path):
        """clear should remove the cache file."""
        with patch("time.time", return_value=1000.0):
            token_cache.save_token(
                access_token="test_token",
                expires_in=3600,
                provider="test",
            )
            assert (tmp_path / "token_cache.json").exists()

        token_cache.clear()
        assert not (tmp_path / "token_cache.json").exists()

    def test_clear_nonexistent_file(self, token_cache):
        """clear should not raise error if file doesn't exist."""
        token_cache.clear()  # Should not raise

    def test_corrupt_cache_file_returns_none(self, token_cache, tmp_path):
        """Corrupt cache file should return None and be cleared."""
        cache_file = tmp_path / "token_cache.json"
        tmp_path.mkdir(exist_ok=True)
        cache_file.write_text("invalid json {{{")

        token = token_cache.get_token_raw()
        assert token is None
        # File should be cleared
        assert not cache_file.exists()

    def test_cache_file_missing_keys_returns_none(self, token_cache, tmp_path):
        """Cache file with missing keys should return None and be cleared."""
        cache_file = tmp_path / "token_cache.json"
        tmp_path.mkdir(exist_ok=True)
        cache_file.write_text(json.dumps({"access_token": "test"}))  # Missing other keys

        token = token_cache.get_token_raw()
        assert token is None
        assert not cache_file.exists()

    def test_get_token_info(self, token_cache):
        """get_token_info should return metadata without access token."""
        with patch("time.time", return_value=1000.0):
            token_cache.save_token(
                access_token="test_token",
                expires_in=3600,
                provider="test",
            )

        with patch("time.time", return_value=1100.0):
            info = token_cache.get_token_info()
            assert info is not None
            assert info["provider"] == "test"
            assert info["expires_in_seconds"] == 3500  # 3600 - 100
            assert "access_token" not in info

    def test_get_token_info_no_token(self, token_cache):
        """get_token_info should return None when no valid token."""
        info = token_cache.get_token_info()
        assert info is None

    @pytest.mark.skipif(
        platform.system() == "Windows", reason="File permissions not supported on Windows"
    )
    def test_file_permissions(self, token_cache, tmp_path):
        """Cache file should have restricted permissions (0o600)."""

        with patch("time.time", return_value=1000.0):
            token_cache.save_token(
                access_token="test_token",
                expires_in=3600,
                provider="test",
            )

        cache_file = tmp_path / "token_cache.json"
        mode = os.stat(cache_file).st_mode & 0o777
        assert mode == 0o600

    def test_directory_created_if_not_exists(self, tmp_path, monkeypatch):
        """Cache directory should be created if it doesn't exist."""
        cache_dir = tmp_path / "subdir"
        monkeypatch.setattr("statgpt.cli.shared.token_cache._get_data_dir", lambda: cache_dir)
        token_cache = TokenCache()

        with patch("time.time", return_value=1000.0):
            token_cache.save_token(
                access_token="test_token",
                expires_in=3600,
                provider="test",
            )

        assert cache_dir.exists()
        assert (cache_dir / "token_cache.json").exists()
