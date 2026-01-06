"""Tests for CLI settings module."""

import os

from statgpt.cli.settings import CLISettings


def make_settings(**kwargs):
    """Create CLISettings without loading from .env file."""
    return CLISettings(_env_file=None, **kwargs)


class TestCLISettingsDefaults:
    """Tests for CLISettings default values."""

    def test_default_admin_url(self, clean_cli_env):
        """admin_url should default to localhost:8000."""
        settings = make_settings()
        assert settings.admin_url == "http://localhost:8000"

    def test_default_auth_provider(self, clean_cli_env):
        """auth_provider should default to azure."""
        settings = make_settings()
        assert settings.auth_provider == "azure"

    def test_default_log_level(self, clean_cli_env):
        """log_level should default to INFO."""
        settings = make_settings()
        assert settings.log_level == "INFO"

    def test_optional_fields_none(self, clean_cli_env):
        """Optional fields should be None by default."""
        settings = make_settings()
        assert settings.config_dir is None
        assert settings.dial_url is None
        assert settings.dial_api_key is None
        assert settings.data_dir is None


class TestCLISettingsEnvOverride:
    """Tests for CLISettings environment variable overrides."""

    def test_admin_url_from_env(self, clean_cli_env, monkeypatch):
        """admin_url should be overridden by environment variable."""
        monkeypatch.setenv("STATGPT_CLI_ADMIN_URL", "http://custom:9000")
        settings = make_settings()
        assert settings.admin_url == "http://custom:9000"

    def test_auth_provider_from_env(self, clean_cli_env, monkeypatch):
        """auth_provider should be overridden by environment variable."""
        monkeypatch.setenv("STATGPT_CLI_AUTH_PROVIDER", "keycloak")
        settings = make_settings()
        assert settings.auth_provider == "keycloak"

    def test_log_level_from_env(self, clean_cli_env, monkeypatch):
        """log_level should be overridden by environment variable."""
        monkeypatch.setenv("STATGPT_CLI_LOG_LEVEL", "DEBUG")
        settings = make_settings()
        assert settings.log_level == "DEBUG"

    def test_config_dir_from_env(self, clean_cli_env, monkeypatch):
        """config_dir should be set from environment variable."""
        monkeypatch.setenv("STATGPT_CLI_CONFIG_DIR", "/custom/config")
        settings = make_settings()
        assert settings.config_dir == "/custom/config"

    def test_dial_settings_from_env(self, clean_cli_env, monkeypatch):
        """DIAL settings should be set from environment variables."""
        monkeypatch.setenv("STATGPT_CLI_DIAL_URL", "http://dial:8080")
        monkeypatch.setenv("STATGPT_CLI_DIAL_API_KEY", "secret-key")
        settings = make_settings()
        assert settings.dial_url == "http://dial:8080"
        assert settings.dial_api_key == "secret-key"

    def test_azure_auth_settings_from_env(self, clean_cli_env, monkeypatch):
        """Azure auth settings should be set from environment variables."""
        monkeypatch.setenv("STATGPT_CLI_AUTH_AZURE_CLIENT_ID", "client-123")
        monkeypatch.setenv(
            "STATGPT_CLI_AUTH_AZURE_AUTHORITY", "https://login.microsoftonline.com/tenant"
        )
        monkeypatch.setenv("STATGPT_CLI_AUTH_AZURE_SCOPE", "api://scope/.default")
        settings = make_settings()
        assert settings.auth_azure_client_id == "client-123"
        assert settings.auth_azure_authority == "https://login.microsoftonline.com/tenant"
        assert settings.auth_azure_scope == "api://scope/.default"


class TestCLIDataDir:
    """Tests for cli_data_dir property."""

    def test_cli_data_dir_default(self, clean_cli_env):
        """cli_data_dir should default to ~/.statgpt."""
        settings = make_settings()
        expected = os.path.expanduser("~/.statgpt")
        assert settings.cli_data_dir == expected

    def test_cli_data_dir_custom(self, clean_cli_env, monkeypatch):
        """cli_data_dir should use custom data_dir if set."""
        monkeypatch.setenv("STATGPT_CLI_DATA_DIR", "~/custom/data")
        settings = make_settings()
        expected = os.path.expanduser("~/custom/data")
        assert settings.cli_data_dir == expected

    def test_cli_data_dir_expands_tilde(self, clean_cli_env, monkeypatch):
        """cli_data_dir should expand ~ in path."""
        monkeypatch.setenv("STATGPT_CLI_DATA_DIR", "~/.custom-statgpt")
        settings = make_settings()
        assert "~" not in settings.cli_data_dir
        assert settings.cli_data_dir.startswith("/")


class TestGetSettingSource:
    """Tests for get_setting_source method."""

    def test_source_from_env(self, clean_cli_env, monkeypatch):
        """Should return 'env' when value is from environment variable."""
        monkeypatch.setenv("STATGPT_CLI_ADMIN_URL", "http://custom:9000")
        settings = make_settings()
        assert settings.get_setting_source("admin_url") == "env"

    def test_source_default(self, clean_cli_env):
        """Should return 'default' when using default value."""
        settings = make_settings()
        assert settings.get_setting_source("admin_url") == "default"

    def test_source_not_set(self, clean_cli_env):
        """Should return 'not set' when optional field is None."""
        settings = make_settings()
        assert settings.get_setting_source("config_dir") == "not set"

    def test_source_unknown_field(self, clean_cli_env):
        """Should return 'unknown' for non-existent field."""
        settings = make_settings()
        assert settings.get_setting_source("nonexistent_field") == "unknown"

    def test_source_env_for_optional_field(self, clean_cli_env, monkeypatch):
        """Should return 'env' when optional field is set via env."""
        monkeypatch.setenv("STATGPT_CLI_CONFIG_DIR", "/custom/config")
        settings = make_settings()
        assert settings.get_setting_source("config_dir") == "env"


class TestGetAuthSettingsForProvider:
    """Tests for get_auth_settings_for_provider method."""

    def test_azure_provider_settings(self, clean_cli_env, monkeypatch):
        """Should return all azure auth settings."""
        monkeypatch.setenv("STATGPT_CLI_AUTH_AZURE_CLIENT_ID", "client-123")
        monkeypatch.setenv("STATGPT_CLI_AUTH_AZURE_AUTHORITY", "https://authority")
        monkeypatch.setenv("STATGPT_CLI_AUTH_AZURE_SCOPE", "scope")
        monkeypatch.setenv("STATGPT_CLI_AUTH_AZURE_CLIENT_SECRET", "secret")
        monkeypatch.setenv("STATGPT_CLI_AUTH_AZURE_USERNAME", "user")
        monkeypatch.setenv("STATGPT_CLI_AUTH_AZURE_PASSWORD", "pass")

        settings = make_settings()
        azure_settings = settings.get_auth_settings_for_provider("azure")

        assert azure_settings["client_id"] == "client-123"
        assert azure_settings["authority"] == "https://authority"
        assert azure_settings["scope"] == "scope"
        assert azure_settings["client_secret"] == "secret"
        assert azure_settings["username"] == "user"
        assert azure_settings["password"] == "pass"

    def test_azure_provider_settings_partial(self, clean_cli_env, monkeypatch):
        """Should return None for unset azure settings."""
        monkeypatch.setenv("STATGPT_CLI_AUTH_AZURE_CLIENT_ID", "client-123")
        settings = make_settings()
        azure_settings = settings.get_auth_settings_for_provider("azure")

        assert azure_settings["client_id"] == "client-123"
        assert azure_settings["authority"] is None
        assert azure_settings["scope"] is None

    def test_unknown_provider_empty(self, clean_cli_env):
        """Should return empty dict for unknown provider."""
        settings = make_settings()
        unknown_settings = settings.get_auth_settings_for_provider("unknown")
        assert unknown_settings == {}

    def test_azure_settings_keys(self, clean_cli_env):
        """Should have correct keys in azure settings dict."""
        settings = make_settings()
        azure_settings = settings.get_auth_settings_for_provider("azure")

        expected_keys = {
            "client_id",
            "authority",
            "scope",
            "client_secret",
            "username",
            "password",
        }
        assert set(azure_settings.keys()) == expected_keys
