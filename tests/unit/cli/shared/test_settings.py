"""Tests for CLI settings module."""

import os
import types
from typing import Union, get_args, get_origin

from statgpt.cli.settings import CLISettings, FieldMeta, SettingsSection


def make_settings(**kwargs):
    """Create CLISettings without loading from .env file."""
    return CLISettings(_env_file=None, **kwargs)


class TestCLISettingsDefaults:
    """Tests for CLISettings default values."""

    def test_default_admin_url(self, clean_cli_env):
        settings = make_settings()
        assert settings.admin_url == "http://localhost:8000"

    def test_auth_provider_optional(self, clean_cli_env):
        """auth_provider is optional for local development."""
        settings = make_settings()
        assert settings.auth_provider is None

    def test_default_log_level(self, clean_cli_env):
        settings = make_settings()
        assert settings.log_level == "INFO"

    def test_optional_fields_none(self, clean_cli_env):
        settings = make_settings()
        assert settings.config_dir is None
        assert settings.dial_url is None
        assert settings.dial_api_key is None
        assert settings.data_dir is None


class TestCLISettingsEnvOverride:
    """Tests for CLISettings environment variable overrides."""

    def test_admin_url_from_env(self, clean_cli_env, monkeypatch):
        monkeypatch.setenv("STATGPT_CLI_ADMIN_URL", "http://custom:9000")
        settings = make_settings()
        assert settings.admin_url == "http://custom:9000"

    def test_auth_provider_from_env(self, clean_cli_env, monkeypatch):
        monkeypatch.setenv("STATGPT_CLI_AUTH_PROVIDER", "keycloak")
        settings = make_settings()
        assert settings.auth_provider == "keycloak"

    def test_log_level_from_env(self, clean_cli_env, monkeypatch):
        monkeypatch.setenv("STATGPT_CLI_LOG_LEVEL", "DEBUG")
        settings = make_settings()
        assert settings.log_level == "DEBUG"

    def test_config_dir_from_env(self, clean_cli_env, monkeypatch):
        monkeypatch.setenv("STATGPT_CLI_CONFIG_DIR", "/custom/config")
        settings = make_settings()
        assert settings.config_dir == "/custom/config"

    def test_dial_settings_from_env(self, clean_cli_env, monkeypatch):
        monkeypatch.setenv("STATGPT_CLI_DIAL_URL", "http://dial:8080")
        monkeypatch.setenv("STATGPT_CLI_DIAL_API_KEY", "secret-key")
        settings = make_settings()
        assert settings.dial_url == "http://dial:8080"
        assert settings.dial_api_key == "secret-key"

    def test_azure_auth_settings_from_env(self, clean_cli_env, monkeypatch):
        monkeypatch.setenv("STATGPT_CLI_AUTH_AZURE_CLIENT_ID", "client-123")
        monkeypatch.setenv(
            "STATGPT_CLI_AUTH_AZURE_AUTHORITY", "https://login.microsoftonline.com/tenant"
        )
        monkeypatch.setenv("STATGPT_CLI_AUTH_AZURE_SCOPE", "api://scope/.default")
        settings = make_settings()
        assert settings.auth_azure_client_id == "client-123"
        assert settings.auth_azure_authority == "https://login.microsoftonline.com/tenant"
        assert settings.auth_azure_scope == "api://scope/.default"

    def test_max_embeddings_from_env(self, clean_cli_env, monkeypatch):
        monkeypatch.setenv("STATGPT_CLI_MAX_EMBEDDINGS", "100")
        settings = make_settings()
        assert settings.max_embeddings == 100

    def test_max_embeddings_empty_string_becomes_none(self, clean_cli_env, monkeypatch):
        """Empty string for max_embeddings should be treated as None."""
        monkeypatch.setenv("STATGPT_CLI_MAX_EMBEDDINGS", "")
        settings = make_settings()
        assert settings.max_embeddings is None


class TestCLIDataDir:
    """Tests for cli_data_dir property."""

    def test_cli_data_dir_default(self, clean_cli_env):
        settings = make_settings()
        expected = os.path.expanduser("~/.statgpt")
        assert settings.cli_data_dir == expected

    def test_cli_data_dir_custom(self, clean_cli_env, monkeypatch):
        monkeypatch.setenv("STATGPT_CLI_DATA_DIR", "~/custom/data")
        settings = make_settings()
        expected = os.path.expanduser("~/custom/data")
        assert settings.cli_data_dir == expected

    def test_cli_data_dir_expands_tilde(self, clean_cli_env, monkeypatch):
        monkeypatch.setenv("STATGPT_CLI_DATA_DIR", "~/.custom-statgpt")
        settings = make_settings()
        assert "~" not in settings.cli_data_dir
        assert os.path.isabs(settings.cli_data_dir)


class TestGetSettingSource:
    """Tests for get_setting_source method."""

    def test_source_from_env(self, clean_cli_env, monkeypatch):
        monkeypatch.setenv("STATGPT_CLI_ADMIN_URL", "http://custom:9000")
        settings = make_settings()
        assert settings.get_setting_source("admin_url") == "env"

    def test_source_default(self, clean_cli_env):
        settings = make_settings()
        assert settings.get_setting_source("admin_url") == "default"

    def test_source_not_set(self, clean_cli_env):
        settings = make_settings()
        assert settings.get_setting_source("config_dir") == "not set"

    def test_source_unknown_field(self, clean_cli_env):
        settings = make_settings()
        assert settings.get_setting_source("nonexistent_field") == "unknown"

    def test_source_env_for_optional_field(self, clean_cli_env, monkeypatch):
        monkeypatch.setenv("STATGPT_CLI_CONFIG_DIR", "/custom/config")
        settings = make_settings()
        assert settings.get_setting_source("config_dir") == "env"


class TestGetAuthSettingsForProvider:
    """Tests for get_auth_settings_for_provider method."""

    def test_azure_provider_settings(self, clean_cli_env, monkeypatch):
        monkeypatch.setenv("STATGPT_CLI_AUTH_AZURE_CLIENT_ID", "client-123")
        monkeypatch.setenv("STATGPT_CLI_AUTH_AZURE_AUTHORITY", "https://authority")
        monkeypatch.setenv("STATGPT_CLI_AUTH_AZURE_SCOPE", "scope")
        monkeypatch.setenv("STATGPT_CLI_AUTH_AZURE_CLIENT_SECRET", "secret")

        settings = make_settings()
        azure_settings = settings.get_auth_settings_for_provider("azure")

        assert azure_settings["client_id"] == "client-123"
        assert azure_settings["authority"] == "https://authority"
        assert azure_settings["scope"] == "scope"
        assert azure_settings["client_secret"] == "secret"

    def test_azure_provider_settings_partial(self, clean_cli_env, monkeypatch):
        monkeypatch.setenv("STATGPT_CLI_AUTH_AZURE_CLIENT_ID", "client-123")
        settings = make_settings()
        azure_settings = settings.get_auth_settings_for_provider("azure")

        assert azure_settings["client_id"] == "client-123"
        assert azure_settings["authority"] is None
        assert azure_settings["scope"] is None

    def test_unknown_provider_empty(self, clean_cli_env):
        settings = make_settings()
        unknown_settings = settings.get_auth_settings_for_provider("unknown")
        assert unknown_settings == {}

    def test_azure_settings_keys(self, clean_cli_env):
        settings = make_settings()
        azure_settings = settings.get_auth_settings_for_provider("azure")

        expected_keys = {
            "client_id",
            "authority",
            "scope",
            "client_secret",
        }
        assert set(azure_settings.keys()) == expected_keys


class TestFieldMetaAnnotations:
    """Tests for FieldMeta annotations on CLISettings fields."""

    def test_all_fields_have_section(self):
        missing_meta = []
        invalid_section = []

        for field_name, field_info in CLISettings.model_fields.items():
            extra = field_info.json_schema_extra
            if not isinstance(extra, dict):
                missing_meta.append(field_name)
                continue

            try:
                meta = FieldMeta.model_validate(extra)
                if meta.section not in SettingsSection:
                    invalid_section.append(field_name)
            except Exception:
                missing_meta.append(field_name)

        assert not missing_meta, f"Fields missing FieldMeta: {missing_meta}"
        assert not invalid_section, f"Fields with invalid section: {invalid_section}"

    def test_all_sections_have_fields(self):
        sections_with_fields: set[SettingsSection] = set()

        for field_info in CLISettings.model_fields.values():
            extra = field_info.json_schema_extra
            if isinstance(extra, dict):
                try:
                    meta = FieldMeta.model_validate(extra)
                    sections_with_fields.add(meta.section)
                except Exception:
                    pass

        all_sections = set(SettingsSection)
        empty_sections = all_sections - sections_with_fields

        assert not empty_sections, f"Sections with no fields: {empty_sections}"

    def test_secret_fields_are_optional(self):
        non_optional_secrets = []

        for field_name, field_info in CLISettings.model_fields.items():
            extra = field_info.json_schema_extra
            if not isinstance(extra, dict):
                continue

            try:
                meta = FieldMeta.model_validate(extra)
                if meta.secret:
                    annotation = field_info.annotation
                    origin = get_origin(annotation)
                    is_union = origin is Union or isinstance(annotation, types.UnionType)
                    is_optional = is_union and type(None) in get_args(annotation)
                    if not is_optional:
                        non_optional_secrets.append(field_name)
            except Exception:
                pass

        assert not non_optional_secrets, f"Secret fields should be optional: {non_optional_secrets}"
