from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from statgpt.common.schemas import AuditScope

_DEFAULT_AUDIT_SCOPE = f"{AuditScope.CONFIG.value},{AuditScope.EX_IM.value}"


class AppSettings(BaseSettings):
    """Settings for the admin backend application."""

    model_config = SettingsConfigDict()

    otel_service_name: str = Field(
        default="statgpt-admin-backend",
        alias="OTEL_APP_SERVICE_NAME",
        description="OpenTelemetry service name",
    )
    beta_mcp_enabled: bool = Field(
        default=False,
        alias="BETA_MCP_ENABLED",
        description="Flag to enable the beta MCP features",
    )
    audit_scope: str = Field(
        default=_DEFAULT_AUDIT_SCOPE,
        alias="AUDIT_SCOPE",
        description=(
            "Comma-separated audit scopes to record. Defaults to 'config,ex_im'; "
            "add 'reindex' and/or 'ds_link' to enable verbose auditing."
        ),
    )

    @property
    def enabled_audit_scopes(self) -> set[AuditScope]:
        return {AuditScope(token.strip()) for token in self.audit_scope.split(",") if token.strip()}

    @model_validator(mode="after")
    def _validate_audit_scope(self) -> "AppSettings":
        # Fail fast on an invalid AUDIT_SCOPE value instead of at first audit write.
        _ = self.enabled_audit_scopes
        return self


APP_SETTINGS = AppSettings()
