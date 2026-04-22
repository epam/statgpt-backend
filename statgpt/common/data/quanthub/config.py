from enum import StrEnum

from pydantic import Field, SecretStr, model_validator

from statgpt.common.config import utils as config_utils
from statgpt.common.data.base.property_source import PropertySource, PropertySourceEnum
from statgpt.common.data.sdmx.common.config import SdmxDataSetConfig, SdmxDataSourceConfig
from statgpt.common.schemas.base import BaseYamlModel


class QuanthubDataSetConfig(SdmxDataSetConfig):
    """Configuration for a Quanthub SDMX dataset."""

    updated_at: list[PropertySource] = Field(
        min_length=1,
        default_factory=lambda: [
            PropertySource(source=PropertySourceEnum.ANNOTATION, field="lastUpdatedAt"),
            PropertySource(source=PropertySourceEnum.CITATION, field="last_updated"),
        ],
        description=(
            "A list of sources ordered by priority for obtaining the dataset's update date."
            " By default, it first tries to get it from the 'lastUpdatedAt' annotation,"
            " and if not found, from the 'last_updated' field in the citation."
        ),
    )


class BasicAuthCredentials(BaseYamlModel):
    password: str = Field()
    username: str = Field()

    def get_password(self) -> str:
        return config_utils.replace_env(self.password)

    def get_username(self) -> str:
        return config_utils.replace_env(self.username)


class AuthGrantType(StrEnum):
    ROPC = "ropc"
    CLIENT_CREDENTIALS = "client_credentials"
    MSI = "msi"


class UserAuthType(StrEnum):
    FORWARD_DIAL_TOKEN = "forward_dial_token"
    OBO_FLOW = "obo_flow"

    USE_SYSTEM_USER = "use_system_user"
    """User authorization is not used, instead all users use system authorization."""


class GrantConfig(BaseYamlModel):
    """Abstract class"""

    target_scope: str = Field()

    def get_target_scope(self) -> str:
        return config_utils.replace_env(self.target_scope)


class OboFlowConfig(GrantConfig):
    pass


class RopcGrantConfig(GrantConfig):
    system_user_credentials: BasicAuthCredentials = Field()


class MsiGrantConfig(GrantConfig):
    pass


class ClientCredentialGrantConfig(GrantConfig):
    pass


class AuthConfig(BaseYamlModel):
    user_auth_type: str = Field(default="use_system_user")
    system_auth_type: str

    msi: MsiGrantConfig | None = Field(default=None)
    ropc: RopcGrantConfig | None = Field(default=None)
    client_credentials: ClientCredentialGrantConfig | None = Field(default=None)

    obo_flow: OboFlowConfig | None = Field(default=None)

    def get_user_auth_type(self) -> UserAuthType:
        return UserAuthType(config_utils.replace_env(self.user_auth_type))

    def get_system_auth_type(self) -> AuthGrantType:
        return AuthGrantType(config_utils.replace_env(self.system_auth_type))

    def get_msi_config(self) -> MsiGrantConfig:
        if not self.msi:
            raise ValueError("MSI grant not configured")
        return self.msi

    def get_ropc_config(self) -> RopcGrantConfig:
        if not self.ropc:
            raise ValueError("ROPC grant not configured")
        return self.ropc

    def get_client_credentials_config(self) -> ClientCredentialGrantConfig:
        if not self.client_credentials:
            raise ValueError("Client Credentials grant not configured")
        return self.client_credentials

    @model_validator(mode="after")
    def validate_auth_config(self) -> "AuthConfig":
        if all([not self.msi, not self.ropc, not self.client_credentials]):
            raise ValueError("At least one configuration(MSI, ROPC or ClientCredentials) expected")

        grant_type_config_mapping = {
            AuthGrantType.MSI: self.msi,
            AuthGrantType.ROPC: self.ropc,
            AuthGrantType.CLIENT_CREDENTIALS: self.client_credentials,
        }

        selected_grant_type = self.get_system_auth_type()
        config_for_selected_grant_type = grant_type_config_mapping.get(selected_grant_type, None)
        if config_for_selected_grant_type is None:
            raise ValueError(f"Missing config for selected grant type: {selected_grant_type.value}")

        if self.get_user_auth_type() == UserAuthType.OBO_FLOW and not self.obo_flow:
            raise ValueError("OBO flow selected but obo_flow config is missing")

        return self


class QuanthubSdmxDataSourceConfig(SdmxDataSourceConfig):
    api_key_header: str = Field(
        default="Ocp-Apim-Subscription-Key", description="The API key header"
    )
    api_key: str = Field(
        default="",
        description="The API key value or the environment variable in the following format: '$env:{ENV_VAR_NAME}'",
    )

    auth_enabled: bool = Field(default=False)
    auth_config: AuthConfig | None = Field(
        description="The configuration for data query authorization", default=None
    )
    annotations_url: str | None = Field(
        default=None,
        description="The SDMX 3.0 URL for annotations. If not set, loading dynamic annotations is disabled.",
    )
    attributes_url: str | None = Field(
        default=None,
        description="The SDMX 3.0 URL for attributes. If not set, loading attributes is disabled.",
    )
    availability_via_post_url: str | None = Field(
        default=None,
        description="The SDMX 3.0 URL for availability via POST. If not set, the default availability endpoint will be used.",
    )

    @property
    def has_api_key_header(self) -> bool:
        """Check if the API key header name and value are set."""
        return bool(self.api_key_header) and bool(self.api_key)

    def get_api_key(self) -> SecretStr:
        api_key = config_utils.replace_env(self.api_key)
        return SecretStr(api_key)

    def get_annotations_url(self) -> str | None:
        return config_utils.replace_env(self.annotations_url) if self.annotations_url else None

    def get_attributes_url(self) -> str | None:
        return config_utils.replace_env(self.attributes_url) if self.attributes_url else None

    def get_availability_via_post_url(self) -> str | None:
        if self.availability_via_post_url:
            return config_utils.replace_env(self.availability_via_post_url)
        return None
