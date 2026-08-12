from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DiscoveryUploadSettings(BaseSettings):
    """Limits applied to a discovery dataset file upload."""

    model_config = SettingsConfigDict(env_prefix="discovery_upload_")

    max_file_size_bytes: int = Field(
        10 * 1024 * 1024,
        description="Reject an upload larger than this, before parsing it.",
    )
    max_rows: int = Field(
        10_000,
        description="Reject a file with more data rows than this.",
    )
    max_reported_problems: int = Field(
        200,
        description=(
            "Cap on the problems listed in an error response. Beyond it the response is"
            " marked as truncated."
        ),
    )
