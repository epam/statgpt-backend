from datetime import datetime
from functools import cached_property
from zoneinfo import ZoneInfo

from pydantic import ConfigDict, Field, field_validator

from common.schemas.dial import ExtraAllowModel


class StatGPTConfiguration(ExtraAllowModel):
    """
    Dynamic DIAL configuration for StatGPT application.
    """

    model_config = ConfigDict(populate_by_name=True, extra='allow')

    timezone: str = Field(
        description="Timezone in IANA format, e.g. 'Europe/Berlin', 'America/New_York'. "
        "Used to interpret and display dates and times.",
        default="UTC",
    )

    @field_validator('timezone')
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        """Validate that the timezone string is a valid IANA timezone."""
        try:
            ZoneInfo(v)
        except Exception:
            # Fallback to UTC if timezone is invalid
            return "UTC"
        return v

    @cached_property
    def tzinfo(self) -> ZoneInfo:
        """Get the ZoneInfo object for the configured timezone."""
        return ZoneInfo(self.timezone)

    def get_current_timestamp(self) -> str:
        """Get the current timestamp in the configured timezone."""
        tz = self.tzinfo
        return datetime.now(tz).isoformat()

    def get_current_date(self) -> str:
        """Get current date in the configured timezone in '%Y-%m-%d' format."""
        tz = self.tzinfo
        return datetime.now(tz).strftime('%Y-%m-%d')
