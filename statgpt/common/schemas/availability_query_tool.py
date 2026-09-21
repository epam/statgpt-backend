from pydantic import Field

from .tool_details import BaseToolDetails


class AvailabilityQueryToolDetails(BaseToolDetails):
    hard_limit: int = Field(
        default=500,
        ge=1,
        description=(
            "Absolute ceiling on the number of codes returned per dimension. Caps the caller's"
            " codes_per_dimension (including 'no limit') so a single call can never return an"
            " unbounded list."
        ),
    )
