from pydantic import Field

from .tool_details import BaseToolDetails


class DatasetStructureToolDetails(BaseToolDetails):
    include_provider_agencies: bool = Field(
        default=False,
        description="Whether to include the provider agencies in the response.",
    )
