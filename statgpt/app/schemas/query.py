from pydantic import Field

from statgpt.common.schemas.query import JsonQuery, JsonQueryWithMetadata


class AppJsonQuery(JsonQuery):
    disabled: bool = Field(
        default=False,
        description="If true, this dataset query is excluded from processing.",
    )


class AppJsonQueryWithMetadata(JsonQueryWithMetadata):
    disabled: bool = Field(
        default=False,
        description="If true, this dataset query is excluded from processing.",
    )

    @classmethod
    def from_common(
        cls, query: JsonQueryWithMetadata, disabled: bool = False
    ) -> "AppJsonQueryWithMetadata":
        return cls(**query.model_dump(), disabled=disabled)
