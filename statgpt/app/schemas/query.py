from pydantic import Field, computed_field

from statgpt.common.schemas.query import JsonQuery, JsonQueryWithMetadata
from statgpt.common.schemas.record_id import compose_record_id


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

    @computed_field(  # type: ignore[prop-decorator]
        alias="recordId",
        description=(
            "Stable, opaque identifier of this queried record (dataflow + series key)."
            " Same logical record keeps the same id across sessions and releases."
            " Treat it as opaque and pass it back verbatim to reference the exact same record"
            " in a follow-up call; do not assemble or edit it by hand."
            " Format: 'AGENCY:RESOURCE(VERSION)/SERIES_KEY'."
        ),
    )
    @property
    def record_id(self) -> str:
        return compose_record_id(self)

    @classmethod
    def from_common(
        cls, query: JsonQueryWithMetadata, disabled: bool = False
    ) -> "AppJsonQueryWithMetadata":
        return cls(**query.model_dump(), disabled=disabled)
