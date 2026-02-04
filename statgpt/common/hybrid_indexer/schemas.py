from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr


class BaseIndex(BaseModel):
    model_config = ConfigDict(extra='allow')

    id: StrictStr
    dataset_id: StrictStr
    dataset_name: StrictStr
    version_id: StrictInt
    series: StrictStr = Field(description="Dump of dimension queries in JSON format.")
    name: StrictStr
    name_normalized: StrictStr
    where: list[dict[StrictStr, StrictStr]]


class MatchingIndex(BaseIndex):
    pass


class IndicatorIndex(BaseIndex):
    primary: StrictStr
    primary_normalized: StrictStr


class NormalizationOutput(BaseModel):
    reasoning: list[str] = Field(
        description="Step-by-step reasoning following the normalization instructions"
    )
    normalized: str = Field(description="The normalized indicator text")


class HarmonizationOutput(BaseModel):
    reasoning: list[str] = Field(
        description="Step-by-step reasoning for extracting the primary part"
    )
    primary: str = Field(description="The primary part of the indicator name")
