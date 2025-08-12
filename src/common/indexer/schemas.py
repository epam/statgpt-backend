from pydantic import BaseModel, ConfigDict, Field, StrictStr


class MatchingIndex(BaseModel):
    id: StrictStr
    dataset_id: StrictStr
    dataset_name: StrictStr
    series: StrictStr = Field(description="Dump of dimension queries in JSON format.")
    name: StrictStr
    name_normalized: StrictStr
    where: list[dict[StrictStr, StrictStr]]

    model_config = ConfigDict(extra='forbid')


class IndicatorIndex(BaseModel):
    id: StrictStr
    dataset_id: StrictStr
    dataset_name: StrictStr
    series: StrictStr = Field(description="Dump of dimension queries in JSON format.")
    name: StrictStr
    name_normalized: StrictStr
    where: list[dict[StrictStr, StrictStr]]
    primary: StrictStr
    primary_normalized: StrictStr

    model_config = ConfigDict(extra='forbid')
