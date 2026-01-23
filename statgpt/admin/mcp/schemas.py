from pydantic import BaseModel, Field


class DataSource(BaseModel):
    id: int
    title: str
    description: str | None
    type: str


class DataSetPreview(BaseModel):
    id_in_source: str = Field(description="ID of the dataset in the data source")
    title: str
    # description: str | None
