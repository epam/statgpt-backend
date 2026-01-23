from pydantic import BaseModel, Field


class DataSource(BaseModel):
    id: int
    title: str
    description: str | None
    type: str


class DataSetPreview(BaseModel):
    urn: str = Field(description="URN of the dataset")
    title: str
