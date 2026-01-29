from pydantic import BaseModel, Field, computed_field


class DataSource(BaseModel):
    id: int
    title: str
    description: str | None
    type: str


class AvailableDataSources(BaseModel):
    data_sources: list[DataSource]

    @computed_field
    @property
    def count(self) -> int:
        return len(self.data_sources)


class DataSetPreview(BaseModel):
    urn: str = Field(description="URN of the dataset")
    title: str


class AvailableDatasets(BaseModel):
    datasets: list[DataSetPreview]

    @computed_field
    @property
    def count(self) -> int:
        return len(self.datasets)
