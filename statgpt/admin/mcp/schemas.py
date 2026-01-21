from pydantic import BaseModel


class DataSource(BaseModel):
    id: int
    title: str
    description: str | None
    type: str
