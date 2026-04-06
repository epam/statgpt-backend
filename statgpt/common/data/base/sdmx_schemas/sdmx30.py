from pydantic import BaseModel, Field
from sdmx.model.v21 import Annotation


class Sdmx30AnnotationModel(BaseModel):
    id: str | None = Field(default=None)
    title: str | None = Field(default=None)
    type: str | None = Field(default=None)
    value: str | None = Field(default=None)
    text: str | None = Field(default=None)

    def to_sdmx1(self) -> Annotation:
        return Annotation(
            id=self.id,
            title=self.title,
            type=self.type,
            text=self.text,
            # The `value` field was added by SDMX 3.0.0, so it's not included here.
        )
