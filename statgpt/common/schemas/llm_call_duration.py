from pydantic import BaseModel, Field


class LLMCallDurationItem(BaseModel):
    deployment: str
    duration_s: float = Field(ge=0)

    @property
    def id(self) -> str:
        return self.deployment

    def __add__(self, other: 'LLMCallDurationItem') -> 'LLMCallDurationItem':
        if not isinstance(other, LLMCallDurationItem):
            return NotImplemented

        if self.id != other.id:
            raise ValueError("Cannot add LLMCallDurationItem with different id")

        return LLMCallDurationItem(
            deployment=self.deployment,
            duration_s=self.duration_s + other.duration_s,
        )

    def to_rounded_dict(self) -> dict:
        return {**self.model_dump(), 'duration_s': round(self.duration_s, 3)}
