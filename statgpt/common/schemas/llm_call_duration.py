from pydantic import BaseModel, Field


class LLMCallDurationItem(BaseModel):
    deployment: str
    model: str
    duration_s: float = Field(ge=0)

    @property
    def id(self) -> str:
        return f"{self.deployment}_{self.model}"

    def __add__(self, other: 'LLMCallDurationItem') -> 'LLMCallDurationItem':
        if not isinstance(other, LLMCallDurationItem):
            return NotImplemented

        if self.id != other.id:
            raise ValueError("Cannot add LLMCallDurationItem with different id")

        return LLMCallDurationItem(
            deployment=self.deployment,
            model=self.model,
            duration_s=round(self.duration_s + other.duration_s, 3),
        )
