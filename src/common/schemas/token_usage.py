from pydantic import BaseModel, Field, computed_field


class TokenUsageBase(BaseModel):
    deployment: str
    model: str
    prompt_tokens: int
    completion_tokens: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class TokenUsageItem(TokenUsageBase):

    @property
    def id(self) -> str:
        return f"{self.deployment}_{self.model}"

    def __add__(self, other) -> 'TokenUsageItem':
        if not isinstance(other, TokenUsageItem):
            return NotImplemented

        if self.id != other.id:
            raise ValueError("Cannot add TokenUsageItem with different id")

        return TokenUsageItem(
            deployment=self.deployment,
            model=self.model,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


class TokenUsagePricedItem(TokenUsageBase):
    costs: float | None = Field(ge=0, description="The total cost of the token usage")
