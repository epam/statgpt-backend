from pydantic import BaseModel, Field

from .base import BaseYamlModel, DbDefaultBase


class GlossaryTermBase(BaseYamlModel):
    """A model representing a single term from the glossary."""

    term: str
    definition: str
    domain: str = Field(
        description="The domain of the term. For example, 'Insurance', 'Economics', 'Statistics', etc."
    )
    source: str = Field(
        description="The source of the definition. For example, 'IMF', 'Swiss Re', 'World Bank', etc."
    )


class GlossaryTerm(GlossaryTermBase, DbDefaultBase):
    """A model representing a single term from the glossary."""

    channel_id: int = Field(description="The ID of the channel this term belongs to.")


class GlossaryTermUpdate(BaseModel):
    term: str | None = Field(default=None)
    definition: str | None = Field(default=None)
    domain: str | None = Field(
        default=None,
        description="The domain of the term. For example, 'Insurance', 'Economics', 'Statistics', etc.",
    )
    source: str | None = Field(
        default=None,
        description="The source of the definition. For example, 'IMF', 'Swiss Re', 'World Bank', etc.",
    )


class GlossaryTermUpdateBulk(GlossaryTermUpdate):
    id: int = Field(description="The ID of the term to update.")
