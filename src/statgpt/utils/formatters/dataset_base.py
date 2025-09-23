from abc import ABC, abstractmethod

from pydantic import BaseModel

from common.auth.auth_context import AuthContext
from common.data.base import DataSet
from common.schemas.enums import LocaleEnum

from .base import BaseFormatter
from .citation import CitationFormatterConfig


class DatasetFormatterConfig(BaseModel):
    locale: LocaleEnum = LocaleEnum.EN
    include_name: bool = True
    add_source_id: bool = True
    source_id_name: str | None = None
    add_entity_id: bool = False
    entity_id_name: str | None = None
    use_description: bool = True
    highlight_name_in_bold: bool = True
    official_dataset_label: str | None = None
    citation: CitationFormatterConfig | None = None

    @classmethod
    def create_citation_only(
        cls, locale: LocaleEnum, citation=CitationFormatterConfig(as_md_list=True)
    ):
        return cls(
            locale=locale,
            include_name=False,
            add_source_id=False,
            add_entity_id=False,
            use_description=False,
            citation=citation,
        )


class BaseDatasetFormatter(BaseFormatter[DataSet], ABC):
    def __init__(self, config: DatasetFormatterConfig, auth_context: AuthContext):
        super().__init__("dataset", config.locale)
        self.config = config
        self.auth_context = auth_context

    async def _get_dataset_update_at(self, dataset: DataSet) -> str:
        last_updated = ""
        if last_updated_date := await dataset.updated_at(self.auth_context):
            last_updated = last_updated_date.strftime("%b %Y")
        return last_updated

    @abstractmethod
    async def format(self, dataset: DataSet) -> str:
        pass
