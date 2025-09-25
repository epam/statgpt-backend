from __future__ import annotations

from typing import Any, Iterable

from langchain_core.documents import Document
from pydantic import BaseModel

from common.data.base import BaseIndicator
from common.data.base.category import DimensionCategory
from common.settings.document import IndicatorDocumentMetadataFields

from .category import CodeCategory, DimensionCodeCategory
from .constants import SdmxConstants


class CodeIndicator(BaseIndicator):

    _code_category: CodeCategory

    def __init__(self, code: CodeCategory):
        super().__init__()
        self._code_category = code

    @property
    def entity_id(self) -> str:
        return self._code_category.entity_id

    @property
    def dimension_id(self) -> str | None:
        return self._code_category.dimension_id

    @property
    def query_id(self) -> str:
        return self._code_category.query_id

    @property
    def name(self) -> str:
        return self._code_category.name

    @property
    def description(self) -> str | None:
        return self._code_category.description

    @property
    def source_id(self) -> str:
        return self._code_category.source_id

    # def to_document(self, include_description: bool = False) -> Document:
    # return self._code_category.to_document()

    def get_document_content(self, include_description: bool = False) -> str:
        return self._code_category.get_document_content(include_description)

    def get_document_metadata(self) -> dict:
        return self._code_category.get_document_metadata()

    @classmethod
    def from_document(cls, document: Document) -> 'CodeIndicator':
        # TODO: bad design, refactor this:
        code_category: CodeCategory
        if document.metadata.get(SdmxConstants.METADATA_DIMENSION_ID):
            code_category = DimensionCodeCategory.from_document(document)
        else:
            code_category = CodeCategory.from_document(document)

        return cls(code_category)

    def get_debug_summary_dict(self) -> dict:
        return {
            'code_id': self.query_id,
            'entity_id': self.entity_id,
            'name': self.name,
        }


class ComplexIndicatorComponentDetails(BaseModel):
    dimension_id: str
    dimension_name: str
    query_id: str
    name: str


class ComplexIndicator(BaseIndicator):
    """An indicator that represents a combination of multiple indicators."""

    # TODO: we depend on specific classes, which is probably not good. Can refactor if needed.

    CONTENT_SEPARATOR = "; "

    def __init__(self, indicators: Iterable[CodeIndicator]):
        super().__init__()
        # NOTE: preserve the passed order of indicators
        self._indicators = list(indicators)

    @property
    def query_id(self) -> str:
        """TODO: is it correct?"""
        return self.CONTENT_SEPARATOR.join(indicator.query_id for indicator in self._indicators)

    @property
    def name(self) -> str:
        """TODO: is it correct?"""
        return self.CONTENT_SEPARATOR.join(indicator.name for indicator in self._indicators)

    @property
    def indicators(self) -> list[CodeIndicator]:
        return self._indicators

    def get_components_details(self):
        details = []
        for indicator in self._indicators:
            category = indicator._code_category
            if not isinstance(category, DimensionCategory):
                raise TypeError(f"Expected DimensionCategory, got {type(category)}")
            cur_data = ComplexIndicatorComponentDetails(
                dimension_id=category.dimension_id,
                dimension_name=category.dimension_name,
                query_id=indicator.query_id,
                name=indicator.name,
            )
            details.append(cur_data)
        return details

    def get_document_content(self) -> str:
        component_details = self.get_components_details()
        parts = []
        for cur_component in component_details:
            cur_part = f'{cur_component.dimension_name}: {cur_component.name}'
            cur_part = cur_part.replace(self.CONTENT_SEPARATOR, ", ")
            parts.append(cur_part)
        content = self.CONTENT_SEPARATOR.join(parts)
        return content

    def get_document_metadata(
        self, additional_metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        metadata = {
            IndicatorDocumentMetadataFields.INDICATORS_METADATA: [
                indicator.get_document_metadata() for indicator in self._indicators
            ]
        }
        metadata.update(additional_metadata or {})
        return metadata

    def to_document(self, additional_metadata: dict[str, Any] | None = None) -> Document:
        return Document(
            page_content=self.get_document_content(),
            metadata=self.get_document_metadata(additional_metadata),
        )

    @classmethod
    def from_document(cls, document: Document) -> ComplexIndicator:
        indicators = []
        components_metadata = document.metadata[IndicatorDocumentMetadataFields.INDICATORS_METADATA]

        for cur_metadata in components_metadata:
            # page_content is not used here, since
            # CodeIndicator could be instantiated from metadata only.
            cur_ind_component = CodeIndicator.from_document(
                Document(page_content='', metadata=cur_metadata)
            )
            indicators.append(cur_ind_component)

        return cls(indicators)
