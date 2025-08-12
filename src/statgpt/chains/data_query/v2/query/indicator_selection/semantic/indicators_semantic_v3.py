from common.schemas import DataQueryDetails

from .indicators_semantic_v2 import IndicatorSelectionSemanticV2ChainFactory
from .packed_indicators_selection import PackedIndicatorsSelectionV2ChainFactory


class IndicatorSelectionSemanticV3ChainFactory(IndicatorSelectionSemanticV2ChainFactory):
    """
    Same as IndicatorSelectionSemanticV2ChainFactory, but uses PackedIndicatorsSelectionV2ChainFactory.
    """

    def __init__(
        self,
        config: DataQueryDetails,
        vector_search_top_k: int = 100,
    ):
        super().__init__(config, vector_search_top_k=vector_search_top_k)

    def get_indicator_selection_chain_factory(self):
        return PackedIndicatorsSelectionV2ChainFactory(candidates_key=self._candidates_key)
