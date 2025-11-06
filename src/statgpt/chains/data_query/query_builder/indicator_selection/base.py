from abc import ABC

from common.schemas import DataQueryDetails
from statgpt.chains import ChainFactory


class IndicatorSelectionBase(ChainFactory, ABC):
    pass

    # NOTE: can add retrieval stages here


class SemanticIndicatorSelectionBase(IndicatorSelectionBase, ABC):

    def __init__(self, config: DataQueryDetails):
        self._config: DataQueryDetails = config
