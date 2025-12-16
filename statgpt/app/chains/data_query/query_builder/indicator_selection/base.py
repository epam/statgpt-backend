from abc import ABC

from statgpt.app.chains import ChainFactory
from statgpt.common.schemas import DataQueryDetails


class IndicatorSelectionBase(ChainFactory, ABC):
    pass

    # NOTE: can add retrieval stages here


class SemanticIndicatorSelectionBase(IndicatorSelectionBase, ABC):

    def __init__(self, config: DataQueryDetails):
        self._config: DataQueryDetails = config
