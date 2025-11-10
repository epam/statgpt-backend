from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, RunnableParallel

from common.config import multiline_logger as logger
from common.schemas.data_query_tool import SpecialDimensionsProcessor
from common.schemas.enums import SpecialDimensionsProcessorType
from statgpt.chains.data_query.query_builder.special_dimensions_selection import (
    LHCLChainFactory,
    SpecialDimensionChainFactoryBase,
)
from statgpt.utils.callbacks import StageCallback

from .base import DimensionSearchChainFactoryBase


class SpecialDimensionsSearchChainFactory(DimensionSearchChainFactoryBase):
    _FACTORIES: dict[SpecialDimensionsProcessorType, type[SpecialDimensionChainFactoryBase]] = {
        SpecialDimensionsProcessorType.LHCL: LHCLChainFactory,
    }

    @classmethod
    def _get_special_dimension_factory(
        cls, processor: SpecialDimensionsProcessor
    ) -> SpecialDimensionChainFactoryBase:
        factory_type = cls._FACTORIES[processor.type]
        return factory_type(processor=processor)

    def create(self) -> Runnable:
        processors = self._config.special_dimensions_processors

        if len(processors) == 0:
            logger.info('no special dimension processors are present for data query tool')
            return RunnableLambda(lambda _: {})

        chains_dict = {}
        for processor in processors:
            factory = self._get_special_dimension_factory(processor=processor)
            if not factory:
                raise NotImplementedError(
                    f'Unsupported special dimension processor type: {processor.type}'
                )
            chains_dict[processor.id] = factory.create_chain()

        chain = RunnableParallel(chains_dict)
        logger.info(
            f'created processors for {len(chains_dict)} following special dimensions: '
            f'{list(chains_dict.keys())}'
        )

        return chain.with_config(
            config=RunnableConfig(
                callbacks=[
                    StageCallback(
                        stage_name="Selecting Special Dimensions",
                        content_appender=None,
                        debug_only=True,
                    )
                ]
            )
        )
