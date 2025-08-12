import typing as t
from abc import abstractmethod

from langchain_core.runnables import Runnable, RunnableLambda

from common import utils
from statgpt.chains.parameters import ChainParameters
from statgpt.config import ChainParametersConfig
from statgpt.schemas import BatchedSelectionOutputBase
from statgpt.services import ScoredDimensionCandidate

from .chain_factory import ChainFactory


class BatchedSelectionInnerChainFactory(ChainFactory):
    @staticmethod
    @abstractmethod
    def get_output_type() -> t.Type[BatchedSelectionOutputBase]:
        raise NotImplementedError

    @staticmethod
    def get_batch_ix(inputs: dict) -> int:
        return inputs[CandidatesSelectionBatchedChainFactory.BATCH_IX_KEY]

    @staticmethod
    def get_batch_size(inputs: dict) -> int:
        return inputs[CandidatesSelectionBatchedChainFactory.BATCH_SIZE_KEY]


class CandidatesSelectionBatchedChainFactory:
    BATCH_IX_KEY = 'batch_ix'
    BATCH_SIZE_KEY = 'batch_size'

    def __init__(
        self,
        inner_chain_factory: BatchedSelectionInnerChainFactory,
        candidates_key: str,
        batch_size: int,
    ):
        self._inner_chain_factory = inner_chain_factory
        self._candidates_key = candidates_key
        self._batch_size = batch_size

    async def _batch_run_and_gather(self, inputs: dict) -> BatchedSelectionOutputBase:
        auth_context = ChainParameters.get_auth_context(inputs)

        candidates: list[ScoredDimensionCandidate] = inputs[self._candidates_key]
        candidates_batched = utils.batched(candidates, n=self._batch_size)
        batch_inputs = [
            {
                self.BATCH_IX_KEY: ix,
                self.BATCH_SIZE_KEY: self._batch_size,
                self._candidates_key: batch,
                'normalized_query': inputs['normalized_query'],
                ChainParametersConfig.AUTH_CONTEXT: auth_context,
            }
            for ix, batch in enumerate(candidates_batched)
        ]

        inner_chain = self._inner_chain_factory.create_chain()
        res: list[BatchedSelectionOutputBase] = await inner_chain.abatch(batch_inputs)
        # NOTE: 'res' could be an empty list -
        # we would need to know what default object to return.
        # since different BatchedSelectionOutputBase subclasses have different defaults,
        # we use 'combine_batch_outputs' classmethod of the type provided.
        res_gathered = self._inner_chain_factory.get_output_type().combine_batch_outputs(res)

        return res_gathered

    def create_chain(self) -> Runnable:
        return RunnableLambda(self._batch_run_and_gather)
