from langchain_core.runnables import Runnable, RunnablePassthrough

from statgpt.chains.data_query.parameters import DataQueryParameters
from statgpt.chains.parameters import ChainParameters


class NoDataChain:

    _DEFAULT_MESSAGE: str = "No relevant data found for the query. Try to change the query."

    _message: str

    def __init__(self, message: str | None = None):
        self._message = message if message else self._DEFAULT_MESSAGE

    async def create_chain(self, inputs: dict) -> Runnable:
        target = ChainParameters.get_target(inputs)
        target.append_content(self._message)
        return RunnablePassthrough.assign(
            **{DataQueryParameters.RESPONSE_FIELD: lambda _: self._message}
        )
