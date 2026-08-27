from langchain_core.runnables import Runnable, RunnablePassthrough

from statgpt.app.chains.data_query.parameters import DataQueryParameters
from statgpt.app.chains.parameters import ChainParameters
from statgpt.common.schemas.data_query_tool import DataQueryMessages


class NoDataChain:

    _DEFAULT_MESSAGE: str = "No relevant data found for the query. Try to change the query."

    def __init__(self, messages: DataQueryMessages):
        self._messages = messages

    def _get_message(self, inputs: dict) -> str:
        configured = self._messages.get_no_data(ChainParameters.get_invocation_source(inputs))
        return configured if configured else self._DEFAULT_MESSAGE

    async def create_chain(self, inputs: dict) -> Runnable:
        message = self._get_message(inputs)
        target = ChainParameters.get_target(inputs)
        target.append_content(message)
        return RunnablePassthrough.assign(**{DataQueryParameters.RESPONSE_FIELD: lambda _: message})
