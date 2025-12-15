from langchain_core.runnables import RunnablePassthrough

from statgpt.app.chains.data_query.parameters import DataQueryParameters
from statgpt.app.schemas.query_builder import ChainState
from statgpt.app.utils.formatters import DatasetQueryFormatter, DatasetQueryFormatterConfig


class InvalidSelectedTimePeriodChain:
    _DEFAULT_MESSAGE: str = (
        "## Result of query construction"
        "\n\nThe created query contains data according to the selected filters,"
        " but the values are only available for a different time period."
        " Please adjust the time period or modify the query."
        "\n\n## Constructed queries for datasets"
    )

    def __init__(self, message: str | None):
        self._message: str = message or self._DEFAULT_MESSAGE

    async def _get_response_content(self, inputs: dict) -> str:
        chain_state = ChainState.model_validate(inputs)

        query_formatter = DatasetQueryFormatter(
            config=DatasetQueryFormatterConfig(
                locale=chain_state.data_service.channel_config.locale,
                include_missing_dimensions=False,
                include_default_queries=True,
                include_auto_selects=True,
                include_is_official=False,
            ),
            auth_context=chain_state.auth_context,
        )
        formatted_queries = await query_formatter.format_queries(
            dataset_queries=chain_state.dataset_queries,
            datasets_dict=chain_state.datasets_dict,
            availability_queries=chain_state.strong_availability,  # type: ignore[arg-type]
        )
        result: str = self._message + '\n\n' + formatted_queries
        chain_state.target.append_content(result)
        return result

    def create_chain(self):
        return RunnablePassthrough.assign(
            **{
                DataQueryParameters.RESPONSE_FIELD: self._get_response_content,
            }
        )
