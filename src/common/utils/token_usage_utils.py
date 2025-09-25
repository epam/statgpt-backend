import pandas as pd

from common.schemas.token_usage import TokenUsageItem, TokenUsagePricedItem


class TokenUsageCostCalculator:
    def __init__(self, model_to_pricing_map: dict) -> None:
        self._model_to_pricing_map = model_to_pricing_map

    def get_token_usage_with_costs(
        self, token_usage: list[TokenUsageItem]
    ) -> list[TokenUsagePricedItem]:
        return [
            TokenUsagePricedItem(
                deployment=item.deployment,
                model=item.model,
                prompt_tokens=item.prompt_tokens,
                completion_tokens=item.completion_tokens,
                costs=self._calculate_usage_cost(item),
            )
            for item in token_usage
        ]

    def _calculate_usage_cost(self, item: TokenUsageItem) -> float | None:
        """Calculate the cost of the token usage."""
        if model_pricing := self._model_to_pricing_map.get(item.model):
            return (
                item.prompt_tokens * model_pricing.prompt
                + item.completion_tokens * model_pricing.completion
            )
        return None


class TokenUsageDisplayer:
    PRETTY_COLUMN_NAMES = {
        'deployment': 'Deployment',
        'model': 'Model',
        'prompt_tokens': 'Prompt Tokens',
        'completion_tokens': 'Completion Tokens',
        'total_tokens': 'Total Tokens',
        'costs': 'Total Price, $',
    }
    NA_REPLACEMENT = '-'

    @classmethod
    def as_markdown_table(cls, token_usage: list[TokenUsagePricedItem]) -> str:
        """Convert token usage statistics to a string representation of a Markdown table."""
        df = pd.DataFrame(
            [item.model_dump(by_alias=True) for item in token_usage],
            columns=list(cls.PRETTY_COLUMN_NAMES.keys()),
        )

        total_row = cls._get_total_row(df)
        df = pd.concat([df, total_row], ignore_index=True)

        df.rename(columns=cls.PRETTY_COLUMN_NAMES, inplace=True)
        df.fillna(cls.NA_REPLACEMENT, inplace=True)
        return df.to_markdown(index=False)

    @staticmethod
    def _get_total_row(df: pd.DataFrame) -> pd.DataFrame:
        totals = df.sum(numeric_only=True)

        return pd.DataFrame(
            {
                'deployment': '',
                'model': 'TOTAL:',
                'prompt_tokens': totals.get('prompt_tokens'),
                'completion_tokens': totals.get('completion_tokens'),
                'total_tokens': totals.get('total_tokens'),
                'costs': totals.get('costs'),
            },
            index=[0],
        )

    @classmethod
    def as_string_list(cls, token_usage: list[TokenUsagePricedItem]) -> str:
        """Convert token usage statistics to a string representation of a Markdown list."""

        res = ''

        for item in token_usage:
            res += f"### {item.model}\n"
            res += f"* Prompt Tokens: {item.prompt_tokens}\n"
            res += f"* Completion Tokens: {item.completion_tokens}\n"
            res += f"* Total Tokens: {item.total_tokens}\n"

            if item.costs is None:
                res += "* Total Price: <NO DATA>\n\n"
            else:
                res += f"* Total Price, $: {item.costs:.3f}\n\n"

        total_prompt_tokens = sum(item.prompt_tokens for item in token_usage if item.prompt_tokens)
        total_completion_tokens = sum(
            item.completion_tokens for item in token_usage if item.completion_tokens
        )
        total_total_tokens = total_prompt_tokens + total_completion_tokens
        total_costs = sum(item.costs for item in token_usage if item.costs)

        res += '-' * 15 + '\n'
        res += "## TOTAL:\n"
        res += f"* Prompt Tokens: {total_prompt_tokens}\n"
        res += f"* Completion Tokens: {total_completion_tokens}\n"
        res += f"* Total Tokens: {total_total_tokens}\n"
        res += f"* Total Price, $: {total_costs:.3f}\n"

        return res
