"""Tests for the `ChainParameters` getters (#831)."""

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.config import ChainParametersConfig as ParamsConfig


def test_out_of_scope_getters_return_none_when_the_check_did_not_run() -> None:
    assert ChainParameters.is_out_of_scope({}) is None
    assert ChainParameters.get_out_of_scope_reasoning({}) is None


def test_out_of_scope_getters_return_the_saved_result() -> None:
    inputs = {
        ParamsConfig.OUT_OF_SCOPE: True,
        ParamsConfig.OUT_OF_SCOPE_REASONING: "The question is about cooking.",
    }

    assert ChainParameters.is_out_of_scope(inputs) is True
    assert ChainParameters.get_out_of_scope_reasoning(inputs) == "The question is about cooking."
