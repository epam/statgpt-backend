from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.config import ChainParametersConfig
from statgpt.common.schemas.enums import InvocationSource


class TestGetInvocationSource:
    """Tools branch on this to tailor their response to the calling audience."""

    def test_defaults_to_agent_when_absent(self):
        """Entry points that predate the key keep the chat-completion behavior."""
        assert ChainParameters.get_invocation_source({}) is InvocationSource.AGENT

    def test_reads_explicit_mcp(self):
        inputs = {ChainParametersConfig.INVOCATION_SOURCE: InvocationSource.MCP}
        assert ChainParameters.get_invocation_source(inputs) is InvocationSource.MCP

    def test_reads_explicit_agent(self):
        inputs = {ChainParametersConfig.INVOCATION_SOURCE: InvocationSource.AGENT}
        assert ChainParameters.get_invocation_source(inputs) is InvocationSource.AGENT
