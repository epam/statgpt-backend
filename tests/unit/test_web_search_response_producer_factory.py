from statgpt.app.chains.tools import StatGptTool
from statgpt.app.chains.web_search.response_producer import (
    RagResponseProducer,
    UrlOnlyResponseProducer,
)
from statgpt.common.schemas import ChannelConfig, WebSearchAgentTool
from statgpt.common.schemas.tool_details import WebSearchAgentDetails


def _channel_config() -> ChannelConfig:
    return ChannelConfig.model_validate(
        {
            "supreme_agent": {
                "name": "Test Bot",
                "domain": "test",
                "terminology_domain": "test",
            }
        }
    )


def test_web_search_agent_urls_only_true_instantiates_url_only_response_producer():
    tool_config = WebSearchAgentTool(
        name="web_search_agent",
        description="Web search agent tool",
        details=WebSearchAgentDetails(
            deployment_id="dep",
            urls_only=True,
        ),
    )

    tool = StatGptTool.from_config(tool_config=tool_config, channel_config=_channel_config())
    assert isinstance(tool._response_producer, UrlOnlyResponseProducer)  # type: ignore[attr-defined]


def test_web_search_agent_urls_only_false_instantiates_rag_response_producer():
    tool_config = WebSearchAgentTool(
        name="web_search_agent",
        description="Web search agent tool",
        details=WebSearchAgentDetails(
            deployment_id="dep",
            urls_only=False,
        ),
    )

    tool = StatGptTool.from_config(tool_config=tool_config, channel_config=_channel_config())
    producer = tool._response_producer  # type: ignore[attr-defined]
    assert isinstance(producer, RagResponseProducer)
    assert producer._stream_content is True
    assert producer._attachments_metadata is False
