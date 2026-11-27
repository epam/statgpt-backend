import datetime
import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from statgpt.common.schemas.base import ListResponse
from statgpt.common.schemas.channel import Channel, ChannelConfig
from statgpt.common.schemas.onboarding import (
    OnboardingConfig,
    OnboardingTopic,
    PredefinedDataQueryResponse,
    PredefinedTextResponse,
)
from statgpt.common.schemas.query import JsonQuery

_DETAILS = {
    "supremeAgent": {"name": "Test", "domain": "Statistics", "terminologyDomain": "Statistics"},
    "tokenUsage": {},
    "outOfScope": None,
}


def _channel(details: ChannelConfig | dict[str, Any] | None = None) -> Channel:
    now = datetime.datetime(2026, 1, 1)
    return Channel.model_validate(
        {
            "id": 1,
            "created_at": now,
            "updated_at": now,
            "title": "Test",
            "deployment_id": "test",
            "llm_model": "test-model",
            "details": details if details is not None else _DETAILS,
        }
    )


@pytest.mark.parametrize("by_alias", [False, True])
def test_json_sparse_policy_is_local_to_channel_details(by_alias):
    channel = _channel()
    key = "supremeAgent" if by_alias else "supreme_agent"
    dump = channel.model_dump(mode="json", by_alias=by_alias)

    assert set(dump["details"]) == {
        key,
        "tokenUsage" if by_alias else "token_usage",
        "outOfScope" if by_alias else "out_of_scope",
    }
    assert len(dump["details"][key]) == 3
    assert dump["description"] == ""
    assert dump["id"] == 1
    assert channel.details.supreme_agent.max_agent_iterations == 5
    assert channel.details.token_usage.debug_only is True
    assert "locale" in channel.model_dump()["details"]
    assert "locale" in channel.details.model_dump(mode="json")
    assert json.loads(channel.model_dump_json(by_alias=by_alias)) == dump
    assert Channel.model_validate(dump).model_dump(mode="json", by_alias=by_alias) == dump


def test_field_serializer_honors_nested_field_selection():
    channel = _channel()
    assert channel.model_dump(
        mode="json", by_alias=True, include={"details": {"supreme_agent": {"name"}}}
    ) == {"details": {"supremeAgent": {"name": "Test"}}}
    dump = channel.model_dump(
        mode="json",
        by_alias=True,
        exclude={"details": {"supreme_agent": {"domain"}}},
        exclude_none=True,
    )
    assert dump["details"] == {
        "supremeAgent": {"name": "Test", "terminologyDomain": "Statistics"},
        "tokenUsage": {},
    }


@pytest.mark.parametrize(
    "details", [_DETAILS, {**_DETAILS, "locale": "en", "namedEntityTypes": []}]
)
def test_single_list_and_nested_http_responses_preserve_config_keys(details):
    channel = _channel(details)
    app = FastAPI()

    class NestedResponse(BaseModel):
        channel: Channel

    @app.get("/single", response_model=Channel)
    def single():
        return channel

    @app.get("/list", response_model=ListResponse[Channel])
    def list_channels():
        return ListResponse[Channel](data=[channel], limit=10, offset=0, count=1, total=1)

    @app.get("/nested", response_model=NestedResponse)
    def nested():
        return NestedResponse(channel=channel)

    with TestClient(app) as client:
        single_json = client.get("/single").json()
        list_json = client.get("/list").json()
        nested_json = client.get("/nested").json()

    assert single_json["details"] == details
    assert list_json["data"][0] == single_json
    assert nested_json["channel"] == single_json
    assert single_json["description"] == ""
    assert list_json["limit"] == 10
    assert (
        Channel.model_validate(single_json).model_dump(mode="json", by_alias=True)["details"]
        == details
    )


@pytest.mark.parametrize(
    "response",
    [
        PredefinedTextResponse(text="Welcome"),
        PredefinedDataQueryResponse(text="Data", query=JsonQuery(urn="TEST:DATA(1.0)", filters=[])),
    ],
)
def test_typed_onboarding_discriminators_survive_sparse_exchange(response):
    onboarding = OnboardingConfig(
        intro_message="Welcome",
        completion_message="Done",
        topic_selection_prompt="Choose",
        topics={
            "topic": OnboardingTopic(
                id="topic", short_title="Topic", question="Question", response=response
            )
        },
    )
    details = ChannelConfig.model_validate(_DETAILS)
    details.onboarding = onboarding
    channel = _channel(details)
    dump = channel.model_dump(mode="json", by_alias=True)["details"]
    exported_response = dump["onboarding"]["topics"]["topic"]["response"]

    assert "response_type" in response.model_fields_set
    assert exported_response["response_type"] == response.response_type
    assert "completionButtonTitle" not in dump["onboarding"]
    assert "subtopics" not in dump["onboarding"]["topics"]["topic"]
    assert (
        ChannelConfig.model_validate(dump).model_dump(
            mode="json", by_alias=True, exclude_unset=True
        )
        == dump
    )


def test_untyped_onboarding_still_requires_discriminator():
    with pytest.raises(ValidationError, match="union_tag_not_found"):
        OnboardingTopic.model_validate(
            {
                "id": "topic",
                "shortTitle": "Topic",
                "question": "Question",
                "response": {"text": "Welcome"},
            }
        )
