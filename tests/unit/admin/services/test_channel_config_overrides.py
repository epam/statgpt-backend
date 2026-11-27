import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from pydantic import ValidationError

from statgpt.admin.services.channel import AdminPortalChannelService
from statgpt.admin.settings.exim import JobsConfig
from statgpt.common import models, schemas
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.schemas.onboarding import (
    OnboardingConfig,
    OnboardingTopic,
    PredefinedDataQueryResponse,
    PredefinedTextResponse,
)
from statgpt.common.schemas.query import JsonQuery

_AGENT = {"name": "Test", "domain": "Statistics", "terminologyDomain": "Statistics"}


@pytest.fixture
def service() -> AdminPortalChannelService:
    session = MagicMock()
    session.flush = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock())
    return AdminPortalChannelService(session)


def _channel(details: dict[str, Any]) -> models.Channel:
    now = datetime.datetime(2026, 1, 1)
    return models.Channel(
        id=1,
        created_at=now,
        updated_at=now,
        title="Test",
        description="",
        deployment_id="test",
        llm_model="test-model",
        details=details,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"locale": "en", "outOfScope": None, "namedEntityTypes": []},
        {"tokenUsage": {}, "mcp": {"toolNamePrefix": ""}},
        {"supremeAgent": {**_AGENT, "maxAgentIterations": 5, "llmModelConfig": {}}},
        {"conversationStarters": {"introText": None, "buttons": [{"title": "", "text": ""}]}},
    ],
)
async def test_create_preserves_only_supplied_config_fields(service, overrides):
    details = {"supremeAgent": _AGENT, **overrides}
    data = schemas.ChannelBase(
        title="Test", deployment_id="test", llm_model="test-model", details=details
    )

    item = await service._create_channel_model(data)

    assert item.details == details
    assert item.description == ""
    assert data.details.locale == "en"
    assert data.details.token_usage.debug_only is True
    service._session.add.assert_called_once_with(item)


async def test_replacement_removes_prior_nested_overrides(service):
    stored = _channel({"supremeAgent": {**_AGENT, "maxAgentIterations": 7}, "locale": "en"})
    replacement = {"supremeAgent": _AGENT, "outOfScope": None, "tokenUsage": {}}
    service._get_item_or_raise = AsyncMock(return_value=stored)
    service._session.execute.return_value.scalar_one.return_value = _channel(replacement)

    result = await AdminPortalChannelService.update.__wrapped__(
        service, stored.id, schemas.ChannelUpdate(details=replacement)
    )

    params = service._session.execute.call_args.args[0].compile().params
    assert params["details"] == replacement
    assert result.id == stored.id
    assert result.details.supreme_agent.max_agent_iterations == 5


async def test_metadata_update_leaves_config_untouched(service):
    details = {"supremeAgent": _AGENT, "locale": "en"}
    stored = _channel(details)
    service._get_item_or_raise = AsyncMock(return_value=stored)
    service._session.execute.return_value.scalar_one.return_value = stored

    await AdminPortalChannelService.update.__wrapped__(
        service, stored.id, schemas.ChannelUpdate(title="Renamed")
    )

    params = service._session.execute.call_args.args[0].compile().params
    assert params["title"] == "Renamed"
    assert "details" not in params
    assert stored.details == details


async def test_explicit_null_details_retains_existing_update_semantics(service):
    stored = _channel({"supremeAgent": _AGENT})
    service._get_item_or_raise = AsyncMock(return_value=stored)
    service._session.execute.return_value.scalar_one.return_value = stored

    await AdminPortalChannelService.update.__wrapped__(
        service, stored.id, schemas.ChannelUpdate(details=None)
    )

    assert service._session.execute.call_args.args[0].compile().params["details"] is None


@pytest.mark.parametrize(
    "response",
    [
        PredefinedTextResponse(text="Welcome"),
        PredefinedDataQueryResponse(text="Data", query=JsonQuery(urn="TEST:DATA(1.0)", filters=[])),
    ],
)
async def test_typed_onboarding_survives_create_replacement_and_export(service, tmp_path, response):
    details = schemas.ChannelConfig(
        supreme_agent=schemas.SupremeAgentConfig.model_validate(_AGENT),
        onboarding=OnboardingConfig(
            intro_message="Welcome",
            completion_message="Done",
            topic_selection_prompt="Choose",
            topics={
                "topic": OnboardingTopic(
                    id="topic", short_title="Topic", question="Question", response=response
                )
            },
        ),
    )
    data = schemas.ChannelBase(
        title="Test", deployment_id="test", llm_model="test-model", details=details
    )
    created = await service._create_channel_model(data)
    expected = details.model_dump(mode="json", by_alias=True, exclude_unset=True)
    assert created.details == expected
    assert (
        created.details["onboarding"]["topics"]["topic"]["response"]["response_type"]
        == response.response_type
    )
    assert "completionButtonTitle" not in created.details["onboarding"]

    stored = _channel(created.details)
    service._get_item_or_raise = AsyncMock(return_value=stored)
    service._session.execute.return_value.scalar_one.return_value = stored
    result = await AdminPortalChannelService.update.__wrapped__(
        service, stored.id, schemas.ChannelUpdate(details=details)
    )
    assert service._session.execute.call_args.args[0].compile().params["details"] == expected
    assert result.model_dump(mode="json", by_alias=True)["details"] == expected

    service.get_model_by_id = AsyncMock(return_value=stored)
    await service.export_channel_to_folder(
        stored.id, str(tmp_path), schemas.ExportScope.CONFIGS, MagicMock(spec=AuthContext)
    )
    exported = yaml.safe_load((tmp_path / JobsConfig.CHANNEL_FILE).read_text())
    assert (
        schemas.ChannelConfig.model_validate(exported["details"]).model_dump(
            mode="json", by_alias=True, exclude_unset=True
        )
        == expected
    )


def test_invalid_replacement_still_fails_validation():
    with pytest.raises(ValidationError):
        schemas.ChannelUpdate(details={"supremeAgent": {"name": "Missing required fields"}})
