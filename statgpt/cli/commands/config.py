"""Configuration generation commands for StatGPT CLI."""

import json
import os
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from statgpt.cli.commands.base import Command, CommandArg, CommandGroup
from statgpt.cli.shared import cli_settings, console, print_error, print_info, print_success
from statgpt.common.config.utils import replace_env

MODEL_TYPE_TO_PATH = {
    "chat": "/chat/completions",
    "embedding": "/embeddings",
}


# ============================================================================
# Pydantic Models for DIAL API
# ============================================================================


class DIALBaseModel(BaseModel):
    """Base model with camelCase alias generation."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )


class DIALModelCapabilities(DIALBaseModel):
    """Model capabilities from DIAL API."""

    scale_types: list[str]
    completion: bool
    chat_completion: bool
    embeddings: bool
    fine_tune: bool
    inference: bool


class DIALLimits(DIALBaseModel):
    """Model limits from DIAL API."""

    max_total_tokens: int | None = None


class DIALModelPricing(DIALBaseModel):
    """Model pricing from DIAL API."""

    unit: str
    prompt: str
    completion: str | None = None


class DIALModelFeatures(DIALBaseModel):
    """Model features from DIAL API."""

    rate: bool = False
    tokenize: bool = False
    truncate_prompt: bool = False
    configuration: bool = False
    system_prompt: bool = False
    tools: bool = False
    seed: bool = False
    url_attachments: bool = False
    folder_attachments: bool = False

    def to_config(self) -> dict[str, bool]:
        """Convert to config format."""
        value = {
            "systemPromptSupported": self.system_prompt,
            "toolsSupported": self.tools,
            "urlAttachmentsSupported": self.url_attachments,
            "folderAttachmentsSupported": self.folder_attachments,
        }
        return {k: v for k, v in value.items() if v}


class DIALDeploymentBase(DIALBaseModel):
    """Base deployment model."""

    id: str = Field(description="The deployment ID")
    display_name: str | None = Field(None, description="The deployment display name")
    display_version: str | None = Field(None, description="The model display version")
    icon_url: str | None = Field(None, description="The deployment icon URL")
    description: str | None = Field(None, description="The deployment description")
    description_keywords: list[str] | None = Field(None, description="Description keywords")
    input_attachment_types: list[str] | None = Field(None, description="Input attachment types")


class DIALModel(DIALDeploymentBase):
    """Model deployment from DIAL API."""

    tokenizer_model: str | None = Field(None, description="The model tokenizer model")
    capabilities: DIALModelCapabilities = Field(description="The model capabilities")
    limits: DIALLimits | None = Field(None, description="The model limits")
    features: DIALModelFeatures = Field(default_factory=DIALModelFeatures)
    pricing: DIALModelPricing | None = Field(None, description="The model pricing")


class DIALApplication(DIALDeploymentBase):
    """Application deployment from DIAL API."""

    application: str = Field(description="The application name")
    features: DIALModelFeatures = Field(default_factory=DIALModelFeatures)


# ============================================================================
# Config Generation Logic
# ============================================================================


def _get_dial_models(dial_url: str, api_key: str) -> list[dict[str, Any]]:
    """Fetch models from DIAL API."""
    headers = {"Api-Key": api_key}
    response = httpx.get(f"{dial_url}/openai/models", headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()["data"]


def _get_dial_applications(dial_url: str, api_key: str) -> list[dict[str, Any]]:
    """Fetch applications from DIAL API."""
    headers = {"Api-Key": api_key}
    response = httpx.get(f"{dial_url}/openai/applications", headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()["data"]


def _replace_envs(config_models: dict[str, Any]) -> dict[str, Any]:
    """Replace environment variable placeholders in config."""
    for model_config in config_models.values():
        for upstream in model_config.get("upstreams", []):
            upstream["endpoint"] = replace_env(upstream["endpoint"])
            upstream["key"] = replace_env(upstream["key"])
    return config_models


def _to_config_model(
    model: dict[str, Any], dial_url: str, api_key: str
) -> tuple[str, dict[str, Any]] | None:
    """Convert DIAL model to config format."""
    parsed = DIALModel.model_validate(model)
    capabilities = parsed.capabilities

    if capabilities.chat_completion:
        type_ = "chat"
    elif capabilities.embeddings:
        type_ = "embedding"
    else:
        return None

    model_config: dict[str, Any] = {
        "type": type_,
        "endpoint": f"http://adapter-dial:5000/openai/deployments/{parsed.id}{MODEL_TYPE_TO_PATH[type_]}",
        "upstreams": [
            {
                "endpoint": f"{dial_url}/openai/deployments/{parsed.id}{MODEL_TYPE_TO_PATH[type_]}",
                "key": api_key,
            }
        ],
    }

    serialized = parsed.model_dump(exclude_none=True, exclude_defaults=True, by_alias=True)

    for field in [
        "displayName",
        "displayVersion",
        "description",
        "descriptionKeywords",
        "iconUrl",
        "inputAttachmentTypes",
        "tokenizerModel",
        "limits",
        "pricing",
    ]:
        if field in serialized and serialized[field]:
            model_config[field] = serialized[field]

    if parsed.features:
        features = parsed.features.to_config()
        if features:
            model_config["features"] = features

    return parsed.id, model_config


def _to_config_application(
    application: dict[str, Any], dial_url: str, api_key: str
) -> tuple[str, dict[str, Any]]:
    """Convert DIAL application to config format."""
    parsed = DIALApplication.model_validate(application)

    app_config: dict[str, Any] = {
        "type": "chat",
        "endpoint": f"http://adapter-dial:5000/openai/deployments/{parsed.id}/chat/completions",
        "upstreams": [
            {
                "endpoint": f"{dial_url}/openai/deployments/{parsed.id}/chat/completions",
                "key": api_key,
            }
        ],
    }

    serialized = parsed.model_dump(exclude_none=True, exclude_defaults=True, by_alias=True)

    for field in [
        "displayName",
        "displayVersion",
        "description",
        "descriptionKeywords",
        "iconUrl",
        "inputAttachmentTypes",
    ]:
        if field in serialized and serialized[field]:
            app_config[field] = serialized[field]

    return parsed.id, app_config


async def generate_handler(
    template: str | None = None,
    config: str | None = None,
    applications: str | None = None,
) -> None:
    """Generate DIAL Core configuration from remote DIAL deployments."""
    dial_url = cli_settings.remote_dial_url
    api_key = cli_settings.remote_dial_api_key

    if not dial_url or not api_key:
        print_error(
            "Remote DIAL credentials not configured.\n"
            "Set STATGPT_CLI_REMOTE_DIAL_URL and STATGPT_CLI_REMOTE_DIAL_API_KEY"
        )
        return

    if not template:
        print_error("Template file is required: --template <path>")
        return

    if not config:
        print_error("Output config file is required: --config <path>")
        return

    if not os.path.exists(template):
        print_error(f"Template file not found: {template}")
        return

    # Parse application IDs to include
    app_ids: set[str] = set()
    if applications:
        app_ids = {aid.strip() for aid in applications.split(",") if aid.strip()}

    print_info(f"Fetching deployments from: {dial_url}")

    try:
        # Load template
        with open(template) as f:
            config_template = json.load(f)

        # Fetch models and applications
        dial_models = _get_dial_models(dial_url, api_key)
        dial_applications = _get_dial_applications(dial_url, api_key)

        print_info(f"Found {len(dial_models)} models and {len(dial_applications)} applications")

        # Convert models
        config_models: dict[str, Any] = {}
        for model in dial_models:
            result = _to_config_model(model, dial_url, api_key)
            if result:
                model_id, model_config = result
                config_models[model_id] = model_config
                console.print(f"  [green]\u2713[/green] Model: {model_id}")

        # Convert applications (only those in app_ids if specified)
        config_applications: dict[str, Any] = {}
        for app in dial_applications:
            app_id, app_config = _to_config_application(app, dial_url, api_key)
            if not app_ids or app_id in app_ids:
                config_applications[app_id] = app_config
                console.print(f"  [green]\u2713[/green] Application: {app_id}")

        # Update template
        _replace_envs(config_template.get("models", {}))
        config_template.setdefault("models", {}).update(config_models)
        config_template["models"].update(config_applications)

        # Update role limits
        limits: dict[str, Any] = (
            config_template.get("roles", {}).get("default", {}).get("limits", {})
        )
        limits.clear()
        limits.update({model_id: {} for model_id in config_template.get("models", {}).keys()})
        limits.update({app_id: {} for app_id in config_template.get("applications", {}).keys()})

        # Write output
        os.makedirs(os.path.dirname(config) or ".", exist_ok=True)
        with open(config, "w") as f:
            json.dump(config_template, f, indent=2)

        print_success(f"Configuration saved to: {config}")

    except httpx.HTTPError as e:
        print_error(f"Failed to fetch from DIAL API: {e}")
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in template: {e}")
    except Exception as e:
        print_error(f"Failed to generate config: {e}")
        raise


# ============================================================================
# Command Definitions
# ============================================================================


generate_command = Command(
    name="generate",
    description="Generate DIAL Core configuration from remote DIAL deployments",
    handler=generate_handler,
    args=[
        CommandArg(
            name="template",
            description="Path to template configuration file",
            required=True,
        ),
        CommandArg(
            name="config",
            description="Path to output configuration file",
            required=True,
        ),
        CommandArg(
            name="applications",
            description="Comma-separated list of application IDs to include",
        ),
    ],
)

# Command group
config_group = CommandGroup(
    name="config",
    description="Configuration generation",
)
config_group.add_command(generate_command)
