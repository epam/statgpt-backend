from fastmcp.prompts import Message


def add_config_for_dataset(user_request: str):
    """Prompt messages for dataset configuration creation from user request."""
    return [
        Message(
            content=f"""
You are a dataset configuration assistant.

Given a single user request, create a complete dataset configuration YAML.
USER REQUEST: {user_request}

The request may mention a dataset topic/title and optionally a client/data source(Clients and data sources are the same thing).
If the client is not specified, ask user to specify the client and then search across all clients/data sources to find the best-matching client and its dataset.
Identify the dataset(URN, title, etc). But If there are several similar datasets that match user request
you must to ask clarification questions(which to choose) listing these similar datasets and their URNs, titles.


Use existing dataset configurations for the same client as references.
Prefer English configs and reuse shared anchors (settings, details, provider).
Use the client’s channel configuration to identify named-entity dimension types.

When generating dimensions config:
- Use existing YAML configs for the same client as the source of truth for dimension structure; derive - their patterns and infer the logic from similar datasets.
- Infer dimension types and required flags by dataset structure and sample dimension values.
  Use typical full combinations to identify which dimensions are essential and which can be optional.
- If a dimension is a dataset specific (related only for this dataset/its ds domain), then its dimension type: INDICATOR (all other dimensions should be NON_INDICATOR but if dimension is in named entity types it should be NON_INDICATOR)
- Mark a dimension as isRequired: true only if omitting it makes the dataset ambiguous or incomplete; otherwise, keep it isRequired: false.


For the Description:
Write a dataset description based on:
- the dataset title,
- indicators meanings,
- example dimension combinations that show what the dataset measures.

Output:
Correct config should pass config validation.
If it passes validation, return a complete, production-ready dataset configuration YAML consistent with existing client configs.
""",
            role="user",
        )
    ]
