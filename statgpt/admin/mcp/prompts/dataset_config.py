from fastmcp.prompts import Message
from fastmcp.server.providers import LocalProvider

mcp_prompts = LocalProvider()


@mcp_prompts.prompt()
def validate_generated_config(user_request: str):
    return [
        Message(
            content=f"""
You are a dataset configuration assistant. When you don't know something and you can't find the answer in the user request, ask user for clarification.
USER REQUEST: {user_request}

User will ask you to validate the generated dataset configuration YAML(it is either presented in file or take it from history).
Given a generated dataset configuration YAML, validate it and return the validation result.
Config should pass config validation(Do not forget that your config should have common dimensions and attributes(and other anchors) included).
Output validation results/status. If it does not pass provide explanation why it does not pass.
""",
            role="user",
        )
    ]


@mcp_prompts.prompt()
def add_dataset_config(user_request: str):
    """Prompt messages for dataset configuration creation from user request."""
    return [
        Message(
            content=f"""
You are a dataset configuration assistant. When you don't know something and you can't find the answer in the user request, ask user for clarification.

Given a single user request, create a complete dataset configuration YAML.
USER REQUEST: {user_request}

The request may mention a dataset title/URN and optionally a client/data source(Clients and data sources are the same thing).
If the client is not specified, ask user to specify the client and then search across all clients/data sources to find the matching client and its dataset.
But If there are several similar datasets that match user request you must ask user to choose one from the list of matching datasets.
(You are not allowed to generate/create your own URN or titles, you must use the ones you found by search.)


Use existing dataset configurations for the same client as references of how configuration should look like.
Prefer English configs and reuse shared anchors (settings, details, provider).
Use the client’s channel configuration to identify named entity types.

If you encounter errors in the tools related to getting dataset info, ask user for clarification.
For the title review how it is done in the existing configs. And use the same pattern.

When generating dimensions config:
- Use existing YAML configs for the same client as the source of truth for dimension structure; derive - their patterns and infer the logic from similar datasets.
- Infer dimension types and required flags by dataset structure and sample dimension values.
  Use dimension name and sample values to identify which dimensions are essential and which can be optional.
- If dimension can be explained to average human and is general(not dataset specific) dimension (or in named entity types it should be NON_INDICATOR) then it should be NON_INDICATOR,
  otherwise if this dimension is specialized for this dataset then it should be INDICATOR. (For better understanding look at previous dataset configurations)
- isRequired flag means that at least one marked dimension must have non-empty terms in a query.
  Set it to `true` only for core crucial indicator dimensions that define what is being measured.
  Do not set it for dimensions that answer "in what breakdown/aspect is something measured" (e.g. unit_of_measure, adjustment) or for non-indicators.
  Each dataset must have at least one required dimension.


After generating all dimensions, review named entity types and update them as needed:
for each NON_INDICATOR dimension from generated config:
- Check if the dimension can be mapped to any existing named entity type.
- If it can, it is okay to leave it as is.
- If it cannot, then create new named entity type for this dimension.

For the Description:
1) Use the description from the dataset structure if it is available and contains meaningful information, but do not fill it just set is as None/null as we will retrieve it from the dataset structure later.
2) If the description from the dataset structure is not available/not meaningful, write a dataset description based on:
   - the dataset title,
   - indicators meanings,
   - example dimension combinations that show what the dataset measures.

Output:
Save the configuration to a datasets configs file, then validate it. The configuration must pass validation.
If validation succeeds, return a complete, production-ready dataset YAML that matches existing client configurations.
If not return the validation results/status to user and ask him for clarification.
""",
            role="user",
        )
    ]
