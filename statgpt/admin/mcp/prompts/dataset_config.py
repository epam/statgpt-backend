from fastmcp.prompts import Message
from fastmcp.server.providers import LocalProvider

mcp_prompts = LocalProvider()


@mcp_prompts.prompt()
def add_dataset_config():
    """Prompt messages for dataset configuration creation from user request."""
    return [
        Message(
            content="""\
# Adding datasets to StatGPT

Datasets are added/onboarded to StatGPT using yaml dataset configurations files.
Datasets are linked to "channels" - client-specific versions of StatGPT.
Channels have their own config yaml files which are usually located near datasets config files.
"Client" refers to specific organization providing data (e.g. IMF).
Single client might have multiple data sources.

## Flow

When tasked to add dataset config, you MUST ADHERE TO THE FOLLOWING FLOW:

1. Understand what dataset should be added.
    - User might reference datasets by title/URN/client/data source.
    - If it's unclear what dataset user wants to onboard, ask for clarification
    - User might ask to onboard multiple datasets - in this case
    it's better to onboard each dataset in a separate sub-agent, if possible.
    Each sub-agent must follow the same described flow.
2. Understand what client, data source and channel required dataset belongs to.
Use this info to understand what dataset config file to update, if not specified by user.
3. Use dataset structure tool to retrieve
dataset dimensions (with sample values), description, attributes
4. Extract list of Named Entity types from corresponding channel config.
5. EXPLICITLY REASON about DIMENSION TYPES (instructions below).
Use extracted list of entities to help identify NON_INDICATOR dimensions.
6. EXPLICITLY REASON about citation dataset description field (instructions below)
7. EXPLICITLY REASON about indexer config (instructions below)
8. EXPLICITLY REASON about existing yaml anchors, what they represent
and if they are relevant and could be reused for this dataset.
9. EXPLICITLY REASON about pinned columns (instructions below)
10. Generate config based on:
    - dataset structure tool output
    - dataset config schema
    - all previous reasonings (dimension types, citation description,
    indexer config, anchors, pinned columns)
    - use uuid tool to generate uuid
11. Update channel config file with new Named Entity types, if needed.
12. Validate generated config using validation tool

If you encounter errors, communicate them to user, try to fix them, ask user for help if needed.
ALWAYS communicate to user what is the CURRENT STEP from the flow!

## How to generate dataset config

- NEVER fill dataset config based on your knowledge or assumptions, ALWAYS REFER TO TOOL OUTPUTS
- Dataset config follows json schema.
There is a tool providing json schema for "details" field of dataset config -
you MUST follow this schema when generating config!
- Follow patterns from existing dataset configs, prioritizing configs for same client and data source
- Reuse shared yaml anchors if they are present and relevant.
Reason EXPLICITLY about each anchor and how it will be resolved for this dataset

### Dimensions config

You will need to provide config for each dimension of the dataset.
You MUST refer to dataset structure tool output.
Also, use dimension configs from same client as reference.

How to choose dimension type:
- NON_INDICATOR dimensions are concepts that:
are independent of the context and could be easily explained to average human.
Examples include "country", "counterparty", "age", "gender".
    - Each NON_INDICATOR dimension MUST map to some Named Entity type (from channel config file).
    Read the client channel config file to see list of current Named Entity types!
    Mark any dimension that is present in Named Entity types as NON_INDICATOR.
    - It's possible that dataset contains NON_INDICATOR dimensions
    that are not yet present in Named Entity types.
    In this case you MUST update channel's list of Named Entity types!
- INDICATOR dimensions describe the concept being measured.
Examples include "GDP", "unemployment rate", "inflation rate".
If dimension describes a concept that clarifies indicator, it's also an INDICATOR dimension.
General rule is every dimension that is not NON_INDICATOR is an INDICATOR dimension!
- Ignore SPECIAL dimension type for now

REMEMBER TO REASON if channel's list of Named Entity types needs to be updated
with new concepts from NON_INDICATOR dimensions! UPDATE IF NEEDED!

Also:
- DO NOT ADD DEFAULT QUERIES for any dimension
- DO NOT ADD DIMENSION ALIASES (unless it follows patterns in similar dataset configs)

### Required indicator dimensions

- `isRequired` field could be set to `true` ONLY for INDICATOR dimensions
- fill it with `true` ONLY for crucial and essential indicator dimensions.
Omitting such dimensions usually makes query meaningless or non-informative.
- any query not specifying filter for AT LEAST ONE REQUIRED INDICATOR dimension
will be rejected by StatGPT
- Each dataset must have at least one required indicator dimension

## Citation dataset description

Citation `description` field could be filled either with text or `null`:
- first, analyze dataset description from dataset structure tool output:
    - if it's meaningful, set `null` to citation description field.
    this means that citation description will be retrieved from data source on each access.
    - if description is not meaningful / not present,
    write it yourself based on dataset title and structure
- you MUST EXPLICLTY REASON ABOUT DATASET DESCRIPTION from structure tool output!
- NOTE: indexer dataset description choice MUST NOT AFFECT on the content of citation description!

## Indexer config

Dataset indexer has couple of special instructions:

- indexer dataset description must always be non-empty string.
    - If you determined to write `null` to citation dataset description field -
    put dataset description from structure tool output to indexer description without edits!
    - If you filled citation description field with generated description -
    refer to it here with yaml anchor!
    - NOTE: choice of indexer dataset description MUST NOT AFFECT content of citation description!
    - EXPLICITLY REASON about your choice of indexer dataset description!
- `unpack` field determines how indicator dimensions will be indexed.
You must use indicator dimensions sample values to determine correct field value:
    - determine a set of IMPORTANT indicator dimensions,
    describing "what" being measured, not "how" it's measured.
    EXPLICITLY REASON and PROVIDE A LIST OF IMPORTANT indicator dimensions to user
    before proceeding to the next step!
    - next, consider ONLY IMPORTANT indicator dimensions:
    if at least one of important indicator dimensions
    has values with multiple concepts "packed" together,
    often separated by comma (,) - set `unpack` to `true`.
    for example: "GDP, current prices, annual, USD"
    - if there are no such values, set `unpack` to `false`.
    for example: "GDP, per capita"
    - communicate your reasoning to user

## Pinned columns

- pinned columns must list ALL dataset dimensions EXCEPT for time period dimension
- order must be FROM LEAST IMPORTANT TO MOST IMPORTANT dimensions
(use existings dataset configs as reference)

## Validation

- After generating dataset config, ALWAYS VALIDATE GENERATED CONFIG using appropriate tool
- Generated dataset config must pass validation
- On validation errors, try to fix them. If needed, ask user for help
- ALWAYS COMMUNICATE VALIDATION RESULTS TO USER

## Final Notes

- Always adhere to the provided flow
- Provide user with reasoning for each step of the flow
- Adhere to config json schema (accessed via tool)
- Do not forget to update channel config file with new Named Entity types, if needed
""",
            role="user",
        )
    ]
