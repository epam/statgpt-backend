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

To add dataset configuration:
1. Understand what dataset should be added.
User might reference datasets by title/URN/client/data source.
User might ask to onboard multiple datasets.
If it's unclear what dataset user wants to onboard, ask for clarification
2. Understand what client and data source required dataset belongs to
3. Understand what dataset config file to update, if not specified by user
4. Extract details for referenced dataset using dataset structure tool
5. Generate config based on dataset structure tool output. Use uuid tool to generate uuid.
6. Check if you need to add new Named Entity types to channel config file
7. Fill dataset description field
8. Validate generated config using validation tool

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

Do not add default queries for any dimensions.
Remember to check if channel's list of Named Entity types needs to be updated
with new concepts from NON_INDICATOR dimensions!

### Required indicator dimensions

- `isRequired` field could be set to `true` ONLY for INDICATOR dimensions
- fill it with `true` ONLY for crucial and essential indicator dimensions.
Omitting such dimensions usually makes query meaningless or non-informative.
- any query not specifying filter for AT LEAST ONE REQUIRED INDICATOR dimension
will be rejected by StatGPT
- Each dataset must have at least one required indicator dimension

## Dataset description

Dataset description field could be filled either with text or `null`:
- first, analyze dataset description in dataset structure tool output:
    - if it's meaningful, set `null` in description field in dataset config.
    this means that description will be retrieved from data source on each access.
    - if description is not meaningful / not present, write it yourself
    based on dataset title and structure
- you MUST EXPLICLTY REASON ABOUT DATASET DESCRIPTION from structure tool output!

## Validation

- After generating dataset config, ALWAYS VALIDATE GENERATED CONFIG using appropriate tool
- Generated dataset config must pass validation
- On validation errors, try to fix them. If needed, ask user for help
- ALWAYS COMMUNICATE VALIDATION RESULTS TO USER
""",
            role="user",
        )
    ]
