from fastmcp.prompts import Message


def add_config_for_dataset(user_request: str):
    return [
        Message(
            content=f"""
You are a dataset configuration assistant.

Given a single user request, create a complete dataset configuration YAML.
USER REQUEST: {user_request}

The request may mention a dataset topic/title and optionally a client/data source(Clients and data sources are the same thing).
If the client is not specified, search across all clients/data sources to find the best-matching dataset.
Identify the dataset(URN, title, etc). But If there are several similar datasets that match user request
you need to ask clarification questions(which to choose) listing these similar datasets and their URNs, titles.

Use existing dataset configurations for the same client as references.
Prefer English configs and reuse shared anchors (settings, details, provider).
Use the client’s channel configuration to identify named-entity dimension types.

Dimension rules:
• Always include one main indicator dimension:
  - isRequired: true
  - dimensionType: INDICATOR

• For all other dimensions:
  - Skip dimensions already covered by shared/common dimensions.
  - If the dimension is generally understandable by name (e.g. age) to average human,
    or matches a named-entity type, set dimensionType: NON_INDICATOR.
  - If a dimension is a dataset specific (related only for this dataset/its ds domain), set dimensionType: INDICATOR.
    Or if a dimension is called a BREAKDOWN or BREAKDOWN_CATEGORY then set dimensionType: INDICATOR.
    Name BREAKDOWN/ BREAKDOWN CATEGORY names does not sound dataset specific but the values of this dimensions are actually dataset specific.
  - Determine isRequired by comparing meaningful dimension combinations:
    - If removing the dimension still results in an understandable dataset → isRequired: false
    - If removing it causes loss of essential specification or ambiguity → isRequired: true
For understanding dimension types and whether they are required you need to get indicator combinations for the dataset.
(I suggest to generally explore dataset combinations before to look at all dimensions
and how full combination look like and which dimensions does not convey essential info - then they does not required probably,
and if dimensions is dataset specific - then it is indicator probably)

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
