import os

from common.utils.files import read_yaml


class AvailableDatasetsPrompt:
    """
    Defult prompt for Available Dataset agent.
    Has the lowest priority, used if no other prompts are found.
    """

    SYSTEM_PROMPT: str


class DialRagPrompts:
    PREFILTER_SYSTEM_PROMPT_DATE: str
    PREFILTER_SYSTEM_PROMPT_LATEST: str
    PREFILTER_SYSTEM_PROMPT_PUBLICATIONS: str
    PREFILTER_SYSTEM_PROMPT_LAST_N_PUBLICATIONS: str


class RouterPrompts:
    """
    Default prompt for Router.
    Has the lowest priority, used if no other prompts are found.
    """

    SYSTEM_PROMPT: str


class SupremeAgentPrompts:
    """
    Default prompt for Supreme Agent.
    Has the lowest priority, used if no other prompts are found.
    """

    SYSTEM_PROMPT: str


class NotSupportedScenariosPrompts:
    """
    Default prompt for Not Supported Scenarios agent.
    """

    CHECKER_PROMPT: str
    RESPONSE_PROMPT: str


def load_prompts():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_dir = os.path.join(script_dir, "assets")

    available_datasets_path = os.path.join(config_dir, "available_datasets.yaml")
    dial_rag_path = os.path.join(config_dir, "dial_rag.yaml")
    router_path = os.path.join(config_dir, "router.yaml")
    supreme_agent_path = os.path.join(config_dir, "supreme_agent.yaml")
    not_supported_scenarios_path = os.path.join(config_dir, "not_supported_scenarios.yaml")

    available_dataset_config = read_yaml(available_datasets_path)
    dial_rag_config = read_yaml(dial_rag_path)
    router_config = read_yaml(router_path)
    supreme_agent_config = read_yaml(supreme_agent_path)
    not_supported_scenarios_config = read_yaml(not_supported_scenarios_path)

    # available datasets prompts
    AvailableDatasetsPrompt.SYSTEM_PROMPT = available_dataset_config['systemPrompt']

    # DIAL RAG prompts
    DialRagPrompts.PREFILTER_SYSTEM_PROMPT_DATE = dial_rag_config["prefilterSystemPromptDate"]
    DialRagPrompts.PREFILTER_SYSTEM_PROMPT_LATEST = dial_rag_config["prefilterSystemPromptLatest"]
    DialRagPrompts.PREFILTER_SYSTEM_PROMPT_PUBLICATIONS = dial_rag_config[
        "prefilterSystemPromptPublicationTypes"
    ]
    DialRagPrompts.PREFILTER_SYSTEM_PROMPT_LAST_N_PUBLICATIONS = dial_rag_config[
        "prefilterSystemPromptLastNPublications"
    ]

    # router prompts
    RouterPrompts.SYSTEM_PROMPT = router_config["systemPrompt"]

    # supreme agent prompts
    SupremeAgentPrompts.SYSTEM_PROMPT = supreme_agent_config["systemPrompt"]

    # not supported scenarios prompts
    NotSupportedScenariosPrompts.CHECKER_PROMPT = not_supported_scenarios_config["checkerPrompt"]
    NotSupportedScenariosPrompts.RESPONSE_PROMPT = not_supported_scenarios_config["responsePrompt"]


load_prompts()
