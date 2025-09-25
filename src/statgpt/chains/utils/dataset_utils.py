from common.schemas import DataSet
from statgpt.chains.parameters import ChainParameters
from statgpt.config import StateVarsConfig
from statgpt.utils.dial_stages import optional_timed_stage


async def get_available_datasets(inputs: dict) -> dict[str, DataSet]:
    data_service = ChainParameters.get_data_service(inputs)
    auth_context = ChainParameters.get_auth_context(inputs)
    choice = ChainParameters.get_choice(inputs)
    state = ChainParameters.get_state(inputs)
    debug = state.get(StateVarsConfig.SHOW_DEBUG_STAGES, False)

    name = '[DEBUG] Get available datasets'
    with optional_timed_stage(choice=choice, name=name, enabled=debug):
        datasets = await data_service.list_available_datasets(auth_context)
        return {ds.entity_id: ds for ds in datasets}


async def get_dataset_by_source_id(inputs: dict, dataset_id: str) -> DataSet | None:
    data_service = ChainParameters.get_data_service(inputs)
    auth_context = ChainParameters.get_auth_context(inputs)
    choice = ChainParameters.get_choice(inputs)
    state = ChainParameters.get_state(inputs)
    debug = state.get(StateVarsConfig.SHOW_DEBUG_STAGES, False)

    name = f'[DEBUG] Get dataset by ID: {dataset_id}'
    with optional_timed_stage(choice=choice, name=name, enabled=debug):
        dataset = await data_service.get_dataset_by_source_id(auth_context, dataset_id)
        return dataset
