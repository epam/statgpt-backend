from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.services.chat_facade import VersionedDataSet
from statgpt.app.utils.dial_stages import optional_timed_stage
from statgpt.common.data.base import DataSet


async def get_available_datasets(inputs: dict) -> dict[str, VersionedDataSet]:
    data_service = ChainParameters.get_data_service(inputs)
    auth_context = ChainParameters.get_auth_context(inputs)
    choice = ChainParameters.get_choice(inputs)
    state = ChainParameters.get_state(inputs)
    debug = state.show_debug_stages

    name = '[DEBUG] Get available datasets'
    with optional_timed_stage(choice=choice, name=name, enabled=debug):
        datasets = await data_service.list_available_datasets(auth_context)
        return {ds.data.entity_id: ds for ds in datasets}


async def get_dataset_by_source_id(inputs: dict, dataset_id: str) -> DataSet | None:
    data_service = ChainParameters.get_data_service(inputs)
    auth_context = ChainParameters.get_auth_context(inputs)
    choice = ChainParameters.get_choice(inputs)
    state = ChainParameters.get_state(inputs)
    debug = state.show_debug_stages

    name = f'[DEBUG] Get dataset by ID: {dataset_id}'
    with optional_timed_stage(choice=choice, name=name, enabled=debug):
        dataset = await data_service.get_dataset_by_source_id(auth_context, dataset_id)
        return dataset
