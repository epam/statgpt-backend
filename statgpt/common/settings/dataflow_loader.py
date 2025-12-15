from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataflowLoaderSettings(BaseSettings):
    """
    Settings for DataflowLoader
    """

    model_config = SettingsConfigDict(env_prefix="dataflow_loader_")

    dataset_concurrency_limit: int = Field(
        10, description="Maximum concurrency of dataset fetching tasks"
    )
    code_list_concurrency_limit: int = Field(
        10, description="Maximum concurrency of code list fetching tasks"
    )
    concept_scheme_concurrency_limit: int = Field(
        10, description="Maximum concurrency of concept scheme fetching tasks"
    )
