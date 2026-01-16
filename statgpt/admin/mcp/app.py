from fastmcp import FastMCP
from fastmcp.dependencies import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from statgpt.admin.mcp.prompts.config_creation import add_config_for_dataset
from statgpt.admin.mcp.services.dataset_exploration import (
    generate_id,
    get_data_sources_list,
    get_dataset_structure,
    get_datasets,
    validate_dataset_config,
)
from statgpt.common.models import get_session_contex_manager
from statgpt.common.schemas.data_source import DataSource

mcp = FastMCP(name="statgpt_dataset_exploration")

# Register tools
mcp.tool()(get_datasets)
mcp.tool()(get_dataset_structure)
mcp.tool()(validate_dataset_config)
mcp.tool()(generate_id)


# Register tools with dependencies
@mcp.tool
async def get_data_sources(
    session: AsyncSession = Depends(get_session_contex_manager),  # type: ignore[arg-type]
) -> list[DataSource]:
    """Return list of all data sources."""
    return await get_data_sources_list(session, limit=None, offset=0, data_source_id=None)


# Register prompts
mcp.prompt()(add_config_for_dataset)
