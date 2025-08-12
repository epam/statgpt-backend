from datetime import datetime

from langchain_core.tools import tool


@tool
def datetime_now():
    """Get current date and time."""
    return datetime.now().isoformat()
