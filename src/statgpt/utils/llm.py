from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def langchain_message_to_str(message: BaseMessage):
    if isinstance(message, HumanMessage):
        role = 'User'
    elif isinstance(message, AIMessage):
        role = 'AI'
    else:
        raise ValueError(f"Unexpected message type: {type(message)}")
    return f"{role}: {message.content}"


def langchain_history_to_str(history: list[BaseMessage]):
    return '\n'.join(langchain_message_to_str(m) for m in history)


def wrap_in_braces(text: str):
    """Wrap input text in braces. Useful to pass dynamic field names to langchain prompt templates"""
    return f'{{{text}}}'
