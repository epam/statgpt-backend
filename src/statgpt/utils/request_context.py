from pydantic import BaseModel


# TODO: remove the `RequestContext` class from the code base.
class RequestContext(BaseModel):
    api_key: str
    inputs: dict | None = None
