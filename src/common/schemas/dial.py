from pydantic import BaseModel


class Pricing(BaseModel):
    unit: str
    prompt: float
    completion: float
