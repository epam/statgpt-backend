from dataclasses import dataclass


@dataclass(frozen=True)
class User:
    id: str
    username: str
    name: str
