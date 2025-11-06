from common.schemas import DataQueryDetails

from .base import BaseQueryConstructor
from .composite import CompositeQueryConstructor
from .iterative import IterativeQueryConstructor
from .simple import SimpleQueryConstructor


class QueryConstructorFactory:

    @classmethod
    def create(cls, config: DataQueryDetails) -> BaseQueryConstructor:
        return CompositeQueryConstructor(
            config=config,
            constructors=[
                SimpleQueryConstructor(config),
                IterativeQueryConstructor(config),
            ],
        )
