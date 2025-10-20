from common.data.sdmx import Sdmx21DataSet

from .base import BaseQueryConstructor
from .composite import CompositeQueryConstructor
from .iterative import IterativeQueryConstructor
from .simple import SimpleQueryConstructor


class QueryConstructorFactory:

    @classmethod
    def create(cls, dataset: Sdmx21DataSet) -> BaseQueryConstructor:
        return CompositeQueryConstructor(
            constructors=[
                SimpleQueryConstructor(),
                IterativeQueryConstructor(),
            ]
        )
