import abc


class IAuthorizer(abc.ABC):

    @abc.abstractmethod
    async def get_authorization_headers(self):
        pass
