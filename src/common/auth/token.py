class TokenResponseI:
    @property
    def access_token(self) -> str:
        raise NotImplementedError()

    @property
    def expires_at(self) -> int:
        raise NotImplementedError()

    @property
    def refresh_token(self) -> str | None:
        raise NotImplementedError()
