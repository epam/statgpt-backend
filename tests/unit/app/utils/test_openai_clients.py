import asyncio

from openai._models import FinalRequestOptions

from statgpt.app.utils.openai import get_async_client

_API_KEY = 'per-request-key'
_ACCESS_TOKEN = 'user-access-token'


def _request_auth_headers(access_token: str | None) -> dict[str, str]:
    """The auth headers the client actually puts on the wire for a chat completion."""

    client = get_async_client(
        api_key=_API_KEY, azure_endpoint='http://dial-core', access_token=access_token
    )

    async def _build() -> dict[str, str]:
        options = FinalRequestOptions.construct(
            method='post', url='/chat/completions', json_data={}
        )
        request = client._build_request(await client._prepare_options(options))
        return {
            name: value
            for name, value in request.headers.items()
            if name.lower() in ('api-key', 'authorization')
        }

    return asyncio.run(_build())


def test_access_token_is_sent_alongside_api_key():
    headers = _request_auth_headers(_ACCESS_TOKEN)

    assert headers == {'api-key': _API_KEY, 'authorization': f'Bearer {_ACCESS_TOKEN}'}


def test_without_access_token_no_user_token_is_sent():
    """No token to forward: the bearer header keeps the SDK's default (the api key)."""

    headers = _request_auth_headers(None)

    assert headers == {'api-key': _API_KEY, 'authorization': f'Bearer {_API_KEY}'}
