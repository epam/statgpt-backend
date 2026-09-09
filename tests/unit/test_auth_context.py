from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aidial_client import DialException

from statgpt.app.security.auth_context import (
    SystemUserAuthContext,
    UserAuthContext,
    create_auth_context,
)
from statgpt.app.security.exceptions import InsufficientRoleError


@asynccontextmanager
async def _patch_dial_roles(roles: list[str] | None = None, *, raises: Exception | None = None):
    """Patch `dial_client_factory` so `dial.user.info()` yields `roles` (or raises `raises`).

    Yields a mock whose `.bearer_token` records the token the factory was called with, so tests
    can assert DIAL is queried with the caller's access token rather than the api key.
    """
    captured = MagicMock(bearer_token=None)

    @asynccontextmanager
    async def _factory(*args, **kwargs):
        captured.bearer_token = kwargs.get("bearer_token")
        dial = MagicMock()
        dial.user.info = (
            AsyncMock(side_effect=raises)
            if raises
            else AsyncMock(return_value=MagicMock(roles=roles))
        )
        yield dial

    with patch("statgpt.common.auth.auth_context.dial_client_factory", _factory):
        yield captured


@pytest.fixture
def mock_request():
    """Create a mock Request object."""
    request = MagicMock()
    # ai-dial-sdk >= 0.30 uses `.bearer_token` (token WITHOUT "Bearer " prefix)
    request.bearer_token = None
    request.api_key = "test-api-key"
    return request


class TestCreateAuthContext:
    @pytest.mark.asyncio
    async def test_bearer_token_present_returns_user_context(self, mock_request):
        """When bearer_token is present, return UserAuthContext regardless of bearer_token_required."""
        mock_request.bearer_token = "token123"

        context = await create_auth_context(mock_request, bearer_token_required=False)

        assert isinstance(context, UserAuthContext)

    @pytest.mark.asyncio
    async def test_bearer_token_present_with_bearer_token_required_returns_user_context(
        self, mock_request
    ):
        """When bearer_token is present and bearer_token_required=True, still return UserAuthContext."""
        mock_request.bearer_token = "token123"

        context = await create_auth_context(mock_request, bearer_token_required=True)

        assert isinstance(context, UserAuthContext)

    @pytest.mark.asyncio
    async def test_no_bearer_token_and_not_required_returns_user_context(self, mock_request):
        """When no bearer_token and bearer_token_required=False, return UserAuthContext."""
        mock_request.bearer_token = None

        context = await create_auth_context(mock_request, bearer_token_required=False)

        assert isinstance(context, UserAuthContext)

    @pytest.mark.asyncio
    @patch("statgpt.app.security.auth_context._check_roles")
    @patch("statgpt.app.security.auth_context.dial_app_settings")
    async def test_no_jwt_required_with_allowed_role_returns_system_context(
        self, mock_settings, mock_check_roles, mock_request
    ):
        """When no bearer_token, bearer_token_required=True, and user has allowed role."""
        mock_request.bearer_token = None
        mock_settings.system_user_context_roles_set = {"evaluator"}
        mock_check_roles.return_value = True

        context = await create_auth_context(mock_request, bearer_token_required=True)

        assert isinstance(context, SystemUserAuthContext)
        mock_check_roles.assert_called_once_with(mock_request, {"evaluator"})

    @pytest.mark.asyncio
    @patch("statgpt.app.security.auth_context._check_roles")
    @patch("statgpt.app.security.auth_context.dial_app_settings")
    async def test_no_jwt_required_without_allowed_role_raises(
        self, mock_settings, mock_check_roles, mock_request
    ):
        """When no bearer_token, bearer_token_required=True, and user lacks allowed role."""
        mock_request.bearer_token = None
        mock_settings.system_user_context_roles_set = {"evaluator"}
        mock_check_roles.return_value = False

        with pytest.raises(InsufficientRoleError):
            await create_auth_context(mock_request, bearer_token_required=True)

    @pytest.mark.asyncio
    @patch("statgpt.app.security.auth_context.dial_app_settings")
    async def test_no_jwt_required_no_roles_configured_raises(self, mock_settings, mock_request):
        """When no bearer_token, bearer_token_required=True, but no roles configured."""
        mock_request.bearer_token = None
        mock_settings.system_user_context_roles_set = set()

        with pytest.raises(InsufficientRoleError):
            await create_auth_context(mock_request, bearer_token_required=True)


class TestUserAuthContext:
    def test_is_system_returns_false(self, mock_request):
        """UserAuthContext.is_system should return False."""
        context = UserAuthContext(mock_request)
        assert context.is_system is False

    def test_dial_access_token_returns_bearer_token(self, mock_request):
        """dial_access_token should return bearer_token as-is."""
        mock_request.bearer_token = "token123"
        context = UserAuthContext(mock_request)
        assert context.dial_access_token == "token123"

    def test_dial_access_token_returns_none_when_no_bearer_token(self, mock_request):
        """dial_access_token should return None when no bearer_token."""
        mock_request.bearer_token = None
        context = UserAuthContext(mock_request)
        assert context.dial_access_token is None

    @pytest.mark.asyncio
    async def test_get_roles_returns_dial_roles(self, mock_request):
        """Roles come from DIAL's user-info endpoint, queried with the caller's access token."""
        mock_request.bearer_token = "token123"
        context = UserAuthContext(mock_request)
        async with _patch_dial_roles(["viewer", "dr_access"]) as captured:
            assert await context.get_roles() == ["viewer", "dr_access"]
        # The access token — not the api key — is what DIAL is queried with.
        assert captured.bearer_token == "token123"

    @pytest.mark.asyncio
    async def test_get_roles_empty_without_token(self, mock_request):
        """No access token means roles can't be resolved, so an empty list is returned."""
        mock_request.bearer_token = None
        context = UserAuthContext(mock_request)
        async with _patch_dial_roles(["dr_access"]):
            assert await context.get_roles() == []

    @pytest.mark.asyncio
    async def test_get_roles_empty_on_dial_error(self, mock_request):
        """Fail closed: a DIAL error while resolving roles yields an empty list."""
        mock_request.bearer_token = "token123"
        context = UserAuthContext(mock_request)
        async with _patch_dial_roles(raises=DialException("boom")):
            assert await context.get_roles() == []

    @pytest.mark.asyncio
    async def test_has_role_true_when_present(self, mock_request):
        mock_request.bearer_token = "token123"
        context = UserAuthContext(mock_request)
        async with _patch_dial_roles(["viewer", "dr_access", "editor"]):
            assert await context.has_role("dr_access") is True

    @pytest.mark.asyncio
    async def test_has_role_false_when_absent(self, mock_request):
        mock_request.bearer_token = "token123"
        context = UserAuthContext(mock_request)
        async with _patch_dial_roles(["viewer", "editor"]):
            assert await context.has_role("dr_access") is False

    @pytest.mark.asyncio
    async def test_has_role_false_without_token(self, mock_request):
        """Fail closed: no token means roles can't be resolved, so access is denied."""
        mock_request.bearer_token = None
        context = UserAuthContext(mock_request)
        async with _patch_dial_roles(["dr_access"]):
            assert await context.has_role("dr_access") is False


class TestSystemUserAuthContext:
    def test_is_system_returns_true(self, mock_request):
        """SystemUserAuthContext.is_system should return True."""
        context = SystemUserAuthContext(mock_request)
        assert context.is_system is True

    def test_dial_access_token_returns_none(self, mock_request):
        """SystemUserAuthContext.dial_access_token should always return None."""
        mock_request.bearer_token = "token123"
        context = SystemUserAuthContext(mock_request)
        assert context.dial_access_token is None

    @pytest.mark.asyncio
    async def test_role_gating_fails_closed(self, mock_request):
        """A system user exposes no access token, so DIAL role resolution yields nothing and a
        role-gated check is always denied (system users bypass the gate at a higher level)."""
        mock_request.bearer_token = "token123"
        context = SystemUserAuthContext(mock_request)
        async with _patch_dial_roles(["dr_access"]):
            assert await context.get_roles() == []
            assert await context.has_role("dr_access") is False
