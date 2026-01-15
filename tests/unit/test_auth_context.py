from unittest.mock import MagicMock, patch

import pytest

from statgpt.app.security.auth_context import (
    SystemUserAuthContext,
    UserAuthContext,
    create_auth_context,
)
from statgpt.app.security.exceptions import InsufficientRoleError


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
