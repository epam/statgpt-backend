import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from statgpt.admin.services.dataset import AdminPortalDataSetService
from statgpt.common.schemas import (
    ChannelDatasetUpdateStatus,
    ChannelDatasetVersion,
    PreprocessingStatusEnum,
)

_RESOLVED_URN_1_0 = {'agency_id': 'IMF.STA', 'resource_id': 'CPI', 'version': '1.0'}
_RESOLVED_URN_2_0 = {'agency_id': 'IMF.STA', 'resource_id': 'CPI', 'version': '2.0'}
_OTHER_AGENCY_URN = {'agency_id': 'IMF', 'resource_id': 'CPI', 'version': '1.0'}


def _make_handler(
    *,
    indexing_hash: str,
    structure_hash: str,
    resolve_returns: dict | None = None,
    reresolve_returns: dict | None = None,
    merge_returns: dict | None = None,
) -> MagicMock:
    handler = MagicMock()
    handler.resolve_config = AsyncMock(return_value=("resolved", resolve_returns or {}))
    handler.reresolve_config = AsyncMock(return_value=("reresolved", reresolve_returns or {}))
    parsed = MagicMock()
    parsed.indexing_hash = indexing_hash
    handler.parse_data_set_config = MagicMock(return_value=parsed)
    handler.get_structure_hash_and_metadata = AsyncMock(return_value=(structure_hash, {}))
    handler.merge_config_with_resolved = MagicMock(return_value=merge_returns or {})
    return handler


def _make_last_completed(
    *,
    resolved_config: dict | None,
    indexing_config_hash: str = "cfg_hash",
    structure_hash: str = "struct_hash",
) -> ChannelDatasetVersion:
    now = datetime.datetime(2026, 1, 1)
    return ChannelDatasetVersion(
        id=1,
        created_at=now,
        updated_at=now,
        channel_dataset_id=1,
        version=1,
        preprocessing_status=PreprocessingStatusEnum.COMPLETED,
        creation_reason="initial reindex",
        reason_for_failure=None,
        pointer_to=None,
        indexing_config_hash=indexing_config_hash,
        structure_metadata={},
        structure_hash=structure_hash,
        indicator_dimensions_hash="ih",
        non_indicator_dimensions_hash="nh",
        special_dimensions_hash="sh",
        resolved_config=resolved_config,
        indexing_stats=None,
    )


class TestClassifyConfigUpdate:
    """Tests for AdminPortalDataSetService._classify_config_update."""

    @pytest.mark.asyncio
    async def test_user_kept_latest_no_changes_returns_auto_updated(self) -> None:
        """Regression guard for #212: user submits `version: latest`, registry still resolves
        to the same canonical URN as `last_completed.resolved_config`. Must not require reindex."""
        resolved_same = {'urn': _RESOLVED_URN_1_0, 'foo': 'bar'}
        # reresolve_config returns the previous_resolved_config when URN matches.
        handler = _make_handler(
            indexing_hash="cfg_hash",
            structure_hash="struct_hash",
            reresolve_returns=resolved_same,
        )
        last_completed = _make_last_completed(resolved_config=resolved_same)

        status, new_resolved, reasons = await AdminPortalDataSetService._classify_config_update(
            handler=handler,
            current_config={
                'urn': {'agency_id': 'IMF.STA', 'resource_id': 'CPI', 'version': 'latest'}
            },
            last_completed=last_completed,
            auth_context=MagicMock(),
        )

        assert status is ChannelDatasetUpdateStatus.AUTO_UPDATED
        assert reasons == []
        assert new_resolved is resolved_same
        handler.reresolve_config.assert_awaited_once()
        handler.resolve_config.assert_not_called()

    @pytest.mark.asyncio
    async def test_version_changed_returns_needs_reindex(self) -> None:
        """Explicit URN version bump 1.0 → 2.0, even with identical structure, must require reindex."""
        previous_resolved = {'urn': _RESOLVED_URN_1_0, 'foo': 'bar'}
        new_resolved = {'urn': _RESOLVED_URN_2_0, 'foo': 'bar'}
        handler = _make_handler(
            indexing_hash="cfg_hash",
            structure_hash="struct_hash",
            reresolve_returns=new_resolved,
        )
        last_completed = _make_last_completed(resolved_config=previous_resolved)

        status, _, reasons = await AdminPortalDataSetService._classify_config_update(
            handler=handler,
            current_config={'urn': _RESOLVED_URN_2_0},
            last_completed=last_completed,
            auth_context=MagicMock(),
        )

        assert status is ChannelDatasetUpdateStatus.NEEDS_REINDEX
        assert "URN changed" in reasons

    @pytest.mark.asyncio
    async def test_agency_changed_returns_needs_reindex(self) -> None:
        previous_resolved = {'urn': _RESOLVED_URN_1_0}
        new_resolved = {'urn': _OTHER_AGENCY_URN}
        handler = _make_handler(
            indexing_hash="cfg_hash",
            structure_hash="struct_hash",
            reresolve_returns=new_resolved,
        )
        last_completed = _make_last_completed(resolved_config=previous_resolved)

        status, _, reasons = await AdminPortalDataSetService._classify_config_update(
            handler=handler,
            current_config={'urn': _OTHER_AGENCY_URN},
            last_completed=last_completed,
            auth_context=MagicMock(),
        )

        assert status is ChannelDatasetUpdateStatus.NEEDS_REINDEX
        assert "URN changed" in reasons

    @pytest.mark.asyncio
    async def test_structure_hash_drift_returns_needs_reindex(self) -> None:
        """URN unchanged but upstream dataflow structure has drifted since last reindex."""
        resolved_same = {'urn': _RESOLVED_URN_1_0}
        handler = _make_handler(
            indexing_hash="cfg_hash",
            structure_hash="NEW_struct_hash",
            reresolve_returns=resolved_same,
        )
        last_completed = _make_last_completed(
            resolved_config=resolved_same, structure_hash="OLD_struct_hash"
        )

        status, _, reasons = await AdminPortalDataSetService._classify_config_update(
            handler=handler,
            current_config={'urn': _RESOLVED_URN_1_0},
            last_completed=last_completed,
            auth_context=MagicMock(),
        )

        assert status is ChannelDatasetUpdateStatus.NEEDS_REINDEX
        assert "structure hash changed" in reasons
        assert "URN changed" not in reasons

    @pytest.mark.asyncio
    async def test_indexing_hash_changed_returns_needs_reindex(self) -> None:
        """User edited an IndexingField-marked field (e.g. dimensions config)."""
        resolved_same = {'urn': _RESOLVED_URN_1_0}
        handler = _make_handler(
            indexing_hash="NEW_cfg_hash",
            structure_hash="struct_hash",
            reresolve_returns=resolved_same,
        )
        last_completed = _make_last_completed(
            resolved_config=resolved_same, indexing_config_hash="OLD_cfg_hash"
        )

        status, _, reasons = await AdminPortalDataSetService._classify_config_update(
            handler=handler,
            current_config={'urn': _RESOLVED_URN_1_0},
            last_completed=last_completed,
            auth_context=MagicMock(),
        )

        assert status is ChannelDatasetUpdateStatus.NEEDS_REINDEX
        assert "indexing config hash changed" in reasons

    @pytest.mark.asyncio
    async def test_legacy_no_resolved_config_falls_back_to_resolve(self) -> None:
        """When last_completed has no resolved_config (pre-tracking data), use resolve_config and
        treat URN as unchanged (cannot compare)."""
        resolved = {'urn': _RESOLVED_URN_1_0}
        handler = _make_handler(
            indexing_hash="cfg_hash",
            structure_hash="struct_hash",
            resolve_returns=resolved,
        )
        last_completed = _make_last_completed(resolved_config=None)

        status, _, reasons = await AdminPortalDataSetService._classify_config_update(
            handler=handler,
            current_config={'urn': _RESOLVED_URN_1_0},
            last_completed=last_completed,
            auth_context=MagicMock(),
        )

        assert status is ChannelDatasetUpdateStatus.AUTO_UPDATED
        assert reasons == []
        handler.resolve_config.assert_awaited_once()
        handler.reresolve_config.assert_not_called()


class TestApplyConfigInternalUrnInvariant:
    """Verifies that _apply_config_internal preserves the resolved URN (regression guard:
    PR #399 removed IndexingField from urn, so merge_config_with_resolved now takes urn
    from current_config — which may still be "latest")."""

    @pytest.mark.asyncio
    async def test_new_version_resolved_config_has_resolved_urn(self) -> None:
        previous_resolved = {'urn': _RESOLVED_URN_1_0, 'citation': None}
        # current_config still has user's input with "latest":
        current_config = {
            'urn': {'agency_id': 'IMF.STA', 'resource_id': 'CPI', 'version': 'latest'},
            'citation': {'provider': 'IMF', 'last_updated': 'now'},
        }
        # merge_config_with_resolved returns user's URN (because urn is non-indexing
        # after PR #399). The override inside _apply_config_internal must replace it.
        merge_result = {
            'urn': {'agency_id': 'IMF.STA', 'resource_id': 'CPI', 'version': 'latest'},
            'citation': {'provider': 'IMF', 'last_updated': 'now'},
        }
        handler = _make_handler(
            indexing_hash="cfg_hash",
            structure_hash="struct_hash",
            merge_returns=merge_result,
        )
        last_completed = _make_last_completed(resolved_config=previous_resolved)

        captured: dict = {}

        def _capture_add(item):
            captured['version'] = item

        async def _refresh(item):
            item.id = 99
            item.created_at = datetime.datetime(2026, 1, 1)
            item.updated_at = datetime.datetime(2026, 1, 1)
            item.version = 2

        session = MagicMock()
        session.add = _capture_add
        session.flush = AsyncMock(return_value=None)
        session.refresh = AsyncMock(side_effect=_refresh)
        service = AdminPortalDataSetService(session=session)

        channel_dataset = MagicMock()
        channel_dataset.id = 1

        new_version = await service._apply_config_internal(
            channel_dataset=channel_dataset,
            last_completed=last_completed,
            handler=handler,
            current_config=current_config,
            resolved_config={'urn': _RESOLVED_URN_1_0},
        )

        # Invariant: stored resolved_config has the canonical resolved URN, not "latest"
        stored_version = captured['version']
        assert stored_version.resolved_config['urn'] == _RESOLVED_URN_1_0
        # Non-URN fields from merge are preserved (citation kept from current/merge):
        assert stored_version.resolved_config['citation'] == {
            'provider': 'IMF',
            'last_updated': 'now',
        }
        # Pointer reuses prior indexed data:
        assert stored_version.pointer_to == last_completed.version_data_id
        # Returned ChannelDatasetVersion is built from the inserted model:
        assert new_version is not None
