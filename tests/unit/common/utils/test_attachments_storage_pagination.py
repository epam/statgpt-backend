import weakref
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from aidial_client import AsyncDial, DialException
from aidial_client.types.metadata import FileItem, FileMetadata

from statgpt.common.utils.dial import attachments as storage_module
from statgpt.common.utils.dial.attachments import AttachmentsStorage


def _file(name: str) -> FileItem:
    return FileItem(
        bucket="bucket", url=f"files/bucket/folder/{name}", node_type="ITEM", resource_type="FILE"
    )


def _page(items: list[FileItem] | None = None, token: str | None = None) -> FileMetadata:
    return FileMetadata(
        bucket="bucket",
        url="files/bucket/folder/",
        node_type="FOLDER",
        resource_type="FILE",
        items=items,
        next_token=token,
    )


def _storage(get: AsyncMock) -> AttachmentsStorage:
    dial = MagicMock(spec=AsyncDial)
    dial.files = MagicMock()
    dial.files.metadata.get = get
    return AttachmentsStorage(dial)


async def test_iterator_is_lazy_and_fetches_only_the_current_page() -> None:
    first, second, third = _file("one.csv"), _file("two.csv"), _file("three.csv")
    get = AsyncMock(side_effect=[_page([first, second], "next"), _page([third])])
    iterator = _storage(get).iter_files_in_folder("/folder/", bucket="bucket", page_size=2)
    get.assert_not_awaited()

    assert await anext(iterator) is first
    assert await anext(iterator) is second
    assert get.await_count == 1
    assert await anext(iterator) is third
    assert get.await_count == 2
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)

    assert [call.kwargs for call in get.await_args_list] == [
        {"resource": "files", "relative_url": "files/bucket/folder/", "limit": 2, "token": None},
        {"resource": "files", "relative_url": "files/bucket/folder/", "limit": 2, "token": "next"},
    ]


async def test_previous_metadata_pages_are_not_retained() -> None:
    refs: list[weakref.ReferenceType[FileMetadata]] = []

    async def get(**kwargs: Any) -> FileMetadata:
        assert all(ref() is None for ref in refs)
        metadata = _page([_file("one.csv")], "next" if not refs else None)
        refs.append(weakref.ref(metadata))
        return metadata

    storage = _storage(AsyncMock(side_effect=get))
    names = [file.url async for file in storage.iter_files_in_folder("folder", "bucket")]

    assert len(names) == 2
    assert all(ref() is None for ref in refs)


async def test_list_helper_still_combines_pages_and_resolves_appdata_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = _file("one.csv"), _file("two.csv")
    get = AsyncMock(side_effect=[_page([first], "next"), _page(None, "last"), _page([second])])
    resolve = AsyncMock(return_value="bucket/appdata/application")
    monkeypatch.setattr(storage_module, "resolve_bucket", resolve)

    result = await _storage(get).get_files_in_folder("/folder/")

    assert result == [first, second]
    assert isinstance(result, list)
    resolve.assert_awaited_once()
    assert [call.kwargs["token"] for call in get.await_args_list] == [None, "next", "last"]
    assert all(call.kwargs["limit"] == 100 for call in get.await_args_list)
    assert all(
        call.kwargs["relative_url"] == "files/bucket/appdata/application/folder/"
        for call in get.await_args_list
    )


@pytest.mark.parametrize("prior_page", [False, True])
async def test_missing_folder_returns_files_already_seen(prior_page: bool) -> None:
    first = _file("one.csv")
    effects = ([_page([first], "next")] if prior_page else []) + [
        DialException("missing", status_code=404)
    ]
    get = AsyncMock(side_effect=effects)

    assert await _storage(get).get_files_in_folder("folder", "bucket") == (
        [first] if prior_page else []
    )


async def test_listing_errors_other_than_not_found_propagate() -> None:
    get = AsyncMock(side_effect=DialException("unavailable", status_code=503))
    with pytest.raises(DialException):
        await _storage(get).get_files_in_folder("folder", "bucket")
