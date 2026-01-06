"""Tests for CLI interactive prompts module."""

from unittest.mock import AsyncMock, patch

import pytest

from statgpt.cli.shared.prompts import (
    CheckboxSelector,
    RadioSelector,
    select_clients_interactive,
    select_datasets_interactive,
)


class TestCheckboxSelectorFiltering:
    """Tests for CheckboxSelector._get_filtered_items() method."""

    def test_no_filter_returns_all_items(self):
        """Empty filter should return all items."""
        items = [("a", "Apple"), ("b", "Banana"), ("c", "Cherry")]
        selector = CheckboxSelector(items)
        selector._filter_text = ""
        result = selector._get_filtered_items()
        assert result == items

    def test_filter_matches_label(self):
        """Filter should match against label."""
        items = [("a", "Apple"), ("b", "Banana"), ("c", "Cherry")]
        selector = CheckboxSelector(items)
        selector._filter_text = "app"
        result = selector._get_filtered_items()
        assert result == [("a", "Apple")]

    def test_filter_matches_value(self):
        """Filter should match against value."""
        items = [("apple", "Fruit 1"), ("banana", "Fruit 2"), ("cherry", "Fruit 3")]
        selector = CheckboxSelector(items)
        selector._filter_text = "apple"
        result = selector._get_filtered_items()
        assert result == [("apple", "Fruit 1")]

    def test_filter_case_insensitive(self):
        """Filter should be case insensitive."""
        items = [("a", "Apple"), ("b", "Banana")]
        selector = CheckboxSelector(items)
        selector._filter_text = "APP"
        result = selector._get_filtered_items()
        assert result == [("a", "Apple")]

    def test_filter_no_matches(self):
        """Filter with no matches returns empty list."""
        items = [("a", "Apple"), ("b", "Banana")]
        selector = CheckboxSelector(items)
        selector._filter_text = "xyz"
        result = selector._get_filtered_items()
        assert result == []

    def test_filter_partial_match(self):
        """Filter should match partial strings."""
        items = [("a", "Banana"), ("b", "Orange"), ("c", "Mango")]
        selector = CheckboxSelector(items)
        selector._filter_text = "an"
        result = selector._get_filtered_items()
        assert result == [("a", "Banana"), ("b", "Orange"), ("c", "Mango")]

    def test_filter_matches_multiple(self):
        """Filter can match multiple items."""
        items = [("a", "Apple Pie"), ("b", "Banana"), ("c", "Apple Juice")]
        selector = CheckboxSelector(items)
        selector._filter_text = "apple"
        result = selector._get_filtered_items()
        assert result == [("a", "Apple Pie"), ("c", "Apple Juice")]


class TestCheckboxSelectorDisplayText:
    """Tests for CheckboxSelector._get_display_text() method."""

    def test_title_shown(self):
        """Display text should contain the title."""
        items = [("a", "Apple")]
        selector = CheckboxSelector(items, title="My Title")
        text = selector._get_display_text()
        text_str = "".join(t[1] for t in text)
        assert "My Title" in text_str

    def test_filter_shown_when_enabled(self):
        """Filter input should be shown when enabled."""
        items = [("a", "Apple")]
        selector = CheckboxSelector(items, filter_enabled=True)
        text = selector._get_display_text()
        text_str = "".join(t[1] for t in text)
        assert "Filter:" in text_str

    def test_filter_hidden_when_disabled(self):
        """Filter input should be hidden when disabled."""
        items = [("a", "Apple")]
        selector = CheckboxSelector(items, filter_enabled=False)
        text = selector._get_display_text()
        text_str = "".join(t[1] for t in text)
        assert "Filter:" not in text_str

    def test_cursor_item_has_indicator(self):
        """Item at cursor position should have > indicator."""
        items = [("a", "Apple"), ("b", "Banana")]
        selector = CheckboxSelector(items)
        selector._cursor = 0
        text = selector._get_display_text()
        text_str = "".join(t[1] for t in text)
        assert " > " in text_str

    def test_selected_item_shows_checked(self):
        """Selected item should show [x]."""
        items = [("a", "Apple"), ("b", "Banana")]
        selector = CheckboxSelector(items)
        selector._selected.add("a")
        text = selector._get_display_text()
        text_str = "".join(t[1] for t in text)
        assert "[x]" in text_str

    def test_unselected_item_shows_unchecked(self):
        """Unselected item should show [ ]."""
        items = [("a", "Apple"), ("b", "Banana")]
        selector = CheckboxSelector(items)
        text = selector._get_display_text()
        text_str = "".join(t[1] for t in text)
        assert "[ ]" in text_str

    def test_no_items_after_filter(self):
        """Should show message when filter returns no items."""
        items = [("a", "Apple")]
        selector = CheckboxSelector(items)
        selector._filter_text = "xyz"
        text = selector._get_display_text()
        text_str = "".join(t[1] for t in text)
        assert "No matching items" in text_str

    def test_help_text_shown(self):
        """Help text should be shown at bottom."""
        items = [("a", "Apple")]
        selector = CheckboxSelector(items)
        text = selector._get_display_text()
        text_str = "".join(t[1] for t in text)
        assert "Space: toggle" in text_str
        assert "Enter: confirm" in text_str


class TestRadioSelectorFiltering:
    """Tests for RadioSelector._get_filtered_items() method."""

    def test_no_filter_returns_all_items(self):
        """Empty filter should return all items."""
        items = [("a", "Apple"), ("b", "Banana")]
        selector = RadioSelector(items)
        selector._filter_text = ""
        result = selector._get_filtered_items()
        assert result == items

    def test_filter_matches_label(self):
        """Filter should match against label."""
        items = [("a", "Apple"), ("b", "Banana")]
        selector = RadioSelector(items)
        selector._filter_text = "ban"
        result = selector._get_filtered_items()
        assert result == [("b", "Banana")]

    def test_filter_case_insensitive(self):
        """Filter should be case insensitive."""
        items = [("a", "Apple"), ("b", "Banana")]
        selector = RadioSelector(items)
        selector._filter_text = "BANANA"
        result = selector._get_filtered_items()
        assert result == [("b", "Banana")]


class TestRadioSelectorDisplayText:
    """Tests for RadioSelector._get_display_text() method."""

    def test_cursor_item_shows_filled_radio(self):
        """Item at cursor should show filled radio (●)."""
        items = [("a", "Apple"), ("b", "Banana")]
        selector = RadioSelector(items)
        selector._cursor = 0
        text = selector._get_display_text()
        text_str = "".join(t[1] for t in text)
        assert "(●)" in text_str

    def test_non_cursor_item_shows_empty_radio(self):
        """Items not at cursor should show empty radio ( )."""
        items = [("a", "Apple"), ("b", "Banana")]
        selector = RadioSelector(items)
        selector._cursor = 0
        text = selector._get_display_text()
        text_str = "".join(t[1] for t in text)
        assert "( )" in text_str

    def test_help_text_different_from_checkbox(self):
        """Radio selector help text should not mention Space toggle."""
        items = [("a", "Apple")]
        selector = RadioSelector(items)
        text = selector._get_display_text()
        text_str = "".join(t[1] for t in text)
        assert "Enter: select" in text_str
        assert "Space: toggle" not in text_str


class TestSelectClientsInteractive:
    """Tests for select_clients_interactive() function."""

    @pytest.mark.asyncio
    async def test_empty_selection_returns_empty_set(self):
        """Empty selection (cancelled) should return empty set."""
        with patch(
            "statgpt.cli.shared.prompts.select_items_interactive",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await select_clients_interactive(["client1", "client2"])
            assert result == set()

    @pytest.mark.asyncio
    async def test_all_selected_returns_none(self):
        """Selecting __all__ should return None."""
        with patch(
            "statgpt.cli.shared.prompts.select_items_interactive",
            new_callable=AsyncMock,
            return_value=["__all__"],
        ):
            result = await select_clients_interactive(["client1", "client2"])
            assert result is None

    @pytest.mark.asyncio
    async def test_specific_clients_returns_set(self):
        """Selecting specific clients should return set of them."""
        with patch(
            "statgpt.cli.shared.prompts.select_items_interactive",
            new_callable=AsyncMock,
            return_value=["client1", "client2"],
        ):
            result = await select_clients_interactive(["client1", "client2", "client3"])
            assert result == {"client1", "client2"}

    @pytest.mark.asyncio
    async def test_all_with_specific_returns_none(self):
        """__all__ takes precedence over specific selections."""
        with patch(
            "statgpt.cli.shared.prompts.select_items_interactive",
            new_callable=AsyncMock,
            return_value=["__all__", "client1"],
        ):
            result = await select_clients_interactive(["client1", "client2"])
            assert result is None

    @pytest.mark.asyncio
    async def test_items_include_all_option(self):
        """Items passed to selector should include 'All clients' option."""
        mock_select = AsyncMock(return_value=[])
        with patch(
            "statgpt.cli.shared.prompts.select_items_interactive",
            mock_select,
        ):
            await select_clients_interactive(["client1", "client2"])
            call_args = mock_select.call_args
            items = call_args[0][0]
            assert items[0] == ("__all__", "All clients")

    @pytest.mark.asyncio
    async def test_clients_sorted(self):
        """Client items should be sorted alphabetically."""
        mock_select = AsyncMock(return_value=[])
        with patch(
            "statgpt.cli.shared.prompts.select_items_interactive",
            mock_select,
        ):
            await select_clients_interactive(["zebra", "apple", "mango"])
            call_args = mock_select.call_args
            items = call_args[0][0]
            # Skip __all__ at index 0
            client_items = items[1:]
            assert client_items == [("apple", "apple"), ("mango", "mango"), ("zebra", "zebra")]


class TestSelectDatasetsInteractive:
    """Tests for select_datasets_interactive() function."""

    @pytest.mark.asyncio
    async def test_returns_set_of_selected(self):
        """Should return set of selected dataset URNs."""
        with patch(
            "statgpt.cli.shared.prompts.select_items_interactive",
            new_callable=AsyncMock,
            return_value=["urn:dataset:1", "urn:dataset:2"],
        ):
            result = await select_datasets_interactive(
                [("urn:dataset:1", "Dataset 1"), ("urn:dataset:2", "Dataset 2")]
            )
            assert result == {"urn:dataset:1", "urn:dataset:2"}

    @pytest.mark.asyncio
    async def test_empty_selection_returns_empty_set(self):
        """Cancelled selection should return empty set."""
        with patch(
            "statgpt.cli.shared.prompts.select_items_interactive",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await select_datasets_interactive([("urn:1", "Dataset 1")])
            assert result == set()

    @pytest.mark.asyncio
    async def test_filter_enabled(self):
        """Dataset selection should have filtering enabled."""
        mock_select = AsyncMock(return_value=[])
        with patch(
            "statgpt.cli.shared.prompts.select_items_interactive",
            mock_select,
        ):
            await select_datasets_interactive([("urn:1", "Dataset 1")])
            call_args = mock_select.call_args
            assert call_args[1]["filter_enabled"] is True
