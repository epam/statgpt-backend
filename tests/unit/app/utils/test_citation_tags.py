import pytest

from statgpt.app.utils.citation_tags import CitationTagRepairer

_ID = '9f81df22e8b442e8a491d163198b7d4a'
_CANONICAL = f'<cit data-id="{_ID}"></cit>'


def _repair(*chunks: str) -> str:
    repairer = CitationTagRepairer()
    return ''.join(repairer.feed(chunk) for chunk in chunks) + repairer.flush()


@pytest.mark.parametrize(
    'tag',
    [
        pytest.param(_CANONICAL, id='canonical'),
        pytest.param(f'<cit data-id="{_ID}></cit>', id='closing-quote-missing'),
        pytest.param(f'<cit data-id={_ID}"></cit>', id='opening-quote-missing'),
        pytest.param(f'<cit data-id={_ID}></cit>', id='unquoted'),
        pytest.param(f"<cit data-id='{_ID}'></cit>", id='single-quoted'),
        pytest.param(f'<cit data-id = "{_ID}" ></cit>', id='extra-whitespace'),
        pytest.param(f'<cit data-id="{_ID}"</cit>', id='opening-tag-unterminated'),
        pytest.param(f'<cit data-id="{_ID}"/>', id='self-closing'),
        pytest.param(f'<cit data-id="{_ID}">\n</cit>', id='whitespace-content'),
    ],
)
def test_tag_is_repaired_into_canonical_form(tag: str):
    assert _repair(f'mid-2024. {tag}\n\nSource') == f'mid-2024. {_CANONICAL}\n\nSource'


def test_opening_tag_with_no_closing_tag_is_closed():
    assert _repair(f'mid-2024. <cit data-id="{_ID}">\n\nSource') == (
        f'mid-2024. {_CANONICAL}\n\nSource'
    )


def test_tags_next_to_each_other_are_repaired_separately():
    text = '<cit data-id="a></cit><cit data-id="b"></cit> <cit data-id=c></cit>'

    assert _repair(text) == (
        '<cit data-id="a"></cit><cit data-id="b"></cit> <cit data-id="c"></cit>'
    )


@pytest.mark.parametrize('split', range(1, len(f'<cit data-id="{_ID}></cit>')))
def test_tag_split_across_chunks_is_repaired_as_a_whole(split: int):
    tag = f'<cit data-id="{_ID}></cit>'

    assert _repair('mid-2024. ', tag[:split], tag[split:], ' Source') == (
        f'mid-2024. {_CANONICAL} Source'
    )


@pytest.mark.parametrize(
    'text',
    [
        pytest.param('a < b and c<d', id='less-than'),
        pytest.param('<b>bold</b> <br/>', id='other-tags'),
        pytest.param('<citation>', id='longer-tag-name'),
        pytest.param('<cit></cit> <cit title="x"></cit>', id='no-data-id'),
        pytest.param('<cit data-id=""></cit>', id='empty-data-id'),
    ],
)
def test_text_that_is_not_a_citation_tag_is_left_as_is(text: str):
    assert _repair(text) == text


def test_text_is_let_through_without_waiting_for_more():
    repairer = CitationTagRepairer()

    assert repairer.feed('Japan exits ') == 'Japan exits '
    assert repairer.feed('a < b <b>') == 'a < b <b>'
    assert repairer.flush() == ''


def test_possible_tag_start_is_held_back_until_decided():
    repairer = CitationTagRepairer()

    assert repairer.feed('mid-2024. <ci') == 'mid-2024. '
    assert repairer.feed(f't data-id="{_ID}') == ''
    assert repairer.feed('></cit> Source') == f'{_CANONICAL} Source'


def test_held_back_text_is_flushed_as_is_at_the_end_of_the_stream():
    repairer = CitationTagRepairer()

    assert repairer.feed(f'mid-2024. <cit data-id="{_ID}') == 'mid-2024. '
    assert repairer.flush() == f'<cit data-id="{_ID}'


def test_unterminated_tag_is_let_through_once_too_long():
    repairer = CitationTagRepairer()
    text = '<cit data-id="x" ' + 'y' * 300

    assert repairer.feed(text) == text
    assert repairer.flush() == ''
