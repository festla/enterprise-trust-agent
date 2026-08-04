import pytest
from pydantic import ValidationError

from app.schemas.context_expansion import (
    AdjacentPageContextExpansion,
    ContextExpansionItem,
)
from app.schemas.enums import ReportType


COMPANY_ID = "haier_smart_home"
REPORT_ID = "haier_smart_home_2024"

DOCUMENT_ID = (
    f"doc_{REPORT_ID}_{'a' * 24}"
)

BASE_CHUNK_ID = (
    f"chunk_{REPORT_ID}_{'b' * 24}"
)

ADJACENT_CHUNK_ID = (
    f"chunk_{REPORT_ID}_{'c' * 24}"
)

UNKNOWN_CHUNK_ID = (
    f"chunk_{REPORT_ID}_{'d' * 24}"
)

OTHER_DOCUMENT_ID = (
    f"doc_{REPORT_ID}_{'e' * 24}"
)

OTHER_CHUNK_ID = (
    f"chunk_{REPORT_ID}_{'f' * 24}"
)

BASE_TEXT = "consolidated income statement header"

ADJACENT_TEXT = "net profit value for fiscal year 2024"


def build_item(
    **overrides: object,
) -> ContextExpansionItem:
    values = {
        "context_order": 1,
        "origin": "retrieved",
        "company_id": COMPANY_ID,
        "report_id": REPORT_ID,
        "fiscal_year": 2024,
        "report_type": (
            ReportType.ANNUAL_REPORT
        ),
        "document_id": DOCUMENT_ID,
        "page_id": (
            f"{DOCUMENT_ID}_page_0121"
        ),
        "pdf_page": 121,
        "printed_page": 119,
        "chunk_id": BASE_CHUNK_ID,
        "chunk_index": 10,
        "text": BASE_TEXT,
        "text_char_count": len(BASE_TEXT),
        "retrieval_rank": 1,
        "retrieval_score": 0.92,
        "anchor_chunk_id": BASE_CHUNK_ID,
        "anchor_retrieval_rank": 1,
        "page_distance": 0,
    }

    values.update(overrides)

    return ContextExpansionItem(
        **values
    )


def build_adjacent_item(
    **overrides: object,
) -> ContextExpansionItem:
    values = {
        "context_order": 2,
        "origin": "adjacent_page",
        "company_id": COMPANY_ID,
        "report_id": REPORT_ID,
        "fiscal_year": 2024,
        "report_type": (
            ReportType.ANNUAL_REPORT
        ),
        "document_id": DOCUMENT_ID,
        "page_id": (
            f"{DOCUMENT_ID}_page_0122"
        ),
        "pdf_page": 122,
        "printed_page": 120,
        "chunk_id": ADJACENT_CHUNK_ID,
        "chunk_index": 11,
        "text": ADJACENT_TEXT,
        "text_char_count": len(
            ADJACENT_TEXT
        ),
        "retrieval_rank": None,
        "retrieval_score": None,
        "anchor_chunk_id": BASE_CHUNK_ID,
        "anchor_retrieval_rank": 1,
        "page_distance": 1,
    }

    values.update(overrides)

    return ContextExpansionItem(
        **values
    )


def build_expansion(
    **overrides: object,
) -> AdjacentPageContextExpansion:
    base_item = build_item()
    adjacent_item = build_adjacent_item()

    values = {
        "schema_version": 1,
        "strategy_id": (
            "adjacent_page_context_v1"
        ),
        "query_id": "q3",
        "original_query": (
            "What is the 2024 net profit?"
        ),
        "semantic_query": (
            "net profit fiscal year 2024"
        ),
        "company_id": COMPANY_ID,
        "report_id": REPORT_ID,
        "fiscal_year": 2024,
        "report_type": (
            ReportType.ANNUAL_REPORT
        ),
        "document_id": DOCUMENT_ID,
        "base_top_k": 5,
        "page_window": 1,
        "items": (
            base_item,
            adjacent_item,
        ),
        "base_chunk_ids": (
            BASE_CHUNK_ID,
        ),
        "expanded_chunk_ids": (
            ADJACENT_CHUNK_ID,
        ),
        "used_chunk_ids": (
            BASE_CHUNK_ID,
            ADJACENT_CHUNK_ID,
        ),
        "base_item_count": 1,
        "expanded_item_count": 1,
        "total_item_count": 2,
        "duplicate_chunk_count": 0,
        "base_char_count": len(
            BASE_TEXT
        ),
        "expanded_char_count": len(
            ADJACENT_TEXT
        ),
        "total_char_count": (
            len(BASE_TEXT)
            + len(ADJACENT_TEXT)
        ),
    }

    values.update(overrides)

    return AdjacentPageContextExpansion(
        **values
    )


def test_accept_valid_adjacent_page_expansion(
) -> None:
    expansion = build_expansion()

    assert expansion.base_item_count == 1
    assert expansion.expanded_item_count == 1
    assert expansion.total_item_count == 2

    assert expansion.base_chunk_ids == (
        BASE_CHUNK_ID,
    )

    assert expansion.expanded_chunk_ids == (
        ADJACENT_CHUNK_ID,
    )

    assert expansion.items[0].origin == (
        "retrieved"
    )

    assert expansion.items[1].origin == (
        "adjacent_page"
    )

    assert (
        expansion.items[1].anchor_chunk_id
        == BASE_CHUNK_ID
    )

    assert (
        expansion.items[1].page_distance
        == 1
    )


def test_reject_incorrect_text_char_count(
) -> None:
    with pytest.raises(
        ValidationError,
        match="text_char_count",
    ):
        build_item(
            text_char_count=999,
        )


def test_reject_retrieved_item_without_rank(
) -> None:
    with pytest.raises(
        ValidationError,
        match="retrieval_rank",
    ):
        build_item(
            retrieval_rank=None,
        )


def test_reject_adjacent_item_with_rank(
) -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "adjacent item cannot contain "
            "retrieval_rank"
        ),
    ):
        build_adjacent_item(
            retrieval_rank=2,
        )


def test_reject_adjacent_item_anchoring_to_itself(
) -> None:
    with pytest.raises(
        ValidationError,
        match="cannot anchor to itself",
    ):
        build_adjacent_item(
            anchor_chunk_id=(
                ADJACENT_CHUNK_ID
            ),
        )


def test_reject_noncontinuous_context_order(
) -> None:
    base_item = build_item()

    adjacent_item = build_adjacent_item(
        context_order=3,
    )

    with pytest.raises(
        ValidationError,
        match="context_order",
    ):
        build_expansion(
            items=(
                base_item,
                adjacent_item,
            ),
        )


def test_reject_duplicate_chunk_ids(
) -> None:
    base_item = build_item()

    duplicate_item = build_item(
        context_order=2,
        retrieval_rank=2,
        anchor_retrieval_rank=2,
    )

    with pytest.raises(
        ValidationError,
        match="duplicate chunk_id",
    ):
        build_expansion(
            items=(
                base_item,
                duplicate_item,
            ),
        )


def test_reject_unknown_adjacent_anchor(
) -> None:
    base_item = build_item()

    adjacent_item = build_adjacent_item(
        anchor_chunk_id=UNKNOWN_CHUNK_ID,
    )

    with pytest.raises(
        ValidationError,
        match="retrieved anchor chunk",
    ):
        build_expansion(
            items=(
                base_item,
                adjacent_item,
            ),
        )


def test_reject_incorrect_page_distance(
) -> None:
    base_item = build_item()

    adjacent_item = build_adjacent_item(
        page_id=(
            f"{DOCUMENT_ID}_page_0123"
        ),
        pdf_page=123,
        printed_page=121,
        page_distance=1,
    )

    with pytest.raises(
        ValidationError,
        match="page_distance",
    ):
        build_expansion(
            items=(
                base_item,
                adjacent_item,
            ),
        )


def test_reject_item_from_other_document(
) -> None:
    base_item = build_item()

    adjacent_item = build_adjacent_item(
        document_id=OTHER_DOCUMENT_ID,
        page_id=(
            f"{OTHER_DOCUMENT_ID}_page_0122"
        ),
        chunk_id=OTHER_CHUNK_ID,
    )

    with pytest.raises(
        ValidationError,
        match="document_id",
    ):
        build_expansion(
            items=(
                base_item,
                adjacent_item,
            ),
            expanded_chunk_ids=(
                OTHER_CHUNK_ID,
            ),
            used_chunk_ids=(
                BASE_CHUNK_ID,
                OTHER_CHUNK_ID,
            ),
        )


def test_reject_base_rank_reordering(
) -> None:
    first_item = build_item(
        retrieval_rank=2,
        anchor_retrieval_rank=2,
    )

    second_text = "second retrieved chunk"

    second_item = build_item(
        context_order=2,
        page_id=(
            f"{DOCUMENT_ID}_page_0130"
        ),
        pdf_page=130,
        printed_page=128,
        chunk_id=OTHER_CHUNK_ID,
        chunk_index=20,
        text=second_text,
        text_char_count=len(
            second_text
        ),
        retrieval_rank=1,
        retrieval_score=0.91,
        anchor_chunk_id=OTHER_CHUNK_ID,
        anchor_retrieval_rank=1,
    )

    with pytest.raises(
        ValidationError,
        match="retrieval rank order",
    ):
        build_expansion(
            items=(
                first_item,
                second_item,
            ),
            base_chunk_ids=(
                BASE_CHUNK_ID,
                OTHER_CHUNK_ID,
            ),
            expanded_chunk_ids=(),
            used_chunk_ids=(
                BASE_CHUNK_ID,
                OTHER_CHUNK_ID,
            ),
            base_item_count=2,
            expanded_item_count=0,
            total_item_count=2,
            base_char_count=(
                len(BASE_TEXT)
                + len(second_text)
            ),
            expanded_char_count=0,
            total_char_count=(
                len(BASE_TEXT)
                + len(second_text)
            ),
        )


@pytest.mark.parametrize(
    (
        "field_name",
        "invalid_value",
        "expected_message",
    ),
    (
        (
            "base_item_count",
            2,
            "base_item_count",
        ),
        (
            "expanded_item_count",
            2,
            "expanded_item_count",
        ),
        (
            "total_item_count",
            3,
            "total_item_count",
        ),
        (
            "base_char_count",
            1,
            "base_char_count",
        ),
        (
            "expanded_char_count",
            1,
            "expanded_char_count",
        ),
        (
            "total_char_count",
            1,
            "total_char_count",
        ),
    ),
)
def test_reject_incorrect_audit_values(
    field_name: str,
    invalid_value: int,
    expected_message: str,
) -> None:
    with pytest.raises(
        ValidationError,
        match=expected_message,
    ):
        build_expansion(
            **{
                field_name: invalid_value,
            }
        )


def test_reject_unknown_expansion_field(
) -> None:
    with pytest.raises(
        ValidationError,
    ):
        build_expansion(
            unexpected_field="not allowed",
        )