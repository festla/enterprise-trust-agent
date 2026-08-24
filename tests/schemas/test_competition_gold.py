from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.competition_gold import (
    CompetitionGoldChunkDataset,
    CompetitionGoldChunkReference,
    CompetitionGoldFactRecord,
)


def _reference(
    chunk_id: str,
    *,
    role: str = "direct",
) -> CompetitionGoldChunkReference:
    return CompetitionGoldChunkReference(
        chunk_id=chunk_id,
        role=role,
    )


def _confirmed_record(
    **overrides: object,
) -> CompetitionGoldFactRecord:
    payload: dict[str, object] = {
        "case_id": "Q101",
        "fact_index": 0,
        "fact": "测试监管事实。",
        "expected_source_id": (
            "src_0000000000000001"
        ),
        "review_status": "confirmed",
        "evidence_mode": "single_chunk",
        "gold_chunks": (
            _reference("chunk:doc_test:00001"),
        ),
    }

    payload.update(overrides)

    return CompetitionGoldFactRecord(
        **payload
    )


def test_accepts_confirmed_single_chunk(
) -> None:
    record = _confirmed_record()

    assert record.review_status == "confirmed"
    assert record.evidence_mode == "single_chunk"
    assert len(record.gold_chunks) == 1
    assert record.gold_chunks[0].role == "direct"


def test_accepts_parent_child_chunks(
) -> None:
    record = _confirmed_record(
        evidence_mode="parent_child",
        gold_chunks=(
            _reference(
                "chunk:doc_test:00001",
                role="parent_context",
            ),
            _reference(
                "chunk:doc_test:00002",
                role="direct",
            ),
        ),
    )

    assert len(record.gold_chunks) == 2


def test_accepts_multi_chunk_evidence(
) -> None:
    record = _confirmed_record(
        evidence_mode="multi_chunk",
        gold_chunks=(
            _reference(
                "chunk:doc_test:00001",
            ),
            _reference(
                "chunk:doc_test:00002",
            ),
        ),
    )

    assert len(record.gold_chunks) == 2


def test_rejects_duplicate_chunk_ids(
) -> None:
    with pytest.raises(
        ValidationError,
        match="重复 chunk_id",
    ):
        _confirmed_record(
            evidence_mode="multi_chunk",
            gold_chunks=(
                _reference(
                    "chunk:doc_test:00001",
                ),
                _reference(
                    "chunk:doc_test:00001",
                ),
            ),
        )


def test_rejects_invalid_single_chunk(
) -> None:
    with pytest.raises(
        ValidationError,
        match="single_chunk",
    ):
        _confirmed_record(
            gold_chunks=(
                _reference(
                    "chunk:doc_test:00001",
                ),
                _reference(
                    "chunk:doc_test:00002",
                ),
            ),
        )


def test_rejects_parent_child_without_parent(
) -> None:
    with pytest.raises(
        ValidationError,
        match="parent_context",
    ):
        _confirmed_record(
            evidence_mode="parent_child",
            gold_chunks=(
                _reference(
                    "chunk:doc_test:00001",
                ),
                _reference(
                    "chunk:doc_test:00002",
                ),
            ),
        )


def test_pending_record_cannot_have_selection(
) -> None:
    with pytest.raises(
        ValidationError,
        match="pending",
    ):
        _confirmed_record(
            review_status="pending",
        )


def test_accepts_empty_pending_record(
) -> None:
    record = CompetitionGoldFactRecord(
        case_id="Q101",
        fact_index=0,
        fact="待审核事实。",
        expected_source_id=(
            "src_0000000000000001"
        ),
    )

    assert record.review_status == "pending"
    assert record.evidence_mode is None
    assert record.gold_chunks == ()


def test_unresolved_requires_review_note(
) -> None:
    with pytest.raises(
        ValidationError,
        match="review_note",
    ):
        CompetitionGoldFactRecord(
            case_id="Q101",
            fact_index=0,
            fact="当前 Corpus 无法支持。",
            expected_source_id=(
                "src_0000000000000001"
            ),
            review_status="unresolved",
        )


def test_dataset_rejects_duplicate_fact_key(
) -> None:
    record = _confirmed_record()

    with pytest.raises(
        ValidationError,
        match="重复",
    ):
        CompetitionGoldChunkDataset(
            corpus_id=(
                "competition_corpus_"
                "0123456789abcdef"
            ),
            candidate_report_sha256=(
                "a" * 64
            ),
            records=(
                record,
                record,
            ),
        )


def test_dataset_requires_sorted_records(
) -> None:
    first = _confirmed_record(
        case_id="Q102",
    )

    second = _confirmed_record(
        case_id="Q101",
    )

    with pytest.raises(
        ValidationError,
        match="排序",
    ):
        CompetitionGoldChunkDataset(
            corpus_id=(
                "competition_corpus_"
                "0123456789abcdef"
            ),
            candidate_report_sha256=(
                "a" * 64
            ),
            records=(
                first,
                second,
            ),
        )