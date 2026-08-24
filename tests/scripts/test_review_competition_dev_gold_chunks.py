from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import (
    review_competition_dev_gold_chunks
    as reviewer,
)


CORPUS_ID = (
    "competition_corpus_"
    "0123456789abcdef"
)

SOURCE_ID = (
    "src_0000000000000001"
)


@dataclass(frozen=True)
class FakeChunk:
    chunk_id: str
    chunk_index: int
    text: str

    source_id: str = SOURCE_ID
    doc_id: str = "doc_test"
    source_type: str = "word"
    chunk_type: str = "text"

    page_start: int | None = None
    page_end: int | None = None

    paragraph_start_index: (
        int | None
    ) = 0

    paragraph_end_index: (
        int | None
    ) = 0

    table_index: int | None = None

    section_path: tuple[
        str,
        ...,
    ] = ()

    article: str | None = None
    table_title: str | None = None


def _write_json(
    path: Path,
    payload: object,
) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _coverage_payload(
) -> dict[str, object]:
    return {
        "audit_version": (
            "competition_evidence_"
            "coverage_dev_v1"
        ),
        "corpus_id": CORPUS_ID,
        "summary": {
            "fact_count": 2,
            "missing_fact_count": 1,
        },
        "cases": [
            {
                "case_id": "Q101",
                "expected_source_id": (
                    SOURCE_ID
                ),
                "fact_coverages": [
                    {
                        "fact": (
                            "缺失事实。"
                        ),
                        "missing": True,
                        "single_chunk_ids": [],
                        "two_chunk_window_ids": [],
                    },
                    {
                        "fact": (
                            "已覆盖事实。"
                        ),
                        "missing": False,
                        "single_chunk_ids": [
                            (
                                "chunk:"
                                "doc_test:"
                                "00001"
                            )
                        ],
                        "two_chunk_window_ids": [],
                    },
                ],
            }
        ],
    }


def _candidate_payload(
    *,
    include_record: bool = True,
) -> dict[str, object]:
    records: list[
        dict[str, object]
    ] = []

    if include_record:
        records.append(
            {
                "case_id": "Q101",
                "fact_index": 0,
                "fact": (
                    "缺失事实。"
                ),
                "expected_source_id": (
                    SOURCE_ID
                ),
                "candidates": [
                    {
                        "rank": 1,
                        "chunk_id": (
                            "chunk:"
                            "doc_test:"
                            "00000"
                        ),
                        "source_id": (
                            SOURCE_ID
                        ),
                        "bigram_recall": 0.8,
                        "bigram_f1": 0.6,
                        "unigram_recall": 0.9,
                        "sequence_ratio": 0.5,
                    }
                ],
            }
        )

    return {
        "report_version": (
            "competition_evidence_"
            "candidates_dev_v1"
        ),
        "corpus_id": CORPUS_ID,
        "missing_fact_count": (
            len(records)
        ),
        "records": records,
    }


def _gold_payload(
    candidate_path: Path,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "gold_version": (
            "competition_dev_"
            "gold_chunks_v1"
        ),
        "split_version": (
            "competition_eval_split_v1"
        ),
        "corpus_id": CORPUS_ID,
        "candidate_report_sha256": (
            _sha256(
                candidate_path
            )
        ),
        "records": [
            {
                "schema_version": 1,
                "case_id": "Q101",
                "fact_index": 0,
                "fact": (
                    "缺失事实。"
                ),
                "expected_source_id": (
                    SOURCE_ID
                ),
                "review_status": (
                    "pending"
                ),
                "evidence_mode": None,
                "gold_chunks": [],
                "review_note": None,
            },
            {
                "schema_version": 1,
                "case_id": "Q101",
                "fact_index": 1,
                "fact": (
                    "已覆盖事实。"
                ),
                "expected_source_id": (
                    SOURCE_ID
                ),
                "review_status": (
                    "pending"
                ),
                "evidence_mode": None,
                "gold_chunks": [],
                "review_note": None,
            },
        ],
    }


@pytest.fixture
def review_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    coverage_path = (
        tmp_path
        / "coverage.json"
    )

    candidate_path = (
        tmp_path
        / "candidates.json"
    )

    gold_path = (
        tmp_path
        / "gold.json"
    )

    corpus_directory = (
        tmp_path
        / CORPUS_ID
    )

    corpus_directory.mkdir()

    _write_json(
        coverage_path,
        _coverage_payload(),
    )

    _write_json(
        candidate_path,
        _candidate_payload(),
    )

    _write_json(
        gold_path,
        _gold_payload(
            candidate_path
        ),
    )

    chunks = (
        FakeChunk(
            (
                "chunk:"
                "doc_test:"
                "00000"
            ),
            0,
            "候选直接证据。",
        ),
        FakeChunk(
            (
                "chunk:"
                "doc_test:"
                "00001"
            ),
            1,
            "已覆盖证据。",
        ),
        FakeChunk(
            (
                "chunk:"
                "doc_test:"
                "00002"
            ),
            2,
            "后一段补充证据。",
        ),
    )

    fake_corpus = (
        SimpleNamespace(
            manifest=(
                SimpleNamespace(
                    corpus_id=(
                        CORPUS_ID
                    )
                )
            ),
            chunks=chunks,
        )
    )

    monkeypatch.setattr(
        reviewer,
        (
            "load_competition_"
            "chunk_corpus"
        ),
        lambda _: fake_corpus,
    )

    return SimpleNamespace(
        coverage_path=coverage_path,
        candidate_path=candidate_path,
        gold_path=gold_path,
        corpus_directory=(
            corpus_directory
        ),
    )


def _load(
    files: SimpleNamespace,
) -> (
    reviewer
    .CompetitionGoldReviewSession
):
    return (
        reviewer
        .load_review_session(
            coverage_path=(
                files.coverage_path
            ),
            candidate_path=(
                files.candidate_path
            ),
            gold_path=(
                files.gold_path
            ),
            corpus_directory=(
                files.corpus_directory
            ),
            expected_fact_count=2,
        )
    )


def test_merges_missing_candidates_and_exact_coverage(
    review_files: SimpleNamespace,
) -> None:
    session = _load(
        review_files
    )

    missing_ids = tuple(
        option.chunk.chunk_id
        for option
        in session.options_by_key[
            ("Q101", 0)
        ]
    )

    covered_ids = tuple(
        option.chunk.chunk_id
        for option
        in session.options_by_key[
            ("Q101", 1)
        ]
    )

    assert missing_ids == (
        "chunk:doc_test:00000",
        "chunk:doc_test:00001",
    )

    assert covered_ids == (
        "chunk:doc_test:00001",
        "chunk:doc_test:00000",
        "chunk:doc_test:00002",
    )


def test_rejects_candidate_sha256_mismatch(
    review_files: SimpleNamespace,
) -> None:
    payload = _gold_payload(
        review_files.candidate_path
    )

    payload[
        "candidate_report_sha256"
    ] = "0" * 64

    _write_json(
        review_files.gold_path,
        payload,
    )

    with pytest.raises(
        RuntimeError,
        match="Candidate SHA256",
    ):
        _load(
            review_files
        )


def test_rejects_candidate_keys_not_equal_missing(
    review_files: SimpleNamespace,
) -> None:
    _write_json(
        review_files.candidate_path,
        _candidate_payload(
            include_record=False
        ),
    )

    _write_json(
        review_files.gold_path,
        _gold_payload(
            review_files.candidate_path
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Missing Fact",
    ):
        _load(
            review_files
        )


def test_single_chunk_decision_saves_atomically_with_backup(
    review_files: SimpleNamespace,
) -> None:
    original = (
        review_files
        .gold_path
        .read_bytes()
    )

    session = _load(
        review_files
    )

    (
        reviewer
        .apply_review_decision(
            session,
            record_index=0,
            review_status=(
                "confirmed"
            ),
            evidence_mode=(
                "single_chunk"
            ),
            selections=(
                (
                    (
                        "chunk:"
                        "doc_test:"
                        "00000"
                    ),
                    "direct",
                ),
            ),
        )
    )

    saved = (
        reviewer
        .CompetitionGoldChunkDataset
        .model_validate_json(
            review_files
            .gold_path
            .read_bytes()
        )
    )

    assert (
        saved.records[0]
        .review_status
        == "confirmed"
    )

    assert (
        saved.records[0]
        .evidence_mode
        == "single_chunk"
    )

    assert (
        saved.records[0]
        .gold_chunks[0]
        .chunk_id
        == "chunk:doc_test:00000"
    )

    backup_path = Path(
        f"{review_files.gold_path}.bak"
    )

    assert (
        backup_path.read_bytes()
        == original
    )


def test_parent_child_and_multi_chunk_roles(
    review_files: SimpleNamespace,
) -> None:
    session = _load(
        review_files
    )

    (
        reviewer
        .apply_review_decision(
            session,
            record_index=0,
            review_status=(
                "confirmed"
            ),
            evidence_mode=(
                "parent_child"
            ),
            selections=(
                (
                    (
                        "chunk:"
                        "doc_test:"
                        "00001"
                    ),
                    "parent_context",
                ),
                (
                    (
                        "chunk:"
                        "doc_test:"
                        "00000"
                    ),
                    "direct",
                ),
            ),
        )
    )

    (
        reviewer
        .apply_review_decision(
            session,
            record_index=1,
            review_status=(
                "confirmed"
            ),
            evidence_mode=(
                "multi_chunk"
            ),
            selections=(
                (
                    (
                        "chunk:"
                        "doc_test:"
                        "00001"
                    ),
                    "direct",
                ),
                (
                    (
                        "chunk:"
                        "doc_test:"
                        "00002"
                    ),
                    "supporting",
                ),
            ),
        )
    )

    assert [
        item.role
        for item
        in session.dataset
        .records[0]
        .gold_chunks
    ] == [
        "parent_context",
        "direct",
    ]

    assert [
        item.role
        for item
        in session.dataset
        .records[1]
        .gold_chunks
    ] == [
        "direct",
        "supporting",
    ]


def test_quit_does_not_modify_gold(
    review_files: SimpleNamespace,
) -> None:
    original = (
        review_files
        .gold_path
        .read_bytes()
    )

    session = _load(
        review_files
    )

    output: list[str] = []

    (
        reviewer
        .run_interactive_review(
            session,
            input_fn=lambda _: "q",
            output_fn=output.append,
            preview_chars=120,
        )
    )

    assert (
        review_files
        .gold_path
        .read_bytes()
        == original
    )

    assert not Path(
        f"{review_files.gold_path}.bak"
    ).exists()

    assert any(
        "安全退出" in line
        for line in output
    )