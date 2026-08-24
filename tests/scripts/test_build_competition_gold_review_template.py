from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import (
    build_competition_gold_review_template
    as template_script,
)


CORPUS_ID = (
    "competition_corpus_"
    "0123456789abcdef"
)

SOURCE_ID = (
    "src_0000000000000001"
)


def _write_json(
    path: Path,
    payload: object,
) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_split(
    path: Path,
    *,
    dev_case_ids: list[str],
) -> None:
    _write_json(
        path,
        {
            "dev_case_ids": dev_case_ids,
        },
    )


def _coverage_payload(
    *,
    case_id: str = "Q101",
    corpus_id: str = CORPUS_ID,
) -> dict[str, object]:
    return {
        "audit_version": (
            "competition_evidence_coverage_dev_v1"
        ),
        "corpus_id": corpus_id,
        "text_dev_case_count": 1,
        "summary": {
            "fact_count": 2,
            "missing_fact_count": 1,
        },
        "cases": [
            {
                "case_id": case_id,
                "expected_source_id": (
                    SOURCE_ID
                ),
                "fact_coverages": [
                    {
                        "fact": "需要人工审核的事实。",
                        "missing": True,
                    },
                    {
                        "fact": "已经严格覆盖的事实。",
                        "missing": False,
                    },
                ],
            }
        ],
    }


def _candidate_payload(
    *,
    case_id: str = "Q101",
    corpus_id: str = CORPUS_ID,
    source_id: str = SOURCE_ID,
) -> dict[str, object]:
    return {
        "report_version": (
            "competition_evidence_candidates_dev_v1"
        ),
        "corpus_id": corpus_id,
        "missing_fact_count": 1,
        "records": [
            {
                "case_id": case_id,
                "fact_index": 0,
                "fact": "需要人工审核的事实。",
                "expected_source_id": (
                    SOURCE_ID
                ),
                "candidates": [
                    {
                        "chunk_id": (
                            "chunk:doc_test:00001"
                        ),
                        "source_id": source_id,
                    }
                ],
            }
        ],
    }


def test_builds_pending_dev_gold_template(
    tmp_path: Path,
) -> None:
    split_path = tmp_path / "split.json"
    coverage_path = tmp_path / "coverage.json"
    candidate_path = tmp_path / "candidates.json"
    output_path = tmp_path / "gold.json"

    _write_split(
        split_path,
        dev_case_ids=["Q101"],
    )

    _write_json(
        coverage_path,
        _coverage_payload(),
    )

    _write_json(
        candidate_path,
        _candidate_payload(),
    )

    result = (
        template_script
        .run_competition_gold_review_template(
            split_path=split_path,
            coverage_path=coverage_path,
            candidate_path=candidate_path,
            output_path=output_path,
            expected_text_dev_cases=1,
            expected_fact_count=2,
            expected_missing_fact_count=1,
        )
    )

    assert result.dev_case_count == 1
    assert result.fact_count == 2
    assert result.missing_fact_count == 1

    assert all(
        record.review_status == "pending"
        for record
        in result.dataset.records
    )

    assert output_path.is_file()

    payload = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert len(payload["records"]) == 2

    expected_sha256 = hashlib.sha256(
        candidate_path.read_bytes()
    ).hexdigest()

    assert (
        payload["candidate_report_sha256"]
        == expected_sha256
    )


def test_rejects_non_dev_case(
    tmp_path: Path,
) -> None:
    split_path = tmp_path / "split.json"
    coverage_path = tmp_path / "coverage.json"
    candidate_path = tmp_path / "candidates.json"

    _write_split(
        split_path,
        dev_case_ids=["Q101"],
    )

    _write_json(
        coverage_path,
        _coverage_payload(
            case_id="Q999",
        ),
    )

    _write_json(
        candidate_path,
        _candidate_payload(
            case_id="Q999",
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="非 Dev Case",
    ):
        (
            template_script
            .run_competition_gold_review_template(
                split_path=split_path,
                coverage_path=coverage_path,
                candidate_path=candidate_path,
                output_path=(
                    tmp_path / "gold.json"
                ),
                expected_text_dev_cases=1,
                expected_fact_count=2,
                expected_missing_fact_count=1,
            )
        )


def test_rejects_candidate_corpus_mismatch(
    tmp_path: Path,
) -> None:
    split_path = tmp_path / "split.json"
    coverage_path = tmp_path / "coverage.json"
    candidate_path = tmp_path / "candidates.json"

    _write_split(
        split_path,
        dev_case_ids=["Q101"],
    )

    _write_json(
        coverage_path,
        _coverage_payload(),
    )

    _write_json(
        candidate_path,
        _candidate_payload(
            corpus_id=(
                "competition_corpus_"
                "0000000000000000"
            ),
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="corpus_id 不一致",
    ):
        (
            template_script
            .run_competition_gold_review_template(
                split_path=split_path,
                coverage_path=coverage_path,
                candidate_path=candidate_path,
                output_path=(
                    tmp_path / "gold.json"
                ),
                expected_text_dev_cases=1,
                expected_fact_count=2,
                expected_missing_fact_count=1,
            )
        )


def test_rejects_missing_candidate_record(
    tmp_path: Path,
) -> None:
    split_path = tmp_path / "split.json"
    coverage_path = tmp_path / "coverage.json"
    candidate_path = tmp_path / "candidates.json"

    _write_split(
        split_path,
        dev_case_ids=["Q101"],
    )

    _write_json(
        coverage_path,
        _coverage_payload(),
    )

    candidate_payload = _candidate_payload()
    candidate_payload["missing_fact_count"] = 0
    candidate_payload["records"] = []

    _write_json(
        candidate_path,
        candidate_payload,
    )

    with pytest.raises(
        RuntimeError,
        match="记录不一致",
    ):
        (
            template_script
            .run_competition_gold_review_template(
                split_path=split_path,
                coverage_path=coverage_path,
                candidate_path=candidate_path,
                output_path=(
                    tmp_path / "gold.json"
                ),
                expected_text_dev_cases=1,
                expected_fact_count=2,
                expected_missing_fact_count=1,
            )
        )


def test_rejects_candidate_source_mismatch(
    tmp_path: Path,
) -> None:
    split_path = tmp_path / "split.json"
    coverage_path = tmp_path / "coverage.json"
    candidate_path = tmp_path / "candidates.json"

    _write_split(
        split_path,
        dev_case_ids=["Q101"],
    )

    _write_json(
        coverage_path,
        _coverage_payload(),
    )

    _write_json(
        candidate_path,
        _candidate_payload(
            source_id=(
                "src_0000000000000999"
            ),
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="来源",
    ):
        (
            template_script
            .run_competition_gold_review_template(
                split_path=split_path,
                coverage_path=coverage_path,
                candidate_path=candidate_path,
                output_path=(
                    tmp_path / "gold.json"
                ),
                expected_text_dev_cases=1,
                expected_fact_count=2,
                expected_missing_fact_count=1,
            )
        )