from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas.competition_chunk import (
    CompetitionChunkDocument,
)
from app.schemas.competition_evidence import (
    CompetitionKnowledgeSource,
)
from app.schemas.competition_text import (
    CompetitionTextBlock,
    CompetitionTextDocument,
)
from app.services.competition_corpus import (
    build_competition_chunk_corpus,
)
from app.services.competition_text_chunker import (
    build_competition_text_chunks,
)
from scripts import (
    report_competition_evidence_candidates_dev
    as report_script,
)


SOURCE_ID = "src_0000000000000001"


def _build_corpus(tmp_path: Path):
    source = CompetitionKnowledgeSource(
        source_id=SOURCE_ID,
        doc_id=(
            "doc_src_0000000000000001_"
            "0123456789abcdef01234567"
        ),
        title="测试制度",
        source_type="word",
        relative_path="word/测试制度.docx",
        sha256="a" * 64,
    )

    document = CompetitionTextDocument(
        source=source,
        blocks=(
            CompetitionTextBlock(
                block_id=(
                    f"block:{source.doc_id}:"
                    "00000"
                ),
                source_id=source.source_id,
                doc_id=source.doc_id,
                source_type="word",
                block_index=0,
                block_type="paragraph",
                paragraph_index=0,
                text=(
                    "商业银行资本充足率"
                    "应当符合监管规定。"
                ),
            ),
        ),
    )

    chunk_document = CompetitionChunkDocument(
        source=source,
        chunks=(
            build_competition_text_chunks(
                document
            )
        ),
    )

    return build_competition_chunk_corpus(
        documents=(chunk_document,),
        output_root=(
            tmp_path / "corpora"
        ),
    )


def _write_audit(
    *,
    path: Path,
    corpus_id: str,
    fact: str = (
        "商业银行应满足资本充足率"
        "监管要求。"
    ),
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "audit_version": (
                    "competition_evidence_coverage_dev_v1"
                ),
                "corpus_id": corpus_id,
                "cases": [
                    {
                        "case_id": "Q101",
                        "source_type": "word",
                        "qa_type": "单事实检索",
                        "difficulty": "easy",
                        "expected_source_id": (
                            SOURCE_ID
                        ),
                        "fact_coverages": [
                            {
                                "fact": fact,
                                "missing": True,
                            },
                            {
                                "fact": "已经覆盖的事实",
                                "missing": False,
                            },
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_runner_builds_missing_fact_candidates(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path)

    audit_path = tmp_path / "audit.json"
    _write_audit(
        path=audit_path,
        corpus_id=(
            corpus.manifest.corpus_id
        ),
    )

    output_path = tmp_path / "report.json"

    result = (
        report_script
        .run_competition_evidence_candidate_report(
            audit_path=audit_path,
            corpus_directory=(
                corpus.corpus_directory
            ),
            output_path=output_path,
            top_k=5,
        )
    )

    assert len(result.records) == 1

    record = result.records[0]

    assert record.missing_fact.case_id == "Q101"
    assert record.missing_fact.fact_index == 0
    assert record.candidates

    assert (
        record.candidates[0]
        .chunk.source_id
        == SOURCE_ID
    )

    assert output_path.is_file()

    payload = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert payload["report_version"] == (
        "competition_evidence_candidates_dev_v1"
    )

    assert payload["missing_fact_count"] == 1
    assert payload["top_k"] == 5

    assert (
        payload["records"][0]
        ["candidates"][0]["text"]
    )


def test_runner_rejects_corpus_identity_mismatch(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path)

    audit_path = tmp_path / "audit.json"
    _write_audit(
        path=audit_path,
        corpus_id=(
            "competition_corpus_"
            "0000000000000000"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="corpus_id 不一致",
    ):
        (
            report_script
            .run_competition_evidence_candidate_report(
                audit_path=audit_path,
                corpus_directory=(
                    corpus.corpus_directory
                ),
                output_path=(
                    tmp_path / "report.json"
                ),
            )
        )


def test_runner_rejects_invalid_top_k(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        (
            report_script
            .run_competition_evidence_candidate_report(
                audit_path=(
                    tmp_path / "audit.json"
                ),
                corpus_directory=(
                    tmp_path / "corpus"
                ),
                output_path=(
                    tmp_path / "report.json"
                ),
                top_k=0,
            )
        )


def test_loader_rejects_invalid_cases(
    tmp_path: Path,
) -> None:
    audit_path = tmp_path / "audit.json"

    audit_path.write_text(
        json.dumps(
            {
                "audit_version": (
                    "competition_evidence_coverage_dev_v1"
                ),
                "corpus_id": (
                    "competition_corpus_"
                    "0000000000000000"
                ),
                "cases": "invalid",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="缺少 cases",
    ):
        report_script.load_missing_evidence_facts(
            audit_path
        )