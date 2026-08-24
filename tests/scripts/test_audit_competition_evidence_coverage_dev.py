from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas.competition import (
    CompetitionQaCase,
    CompetitionSourceRecord,
)
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
    audit_competition_evidence_coverage_dev
    as audit_script,
)


def _case() -> CompetitionQaCase:
    return CompetitionQaCase(
        case_id="Q101",
        source_type="word",
        difficulty="easy",
        difficulty_cn="简单",
        qa_type="单事实检索",
        question="资本充足率应满足什么要求？",
        option_a="应满足监管要求。",
        option_b="不需要满足监管要求。",
        option_c="只需要满足利润要求。",
        option_d="以上均不正确。",
        answer="A",
        answer_text="应满足监管要求。",
        evidence=(
            "商业银行应满足资本充足率"
            "监管要求。"
        ),
        source_title="测试制度",
        file_label="测试制度.docx",
    )


def _build_corpus(tmp_path: Path):
    source = CompetitionKnowledgeSource(
        source_id=(
            "src_0000000000000001"
        ),
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
                    "商业银行应满足资本充足率"
                    "监管要求。"
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

    corpus = build_competition_chunk_corpus(
        documents=(chunk_document,),
        output_root=(
            tmp_path / "corpora"
        ),
    )

    source_record = CompetitionSourceRecord(
        source_id=source.source_id,
        source_type="word",
        actual_filename="测试制度.docx",
        relative_path=source.relative_path,
        extension=".docx",
        size_bytes=100,
    )

    return corpus, source_record


def test_runner_audits_frozen_text_cases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corpus, source_record = (
        _build_corpus(tmp_path)
    )

    case = _case()

    monkeypatch.setattr(
        audit_script,
        "load_competition_qa_excel",
        lambda path: (case,),
    )

    monkeypatch.setattr(
        audit_script,
        "build_competition_source_manifest",
        lambda path: (source_record,),
    )

    qa_path = tmp_path / "QA.xlsx"
    qa_path.write_bytes(b"qa")

    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "dev_case_ids": [
                    case.case_id
                ]
            }
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "audit.json"

    result = (
        audit_script
        .run_competition_evidence_coverage_dev(
            qa_file=qa_path,
            attachments_root=(
                tmp_path / "attachments"
            ),
            split_file=split_path,
            corpus_directory=(
                corpus.corpus_directory
            ),
            output_path=output_path,
            expected_text_case_count=1,
        )
    )

    assert result.dev_case_count == 1
    assert result.text_dev_case_count == 1
    assert result.summary.fact_count == 1

    assert (
        result.summary
        .single_chunk_fact_coverage
        == 1.0
    )

    assert (
        result.summary
        .document_fact_coverage
        == 1.0
    )

    assert output_path.is_file()

    payload = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert payload["audit_version"] == (
        "competition_evidence_coverage_dev_v1"
    )

    assert payload["summary"][
        "single_chunk_fact_coverage"
    ] == 1.0

    assert payload["cases"][0][
        "resolution_strategy"
    ]


def test_runner_rejects_changed_text_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _case()

    monkeypatch.setattr(
        audit_script,
        "load_competition_qa_excel",
        lambda path: (case,),
    )

    qa_path = tmp_path / "QA.xlsx"
    qa_path.write_bytes(b"qa")

    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "dev_case_ids": [
                    case.case_id
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="文本 QA 数量改变",
    ):
        (
            audit_script
            .run_competition_evidence_coverage_dev(
                qa_file=qa_path,
                attachments_root=(
                    tmp_path / "attachments"
                ),
                split_file=split_path,
                corpus_directory=(
                    tmp_path / "corpus"
                ),
                output_path=(
                    tmp_path / "audit.json"
                ),
                expected_text_case_count=69,
            )
        )


def test_runner_rejects_unknown_dev_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _case()

    monkeypatch.setattr(
        audit_script,
        "load_competition_qa_excel",
        lambda path: (case,),
    )

    qa_path = tmp_path / "QA.xlsx"
    qa_path.write_bytes(b"qa")

    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "dev_case_ids": [
                    "Q999"
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="未知 QA",
    ):
        (
            audit_script
            .run_competition_evidence_coverage_dev(
                qa_file=qa_path,
                attachments_root=(
                    tmp_path / "attachments"
                ),
                split_file=split_path,
                corpus_directory=(
                    tmp_path / "corpus"
                ),
                output_path=(
                    tmp_path / "audit.json"
                ),
                expected_text_case_count=None,
            )
        )