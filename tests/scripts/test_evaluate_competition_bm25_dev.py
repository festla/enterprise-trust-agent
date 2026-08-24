from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.rag.tokenization import (
    DeterministicChineseBigramTokenizer,
)
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
from app.services.competition_bm25_index import (
    build_competition_bm25_index,
)
from app.services.competition_corpus import (
    build_competition_chunk_corpus,
)
from app.services.competition_text_chunker import (
    build_competition_text_chunks,
)
from scripts import (
    evaluate_competition_bm25_dev
    as baseline_script,
)


def _case() -> CompetitionQaCase:
    return CompetitionQaCase(
        case_id="Q101",
        source_type="word",
        difficulty="easy",
        difficulty_cn="简单",
        qa_type="单事实检索",
        question=(
            "商业银行资本充足率"
            "应满足什么要求？"
        ),
        option_a=(
            "商业银行应满足"
            "资本充足率监管要求。"
        ),
        option_b="不需要满足监管要求。",
        option_c="只需要满足流动性要求。",
        option_d="只需要满足利润要求。",
        answer="A",
        answer_text=(
            "商业银行应满足"
            "资本充足率监管要求。"
        ),
        evidence=(
            "商业银行应满足"
            "资本充足率监管要求。"
        ),
        source_title="测试制度",
        file_label="测试制度.docx",
    )


def _build_index(
    tmp_path: Path,
):
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
        relative_path=(
            "word/"
            "测试制度_测试制度.docx"
        ),
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
                    "商业银行应满足"
                    "资本充足率监管要求。"
                ),
            ),
        ),
    )

    chunk_document = (
        CompetitionChunkDocument(
            source=source,
            chunks=(
                build_competition_text_chunks(
                    document
                )
            ),
        )
    )

    corpus = (
        build_competition_chunk_corpus(
            documents=(
                chunk_document,
            ),
            output_root=(
                tmp_path / "corpora"
            ),
        )
    )

    index_result = (
        build_competition_bm25_index(
            corpus_directory=(
                corpus.corpus_directory
            ),
            output_root=(
                tmp_path / "indexes"
            ),
            tokenizer=(
                DeterministicChineseBigramTokenizer()
            ),
        )
    )

    source_record = (
        CompetitionSourceRecord(
            source_id=source.source_id,
            source_type="word",
            actual_filename=(
                "测试制度_测试制度.docx"
            ),
            relative_path=(
                source.relative_path
            ),
            extension=".docx",
            size_bytes=100,
        )
    )

    return (
        corpus,
        index_result,
        source_record,
    )


def test_builds_query_without_gold_fields(
) -> None:
    case = _case()

    question_only = (
        baseline_script.build_bm25_query(
            case=case,
            mode="question_only",
        )
    )

    assert question_only == (
        case.question
    )

    with_options = (
        baseline_script.build_bm25_query(
            case=case,
            mode=(
                "question_with_options"
            ),
        )
    )

    assert case.question in with_options
    assert "A. " in with_options
    assert case.option_a in with_options
    assert "B. " in with_options


def test_load_dev_case_ids_validates_duplicates(
    tmp_path: Path,
) -> None:
    split_path = (
        tmp_path / "split.json"
    )

    split_path.write_text(
        json.dumps(
            {
                "dev_case_ids": [
                    "Q101",
                    "Q101",
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="重复 case_id",
    ):
        baseline_script.load_dev_case_ids(
            split_path
        )


def test_runner_evaluates_both_query_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (
        corpus,
        index_result,
        source_record,
    ) = _build_index(tmp_path)

    case = _case()

    monkeypatch.setattr(
        baseline_script,
        "load_competition_qa_excel",
        lambda path: (case,),
    )

    monkeypatch.setattr(
        baseline_script,
        "build_competition_source_manifest",
        lambda path: (
            source_record,
        ),
    )

    split_path = (
        tmp_path / "split.json"
    )

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

    qa_path = tmp_path / "QA.xlsx"
    qa_path.write_bytes(
        b"test-qa"
    )

    output_path = (
        tmp_path / "baseline.json"
    )

    result = (
        baseline_script
        .run_competition_bm25_dev(
            qa_file=qa_path,
            attachments_root=(
                tmp_path / "attachments"
            ),
            split_file=split_path,
            corpus_directory=(
                corpus.corpus_directory
            ),
            index_directory=(
                index_result
                .index_directory
            ),
            output_path=output_path,
            top_k=3,
            cutoffs=(1, 3),
            expected_text_case_count=1,
        )
    )

    assert result.dev_case_count == 1
    assert result.text_dev_case_count == 1

    assert {
        mode.query_mode
        for mode in result.mode_results
    } == {
        "question_only",
        "question_with_options",
    }

    for mode in result.mode_results:
        assert (
            mode.summary.at(1)
            .source_hit_rate
            == 1.0
        )

        assert (
            mode.summary.at(1)
            .all_facts_hit_rate
            == 1.0
        )

    assert output_path.is_file()

    payload = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert payload[
        "baseline_version"
    ] == "competition_bm25_dev_v1"

    assert set(
        payload["modes"]
    ) == {
        "question_only",
        "question_with_options",
    }


def test_runner_rejects_changed_frozen_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _case()

    monkeypatch.setattr(
        baseline_script,
        "load_competition_qa_excel",
        lambda path: (case,),
    )

    split_path = (
        tmp_path / "split.json"
    )

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

    qa_path = tmp_path / "QA.xlsx"
    qa_path.write_bytes(b"qa")

    with pytest.raises(
        RuntimeError,
        match="文本 QA 数量改变",
    ):
        baseline_script.run_competition_bm25_dev(
            qa_file=qa_path,
            attachments_root=(
                tmp_path / "attachments"
            ),
            split_file=split_path,
            corpus_directory=(
                tmp_path / "corpus"
            ),
            index_directory=(
                tmp_path / "index"
            ),
            output_path=(
                tmp_path / "output.json"
            ),
            expected_text_case_count=69,
        )