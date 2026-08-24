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
from app.schemas.competition_gold import (
    CompetitionGoldChunkDataset,
    CompetitionGoldChunkReference,
    CompetitionGoldFactRecord,
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
    evaluate_competition_bm25_gold_dev
    as gold_script,
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


def _build_index(tmp_path: Path):
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

    chunks = build_competition_text_chunks(
        document
    )

    chunk_document = CompetitionChunkDocument(
        source=source,
        chunks=chunks,
    )

    corpus = build_competition_chunk_corpus(
        documents=(chunk_document,),
        output_root=(
            tmp_path / "corpora"
        ),
    )

    index_result = build_competition_bm25_index(
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

    source_record = CompetitionSourceRecord(
        source_id=source.source_id,
        source_type="word",
        actual_filename=(
            "测试制度_测试制度.docx"
        ),
        relative_path=source.relative_path,
        extension=".docx",
        size_bytes=100,
    )

    return (
        corpus,
        index_result,
        source_record,
        chunks[0],
    )


def _write_gold(
    path: Path,
    *,
    corpus_id: str,
    chunk_id: str,
) -> None:
    dataset = CompetitionGoldChunkDataset(
        corpus_id=corpus_id,
        candidate_report_sha256="b" * 64,
        records=(
            CompetitionGoldFactRecord(
                case_id="Q101",
                fact_index=0,
                fact=(
                    "GOLD_SECRET_不得进入检索查询"
                ),
                expected_source_id=(
                    "src_0000000000000001"
                ),
                review_status="confirmed",
                evidence_mode="single_chunk",
                gold_chunks=(
                    CompetitionGoldChunkReference(
                        chunk_id=chunk_id,
                        role="direct",
                    ),
                ),
            ),
        ),
    )

    path.write_text(
        dataset.model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )


def _prepare_runner(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    gold_corpus_id: str | None = None,
    gold_chunk_id: str | None = None,
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
]:
    (
        corpus,
        index_result,
        source_record,
        gold_chunk,
    ) = _build_index(tmp_path)

    case = _case()

    monkeypatch.setattr(
        gold_script,
        "load_competition_qa_excel",
        lambda path: (case,),
    )

    monkeypatch.setattr(
        gold_script,
        "build_competition_source_manifest",
        lambda path: (source_record,),
    )

    qa_path = tmp_path / "QA.xlsx"
    qa_path.write_bytes(b"test-qa")

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

    gold_path = tmp_path / "gold.json"

    _write_gold(
        gold_path,
        corpus_id=(
            gold_corpus_id
            if gold_corpus_id is not None
            else corpus.manifest.corpus_id
        ),
        chunk_id=(
            gold_chunk_id
            if gold_chunk_id is not None
            else gold_chunk.chunk_id
        ),
    )

    output_path = tmp_path / "report.json"

    return (
        qa_path,
        split_path,
        gold_path,
        corpus.corpus_directory,
        index_result.index_directory,
        output_path,
    )


def _run(
    paths: tuple[
        Path,
        Path,
        Path,
        Path,
        Path,
        Path,
    ],
    *,
    expected_fact_count: int = 1,
):
    (
        qa_path,
        split_path,
        gold_path,
        corpus_directory,
        index_directory,
        output_path,
    ) = paths

    return gold_script.run_competition_bm25_gold_dev(
        qa_file=qa_path,
        attachments_root=(
            qa_path.parent / "attachments"
        ),
        split_file=split_path,
        gold_file=gold_path,
        corpus_directory=corpus_directory,
        index_directory=index_directory,
        output_path=output_path,
        top_k=3,
        cutoffs=(1, 3),
        expected_text_case_count=1,
        expected_fact_count=expected_fact_count,
    )


def test_load_gold_rejects_pending_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pending.json"

    dataset = CompetitionGoldChunkDataset(
        corpus_id=(
            "competition_corpus_"
            "0000000000000001"
        ),
        candidate_report_sha256="c" * 64,
        records=(
            CompetitionGoldFactRecord(
                case_id="Q101",
                fact_index=0,
                fact="待审核事实。",
                expected_source_id=(
                    "src_0000000000000001"
                ),
            ),
        ),
    )

    path.write_text(
        dataset.model_dump_json(indent=2),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="未确认记录",
    ):
        gold_script.load_competition_gold_dataset(
            path,
            expected_fact_count=1,
        )


def test_runner_evaluates_both_modes_without_gold_leakage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _prepare_runner(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
    )

    result = _run(paths)

    assert result.text_dev_case_count == 1
    assert result.fact_count == 1

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
            .fact_direct_hit_rate
            == 1.0
        )
        assert (
            mode.summary.at(1)
            .fact_complete_gold_hit_rate
            == 1.0
        )
        assert all(
            "GOLD_SECRET" not in case_result.query
            for case_result
            in mode.case_results
        )

    output_path = paths[-1]
    payload = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert payload[
        "evaluation_version"
    ] == "competition_bm25_gold_dev_v1"
    assert payload[
        "query_uses_gold_fact_text"
    ] is False
    assert payload["fact_count"] == 1
    assert set(payload["modes"]) == {
        "question_only",
        "question_with_options",
    }


def test_runner_rejects_gold_corpus_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _prepare_runner(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        gold_corpus_id=(
            "competition_corpus_"
            "ffffffffffffffff"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="corpus_id 不一致",
    ):
        _run(paths)


def test_runner_rejects_missing_gold_chunk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _prepare_runner(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        gold_chunk_id="chunk:not-in-index",
    )

    with pytest.raises(
        RuntimeError,
        match="不存在于 BM25 Index",
    ):
        _run(paths)


def test_runner_rejects_changed_gold_fact_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _prepare_runner(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
    )

    with pytest.raises(
        RuntimeError,
        match="Gold Fact 数量改变",
    ):
        _run(
            paths,
            expected_fact_count=2,
        )


def test_runner_rejects_top_k_below_cutoff(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="top_k 不能小于",
    ):
        gold_script.run_competition_bm25_gold_dev(
            qa_file=tmp_path / "QA.xlsx",
            attachments_root=(
                tmp_path / "attachments"
            ),
            split_file=tmp_path / "split.json",
            gold_file=tmp_path / "gold.json",
            corpus_directory=tmp_path / "corpus",
            index_directory=tmp_path / "index",
            output_path=tmp_path / "output.json",
            top_k=2,
            cutoffs=(1, 3),
        )