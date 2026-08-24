from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.schemas.competition_chunk import (
    CompetitionChunkDocument,
    CompetitionTextChunk,
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
from app.services.competition_corpus import (
    build_competition_chunk_corpus,
)
from app.services.competition_text_chunker import (
    build_competition_text_chunks,
)
from scripts import (
    evaluate_competition_bm25_query_fusion_dev
    as fusion_script,
)


def _chunk_document(
    *,
    source_number: int,
    text: str,
) -> tuple[
    CompetitionChunkDocument,
    CompetitionTextChunk,
]:
    source_id = (
        f"src_{source_number:016x}"
    )

    source = CompetitionKnowledgeSource(
        source_id=source_id,
        doc_id=(
            f"doc_{source_id}_"
            "0123456789abcdef01234567"
        ),
        title=f"测试来源 {source_number}",
        source_type="word",
        relative_path=(
            f"word/{source_id}.docx"
        ),
        sha256=(
            f"{source_number:x}"[-1]
            * 64
        ),
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
                text=text,
            ),
        ),
    )

    chunk = build_competition_text_chunks(
        document
    )[0]

    return (
        CompetitionChunkDocument(
            source=source,
            chunks=(chunk,),
        ),
        chunk,
    )


def _hit_payload(
    chunk: CompetitionTextChunk,
    *,
    rank: int,
) -> dict[str, object]:
    return {
        "rank": rank,
        "score": 10.0 / rank,
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "doc_id": chunk.doc_id,
        "chunk_type": chunk.chunk_type,
    }


def _summary_payload() -> dict[str, object]:
    return {
        "fact_count": 1,
        "metrics": [
            {
                "cutoff": 1,
                "fact_direct_hit_rate": 0.0,
                "fact_any_gold_hit_rate": 0.0,
                "fact_complete_gold_hit_rate": 0.0,
                "mean_gold_chunk_recall": 0.0,
                "mean_direct_reciprocal_rank": 0.0,
            },
            {
                "cutoff": 2,
                "fact_direct_hit_rate": 1.0,
                "fact_any_gold_hit_rate": 1.0,
                "fact_complete_gold_hit_rate": 1.0,
                "mean_gold_chunk_recall": 1.0,
                "mean_direct_reciprocal_rank": 0.5,
            },
        ],
    }


def _prepare_inputs(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    dict[str, object],
]:
    gold_document, gold_chunk = (
        _chunk_document(
            source_number=1,
            text="直接 Gold 证据。",
        )
    )

    question_document, question_chunk = (
        _chunk_document(
            source_number=2,
            text="仅问题查询候选。",
        )
    )

    option_document, option_chunk = (
        _chunk_document(
            source_number=3,
            text="仅选项查询候选。",
        )
    )

    corpus = build_competition_chunk_corpus(
        documents=(
            gold_document,
            question_document,
            option_document,
        ),
        output_root=tmp_path / "corpora",
    )

    gold_path = tmp_path / "gold.json"

    gold_dataset = CompetitionGoldChunkDataset(
        corpus_id=corpus.manifest.corpus_id,
        candidate_report_sha256="d" * 64,
        records=(
            CompetitionGoldFactRecord(
                case_id="Q101",
                fact_index=0,
                fact="直接 Gold 证据。",
                expected_source_id=(
                    gold_chunk.source_id
                ),
                review_status="confirmed",
                evidence_mode="single_chunk",
                gold_chunks=(
                    CompetitionGoldChunkReference(
                        chunk_id=gold_chunk.chunk_id,
                        role="direct",
                    ),
                ),
            ),
        ),
    )

    gold_path.write_text(
        gold_dataset.model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )

    gold_sha256 = hashlib.sha256(
        gold_path.read_bytes()
    ).hexdigest()

    question_case = {
        "case_id": "Q101",
        "query": "问题查询",
        "expected_source_id": (
            gold_chunk.source_id
        ),
        "retrieved_chunks": [
            _hit_payload(
                question_chunk,
                rank=1,
            ),
            _hit_payload(
                gold_chunk,
                rank=2,
            ),
        ],
    }

    option_case = {
        "case_id": "Q101",
        "query": "问题查询\nA. 选项",
        "expected_source_id": (
            gold_chunk.source_id
        ),
        "retrieved_chunks": [
            _hit_payload(
                option_chunk,
                rank=1,
            ),
            _hit_payload(
                gold_chunk,
                rank=2,
            ),
        ],
    }

    baseline = {
        "schema_version": 1,
        "evaluation_version": (
            "competition_bm25_gold_dev_v1"
        ),
        "qa_sha256": "a" * 64,
        "split_sha256": "b" * 64,
        "gold_sha256": gold_sha256,
        "corpus_id": corpus.manifest.corpus_id,
        "index_id": (
            "competition_bm25_index_"
            "0000000000000001"
        ),
        "text_dev_case_count": 1,
        "fact_count": 1,
        "top_k": 2,
        "cutoffs": [1, 2],
        "query_uses_gold_fact_text": False,
        "modes": {
            "question_only": {
                "summary": _summary_payload(),
                "cases": [question_case],
            },
            "question_with_options": {
                "summary": _summary_payload(),
                "cases": [option_case],
            },
        },
    }

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            baseline,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "fusion.json"

    return (
        baseline_path,
        gold_path,
        corpus.corpus_directory,
        output_path,
        baseline,
    )


def _write_baseline(
    path: Path,
    payload: dict[str, object],
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


def _run(
    prepared: tuple[
        Path,
        Path,
        Path,
        Path,
        dict[str, object],
    ],
):
    (
        baseline_path,
        gold_path,
        corpus_directory,
        output_path,
        _,
    ) = prepared

    return (
        fusion_script
        .run_competition_bm25_query_fusion_dev(
            baseline_path=baseline_path,
            gold_path=gold_path,
            corpus_directory=corpus_directory,
            output_path=output_path,
            top_k=2,
            cutoffs=(1, 2),
            expected_text_case_count=1,
            expected_fact_count=1,
        )
    )


def test_runner_fuses_real_baseline_hits(
    tmp_path: Path,
) -> None:
    prepared = _prepare_inputs(tmp_path)

    result = _run(prepared)

    assert len(result.case_results) == 1
    assert len(result.fact_results) == 1
    assert (
        result.summary.at(1)
        .fact_direct_hit_rate
        == 1.0
    )

    case = result.case_results[0]

    assert case.fusion_hits[0].chunk_id == (
        result.gold_dataset.records[0]
        .gold_chunks[0].chunk_id
    )

    assert case.fusion_hits[0].source_queries == (
        "question_only",
        "question_with_options",
    )

    output_path = prepared[3]
    payload = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert payload[
        "evaluation_version"
    ] == (
        "competition_bm25_query_fusion_dev_v1"
    )
    assert payload[
        "query_uses_gold_fact_text"
    ] is False


def test_rejects_changed_gold_hash(
    tmp_path: Path,
) -> None:
    prepared = _prepare_inputs(tmp_path)
    baseline = prepared[4]
    baseline["gold_sha256"] = "f" * 64
    _write_baseline(prepared[0], baseline)

    with pytest.raises(
        RuntimeError,
        match="SHA256 不一致",
    ):
        _run(prepared)


def test_rejects_corpus_mismatch(
    tmp_path: Path,
) -> None:
    prepared = _prepare_inputs(tmp_path)
    baseline = prepared[4]
    baseline["corpus_id"] = (
        "competition_corpus_"
        "ffffffffffffffff"
    )
    _write_baseline(prepared[0], baseline)

    with pytest.raises(
        RuntimeError,
        match="corpus_id 不一致",
    ):
        _run(prepared)


def test_rejects_mode_case_mismatch(
    tmp_path: Path,
) -> None:
    prepared = _prepare_inputs(tmp_path)
    baseline = prepared[4]

    modes = baseline["modes"]
    assert isinstance(modes, dict)

    options = modes["question_with_options"]
    assert isinstance(options, dict)

    options["cases"] = []
    _write_baseline(prepared[0], baseline)

    with pytest.raises(
        RuntimeError,
        match="与 Gold Case 不一致",
    ):
        _run(prepared)


def test_rejects_unknown_baseline_chunk(
    tmp_path: Path,
) -> None:
    prepared = _prepare_inputs(tmp_path)
    baseline = prepared[4]

    modes = baseline["modes"]
    assert isinstance(modes, dict)

    question = modes["question_only"]
    assert isinstance(question, dict)

    cases = question["cases"]
    assert isinstance(cases, list)

    case = cases[0]
    assert isinstance(case, dict)

    hits = case["retrieved_chunks"]
    assert isinstance(hits, list)

    hit = hits[0]
    assert isinstance(hit, dict)

    hit["chunk_id"] = "chunk:not-in-corpus"
    _write_baseline(prepared[0], baseline)

    with pytest.raises(
        RuntimeError,
        match="不存在的 Chunk",
    ):
        _run(prepared)


def test_rejects_output_top_k_above_baseline(
    tmp_path: Path,
) -> None:
    prepared = _prepare_inputs(tmp_path)

    with pytest.raises(
        RuntimeError,
        match="小于融合输出 top_k",
    ):
        (
            fusion_script
            .run_competition_bm25_query_fusion_dev(
                baseline_path=prepared[0],
                gold_path=prepared[1],
                corpus_directory=prepared[2],
                output_path=prepared[3],
                top_k=3,
                cutoffs=(1, 2),
                expected_text_case_count=1,
                expected_fact_count=1,
            )
        )