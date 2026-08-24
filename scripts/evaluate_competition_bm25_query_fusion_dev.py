from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from app.schemas.competition_chunk import (
    CompetitionTextChunk,
)
from app.schemas.competition_gold import (
    CompetitionGoldChunkDataset,
    CompetitionGoldFactRecord,
)
from app.schemas.competition_query_fusion import (
    CompetitionBM25QueryFusionConfig,
    CompetitionBM25QueryFusionHit,
)
from app.schemas.competition_retrieval import (
    CompetitionBM25Hit,
)
from app.services.competition_bm25_query_fusion import (
    fuse_competition_bm25_queries,
)
from app.services.competition_corpus import (
    CompetitionCorpusBuildResult,
    load_competition_chunk_corpus,
)
from app.services.competition_gold_retrieval_eval import (
    CompetitionGoldFactRetrievalResult,
    CompetitionGoldRetrievalEvalSummary,
    evaluate_competition_gold_fact,
    summarize_competition_gold_retrieval,
)
from scripts.evaluate_competition_bm25_gold_dev import (
    load_competition_gold_dataset,
)


DEFAULT_BASELINE_FILE = Path(
    "data/competition/processed/eval/"
    "competition_bm25_gold_dev_v1.json"
)

DEFAULT_GOLD_FILE = Path(
    "data/competition/processed/eval/"
    "competition_dev_gold_chunks_v1.json"
)

DEFAULT_OUTPUT_FILE = Path(
    "data/competition/processed/eval/"
    "competition_bm25_query_fusion_dev_v1.json"
)

DEFAULT_CUTOFFS = (1, 3, 5, 10)
DEFAULT_TOP_K = 10
DEFAULT_EXPECTED_TEXT_DEV_CASES = 69
DEFAULT_EXPECTED_FACTS = 97

EXPECTED_BASELINE_VERSION = (
    "competition_bm25_gold_dev_v1"
)


@dataclass(frozen=True, slots=True)
class CompetitionBM25QueryFusionCaseResult:
    case_id: str
    expected_source_id: str
    question_only_query: str
    question_with_options_query: str
    question_only_hits: tuple[
        CompetitionBM25Hit,
        ...,
    ]
    question_with_options_hits: tuple[
        CompetitionBM25Hit,
        ...,
    ]
    fusion_hits: tuple[
        CompetitionBM25QueryFusionHit,
        ...,
    ]
    fact_results: tuple[
        CompetitionGoldFactRetrievalResult,
        ...,
    ]


@dataclass(frozen=True, slots=True)
class CompetitionBM25QueryFusionDevResult:
    baseline_path: Path
    baseline_sha256: str
    baseline_payload: dict[str, object]
    gold_path: Path
    gold_sha256: str
    gold_dataset: CompetitionGoldChunkDataset
    corpus: CompetitionCorpusBuildResult
    config: CompetitionBM25QueryFusionConfig
    case_results: tuple[
        CompetitionBM25QueryFusionCaseResult,
        ...,
    ]
    fact_results: tuple[
        CompetitionGoldFactRetrievalResult,
        ...,
    ]
    summary: CompetitionGoldRetrievalEvalSummary
    cutoffs: tuple[int, ...]
    top_k: int
    output_path: Path


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _load_json_object(
    path: Path,
    *,
    label: str,
) -> dict[str, object]:
    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError(
            f"无法读取 {label}：{path}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"{label} 根节点必须是 JSON Object"
        )

    return payload


def _require_string(
    payload: dict[str, object],
    field_name: str,
    *,
    context: str,
) -> str:
    value = payload.get(field_name)

    if (
        not isinstance(value, str)
        or not value
    ):
        raise RuntimeError(
            f"{context} 缺少有效的 "
            f"{field_name}"
        )

    return value


def _require_non_negative_int(
    payload: dict[str, object],
    field_name: str,
    *,
    context: str,
) -> int:
    value = payload.get(field_name)

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise RuntimeError(
            f"{context} 的 {field_name} 无效"
        )

    return value


def _group_gold_records(
    records: tuple[
        CompetitionGoldFactRecord,
        ...,
    ],
) -> dict[
    str,
    tuple[CompetitionGoldFactRecord, ...],
]:
    groups: dict[
        str,
        list[CompetitionGoldFactRecord],
    ] = {}

    for record in records:
        groups.setdefault(
            record.case_id,
            [],
        ).append(record)

    return {
        case_id: tuple(
            sorted(
                group,
                key=lambda record: (
                    record.fact_index
                ),
            )
        )
        for case_id, group
        in sorted(groups.items())
    }


def _load_mode_cases(
    baseline: dict[str, object],
    *,
    mode: str,
) -> dict[str, dict[str, object]]:
    modes = baseline.get("modes")

    if not isinstance(modes, dict):
        raise RuntimeError(
            "BM25 Gold Baseline 缺少 modes"
        )

    mode_payload = modes.get(mode)

    if not isinstance(mode_payload, dict):
        raise RuntimeError(
            "BM25 Gold Baseline 缺少模式："
            f"{mode}"
        )

    raw_cases = mode_payload.get("cases")

    if not isinstance(raw_cases, list):
        raise RuntimeError(
            f"BM25 模式 {mode} 缺少 cases"
        )

    cases: dict[
        str,
        dict[str, object],
    ] = {}

    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise RuntimeError(
                f"BM25 模式 {mode} 包含无效 Case"
            )

        case_id = _require_string(
            raw_case,
            "case_id",
            context=f"BM25 模式 {mode}",
        )

        if case_id in cases:
            raise RuntimeError(
                f"BM25 模式 {mode} 包含重复 Case："
                f"{case_id}"
            )

        cases[case_id] = raw_case

    return cases


def _rebuild_hits(
    raw_case: dict[str, object],
    *,
    chunks_by_id: dict[
        str,
        CompetitionTextChunk,
    ],
    mode: str,
) -> tuple[CompetitionBM25Hit, ...]:
    raw_hits = raw_case.get(
        "retrieved_chunks"
    )

    if not isinstance(raw_hits, list):
        raise RuntimeError(
            f"BM25 模式 {mode} Case 缺少 "
            "retrieved_chunks"
        )

    hits = []

    for raw_hit in raw_hits:
        if not isinstance(raw_hit, dict):
            raise RuntimeError(
                f"BM25 模式 {mode} 包含无效 Hit"
            )

        rank = _require_non_negative_int(
            raw_hit,
            "rank",
            context=f"BM25 模式 {mode} Hit",
        )

        if rank < 1:
            raise RuntimeError(
                f"BM25 模式 {mode} Hit rank "
                "必须大于等于 1"
            )

        score = raw_hit.get("score")

        if (
            isinstance(score, bool)
            or not isinstance(
                score,
                (int, float),
            )
            or score <= 0
        ):
            raise RuntimeError(
                f"BM25 模式 {mode} Hit score 无效"
            )

        chunk_id = _require_string(
            raw_hit,
            "chunk_id",
            context=f"BM25 模式 {mode} Hit",
        )

        chunk = chunks_by_id.get(chunk_id)

        if chunk is None:
            raise RuntimeError(
                "BM25 Baseline 引用了 Corpus 中"
                "不存在的 Chunk："
                f"{chunk_id}"
            )

        source_id = raw_hit.get("source_id")

        if (
            source_id is not None
            and source_id != chunk.source_id
        ):
            raise RuntimeError(
                "BM25 Baseline Hit 的 source_id "
                "与 Corpus 不一致："
                f"{chunk_id}"
            )

        hits.append(
            CompetitionBM25Hit(
                rank=rank,
                score=float(score),
                chunk=chunk,
            )
        )

    return tuple(hits)


def _validate_input_binding(
    *,
    baseline: dict[str, object],
    baseline_path: Path,
    gold_path: Path,
    gold_dataset: CompetitionGoldChunkDataset,
    corpus: CompetitionCorpusBuildResult,
    expected_text_case_count: int | None,
    expected_fact_count: int | None,
    top_k: int,
) -> None:
    if (
        baseline.get("evaluation_version")
        != EXPECTED_BASELINE_VERSION
    ):
        raise RuntimeError(
            "BM25 Gold Baseline 版本不受支持"
        )

    if (
        baseline.get(
            "query_uses_gold_fact_text"
        )
        is not False
    ):
        raise RuntimeError(
            "BM25 Gold Baseline 未通过"
            "查询泄漏门禁"
        )

    baseline_corpus_id = _require_string(
        baseline,
        "corpus_id",
        context="BM25 Gold Baseline",
    )

    if (
        baseline_corpus_id
        != corpus.manifest.corpus_id
        or baseline_corpus_id
        != gold_dataset.corpus_id
    ):
        raise RuntimeError(
            "Baseline、Gold 与 Corpus 的 "
            "corpus_id 不一致"
        )

    baseline_gold_sha256 = _require_string(
        baseline,
        "gold_sha256",
        context="BM25 Gold Baseline",
    )

    if (
        baseline_gold_sha256
        != _sha256_file(gold_path)
    ):
        raise RuntimeError(
            "BM25 Gold Baseline 与当前 Gold "
            "SHA256 不一致"
        )

    baseline_top_k = (
        _require_non_negative_int(
            baseline,
            "top_k",
            context="BM25 Gold Baseline",
        )
    )

    if baseline_top_k < top_k:
        raise RuntimeError(
            "BM25 Gold Baseline 的 top_k "
            "小于融合输出 top_k"
        )

    text_case_count = (
        _require_non_negative_int(
            baseline,
            "text_dev_case_count",
            context="BM25 Gold Baseline",
        )
    )

    fact_count = _require_non_negative_int(
        baseline,
        "fact_count",
        context="BM25 Gold Baseline",
    )

    if (
        expected_text_case_count is not None
        and text_case_count
        != expected_text_case_count
    ):
        raise RuntimeError(
            "Baseline 文本 Dev Case 数量改变："
            f"expected={expected_text_case_count}; "
            f"actual={text_case_count}"
        )

    if (
        expected_fact_count is not None
        and fact_count != expected_fact_count
    ):
        raise RuntimeError(
            "Baseline Gold Fact 数量改变："
            f"expected={expected_fact_count}; "
            f"actual={fact_count}"
        )

    if fact_count != len(gold_dataset.records):
        raise RuntimeError(
            "Baseline 与 Gold 的 Fact 数量不一致"
        )

    if not baseline_path.is_file():
        raise RuntimeError(
            "BM25 Gold Baseline 文件不存在"
        )


def _summary_payload(
    summary: CompetitionGoldRetrievalEvalSummary,
) -> dict[str, object]:
    return asdict(summary)


def _build_breakdowns(
    *,
    results: tuple[
        CompetitionGoldFactRetrievalResult,
        ...,
    ],
    cutoffs: tuple[int, ...],
) -> dict[str, object]:
    groups: dict[
        str,
        list[CompetitionGoldFactRetrievalResult],
    ] = {}

    for result in results:
        groups.setdefault(
            result.evidence_mode,
            [],
        ).append(result)

    return {
        evidence_mode: _summary_payload(
            summarize_competition_gold_retrieval(
                tuple(group),
                cutoffs=cutoffs,
            )
        )
        for evidence_mode, group
        in sorted(groups.items())
    }


def _hit_payload(
    hit: CompetitionBM25Hit,
) -> dict[str, object]:
    return {
        "rank": hit.rank,
        "score": hit.score,
        "chunk_id": hit.chunk_id,
        "source_id": hit.source_id,
        "doc_id": hit.doc_id,
        "chunk_type": hit.chunk_type,
    }


def _fusion_hit_payload(
    hit: CompetitionBM25QueryFusionHit,
) -> dict[str, object]:
    return {
        "rank": hit.rank,
        "score": hit.score,
        "rrf_score": hit.rrf_score,
        "chunk_id": hit.chunk_id,
        "source_id": hit.source_id,
        "doc_id": hit.doc_id,
        "chunk_type": hit.chunk_type,
        "question_only_rank": (
            hit.question_only_rank
        ),
        "question_with_options_rank": (
            hit.question_with_options_rank
        ),
        "source_queries": list(
            hit.source_queries
        ),
    }


def _build_output_payload(
    result: CompetitionBM25QueryFusionDevResult,
) -> dict[str, object]:
    baseline_modes = result.baseline_payload[
        "modes"
    ]

    assert isinstance(baseline_modes, dict)

    return {
        "schema_version": 1,
        "evaluation_version": (
            "competition_bm25_query_fusion_dev_v1"
        ),
        "source_baseline_version": (
            EXPECTED_BASELINE_VERSION
        ),
        "source_baseline_sha256": (
            result.baseline_sha256
        ),
        "qa_sha256": result.baseline_payload.get(
            "qa_sha256"
        ),
        "split_sha256": (
            result.baseline_payload.get(
                "split_sha256"
            )
        ),
        "gold_sha256": result.gold_sha256,
        "corpus_id": (
            result.corpus.manifest.corpus_id
        ),
        "index_id": result.baseline_payload.get(
            "index_id"
        ),
        "fact_count": len(
            result.gold_dataset.records
        ),
        "text_dev_case_count": len(
            result.case_results
        ),
        "evidence_mode_counts": dict(
            sorted(
                Counter(
                    record.evidence_mode
                    for record
                    in result.gold_dataset.records
                ).items()
            )
        ),
        "cutoffs": list(result.cutoffs),
        "top_k": result.top_k,
        "fusion_config": (
            result.config.model_dump(
                mode="json"
            )
        ),
        "query_uses_gold_fact_text": False,
        "baseline_summaries": {
            mode: mode_payload.get("summary")
            for mode, mode_payload
            in baseline_modes.items()
            if isinstance(mode_payload, dict)
        },
        "fusion_summary": (
            _summary_payload(result.summary)
        ),
        "fusion_breakdowns": {
            "evidence_mode": (
                _build_breakdowns(
                    results=result.fact_results,
                    cutoffs=result.cutoffs,
                )
            )
        },
        "cases": [
            {
                "case_id": case.case_id,
                "expected_source_id": (
                    case.expected_source_id
                ),
                "question_only_query": (
                    case.question_only_query
                ),
                "question_with_options_query": (
                    case.question_with_options_query
                ),
                "question_only_hits": [
                    _hit_payload(hit)
                    for hit
                    in case.question_only_hits
                ],
                "question_with_options_hits": [
                    _hit_payload(hit)
                    for hit
                    in case.question_with_options_hits
                ],
                "fusion_hits": [
                    _fusion_hit_payload(hit)
                    for hit in case.fusion_hits
                ],
                "facts": [
                    asdict(fact_result)
                    for fact_result
                    in case.fact_results
                ],
            }
            for case in result.case_results
        ],
    }


def run_competition_bm25_query_fusion_dev(
    *,
    baseline_path: Path,
    gold_path: Path,
    corpus_directory: Path,
    output_path: Path,
    config: (
        CompetitionBM25QueryFusionConfig
        | None
    ) = None,
    top_k: int = DEFAULT_TOP_K,
    cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
    expected_text_case_count: int | None = (
        DEFAULT_EXPECTED_TEXT_DEV_CASES
    ),
    expected_fact_count: int | None = (
        DEFAULT_EXPECTED_FACTS
    ),
) -> CompetitionBM25QueryFusionDevResult:
    normalized_cutoffs = tuple(
        sorted(set(cutoffs))
    )

    if not normalized_cutoffs:
        raise ValueError(
            "cutoffs 不能为空"
        )

    if top_k < max(normalized_cutoffs):
        raise ValueError(
            "top_k 不能小于最大 cutoff"
        )

    baseline = _load_json_object(
        baseline_path,
        label="BM25 Gold Baseline",
    )

    gold_dataset = load_competition_gold_dataset(
        gold_path,
        expected_fact_count=(
            expected_fact_count
        ),
    )

    corpus = load_competition_chunk_corpus(
        corpus_directory
    )

    _validate_input_binding(
        baseline=baseline,
        baseline_path=baseline_path,
        gold_path=gold_path,
        gold_dataset=gold_dataset,
        corpus=corpus,
        expected_text_case_count=(
            expected_text_case_count
        ),
        expected_fact_count=expected_fact_count,
        top_k=top_k,
    )

    question_cases = _load_mode_cases(
        baseline,
        mode="question_only",
    )

    option_cases = _load_mode_cases(
        baseline,
        mode="question_with_options",
    )

    records_by_case = _group_gold_records(
        gold_dataset.records
    )

    if (
        set(question_cases)
        != set(option_cases)
        or set(question_cases)
        != set(records_by_case)
    ):
        raise RuntimeError(
            "两路 BM25 Case 与 Gold Case 不一致"
        )

    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunk in corpus.chunks
    }

    active_config = (
        config
        if config is not None
        else CompetitionBM25QueryFusionConfig()
    )

    case_results = []
    all_fact_results = []

    for case_id in sorted(records_by_case):
        question_case = question_cases[case_id]
        option_case = option_cases[case_id]

        question_source = _require_string(
            question_case,
            "expected_source_id",
            context=f"question_only {case_id}",
        )

        option_source = _require_string(
            option_case,
            "expected_source_id",
            context=(
                f"question_with_options {case_id}"
            ),
        )

        if question_source != option_source:
            raise RuntimeError(
                "两路 BM25 Case 的来源不一致："
                f"{case_id}"
            )

        if any(
            record.expected_source_id
            != question_source
            for record in records_by_case[case_id]
        ):
            raise RuntimeError(
                "BM25 Case 来源与 Gold 不一致："
                f"{case_id}"
            )

        question_hits = _rebuild_hits(
            question_case,
            chunks_by_id=chunks_by_id,
            mode="question_only",
        )

        option_hits = _rebuild_hits(
            option_case,
            chunks_by_id=chunks_by_id,
            mode="question_with_options",
        )

        fusion_hits = (
            fuse_competition_bm25_queries(
                question_only_hits=question_hits,
                question_with_options_hits=(
                    option_hits
                ),
                config=active_config,
                top_k=top_k,
            )
        )

        fact_results = tuple(
            evaluate_competition_gold_fact(
                record=record,
                hits=fusion_hits,
            )
            for record in records_by_case[case_id]
        )

        case_results.append(
            CompetitionBM25QueryFusionCaseResult(
                case_id=case_id,
                expected_source_id=question_source,
                question_only_query=_require_string(
                    question_case,
                    "query",
                    context=f"question_only {case_id}",
                ),
                question_with_options_query=(
                    _require_string(
                        option_case,
                        "query",
                        context=(
                            "question_with_options "
                            f"{case_id}"
                        ),
                    )
                ),
                question_only_hits=question_hits,
                question_with_options_hits=(
                    option_hits
                ),
                fusion_hits=fusion_hits,
                fact_results=fact_results,
            )
        )

        all_fact_results.extend(
            fact_results
        )

    fact_result_tuple = tuple(
        all_fact_results
    )

    result = CompetitionBM25QueryFusionDevResult(
        baseline_path=baseline_path,
        baseline_sha256=_sha256_file(
            baseline_path
        ),
        baseline_payload=baseline,
        gold_path=gold_path,
        gold_sha256=_sha256_file(gold_path),
        gold_dataset=gold_dataset,
        corpus=corpus,
        config=active_config,
        case_results=tuple(case_results),
        fact_results=fact_result_tuple,
        summary=(
            summarize_competition_gold_retrieval(
                fact_result_tuple,
                cutoffs=normalized_cutoffs,
            )
        ),
        cutoffs=normalized_cutoffs,
        top_k=top_k,
        output_path=output_path,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            _build_output_payload(result),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return result


def _format_rate(value: float) -> str:
    return f"{value:.4f}"


def _baseline_metric(
    baseline: dict[str, object],
    *,
    mode: str,
    cutoff: int,
) -> dict[str, object]:
    modes = baseline["modes"]
    assert isinstance(modes, dict)

    mode_payload = modes[mode]
    assert isinstance(mode_payload, dict)

    summary = mode_payload["summary"]
    assert isinstance(summary, dict)

    metrics = summary["metrics"]
    assert isinstance(metrics, list)

    for metric in metrics:
        if (
            isinstance(metric, dict)
            and metric.get("cutoff") == cutoff
        ):
            return metric

    raise RuntimeError(
        f"Baseline 缺少 cutoff={cutoff}"
    )


def print_fusion_result(
    result: CompetitionBM25QueryFusionDevResult,
) -> None:
    print(
        "=== Competition Frozen Dev "
        "BM25 Dual Query RRF ==="
    )
    print(
        "Corpus ID:",
        result.corpus.manifest.corpus_id,
    )
    print(
        "Cases:",
        len(result.case_results),
    )
    print(
        "Gold facts:",
        len(result.fact_results),
    )
    print(
        "RRF rank constant:",
        result.config.rank_constant,
    )

    for metric in result.summary.metrics:
        question = _baseline_metric(
            result.baseline_payload,
            mode="question_only",
            cutoff=metric.cutoff,
        )
        options = _baseline_metric(
            result.baseline_payload,
            mode="question_with_options",
            cutoff=metric.cutoff,
        )

        print()
        print(f"@{metric.cutoff}")
        print(
            "  DirectHit:",
            "question_only=",
            _format_rate(
                float(
                    question[
                        "fact_direct_hit_rate"
                    ]
                )
            ),
            "question_with_options=",
            _format_rate(
                float(
                    options[
                        "fact_direct_hit_rate"
                    ]
                )
            ),
            "dual_query_rrf=",
            _format_rate(
                metric.fact_direct_hit_rate
            ),
        )
        print(
            "  CompleteGold:",
            "question_only=",
            _format_rate(
                float(
                    question[
                        "fact_complete_gold_hit_rate"
                    ]
                )
            ),
            "question_with_options=",
            _format_rate(
                float(
                    options[
                        "fact_complete_gold_hit_rate"
                    ]
                )
            ),
            "dual_query_rrf=",
            _format_rate(
                metric.fact_complete_gold_hit_rate
            ),
        )
        print(
            "  RRF GoldRecall=",
            _format_rate(
                metric.mean_gold_chunk_recall
            ),
            "DirectMRR=",
            _format_rate(
                metric.mean_direct_reciprocal_rank
            ),
        )

    print()
    print(f"Breakdown @{result.top_k}:")

    for evidence_mode in sorted(
        {
            fact_result.evidence_mode
            for fact_result
            in result.fact_results
        }
    ):
        group = tuple(
            fact_result
            for fact_result
            in result.fact_results
            if (
                fact_result.evidence_mode
                == evidence_mode
            )
        )

        metric = (
            summarize_competition_gold_retrieval(
                group,
                cutoffs=(result.top_k,),
            )
        ).at(result.top_k)

        print(
            f"  evidence_mode={evidence_mode}",
            f"facts={len(group)}",
            "DirectHit=",
            _format_rate(
                metric.fact_direct_hit_rate
            ),
            "CompleteGold=",
            _format_rate(
                metric.fact_complete_gold_hit_rate
            ),
            "GoldRecall=",
            _format_rate(
                metric.mean_gold_chunk_recall
            ),
        )

    print()
    print("Saved:", result.output_path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "评测 Competition BM25 "
            "双查询 RRF 融合"
        )
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_FILE,
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=DEFAULT_GOLD_FILE,
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )
    parser.add_argument(
        "--rank-constant",
        type=int,
        default=60,
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    arguments = (
        build_argument_parser()
        .parse_args(argv)
    )

    config = CompetitionBM25QueryFusionConfig(
        rank_constant=arguments.rank_constant,
    )

    result = run_competition_bm25_query_fusion_dev(
        baseline_path=arguments.baseline,
        gold_path=arguments.gold,
        corpus_directory=arguments.corpus,
        output_path=arguments.output,
        config=config,
        top_k=arguments.top_k,
    )

    print_fusion_result(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())