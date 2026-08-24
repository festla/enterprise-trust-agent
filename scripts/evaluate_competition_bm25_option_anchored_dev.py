from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from app.schemas.competition_gold import (
    CompetitionGoldChunkDataset,
)
from app.schemas.competition_option_anchored_merge import (
    CompetitionBM25OptionAnchoredMergeConfig,
    CompetitionBM25OptionAnchoredMergeHit,
)
from app.schemas.competition_retrieval import (
    CompetitionBM25Hit,
)
from app.services.competition_bm25_option_anchored_merge import (
    merge_competition_bm25_option_anchored,
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
from scripts.evaluate_competition_bm25_query_fusion_dev import (
    DEFAULT_BASELINE_FILE,
    DEFAULT_CUTOFFS,
    DEFAULT_EXPECTED_FACTS,
    DEFAULT_EXPECTED_TEXT_DEV_CASES,
    DEFAULT_GOLD_FILE,
    DEFAULT_TOP_K,
    _baseline_metric,
    _format_rate,
    _group_gold_records,
    _load_json_object,
    _load_mode_cases,
    _rebuild_hits,
    _require_string,
    _sha256_file,
    _validate_input_binding,
)


DEFAULT_OUTPUT_FILE = Path(
    "data/competition/processed/eval/"
    "competition_bm25_option_anchored_dev_v1.json"
)


@dataclass(frozen=True, slots=True)
class CompetitionBM25OptionAnchoredCaseResult:
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

    merged_hits: tuple[
        CompetitionBM25OptionAnchoredMergeHit,
        ...,
    ]

    fact_results: tuple[
        CompetitionGoldFactRetrievalResult,
        ...,
    ]


@dataclass(frozen=True, slots=True)
class CompetitionBM25OptionAnchoredDevResult:
    baseline_path: Path
    baseline_sha256: str
    baseline_payload: dict[str, object]

    gold_path: Path
    gold_sha256: str
    gold_dataset: CompetitionGoldChunkDataset

    corpus: CompetitionCorpusBuildResult

    config: CompetitionBM25OptionAnchoredMergeConfig

    case_results: tuple[
        CompetitionBM25OptionAnchoredCaseResult,
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


def _to_eval_hits(
    hits: tuple[
        CompetitionBM25OptionAnchoredMergeHit,
        ...,
    ],
) -> tuple[
    CompetitionBM25Hit,
    ...,
]:
    """
    Gold evaluator 当前类型声明接受
    CompetitionBM25Hit。

    它实际只使用：
        rank
        chunk_id
        chunk

    因此这里显式转换，
    避免让 evaluator 与新的融合 Schema 耦合。
    """

    return tuple(
        CompetitionBM25Hit(
            rank=hit.rank,
            score=hit.score,
            chunk=hit.chunk,
        )
        for hit in hits
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


def _merged_hit_payload(
    hit: CompetitionBM25OptionAnchoredMergeHit,
) -> dict[str, object]:
    return {
        "rank": hit.rank,
        "score": hit.score,
        "chunk_id": hit.chunk_id,
        "source_id": hit.source_id,
        "doc_id": hit.doc_id,
        "chunk_type": hit.chunk_type,
        "layer": hit.layer,
        "question_only_rank": (
            hit.question_only_rank
        ),
        "question_with_options_rank": (
            hit.question_with_options_rank
        ),
    }


def _verify_option_anchor_invariant(
    *,
    option_hits: tuple[
        CompetitionBM25Hit,
        ...,
    ],
    merged_hits: tuple[
        CompetitionBM25OptionAnchoredMergeHit,
        ...,
    ],
    top_k: int,
    case_id: str,
) -> None:
    """
    如果 option 原始结果足够 top_k，
    融合后 Top-K 的 Chunk 集合必须完全相同。

    这是本实验最重要的门禁。
    """

    if len(option_hits) < top_k:
        return

    expected_ids = {
        hit.chunk_id
        for hit in option_hits[:top_k]
    }

    actual_ids = {
        hit.chunk_id
        for hit in merged_hits[:top_k]
    }

    if actual_ids != expected_ids:
        removed = sorted(
            expected_ids - actual_ids
        )

        added = sorted(
            actual_ids - expected_ids
        )

        raise RuntimeError(
            "Option Anchor 不变量失败："
            f"case={case_id}; "
            f"removed={removed}; "
            f"added={added}"
        )


def _build_output_payload(
    result: CompetitionBM25OptionAnchoredDevResult,
) -> dict[str, object]:
    baseline_modes = result.baseline_payload[
        "modes"
    ]

    assert isinstance(
        baseline_modes,
        dict,
    )

    layer_counter = Counter(
        hit.layer
        for case in result.case_results
        for hit in case.merged_hits
    )

    return {
        "schema_version": 1,
        "evaluation_version": (
            "competition_bm25_"
            "option_anchored_dev_v1"
        ),
        "source_baseline_version": (
            result.baseline_payload.get(
                "evaluation_version"
            )
        ),
        "source_baseline_sha256": (
            result.baseline_sha256
        ),
        "qa_sha256": (
            result.baseline_payload.get(
                "qa_sha256"
            )
        ),
        "split_sha256": (
            result.baseline_payload.get(
                "split_sha256"
            )
        ),
        "gold_sha256": (
            result.gold_sha256
        ),
        "corpus_id": (
            result.corpus.manifest.corpus_id
        ),
        "index_id": (
            result.baseline_payload.get(
                "index_id"
            )
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
        "merge_layer_counts": dict(
            sorted(
                layer_counter.items()
            )
        ),
        "cutoffs": list(
            result.cutoffs
        ),
        "top_k": result.top_k,
        "merge_config": (
            result.config.model_dump(
                mode="json"
            )
        ),
        "query_uses_gold_fact_text": False,
        "option_anchor_invariant": True,
        "baseline_summaries": {
            mode: mode_payload.get(
                "summary"
            )
            for mode, mode_payload
            in baseline_modes.items()
            if isinstance(
                mode_payload,
                dict,
            )
        },
        "merge_summary": (
            _summary_payload(
                result.summary
            )
        ),
        "merge_breakdowns": {
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
                "merged_hits": [
                    _merged_hit_payload(hit)
                    for hit
                    in case.merged_hits
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


def run_competition_bm25_option_anchored_dev(
    *,
    baseline_path: Path,
    gold_path: Path,
    corpus_directory: Path,
    output_path: Path,
    config: (
        CompetitionBM25OptionAnchoredMergeConfig
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
) -> CompetitionBM25OptionAnchoredDevResult:
    normalized_cutoffs = tuple(
        sorted(
            set(cutoffs)
        )
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

    gold_dataset = (
        load_competition_gold_dataset(
            gold_path,
            expected_fact_count=(
                expected_fact_count
            ),
        )
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
        expected_fact_count=(
            expected_fact_count
        ),
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
        else CompetitionBM25OptionAnchoredMergeConfig()
    )

    case_results: list[
        CompetitionBM25OptionAnchoredCaseResult
    ] = []

    all_fact_results: list[
        CompetitionGoldFactRetrievalResult
    ] = []

    for case_id in sorted(
        records_by_case
    ):
        question_case = (
            question_cases[case_id]
        )

        option_case = (
            option_cases[case_id]
        )

        question_source = _require_string(
            question_case,
            "expected_source_id",
            context=(
                f"question_only {case_id}"
            ),
        )

        option_source = _require_string(
            option_case,
            "expected_source_id",
            context=(
                "question_with_options "
                f"{case_id}"
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
            for record
            in records_by_case[case_id]
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

        merged_hits = (
            merge_competition_bm25_option_anchored(
                question_only_hits=(
                    question_hits
                ),
                question_with_options_hits=(
                    option_hits
                ),
                config=active_config,
                top_k=top_k,
            )
        )

        _verify_option_anchor_invariant(
            option_hits=option_hits,
            merged_hits=merged_hits,
            top_k=top_k,
            case_id=case_id,
        )

        eval_hits = _to_eval_hits(
            merged_hits
        )

        fact_results = tuple(
            evaluate_competition_gold_fact(
                record=record,
                hits=eval_hits,
            )
            for record
            in records_by_case[case_id]
        )

        case_results.append(
            CompetitionBM25OptionAnchoredCaseResult(
                case_id=case_id,
                expected_source_id=(
                    question_source
                ),
                question_only_query=(
                    _require_string(
                        question_case,
                        "query",
                        context=(
                            "question_only "
                            f"{case_id}"
                        ),
                    )
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
                question_only_hits=(
                    question_hits
                ),
                question_with_options_hits=(
                    option_hits
                ),
                merged_hits=(
                    merged_hits
                ),
                fact_results=(
                    fact_results
                ),
            )
        )

        all_fact_results.extend(
            fact_results
        )

    fact_result_tuple = tuple(
        all_fact_results
    )

    summary = (
        summarize_competition_gold_retrieval(
            fact_result_tuple,
            cutoffs=normalized_cutoffs,
        )
    )

    result = (
        CompetitionBM25OptionAnchoredDevResult(
            baseline_path=baseline_path,
            baseline_sha256=(
                _sha256_file(
                    baseline_path
                )
            ),
            baseline_payload=baseline,
            gold_path=gold_path,
            gold_sha256=(
                _sha256_file(
                    gold_path
                )
            ),
            gold_dataset=gold_dataset,
            corpus=corpus,
            config=active_config,
            case_results=tuple(
                case_results
            ),
            fact_results=(
                fact_result_tuple
            ),
            summary=summary,
            cutoffs=(
                normalized_cutoffs
            ),
            top_k=top_k,
            output_path=output_path,
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            _build_output_payload(
                result
            ),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return result


def print_option_anchored_result(
    result: CompetitionBM25OptionAnchoredDevResult,
) -> None:
    print(
        "=== Competition Frozen Dev "
        "BM25 Option-Anchored Merge ==="
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
        "Promotion question rank limit:",
        (
            result.config
            .promotion_question_rank_limit
        ),
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
        print(
            f"@{metric.cutoff}"
        )

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
            "option_anchored=",
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
            "option_anchored=",
            _format_rate(
                metric.fact_complete_gold_hit_rate
            ),
        )

        print(
            "  GoldRecall=",
            _format_rate(
                metric.mean_gold_chunk_recall
            ),
            "DirectMRR=",
            _format_rate(
                metric.mean_direct_reciprocal_rank
            ),
        )

    print()
    print(
        f"Breakdown @{result.top_k}:"
    )

    evidence_modes = sorted(
        {
            fact_result.evidence_mode
            for fact_result
            in result.fact_results
        }
    )

    for evidence_mode in evidence_modes:
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
                cutoffs=(
                    result.top_k,
                ),
            )
        ).at(
            result.top_k
        )

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

    layer_counter = Counter(
        hit.layer
        for case in result.case_results
        for hit in case.merged_hits
    )

    print()
    print(
        "Merge layers:",
        dict(
            sorted(
                layer_counter.items()
            )
        ),
    )

    print()
    print(
        "Option Anchor invariant: PASS"
    )

    print(
        "Saved:",
        result.output_path,
    )


def build_argument_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "评测 Competition BM25 "
            "带选项保底分层融合"
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
        "--question-candidates",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--promotion-question-rank-limit",
        type=int,
        default=None,
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    arguments = (
        build_argument_parser()
        .parse_args(argv)
    )

    config = (
        CompetitionBM25OptionAnchoredMergeConfig(
            question_candidate_count=(
                arguments.question_candidates
            ),
            promotion_question_rank_limit=(
                arguments
                .promotion_question_rank_limit
            ),
        )
    )

    result = (
        run_competition_bm25_option_anchored_dev(
            baseline_path=(
                arguments.baseline
            ),
            gold_path=arguments.gold,
            corpus_directory=(
                arguments.corpus
            ),
            output_path=(
                arguments.output
            ),
            config=config,
            top_k=arguments.top_k,
        )
    )

    print_option_anchored_result(
        result
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )