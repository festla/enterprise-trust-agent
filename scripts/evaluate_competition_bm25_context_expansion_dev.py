from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from app.schemas.competition_context_expansion import (
    CompetitionContextExpandedHit,
    CompetitionContextExpansionConfig,
)
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
from app.services.competition_context_expansion import (
    evaluate_competition_gold_fact_context_expanded,
    expand_competition_retrieval_context,
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
    "competition_bm25_context_expansion_dev_v1.json"
)


@dataclass(frozen=True, slots=True)
class CompetitionBM25ContextExpansionCaseResult:
    case_id: str
    expected_source_id: str

    seed_hits: tuple[
        CompetitionBM25OptionAnchoredMergeHit,
        ...,
    ]

    expanded_hits: tuple[
        CompetitionContextExpandedHit,
        ...,
    ]

    seed_fact_results: tuple[
        CompetitionGoldFactRetrievalResult,
        ...,
    ]

    expanded_fact_results: tuple[
        CompetitionGoldFactRetrievalResult,
        ...,
    ]


@dataclass(frozen=True, slots=True)
class CompetitionBM25ContextExpansionDevResult:
    baseline_path: Path
    baseline_sha256: str
    baseline_payload: dict[str, object]

    gold_path: Path
    gold_sha256: str
    gold_dataset: CompetitionGoldChunkDataset

    corpus: CompetitionCorpusBuildResult

    merge_config: (
        CompetitionBM25OptionAnchoredMergeConfig
    )

    expansion_config: (
        CompetitionContextExpansionConfig
    )

    case_results: tuple[
        CompetitionBM25ContextExpansionCaseResult,
        ...,
    ]

    seed_fact_results: tuple[
        CompetitionGoldFactRetrievalResult,
        ...,
    ]

    expanded_fact_results: tuple[
        CompetitionGoldFactRetrievalResult,
        ...,
    ]

    seed_summary: (
        CompetitionGoldRetrievalEvalSummary
    )

    expanded_summary: (
        CompetitionGoldRetrievalEvalSummary
    )

    cutoffs: tuple[int, ...]
    top_k: int

    output_path: Path


def _seed_to_eval_hits(
    seed_hits: tuple[
        CompetitionBM25OptionAnchoredMergeHit,
        ...,
    ],
) -> tuple[
    CompetitionBM25Hit,
    ...,
]:
    return tuple(
        CompetitionBM25Hit(
            rank=hit.rank,
            score=hit.score,
            chunk=hit.chunk,
        )
        for hit in seed_hits
    )


def _summary_payload(
    summary: CompetitionGoldRetrievalEvalSummary,
) -> dict[str, object]:
    return asdict(summary)


def _group_summary(
    *,
    results: tuple[
        CompetitionGoldFactRetrievalResult,
        ...,
    ],
    evidence_mode: str,
    cutoffs: tuple[int, ...],
) -> CompetitionGoldRetrievalEvalSummary:
    group = tuple(
        result
        for result in results
        if (
            result.evidence_mode
            == evidence_mode
        )
    )

    if not group:
        raise RuntimeError(
            "没有 Evidence Mode："
            f"{evidence_mode}"
        )

    return (
        summarize_competition_gold_retrieval(
            group,
            cutoffs=cutoffs,
        )
    )


def _build_breakdowns(
    *,
    results: tuple[
        CompetitionGoldFactRetrievalResult,
        ...,
    ],
    cutoffs: tuple[int, ...],
) -> dict[str, object]:
    evidence_modes = sorted(
        {
            result.evidence_mode
            for result in results
        }
    )

    return {
        evidence_mode: (
            _summary_payload(
                _group_summary(
                    results=results,
                    evidence_mode=(
                        evidence_mode
                    ),
                    cutoffs=cutoffs,
                )
            )
        )
        for evidence_mode in evidence_modes
    }


def _seed_hit_payload(
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


def _expanded_hit_payload(
    hit: CompetitionContextExpandedHit,
) -> dict[str, object]:
    return {
        "rank": hit.rank,
        "effective_seed_rank": (
            hit.effective_seed_rank
        ),
        "chunk_id": hit.chunk_id,
        "source_id": hit.source_id,
        "doc_id": hit.doc_id,
        "chunk_type": hit.chunk_type,
        "is_seed": hit.is_seed,
        "origins": [
            origin.model_dump(
                mode="json"
            )
            for origin in hit.origins
        ],
    }


def _validate_monotonicity(
    *,
    seed_summary: CompetitionGoldRetrievalEvalSummary,
    expanded_summary: CompetitionGoldRetrievalEvalSummary,
) -> None:
    """
    Context Expansion 是 Retrieval Seeds 的超集，
    因而任何 Gold Coverage 指标都不允许下降。
    """

    epsilon = 1e-12

    for seed_metric in seed_summary.metrics:
        expanded_metric = (
            expanded_summary.at(
                seed_metric.cutoff
            )
        )

        checks = (
            (
                "DirectHit",
                seed_metric.fact_direct_hit_rate,
                expanded_metric.fact_direct_hit_rate,
            ),
            (
                "AnyGold",
                seed_metric.fact_any_gold_hit_rate,
                expanded_metric.fact_any_gold_hit_rate,
            ),
            (
                "CompleteGold",
                (
                    seed_metric
                    .fact_complete_gold_hit_rate
                ),
                (
                    expanded_metric
                    .fact_complete_gold_hit_rate
                ),
            ),
            (
                "GoldRecall",
                (
                    seed_metric
                    .mean_gold_chunk_recall
                ),
                (
                    expanded_metric
                    .mean_gold_chunk_recall
                ),
            ),
        )

        for (
            metric_name,
            before,
            after,
        ) in checks:
            if after + epsilon < before:
                raise RuntimeError(
                    "Context Expansion 单调性门禁失败："
                    f"metric={metric_name}; "
                    f"cutoff={seed_metric.cutoff}; "
                    f"before={before}; "
                    f"after={after}"
                )


def _build_output_payload(
    result: CompetitionBM25ContextExpansionDevResult,
) -> dict[str, object]:
    expanded_chunk_count = sum(
        len(case.expanded_hits)
        for case in result.case_results
    )

    seed_chunk_count = sum(
        len(case.seed_hits)
        for case in result.case_results
    )

    relation_counts = Counter(
        origin.relation
        for case in result.case_results
        for hit in case.expanded_hits
        for origin in hit.origins
    )

    return {
        "schema_version": 1,
        "evaluation_version": (
            "competition_bm25_"
            "context_expansion_dev_v1"
        ),
        "source_baseline_sha256": (
            result.baseline_sha256
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
        "cutoffs": list(
            result.cutoffs
        ),
        "top_k": result.top_k,
        "merge_config": (
            result.merge_config.model_dump(
                mode="json"
            )
        ),
        "expansion_config": (
            result.expansion_config.model_dump(
                mode="json"
            )
        ),
        "query_uses_gold_fact_text": False,
        "seed_chunk_count": (
            seed_chunk_count
        ),
        "expanded_chunk_count": (
            expanded_chunk_count
        ),
        "mean_seed_chunks_per_case": (
            seed_chunk_count
            / len(result.case_results)
        ),
        "mean_expanded_chunks_per_case": (
            expanded_chunk_count
            / len(result.case_results)
        ),
        "expansion_ratio": (
            expanded_chunk_count
            / seed_chunk_count
        ),
        "origin_relation_counts": dict(
            sorted(
                relation_counts.items()
            )
        ),
        "seed_summary": (
            _summary_payload(
                result.seed_summary
            )
        ),
        "expanded_summary": (
            _summary_payload(
                result.expanded_summary
            )
        ),
        "seed_breakdowns": {
            "evidence_mode": (
                _build_breakdowns(
                    results=(
                        result.seed_fact_results
                    ),
                    cutoffs=result.cutoffs,
                )
            )
        },
        "expanded_breakdowns": {
            "evidence_mode": (
                _build_breakdowns(
                    results=(
                        result.expanded_fact_results
                    ),
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
                "seed_hits": [
                    _seed_hit_payload(hit)
                    for hit in case.seed_hits
                ],
                "expanded_hits": [
                    _expanded_hit_payload(hit)
                    for hit in case.expanded_hits
                ],
                "seed_facts": [
                    asdict(result)
                    for result
                    in case.seed_fact_results
                ],
                "expanded_facts": [
                    asdict(result)
                    for result
                    in case.expanded_fact_results
                ],
            }
            for case in result.case_results
        ],
    }


def run_competition_bm25_context_expansion_dev(
    *,
    baseline_path: Path,
    gold_path: Path,
    corpus_directory: Path,
    output_path: Path,
    merge_config: (
        CompetitionBM25OptionAnchoredMergeConfig
        | None
    ) = None,
    expansion_config: (
        CompetitionContextExpansionConfig
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
) -> CompetitionBM25ContextExpansionDevResult:
    normalized_cutoffs = tuple(
        sorted(
            set(cutoffs)
        )
    )

    if not normalized_cutoffs:
        raise ValueError(
            "cutoffs 不能为空"
        )

    if top_k < max(
        normalized_cutoffs
    ):
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

    active_merge_config = (
        merge_config
        if merge_config is not None
        else (
            CompetitionBM25OptionAnchoredMergeConfig()
        )
    )

    active_expansion_config = (
        expansion_config
        if expansion_config is not None
        else CompetitionContextExpansionConfig(
            seed_candidate_count=top_k,
            previous_window=1,
            next_window=1,
        )
    )

    if (
        active_expansion_config
        .seed_candidate_count
        < max(normalized_cutoffs)
    ):
        raise ValueError(
            "seed_candidate_count 不能小于"
            "最大评测 cutoff"
        )

    case_results: list[
        CompetitionBM25ContextExpansionCaseResult
    ] = []

    all_seed_results: list[
        CompetitionGoldFactRetrievalResult
    ] = []

    all_expanded_results: list[
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

        if (
            question_source
            != option_source
        ):
            raise RuntimeError(
                "两路 BM25 Case 来源不一致："
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

        seed_hits = (
            merge_competition_bm25_option_anchored(
                question_only_hits=(
                    question_hits
                ),
                question_with_options_hits=(
                    option_hits
                ),
                config=active_merge_config,
                top_k=top_k,
            )
        )

        expanded_hits = (
            expand_competition_retrieval_context(
                seed_hits=seed_hits,
                corpus_chunks=(
                    corpus.chunks
                ),
                config=(
                    active_expansion_config
                ),
            )
        )

        seed_eval_hits = (
            _seed_to_eval_hits(
                seed_hits
            )
        )

        seed_fact_results = tuple(
            evaluate_competition_gold_fact(
                record=record,
                hits=seed_eval_hits,
            )
            for record
            in records_by_case[case_id]
        )

        expanded_fact_results = tuple(
            evaluate_competition_gold_fact_context_expanded(
                record=record,
                hits=expanded_hits,
            )
            for record
            in records_by_case[case_id]
        )

        case_results.append(
            CompetitionBM25ContextExpansionCaseResult(
                case_id=case_id,
                expected_source_id=(
                    question_source
                ),
                seed_hits=seed_hits,
                expanded_hits=(
                    expanded_hits
                ),
                seed_fact_results=(
                    seed_fact_results
                ),
                expanded_fact_results=(
                    expanded_fact_results
                ),
            )
        )

        all_seed_results.extend(
            seed_fact_results
        )

        all_expanded_results.extend(
            expanded_fact_results
        )

    seed_result_tuple = tuple(
        all_seed_results
    )

    expanded_result_tuple = tuple(
        all_expanded_results
    )

    seed_summary = (
        summarize_competition_gold_retrieval(
            seed_result_tuple,
            cutoffs=normalized_cutoffs,
        )
    )

    expanded_summary = (
        summarize_competition_gold_retrieval(
            expanded_result_tuple,
            cutoffs=normalized_cutoffs,
        )
    )

    _validate_monotonicity(
        seed_summary=seed_summary,
        expanded_summary=expanded_summary,
    )

    result = (
        CompetitionBM25ContextExpansionDevResult(
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
            merge_config=(
                active_merge_config
            ),
            expansion_config=(
                active_expansion_config
            ),
            case_results=tuple(
                case_results
            ),
            seed_fact_results=(
                seed_result_tuple
            ),
            expanded_fact_results=(
                expanded_result_tuple
            ),
            seed_summary=seed_summary,
            expanded_summary=(
                expanded_summary
            ),
            cutoffs=normalized_cutoffs,
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


def print_context_expansion_result(
    result: CompetitionBM25ContextExpansionDevResult,
) -> None:
    print(
        "=== Competition Frozen Dev "
        "BM25 Context Expansion ==="
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
        len(result.expanded_fact_results),
    )

    print(
        "Seed Top-K:",
        result.expansion_config.seed_candidate_count,
    )

    print(
        "Previous window:",
        result.expansion_config.previous_window,
    )

    print(
        "Next window:",
        result.expansion_config.next_window,
    )

    print()

    for seed_metric in (
        result.seed_summary.metrics
    ):
        expanded_metric = (
            result.expanded_summary.at(
                seed_metric.cutoff
            )
        )

        print(
            f"@{seed_metric.cutoff}"
        )

        print(
            "  DirectHit:",
            "seed=",
            _format_rate(
                seed_metric.fact_direct_hit_rate
            ),
            "expanded=",
            _format_rate(
                expanded_metric.fact_direct_hit_rate
            ),
        )

        print(
            "  CompleteGold:",
            "seed=",
            _format_rate(
                seed_metric
                .fact_complete_gold_hit_rate
            ),
            "expanded=",
            _format_rate(
                expanded_metric
                .fact_complete_gold_hit_rate
            ),
        )

        print(
            "  GoldRecall:",
            "seed=",
            _format_rate(
                seed_metric.mean_gold_chunk_recall
            ),
            "expanded=",
            _format_rate(
                expanded_metric
                .mean_gold_chunk_recall
            ),
        )

        print(
            "  DirectMRR:",
            "seed=",
            _format_rate(
                seed_metric
                .mean_direct_reciprocal_rank
            ),
            "expanded=",
            _format_rate(
                expanded_metric
                .mean_direct_reciprocal_rank
            ),
        )

        print()

    print(
        f"Breakdown @{result.top_k}:"
    )

    evidence_modes = (
        "single_chunk",
        "parent_child",
        "multi_chunk",
    )

    for evidence_mode in evidence_modes:
        seed_summary = _group_summary(
            results=(
                result.seed_fact_results
            ),
            evidence_mode=evidence_mode,
            cutoffs=(result.top_k,),
        )

        expanded_summary = _group_summary(
            results=(
                result.expanded_fact_results
            ),
            evidence_mode=evidence_mode,
            cutoffs=(result.top_k,),
        )

        seed_metric = (
            seed_summary.at(
                result.top_k
            )
        )

        expanded_metric = (
            expanded_summary.at(
                result.top_k
            )
        )

        fact_count = sum(
            result_item.evidence_mode
            == evidence_mode
            for result_item
            in result.expanded_fact_results
        )

        print(
            f"  evidence_mode={evidence_mode}",
            f"facts={fact_count}",
        )

        print(
            "    DirectHit:",
            _format_rate(
                seed_metric.fact_direct_hit_rate
            ),
            "->",
            _format_rate(
                expanded_metric.fact_direct_hit_rate
            ),
        )

        print(
            "    CompleteGold:",
            _format_rate(
                seed_metric
                .fact_complete_gold_hit_rate
            ),
            "->",
            _format_rate(
                expanded_metric
                .fact_complete_gold_hit_rate
            ),
        )

        print(
            "    GoldRecall:",
            _format_rate(
                seed_metric.mean_gold_chunk_recall
            ),
            "->",
            _format_rate(
                expanded_metric.mean_gold_chunk_recall
            ),
        )

    total_seed_chunks = sum(
        len(case.seed_hits)
        for case in result.case_results
    )

    total_expanded_chunks = sum(
        len(case.expanded_hits)
        for case in result.case_results
    )

    print()

    print(
        "Context size:",
        f"seed={total_seed_chunks}",
        f"expanded={total_expanded_chunks}",
        "ratio=",
        _format_rate(
            total_expanded_chunks
            / total_seed_chunks
        ),
    )

    print()

    print(
        "Monotonicity gate: PASS"
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
            "Option-Anchored + Context Expansion"
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
        "--previous-window",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--next-window",
        type=int,
        default=1,
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    arguments = (
        build_argument_parser()
        .parse_args(argv)
    )

    expansion_config = (
        CompetitionContextExpansionConfig(
            seed_candidate_count=(
                arguments.top_k
            ),
            previous_window=(
                arguments.previous_window
            ),
            next_window=(
                arguments.next_window
            ),
        )
    )

    result = (
        run_competition_bm25_context_expansion_dev(
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
            expansion_config=(
                expansion_config
            ),
            top_k=arguments.top_k,
        )
    )

    print_context_expansion_result(
        result
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )