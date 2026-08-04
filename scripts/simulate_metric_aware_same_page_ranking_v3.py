from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from app.services.registry_loader import (
    load_metrics,
)
from scripts.evaluate_complex_diagnostic_dev_v1 import (
    OUTPUT_ROOT,
    run_preflight,
)
from scripts.simulate_same_page_context_v2 import (
    load_jsonl,
    score_candidate,
    target_is_present,
)


REPLAY_PATH = (
    OUTPUT_ROOT
    / "diagnostic_independent_query_replay_v1.jsonl"
)

OUTPUT_PATH = (
    OUTPUT_ROOT
    / "metric_aware_same_page_ranking_v3.jsonl"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "metric_aware_same_page_ranking_summary_v3.json"
)

OUTPUT_TEMP_PATH = OUTPUT_PATH.with_suffix(
    ".tmp.jsonl"
)

SUMMARY_TEMP_PATH = SUMMARY_PATH.with_suffix(
    ".tmp.json"
)

METRICS_PATH = Path(
    "data/processed/registries/metrics.yaml"
)


def normalize_text(text: str) -> str:
    return "".join(
        character.lower()
        for character in text
        if character.isalnum()
    )


def extract_bigrams(text: str) -> set[str]:
    normalized = normalize_text(text)

    return {
        normalized[index : index + 2]
        for index in range(
            max(0, len(normalized) - 1)
        )
    }


def build_metric_hints(
    *,
    metric_id: str,
    metric_registry: Any,
    metric_aliases: list[Any],
) -> tuple[str, ...]:
    metric = metric_registry.require(
        metric_id
    )

    hints = [metric.display_name_cn]

    hints.extend(
        alias.alias
        for alias in metric_aliases
        if alias.metric_id == metric_id
    )

    return tuple(
        dict.fromkeys(
            hint.strip()
            for hint in hints
            if hint.strip()
        )
    )


def add_metric_scores(
    *,
    candidate: dict[str, Any],
    metric_hints: tuple[str, ...],
) -> dict[str, Any]:
    result = dict(candidate)

    normalized_candidate = normalize_text(
        candidate["text"]
    )

    candidate_bigrams = extract_bigrams(
        candidate["text"]
    )

    exact_matches = []
    overlap_counts = []
    recalls = []

    for hint in metric_hints:
        normalized_hint = normalize_text(
            hint
        )

        hint_bigrams = extract_bigrams(
            hint
        )

        exact_matches.append(
            bool(
                normalized_hint
                and normalized_hint
                in normalized_candidate
            )
        )

        overlap_count = len(
            hint_bigrams.intersection(
                candidate_bigrams
            )
        )

        overlap_counts.append(
            overlap_count
        )

        recalls.append(
            (
                overlap_count
                / len(hint_bigrams)
            )
            if hint_bigrams
            else 0.0
        )

    result["metric_exact_match"] = int(
        any(exact_matches)
    )

    result["metric_bigram_overlap_count"] = (
        max(overlap_counts, default=0)
    )

    result["metric_bigram_recall"] = max(
        recalls,
        default=0.0,
    )

    return result


def build_candidate_rows(
    *,
    baseline: dict[str, Any],
    context: Any,
) -> tuple[list[dict[str, Any]], int]:
    expansion = baseline["full_expansion"]

    if expansion is None:
        return [], 0

    base_items = [
        item
        for item in expansion["items"]
        if item["origin"] == "retrieved"
    ]

    adjacent_items = [
        item
        for item in expansion["items"]
        if item["origin"] == "adjacent_page"
    ]

    document_id = expansion["document_id"]

    base_chunk_ids = {
        item["chunk_id"]
        for item in base_items
    }

    existing_candidate_ids = {
        item["chunk_id"]
        for item in adjacent_items
    }

    anchors_by_page: dict[
        int,
        dict[str, Any],
    ] = {}

    for item in sorted(
        base_items,
        key=lambda value: (
            value["retrieval_rank"],
            value["chunk_id"],
        ),
    ):
        anchors_by_page.setdefault(
            item["pdf_page"],
            item,
        )

    candidate_rows = [
        {
            "origin": "adjacent_page",
            "chunk_id": item["chunk_id"],
            "pdf_page": item["pdf_page"],
            "anchor_chunk_id": (
                item["anchor_chunk_id"]
            ),
            "anchor_retrieval_rank": (
                item["anchor_retrieval_rank"]
            ),
            "text": item["text"],
            "text_char_count": (
                item["text_char_count"]
            ),
        }
        for item in adjacent_items
    ]

    source = (
        context.chunk_sources_by_report_id[
            baseline["report_id"]
        ]
    )

    added_same_page_count = 0

    for chunk in source.chunks:
        if chunk.document_id != document_id:
            continue

        anchor = anchors_by_page.get(
            chunk.pdf_page
        )

        if anchor is None:
            continue

        if (
            chunk.chunk_id
            in base_chunk_ids
        ):
            continue

        same_page_candidate = {
            "origin": "same_page_sibling",
            "chunk_id": chunk.chunk_id,
            "pdf_page": chunk.pdf_page,
            "anchor_chunk_id": (
                anchor["chunk_id"]
            ),
            "anchor_retrieval_rank": (
                anchor["retrieval_rank"]
            ),
            "text": chunk.text,
            "text_char_count": len(
                chunk.text
            ),
        }

        if (
            chunk.chunk_id
            in existing_candidate_ids
        ):
            existing_index = next(
                index
                for index, candidate
                in enumerate(candidate_rows)
                if (
                    candidate["chunk_id"]
                    == chunk.chunk_id
                )
            )

            # 同一 chunk 可由多条扩展路径得到时，
            # 保留页距离更近的同页来源。
            candidate_rows[
                existing_index
            ] = same_page_candidate

            continue

        candidate_rows.append(
            same_page_candidate
        )

        existing_candidate_ids.add(
            chunk.chunk_id
        )

        added_same_page_count += 1

    return (
        candidate_rows,
        added_same_page_count,
    )


def current_sort_key(
    candidate: dict[str, Any],
) -> tuple[Any, ...]:
    return (
        -candidate["token_hit_count"],
        -candidate["bigram_overlap_count"],
        -candidate["query_bigram_recall"],
        candidate["anchor_retrieval_rank"],
        candidate["text_char_count"],
        candidate["chunk_id"],
    )


def bigram_first_sort_key(
    candidate: dict[str, Any],
) -> tuple[Any, ...]:
    return (
        -candidate["bigram_overlap_count"],
        -candidate["query_bigram_recall"],
        -candidate["token_hit_count"],
        candidate["anchor_retrieval_rank"],
        candidate["text_char_count"],
        candidate["chunk_id"],
    )


def metric_aware_sort_key(
    candidate: dict[str, Any],
) -> tuple[Any, ...]:
    return (
        -candidate["metric_exact_match"],
        -candidate["metric_bigram_recall"],
        -candidate[
            "metric_bigram_overlap_count"
        ],
        -candidate["bigram_overlap_count"],
        -candidate["query_bigram_recall"],
        -candidate["token_hit_count"],
        candidate["anchor_retrieval_rank"],
        candidate["text_char_count"],
        candidate["chunk_id"],
    )


def select_candidates(
    *,
    ranked_candidates: list[
        dict[str, Any]
    ],
    max_items: int,
    max_chars: int,
) -> tuple[
    list[dict[str, Any]],
    int,
]:
    selected = []
    selected_char_count = 0

    for candidate in ranked_candidates:
        if len(selected) >= max_items:
            continue

        next_char_count = (
            selected_char_count
            + candidate["text_char_count"]
        )

        if next_char_count > max_chars:
            continue

        selected.append(candidate)

        selected_char_count = (
            next_char_count
        )

    return selected, selected_char_count


def candidate_is_target(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    target_chunk_id = baseline[
        "gold_evidence_chunk_id"
    ]

    if target_chunk_id is not None:
        return (
            candidate["chunk_id"]
            == target_chunk_id
        )

    return (
        candidate["pdf_page"]
        == baseline["gold_evidence_pdf_page"]
    )


def audit_candidate(
    *,
    candidate: dict[str, Any],
    selection_rank: int,
) -> dict[str, Any]:
    return {
        "selection_rank": selection_rank,
        "origin": candidate["origin"],
        "chunk_id": candidate["chunk_id"],
        "pdf_page": candidate["pdf_page"],
        "anchor_retrieval_rank": (
            candidate[
                "anchor_retrieval_rank"
            ]
        ),
        "text_char_count": (
            candidate["text_char_count"]
        ),
        "token_hit_count": (
            candidate["token_hit_count"]
        ),
        "bigram_overlap_count": (
            candidate[
                "bigram_overlap_count"
            ]
        ),
        "query_bigram_recall": (
            candidate["query_bigram_recall"]
        ),
        "metric_exact_match": (
            candidate["metric_exact_match"]
        ),
        "metric_bigram_overlap_count": (
            candidate[
                "metric_bigram_overlap_count"
            ]
        ),
        "metric_bigram_recall": (
            candidate["metric_bigram_recall"]
        ),
    }


def evaluate_strategy(
    *,
    baseline: dict[str, Any],
    candidates: list[dict[str, Any]],
    sort_key: Callable[
        [dict[str, Any]],
        tuple[Any, ...],
    ],
    max_items: int,
    max_chars: int,
) -> dict[str, Any]:
    ranked = sorted(
        candidates,
        key=sort_key,
    )

    target_rank = next(
        (
            index
            for index, candidate
            in enumerate(ranked, start=1)
            if candidate_is_target(
                baseline=baseline,
                candidate=candidate,
            )
        ),
        None,
    )

    if baseline["base_target_hit"]:
        selected = []
        selected_char_count = 0
        projected_hit = True
        outcome = "base_resolved"
    else:
        (
            selected,
            selected_char_count,
        ) = select_candidates(
            ranked_candidates=ranked,
            max_items=max_items,
            max_chars=max_chars,
        )

        projected_hit = target_is_present(
            target_chunk_id=baseline[
                "gold_evidence_chunk_id"
            ],
            target_pdf_page=baseline[
                "gold_evidence_pdf_page"
            ],
            chunk_ids={
                candidate["chunk_id"]
                for candidate in selected
            },
            pdf_pages={
                candidate["pdf_page"]
                for candidate in selected
            },
        )

        if projected_hit:
            target_candidate = next(
                candidate
                for candidate in selected
                if candidate_is_target(
                    baseline=baseline,
                    candidate=candidate,
                )
            )

            if (
                target_candidate["origin"]
                == "same_page_sibling"
            ):
                outcome = (
                    "same_page_sibling_recovered"
                )
            else:
                outcome = (
                    "adjacent_page_recovered"
                )
        elif target_rank is not None:
            outcome = "budget_selection_miss"
        else:
            outcome = (
                "candidate_generation_coverage_gap"
            )

    baseline_hit = bool(
        baseline["target_fact_hit"]
    )

    return {
        "projected_hit": projected_hit,
        "recovered": (
            projected_hit
            and not baseline_hit
        ),
        "regressed": (
            baseline_hit
            and not projected_hit
        ),
        "outcome": outcome,
        "target_candidate_rank": target_rank,
        "selected_candidate_count": len(
            selected
        ),
        "selected_char_count": (
            selected_char_count
        ),
        "selected_candidates": [
            audit_candidate(
                candidate=candidate,
                selection_rank=index,
            )
            for index, candidate
            in enumerate(selected, start=1)
        ],
    }


def summarize_strategy(
    *,
    records: list[dict[str, Any]],
    strategy_id: str,
) -> dict[str, Any]:
    results = [
        record["strategies"][strategy_id]
        for record in records
    ]

    diagnostic_pairs = [
        (
            record,
            record["strategies"][
                strategy_id
            ],
        )
        for record in records
        if record["is_diagnostic_target"]
    ]

    control_pairs = [
        (
            record,
            record["strategies"][
                strategy_id
            ],
        )
        for record in records
        if not record["is_diagnostic_target"]
    ]

    hit_count = sum(
        result["projected_hit"]
        for result in results
    )

    diagnostic_hit_count = sum(
        result["projected_hit"]
        for _, result in diagnostic_pairs
    )

    control_hit_count = sum(
        result["projected_hit"]
        for _, result in control_pairs
    )

    return {
        "query_count": len(records),
        "hit_count": hit_count,
        "hit_rate": (
            hit_count / len(records)
            if records
            else 0.0
        ),
        "recovered_query_count": sum(
            result["recovered"]
            for result in results
        ),
        "regressed_query_count": sum(
            result["regressed"]
            for result in results
        ),
        "outcome_counts": dict(
            sorted(
                Counter(
                    result["outcome"]
                    for result in results
                ).items()
            )
        ),
        "diagnostic_query_count": len(
            diagnostic_pairs
        ),
        "diagnostic_hit_count": (
            diagnostic_hit_count
        ),
        "diagnostic_hit_rate": (
            diagnostic_hit_count
            / len(diagnostic_pairs)
            if diagnostic_pairs
            else 0.0
        ),
        "control_query_count": len(
            control_pairs
        ),
        "control_hit_count": (
            control_hit_count
        ),
        "control_hit_rate": (
            control_hit_count
            / len(control_pairs)
            if control_pairs
            else 0.0
        ),
    }


def main() -> None:
    if not REPLAY_PATH.is_file():
        raise RuntimeError(
            f"missing_replay={REPLAY_PATH}"
        )

    for path in (
        OUTPUT_PATH,
        SUMMARY_PATH,
        OUTPUT_TEMP_PATH,
        SUMMARY_TEMP_PATH,
    ):
        if path.exists():
            raise RuntimeError(
                f"output_exists={path}"
            )

    context = run_preflight(
        execution_requested=False
    )

    baselines = load_jsonl(
        REPLAY_PATH
    )

    if len(baselines) != 24:
        raise RuntimeError(
            "baseline_record_count_not_24="
            f"{len(baselines)}"
        )

    metric_registry, metric_aliases = (
        load_metrics(METRICS_PATH)
    )

    selection_config = (
        context.frozen_config[
            "frozen_parameters"
        ]["context_selection"]
    )

    max_items = selection_config[
        "max_expanded_items"
    ]

    max_chars = selection_config[
        "max_expanded_chars"
    ]

    if max_items != 2:
        raise RuntimeError(
            f"unexpected_max_items={max_items}"
        )

    strategy_keys = {
        "current_lexical_v1": (
            current_sort_key
        ),
        "bigram_first_v1": (
            bigram_first_sort_key
        ),
        "metric_aware_v1": (
            metric_aware_sort_key
        ),
    }

    output_records = []

    for baseline in baselines:
        (
            candidate_rows,
            added_same_page_count,
        ) = build_candidate_rows(
            baseline=baseline,
            context=context,
        )

        metric_hints = build_metric_hints(
            metric_id=baseline["metric_id"],
            metric_registry=metric_registry,
            metric_aliases=metric_aliases,
        )

        candidates = []

        for candidate in candidate_rows:
            lexical_candidate = score_candidate(
                query_text=baseline[
                    "semantic_query"
                ],
                candidate=candidate,
            )

            candidates.append(
                add_metric_scores(
                    candidate=lexical_candidate,
                    metric_hints=metric_hints,
                )
            )

        strategy_results = {
            strategy_id: evaluate_strategy(
                baseline=baseline,
                candidates=candidates,
                sort_key=sort_key,
                max_items=max_items,
                max_chars=max_chars,
            )
            for strategy_id, sort_key
            in strategy_keys.items()
        }

        output_records.append(
            {
                "schema_version": (
                    "metric_aware_ranking_"
                    "simulation_v3"
                ),
                "case_id": baseline[
                    "case_id"
                ],
                "query_id": baseline[
                    "query_id"
                ],
                "report_id": baseline[
                    "report_id"
                ],
                "metric_id": baseline[
                    "metric_id"
                ],
                "semantic_query": baseline[
                    "semantic_query"
                ],
                "is_diagnostic_target": (
                    baseline[
                        "is_diagnostic_target"
                    ]
                ),
                "diagnostic_category": (
                    baseline[
                        "diagnostic_category"
                    ]
                ),
                "baseline_hit": bool(
                    baseline["target_fact_hit"]
                ),
                "base_target_hit": bool(
                    baseline["base_target_hit"]
                ),
                "metric_hints": list(
                    metric_hints
                ),
                "candidate_count": len(
                    candidates
                ),
                "added_same_page_candidate_count": (
                    added_same_page_count
                ),
                "strategies": (
                    strategy_results
                ),
            }
        )

    strategy_summaries = {
        strategy_id: summarize_strategy(
            records=output_records,
            strategy_id=strategy_id,
        )
        for strategy_id in strategy_keys
    }

    baseline_hit_count = sum(
        record["baseline_hit"]
        for record in output_records
    )

    summary = {
        "schema_version": (
            "metric_aware_ranking_"
            "simulation_summary_v3"
        ),
        "experiment_type": (
            "pre_implementation_candidate_"
            "ranking_comparison"
        ),
        "query_count": len(
            output_records
        ),
        "baseline_hit_count": (
            baseline_hit_count
        ),
        "candidate_generation_strategy": (
            "adjacent_page_plus_same_page_"
            "sibling_v2"
        ),
        "ranking_gold_information_used": (
            False
        ),
        "ranking_runtime_fields": [
            "query.metric_id",
            "metric.display_name_cn",
            "metric_alias.alias",
            "query.semantic_query",
            "candidate.text",
            "anchor_retrieval_rank",
        ],
        "base_retrieval_changed": False,
        "resolver_changed": False,
        "context_budget_changed": False,
        "max_expanded_items": max_items,
        "max_expanded_chars": max_chars,
        "strategies": strategy_summaries,
    }

    current_summary = strategy_summaries[
        "current_lexical_v1"
    ]

    metric_summary = strategy_summaries[
        "metric_aware_v1"
    ]

    if current_summary["hit_count"] != 9:
        raise RuntimeError(
            "current_baseline_not_reproduced="
            f"{current_summary['hit_count']}"
        )

    if (
        metric_summary[
            "diagnostic_hit_count"
        ]
        != 8
    ):
        raise RuntimeError(
            "diagnostic_not_fully_recovered="
            f"{metric_summary['diagnostic_hit_count']}"
        )

    if metric_summary["hit_count"] < 18:
        raise RuntimeError(
            "metric_aware_hit_count_below_18="
            f"{metric_summary['hit_count']}"
        )

    if (
        metric_summary[
            "regressed_query_count"
        ]
        != 0
    ):
        raise RuntimeError(
            "metric_aware_regression_detected="
            f"{metric_summary['regressed_query_count']}"
        )

    OUTPUT_TEMP_PATH.write_text(
        "".join(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
            for record in output_records
        ),
        encoding="utf-8",
    )

    SUMMARY_TEMP_PATH.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    OUTPUT_TEMP_PATH.replace(
        OUTPUT_PATH
    )

    SUMMARY_TEMP_PATH.replace(
        SUMMARY_PATH
    )

    print("-" * 80)
    print(
        f"query_count={len(output_records)}"
    )
    print(
        f"baseline_hit_count="
        f"{baseline_hit_count}"
    )

    for strategy_id, result in (
        strategy_summaries.items()
    ):
        print("-" * 80)
        print(f"strategy_id={strategy_id}")
        print(
            f"hit_count="
            f"{result['hit_count']}/24"
        )
        print(
            "diagnostic_hit_count="
            f"{result['diagnostic_hit_count']}/8"
        )
        print(
            "control_hit_count="
            f"{result['control_hit_count']}/16"
        )
        print(
            "recovered_query_count="
            f"{result['recovered_query_count']}"
        )
        print(
            "regressed_query_count="
            f"{result['regressed_query_count']}"
        )

    print("-" * 80)

    for record in output_records:
        if not record[
            "is_diagnostic_target"
        ]:
            continue

        current_result = record[
            "strategies"
        ]["current_lexical_v1"]

        metric_result = record[
            "strategies"
        ]["metric_aware_v1"]

        identity = (
            record["case_id"]
            + "/"
            + record["query_id"]
        )

        print(
            f"{identity}: "
            f"current_hit="
            f"{current_result['projected_hit']}, "
            f"current_rank="
            f"{current_result['target_candidate_rank']}, "
            f"metric_hit="
            f"{metric_result['projected_hit']}, "
            f"metric_rank="
            f"{metric_result['target_candidate_rank']}"
        )

    print("-" * 80)
    print(f"output_path={OUTPUT_PATH.resolve()}")
    print(
        f"summary_path="
        f"{SUMMARY_PATH.resolve()}"
    )
    print("base_retrieval_changed=false")
    print("resolver_changed=false")
    print("context_budget_changed=false")
    print("ranking_gold_information_used=false")
    print(
        "metric_aware_ranking_v3_"
        "simulation_passed=true"
    )


if __name__ == "__main__":
    main()