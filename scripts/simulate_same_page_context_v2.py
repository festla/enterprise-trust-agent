from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.services.context_budgeting import (
    score_lexical_context_candidate,
)

from scripts.evaluate_complex_diagnostic_dev_v1 import (
    OUTPUT_ROOT,
    relative_path,
    run_preflight,
    sha256_file,
)


REPLAY_PATH = (
    OUTPUT_ROOT
    / "diagnostic_independent_query_replay_v1.jsonl"
)

OUTPUT_PATH = (
    OUTPUT_ROOT
    / "same_page_candidate_simulation_v2.jsonl"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "same_page_candidate_simulation_summary_v2.json"
)

OUTPUT_TEMP_PATH = (
    OUTPUT_ROOT
    / "same_page_candidate_simulation_v2.tmp.jsonl"
)

SUMMARY_TEMP_PATH = (
    OUTPUT_ROOT
    / "same_page_candidate_simulation_summary_v2.tmp.json"
)


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def score_candidate(
    *,
    query_text: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    (
        token_hit_count,
        bigram_overlap_count,
        query_bigram_recall,
    ) = score_lexical_context_candidate(
        query_text=query_text,
        candidate_text=candidate["text"],
    )

    result = dict(candidate)

    result["token_hit_count"] = (
        token_hit_count
    )
    result["bigram_overlap_count"] = (
        bigram_overlap_count
    )
    result["query_bigram_recall"] = (
        query_bigram_recall
    )

    return result


def candidate_sort_key(
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


def target_is_present(
    *,
    target_chunk_id: str | None,
    target_pdf_page: int,
    chunk_ids: set[str],
    pdf_pages: set[int],
) -> bool:
    if target_chunk_id is not None:
        return target_chunk_id in chunk_ids

    return target_pdf_page in pdf_pages


def summarize(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    count = len(records)

    baseline_hits = sum(
        bool(record["baseline_hit"])
        for record in records
    )

    projected_hits = sum(
        bool(record["projected_hit"])
        for record in records
    )

    recovered = sum(
        bool(record["recovered"])
        for record in records
    )

    regressed = sum(
        bool(record["regressed"])
        for record in records
    )

    return {
        "query_count": count,
        "baseline_hit_count": (
            baseline_hits
        ),
        "baseline_hit_rate": (
            baseline_hits / count
            if count
            else 0.0
        ),
        "projected_hit_count": (
            projected_hits
        ),
        "projected_hit_rate": (
            projected_hits / count
            if count
            else 0.0
        ),
        "hit_count_delta": (
            projected_hits
            - baseline_hits
        ),
        "recovered_query_count": (
            recovered
        ),
        "regressed_query_count": (
            regressed
        ),
        "projected_outcome_counts": dict(
            sorted(
                Counter(
                    record[
                        "projected_outcome"
                    ]
                    for record in records
                ).items()
            )
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
                f"simulation_output_exists={path}"
            )

    context = run_preflight(
        execution_requested=False
    )

    baseline_records = load_jsonl(
        REPLAY_PATH
    )

    if len(baseline_records) != 24:
        raise RuntimeError(
            "baseline_record_count_not_24="
            f"{len(baseline_records)}"
        )

    selection_config = (
        context.frozen_config[
            "frozen_parameters"
        ][
            "context_selection"
        ]
    )

    max_items = selection_config[
        "max_expanded_items"
    ]

    max_chars = selection_config[
        "max_expanded_chars"
    ]

    output_records = []

    for baseline in baseline_records:
        expansion = baseline[
            "full_expansion"
        ]

        if expansion is None:
            base_items = []
            existing_expanded_items = []
            document_id = None
        else:
            base_items = [
                item
                for item in expansion["items"]
                if item["origin"]
                == "retrieved"
            ]

            existing_expanded_items = [
                item
                for item in expansion["items"]
                if item["origin"]
                == "adjacent_page"
            ]

            document_id = expansion[
                "document_id"
            ]

        base_chunk_ids = {
            item["chunk_id"]
            for item in base_items
        }

        base_pdf_pages = {
            item["pdf_page"]
            for item in base_items
        }

        existing_candidate_ids = {
            item["chunk_id"]
            for item
            in existing_expanded_items
        }

        anchors_by_page = {}

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

        candidate_rows = []

        for item in existing_expanded_items:
            candidate_rows.append(
                {
                    "origin": "adjacent_page",
                    "chunk_id": (
                        item["chunk_id"]
                    ),
                    "pdf_page": (
                        item["pdf_page"]
                    ),
                    "anchor_chunk_id": (
                        item["anchor_chunk_id"]
                    ),
                    "anchor_retrieval_rank": (
                        item[
                            "anchor_retrieval_rank"
                        ]
                    ),
                    "text": item["text"],
                    "text_char_count": (
                        item["text_char_count"]
                    ),
                }
            )

        source = (
            context
            .chunk_sources_by_report_id[
                baseline["report_id"]
            ]
        )

        added_same_page_count = 0

        if document_id is not None:
            for chunk in source.chunks:
                if (
                    chunk.document_id
                    != document_id
                ):
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

                if (
                    chunk.chunk_id
                    in existing_candidate_ids
                ):
                    continue

                candidate_rows.append(
                    {
                        "origin": (
                            "same_page_sibling"
                        ),
                        "chunk_id": (
                            chunk.chunk_id
                        ),
                        "pdf_page": (
                            chunk.pdf_page
                        ),
                        "anchor_chunk_id": (
                            anchor["chunk_id"]
                        ),
                        "anchor_retrieval_rank": (
                            anchor[
                                "retrieval_rank"
                            ]
                        ),
                        "text": chunk.text,
                        "text_char_count": (
                            len(chunk.text)
                        ),
                    }
                )

                added_same_page_count += 1

        scored_candidates = [
            score_candidate(
                query_text=(
                    baseline[
                        "semantic_query"
                    ]
                ),
                candidate=candidate,
            )
            for candidate in candidate_rows
        ]

        ranked_candidates = sorted(
            scored_candidates,
            key=candidate_sort_key,
        )

        selected = []
        selected_char_count = 0

        if not baseline[
            "base_target_hit"
        ]:
            for candidate in ranked_candidates:
                if len(selected) >= max_items:
                    continue

                next_char_count = (
                    selected_char_count
                    + candidate[
                        "text_char_count"
                    ]
                )

                if next_char_count > max_chars:
                    continue

                selected.append(
                    candidate
                )

                selected_char_count = (
                    next_char_count
                )

        selected_chunk_ids = {
            candidate["chunk_id"]
            for candidate in selected
        }

        selected_pdf_pages = {
            candidate["pdf_page"]
            for candidate in selected
        }

        target_chunk_id = baseline[
            "gold_evidence_chunk_id"
        ]

        target_pdf_page = baseline[
            "gold_evidence_pdf_page"
        ]

        target_in_candidate_pool = (
            target_is_present(
                target_chunk_id=(
                    target_chunk_id
                ),
                target_pdf_page=(
                    target_pdf_page
                ),
                chunk_ids={
                    candidate["chunk_id"]
                    for candidate
                    in ranked_candidates
                },
                pdf_pages={
                    candidate["pdf_page"]
                    for candidate
                    in ranked_candidates
                },
            )
        )

        target_selected = target_is_present(
            target_chunk_id=target_chunk_id,
            target_pdf_page=target_pdf_page,
            chunk_ids=selected_chunk_ids,
            pdf_pages=selected_pdf_pages,
        )

        baseline_hit = bool(
            baseline["target_fact_hit"]
        )

        if baseline[
            "base_target_hit"
        ]:
            projected_hit = True
            projected_outcome = (
                "base_resolved"
            )

        elif target_selected:
            projected_hit = True

            selected_target = next(
                (
                    candidate
                    for candidate in selected
                    if (
                        (
                            target_chunk_id
                            is not None
                            and candidate[
                                "chunk_id"
                            ]
                            == target_chunk_id
                        )
                        or (
                            target_chunk_id
                            is None
                            and candidate[
                                "pdf_page"
                            ]
                            == target_pdf_page
                        )
                    )
                ),
                None,
            )

            if (
                selected_target is not None
                and selected_target["origin"]
                == "same_page_sibling"
            ):
                projected_outcome = (
                    "same_page_sibling_recovered"
                )
            else:
                projected_outcome = (
                    "adjacent_page_recovered"
                )

        elif target_in_candidate_pool:
            projected_hit = False
            projected_outcome = (
                "budget_selection_miss"
            )

        else:
            projected_hit = False
            projected_outcome = (
                "candidate_generation_coverage_gap"
            )

        selected_audit = [
            {
                "selection_rank": index,
                "origin": candidate[
                    "origin"
                ],
                "chunk_id": candidate[
                    "chunk_id"
                ],
                "pdf_page": candidate[
                    "pdf_page"
                ],
                "anchor_retrieval_rank": (
                    candidate[
                        "anchor_retrieval_rank"
                    ]
                ),
                "text_char_count": (
                    candidate[
                        "text_char_count"
                    ]
                ),
                "token_hit_count": (
                    candidate[
                        "token_hit_count"
                    ]
                ),
                "bigram_overlap_count": (
                    candidate[
                        "bigram_overlap_count"
                    ]
                ),
                "query_bigram_recall": (
                    candidate[
                        "query_bigram_recall"
                    ]
                ),
            }
            for index, candidate in enumerate(
                selected,
                start=1,
            )
        ]

        output_records.append(
            {
                "schema_version": 1,
                "case_id": (
                    baseline["case_id"]
                ),
                "query_id": (
                    baseline["query_id"]
                ),
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
                "metric_id": (
                    baseline["metric_id"]
                ),
                "baseline_diagnosis": (
                    baseline["diagnosis"]
                ),
                "baseline_hit": (
                    baseline_hit
                ),
                "base_target_hit": (
                    baseline[
                        "base_target_hit"
                    ]
                ),
                "added_same_page_candidate_count": (
                    added_same_page_count
                ),
                "candidate_count": (
                    len(ranked_candidates)
                ),
                "target_in_candidate_pool": (
                    target_in_candidate_pool
                ),
                "selected_candidate_count": (
                    len(selected)
                ),
                "selected_char_count": (
                    selected_char_count
                ),
                "selected_candidates": (
                    selected_audit
                ),
                "target_selected": (
                    target_selected
                ),
                "projected_hit": (
                    projected_hit
                ),
                "projected_outcome": (
                    projected_outcome
                ),
                "recovered": (
                    not baseline_hit
                    and projected_hit
                ),
                "regressed": (
                    baseline_hit
                    and not projected_hit
                ),
            }
        )

    target_records = [
        record
        for record in output_records
        if record[
            "is_diagnostic_target"
        ]
    ]

    control_records = [
        record
        for record in output_records
        if not record[
            "is_diagnostic_target"
        ]
    ]

    category_summaries = {}

    for category in sorted({
        record["diagnostic_category"]
        for record in target_records
    }):
        category_summaries[
            category
        ] = summarize(
            [
                record
                for record in target_records
                if record[
                    "diagnostic_category"
                ] == category
            ]
        )

    output_text = "\n".join(
        json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for record in output_records
    ) + "\n"

    summary = {
        "schema_version": 1,
        "experiment_id": (
            "same_page_and_adjacent_"
            "candidate_simulation_v2"
        ),
        "experiment_role": (
            "pre_implementation_candidate_selection"
        ),
        "uses_gold_during_selection": False,
        "uses_gold_only_for_post_selection_scoring": True,
        "query_specific_rules_used": False,
        "base_retrieval_changed": False,
        "candidate_scoring_changed": False,
        "context_budget_changed": False,
        "max_expanded_items": max_items,
        "max_expanded_chars": max_chars,
        "source_replay": (
            relative_path(REPLAY_PATH)
        ),
        "source_replay_sha256": (
            sha256_file(REPLAY_PATH)
        ),
        "all_queries": summarize(
            output_records
        ),
        "diagnostic_targets": summarize(
            target_records
        ),
        "controls": summarize(
            control_records
        ),
        "diagnostic_categories": (
            category_summaries
        ),
    }

    try:
        with OUTPUT_TEMP_PATH.open(
            "x",
            encoding="utf-8",
        ) as file:
            file.write(output_text)

        with SUMMARY_TEMP_PATH.open(
            "x",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.write("\n")

        OUTPUT_TEMP_PATH.replace(
            OUTPUT_PATH
        )

        SUMMARY_TEMP_PATH.replace(
            SUMMARY_PATH
        )

    except Exception:
        OUTPUT_TEMP_PATH.unlink(
            missing_ok=True
        )
        SUMMARY_TEMP_PATH.unlink(
            missing_ok=True
        )
        raise

    all_summary = summary[
        "all_queries"
    ]
    target_summary = summary[
        "diagnostic_targets"
    ]
    control_summary = summary[
        "controls"
    ]

    print("-" * 80)
    print(
        "query_count="
        f"{all_summary['query_count']}"
    )
    print(
        "baseline_hit_count="
        f"{all_summary['baseline_hit_count']}"
    )
    print(
        "projected_hit_count="
        f"{all_summary['projected_hit_count']}"
    )
    print(
        "projected_hit_rate="
        f"{all_summary['projected_hit_rate']:.4f}"
    )
    print(
        "hit_count_delta="
        f"{all_summary['hit_count_delta']}"
    )
    print(
        "recovered_query_count="
        f"{all_summary['recovered_query_count']}"
    )
    print(
        "regressed_query_count="
        f"{all_summary['regressed_query_count']}"
    )
    print(
        "projected_outcome_counts="
        f"{all_summary['projected_outcome_counts']}"
    )
    print("-" * 80)
    print(
        "diagnostic_baseline_hit_count="
        f"{target_summary['baseline_hit_count']}"
    )
    print(
        "diagnostic_projected_hit_count="
        f"{target_summary['projected_hit_count']}"
    )
    print(
        "diagnostic_projected_hit_rate="
        f"{target_summary['projected_hit_rate']:.4f}"
    )
    print(
        "diagnostic_recovered_count="
        f"{target_summary['recovered_query_count']}"
    )
    print(
        "diagnostic_regressed_count="
        f"{target_summary['regressed_query_count']}"
    )
    print(
        "control_projected_hit_count="
        f"{control_summary['projected_hit_count']}"
    )
    print(
        "control_projected_hit_rate="
        f"{control_summary['projected_hit_rate']:.4f}"
    )
    print("-" * 80)

    for record in target_records:
        print(
            f"{record['case_id']}/"
            f"{record['query_id']}: "
            f"baseline="
            f"{record['baseline_hit']}, "
            f"projected="
            f"{record['projected_hit']}, "
            f"outcome="
            f"{record['projected_outcome']}, "
            f"same_page_added="
            f"{record['added_same_page_candidate_count']}"
        )

    print("-" * 80)
    print(f"output_path={OUTPUT_PATH}")
    print(f"summary_path={SUMMARY_PATH}")
    print("base_retrieval_changed=false")
    print("candidate_scoring_changed=false")
    print("context_budget_changed=false")
    print("same_page_v2_simulation_finished=true")


if __name__ == "__main__":
    main()