from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.schemas.complex_plan_eval_result import (
    ComplexRetrievalQueryOutput,
)

from scripts.evaluate_complex_diagnostic_dev_v1 import (
    CASE_PATH,
    FROZEN_CONFIG_PATH,
    MANIFEST_PATH,
    OUTPUT_ROOT,
    build_retriever,
    jsonable,
    relative_path,
    run_preflight,
    sha256_file,
)


REPLAY_PATH = (
    OUTPUT_ROOT
    / "diagnostic_independent_query_replay_v1.jsonl"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "diagnostic_independent_query_replay_summary_v1.json"
)

REPLAY_TEMP_PATH = (
    OUTPUT_ROOT
    / "diagnostic_independent_query_replay_v1.tmp.jsonl"
)

SUMMARY_TEMP_PATH = (
    OUTPUT_ROOT
    / (
        "diagnostic_independent_query_replay_"
        "summary_v1.tmp.json"
    )
)


def only_record(
    values: tuple[Any, ...],
    *,
    record_name: str,
    case_id: str,
    query_id: str,
) -> Any | None:
    if len(values) > 1:
        raise RuntimeError(
            f"multiple_{record_name}="
            f"{case_id}/{query_id}"
        )

    if not values:
        return None

    return values[0]


def build_runtime_query(
    gold_query: Any,
) -> ComplexRetrievalQueryOutput:
    return (
        ComplexRetrievalQueryOutput
        .model_validate(
            {
                "query_id": (
                    gold_query.query_id
                ),
                "semantic_query": (
                    gold_query.semantic_query
                ),
                "company_id": (
                    gold_query.company_id
                ),
                "report_id": (
                    gold_query.report_id
                ),
                "metric_id": (
                    gold_query.metric_id
                ),
                "fiscal_year": (
                    gold_query.fiscal_year
                ),
                "report_type": (
                    gold_query.report_type
                ),
                "statement_type": (
                    gold_query.statement_type
                ),
                "statement_scope": (
                    gold_query.statement_scope
                ),
            }
        )
    )


def context_identity(
    context: Any | None,
) -> tuple[
    set[str],
    set[int],
]:
    if context is None:
        return set(), set()

    chunk_ids = {
        item.chunk_id
        for item in context.items
    }

    pdf_pages = {
        item.pdf_page
        for item in context.items
    }

    return chunk_ids, pdf_pages


def support_is_present(
    *,
    evidence_chunk_id: str | None,
    evidence_pdf_page: int,
    chunk_ids: set[str],
    pdf_pages: set[int],
) -> bool:
    if evidence_chunk_id is not None:
        return (
            evidence_chunk_id
            in chunk_ids
        )

    return evidence_pdf_page in pdf_pages


def diagnose_query(
    *,
    retrieval_status: str,
    target_fact_hit: bool,
    base_target_hit: bool,
    final_target_hit: bool,
    gate_decision: str | None,
    evidence_chunk_id: str | None,
    evidence_pdf_page: int,
    full_chunk_ids: set[str],
    full_pdf_pages: set[int],
    selected_chunk_ids: set[str],
    selected_pdf_pages: set[int],
) -> str:
    if retrieval_status == "failed":
        return "retrieval_runtime_error"

    if target_fact_hit:
        if base_target_hit:
            return "base_resolved"

        if (
            gate_decision
            == "expansion_required"
            and final_target_hit
        ):
            return "expansion_recovered"

        return "resolved_without_expected_audit"

    support_in_full = support_is_present(
        evidence_chunk_id=evidence_chunk_id,
        evidence_pdf_page=evidence_pdf_page,
        chunk_ids=full_chunk_ids,
        pdf_pages=full_pdf_pages,
    )

    support_in_selected = (
        support_is_present(
            evidence_chunk_id=(
                evidence_chunk_id
            ),
            evidence_pdf_page=(
                evidence_pdf_page
            ),
            chunk_ids=selected_chunk_ids,
            pdf_pages=selected_pdf_pages,
        )
    )

    if support_in_selected:
        return "fact_resolution_miss"

    if support_in_full:
        return "budget_selection_miss"

    if (
        evidence_chunk_id is not None
        and evidence_pdf_page
        in full_pdf_pages
    ):
        return "same_page_chunk_coverage_gap"

    return (
        "retrieval_or_page_window_coverage_gap"
    )


def group_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    query_count = len(records)

    runtime_completed_count = sum(
        record["retrieval_status"]
        == "completed"
        for record in records
    )

    target_hit_count = sum(
        bool(record["target_fact_hit"])
        for record in records
    )

    base_hit_count = sum(
        bool(record["base_target_hit"])
        for record in records
    )

    expansion_recovered_count = sum(
        record["diagnosis"]
        == "expansion_recovered"
        for record in records
    )

    support_in_full_count = sum(
        bool(
            record[
                "target_support_in_full_context"
            ]
        )
        for record in records
    )

    support_in_selected_count = sum(
        bool(
            record[
                "target_support_in_selected_context"
            ]
        )
        for record in records
    )

    diagnosis_counts = Counter(
        record["diagnosis"]
        for record in records
    )

    return {
        "query_count": query_count,
        "runtime_completed_count": (
            runtime_completed_count
        ),
        "target_fact_hit_count": (
            target_hit_count
        ),
        "target_fact_hit_rate": (
            target_hit_count / query_count
            if query_count
            else 0.0
        ),
        "base_target_hit_count": (
            base_hit_count
        ),
        "expansion_recovered_count": (
            expansion_recovered_count
        ),
        "target_support_in_full_count": (
            support_in_full_count
        ),
        "target_support_in_selected_count": (
            support_in_selected_count
        ),
        "diagnosis_counts": dict(
            sorted(
                diagnosis_counts.items()
            )
        ),
    }


def main() -> None:
    for path in (
        REPLAY_PATH,
        SUMMARY_PATH,
        REPLAY_TEMP_PATH,
        SUMMARY_TEMP_PATH,
    ):
        if path.exists():
            raise RuntimeError(
                f"replay_output_exists={path}"
            )

    context = run_preflight(
        execution_requested=False
    )

    retriever = build_retriever(
        context
    )

    top_k = (
        context.frozen_config[
            "frozen_parameters"
        ][
            "base_retrieval"
        ][
            "base_top_k"
        ]
    )

    started_at = datetime.now(
        timezone.utc
    )

    records: list[dict[str, Any]] = []

    for case in context.cases:
        for gold_query in (
            case.gold_rewrite
            .retrieval_queries
        ):
            retriever.clear_audit_records()

            runtime_query = (
                build_runtime_query(
                    gold_query
                )
            )

            trace = retriever.retrieve(
                query=runtime_query,
                top_k=top_k,
            )

            expansion = only_record(
                retriever
                .full_expansion_audit_records,
                record_name="full_expansion",
                case_id=case.case_id,
                query_id=gold_query.query_id,
            )

            base_resolution = only_record(
                retriever
                .base_resolution_audit_records,
                record_name="base_resolution",
                case_id=case.case_id,
                query_id=gold_query.query_id,
            )

            selection = only_record(
                retriever
                .budget_selection_audit_records,
                record_name="budget_selection",
                case_id=case.case_id,
                query_id=gold_query.query_id,
            )

            final_resolution = only_record(
                retriever
                .final_resolution_audit_records,
                record_name="final_resolution",
                case_id=case.case_id,
                query_id=gold_query.query_id,
            )

            fact = (
                context.registry_bundle
                .financial_facts
                .get(
                    gold_query.target_fact_id
                )
            )

            if fact is None:
                raise RuntimeError(
                    "missing_gold_fact="
                    f"{gold_query.target_fact_id}"
                )

            evidence = (
                context.registry_bundle
                .evidences
                .get(
                    fact.primary_evidence_id
                )
            )

            if evidence is None:
                raise RuntimeError(
                    "missing_gold_evidence="
                    f"{fact.primary_evidence_id}"
                )

            base_context = None

            if expansion is not None:
                base_items = tuple(
                    item
                    for item
                    in expansion.items
                    if item.origin
                    == "retrieved"
                )

                base_context = type(
                    "BaseContextView",
                    (),
                    {
                        "items": base_items,
                    },
                )()

            selected_context = (
                selection.selected_context
                if selection is not None
                else None
            )

            (
                base_chunk_ids,
                base_pdf_pages,
            ) = context_identity(
                base_context
            )

            (
                full_chunk_ids,
                full_pdf_pages,
            ) = context_identity(
                expansion
            )

            (
                selected_chunk_ids,
                selected_pdf_pages,
            ) = context_identity(
                selected_context
            )

            actual_fact_ids = tuple(
                trace.retrieved_fact_ids
            )

            base_fact_ids = (
                tuple(
                    base_resolution.fact_ids
                )
                if base_resolution is not None
                else ()
            )

            final_fact_ids = (
                tuple(
                    final_resolution.fact_ids
                )
                if final_resolution is not None
                else ()
            )

            target_fact_hit = (
                gold_query.target_fact_id
                in actual_fact_ids
            )

            base_target_hit = (
                gold_query.target_fact_id
                in base_fact_ids
            )

            final_target_hit = (
                gold_query.target_fact_id
                in final_fact_ids
            )

            gate_decision = (
                selection.gate_decision
                if selection is not None
                else None
            )

            support_in_base = (
                support_is_present(
                    evidence_chunk_id=(
                        evidence.chunk_id
                    ),
                    evidence_pdf_page=(
                        evidence.pdf_page
                    ),
                    chunk_ids=base_chunk_ids,
                    pdf_pages=base_pdf_pages,
                )
            )

            support_in_full = (
                support_is_present(
                    evidence_chunk_id=(
                        evidence.chunk_id
                    ),
                    evidence_pdf_page=(
                        evidence.pdf_page
                    ),
                    chunk_ids=full_chunk_ids,
                    pdf_pages=full_pdf_pages,
                )
            )

            support_in_selected = (
                support_is_present(
                    evidence_chunk_id=(
                        evidence.chunk_id
                    ),
                    evidence_pdf_page=(
                        evidence.pdf_page
                    ),
                    chunk_ids=(
                        selected_chunk_ids
                    ),
                    pdf_pages=(
                        selected_pdf_pages
                    ),
                )
            )

            diagnosis = diagnose_query(
                retrieval_status=trace.status,
                target_fact_hit=(
                    target_fact_hit
                ),
                base_target_hit=(
                    base_target_hit
                ),
                final_target_hit=(
                    final_target_hit
                ),
                gate_decision=(
                    gate_decision
                ),
                evidence_chunk_id=(
                    evidence.chunk_id
                ),
                evidence_pdf_page=(
                    evidence.pdf_page
                ),
                full_chunk_ids=(
                    full_chunk_ids
                ),
                full_pdf_pages=(
                    full_pdf_pages
                ),
                selected_chunk_ids=(
                    selected_chunk_ids
                ),
                selected_pdf_pages=(
                    selected_pdf_pages
                ),
            )

            key = (
                case.case_id,
                gold_query.query_id,
            )

            target_metadata = (
                context.targets_by_key.get(
                    key
                )
            )

            records.append(
                {
                    "schema_version": 1,
                    "case_id": (
                        case.case_id
                    ),
                    "query_id": (
                        gold_query.query_id
                    ),
                    "is_diagnostic_target": (
                        target_metadata
                        is not None
                    ),
                    "diagnostic_category": (
                        target_metadata[
                            "diagnostic_category"
                        ]
                        if target_metadata
                        is not None
                        else "control"
                    ),
                    "company_id": (
                        gold_query.company_id
                    ),
                    "report_id": (
                        gold_query.report_id
                    ),
                    "metric_id": (
                        gold_query.metric_id
                    ),
                    "semantic_query": (
                        gold_query.semantic_query
                    ),
                    "gold_target_fact_id": (
                        gold_query
                        .target_fact_id
                    ),
                    "gold_evidence_id": (
                        evidence.evidence_id
                    ),
                    "gold_evidence_chunk_id": (
                        evidence.chunk_id
                    ),
                    "gold_evidence_pdf_page": (
                        evidence.pdf_page
                    ),
                    "runtime_executed": True,
                    "retrieval_status": (
                        trace.status
                    ),
                    "actual_fact_ids": list(
                        actual_fact_ids
                    ),
                    "actual_evidence_ids": list(
                        trace
                        .retrieved_evidence_ids
                    ),
                    "target_fact_hit": (
                        target_fact_hit
                    ),
                    "base_target_hit": (
                        base_target_hit
                    ),
                    "final_target_hit": (
                        final_target_hit
                    ),
                    "gate_decision": (
                        gate_decision
                    ),
                    "target_support_in_base_context": (
                        support_in_base
                    ),
                    "target_support_in_full_context": (
                        support_in_full
                    ),
                    "target_support_in_selected_context": (
                        support_in_selected
                    ),
                    "base_pdf_pages": sorted(
                        base_pdf_pages
                    ),
                    "full_context_pdf_pages": sorted(
                        full_pdf_pages
                    ),
                    "selected_context_pdf_pages": sorted(
                        selected_pdf_pages
                    ),
                    "diagnosis": diagnosis,
                    "error_message": (
                        trace.error_message
                    ),
                    "retrieval_trace": (
                        jsonable(trace)
                    ),
                    "full_expansion": (
                        jsonable(expansion)
                    ),
                    "base_resolution": (
                        jsonable(
                            base_resolution
                        )
                    ),
                    "budget_selection": (
                        jsonable(selection)
                    ),
                    "final_resolution": (
                        jsonable(
                            final_resolution
                        )
                    ),
                }
            )

    completed_at = datetime.now(
        timezone.utc
    )

    if len(records) != 24:
        raise RuntimeError(
            "independent_query_count_not_24="
            f"{len(records)}"
        )

    keys = [
        (
            record["case_id"],
            record["query_id"],
        )
        for record in records
    ]

    if len(keys) != len(set(keys)):
        raise RuntimeError(
            "duplicate_query_key=true"
        )

    target_records = [
        record
        for record in records
        if record[
            "is_diagnostic_target"
        ]
    ]

    control_records = [
        record
        for record in records
        if not record[
            "is_diagnostic_target"
        ]
    ]

    if len(target_records) != 8:
        raise RuntimeError(
            "target_record_count_not_8="
            f"{len(target_records)}"
        )

    if len(control_records) != 16:
        raise RuntimeError(
            "control_record_count_not_16="
            f"{len(control_records)}"
        )

    category_summaries = {}

    for category in sorted({
        record["diagnostic_category"]
        for record in target_records
    }):
        category_records = [
            record
            for record in target_records
            if record[
                "diagnostic_category"
            ] == category
        ]

        category_summaries[
            category
        ] = group_summary(
            category_records
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    replay_text = "\n".join(
        json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for record in records
    ) + "\n"

    try:
        with REPLAY_TEMP_PATH.open(
            "x",
            encoding="utf-8",
        ) as file:
            file.write(replay_text)

        replay_hash = sha256_file(
            REPLAY_TEMP_PATH
        )

        summary = {
            "schema_version": 1,
            "experiment_id": (
                "complex_diagnostic_dev_v1_"
                "independent_query_replay_v1"
            ),
            "experiment_role": (
                "remove_case_short_circuit_bias"
            ),
            "uses_frozen_test_configuration": True,
            "parameter_retuning_during_run": False,
            "query_specific_rules_used": False,
            "gold_fact_ids_available_to_runtime": False,
            "gold_evidence_ids_available_to_runtime": False,
            "gold_pdf_pages_available_to_runtime": False,
            "case_dataset": (
                relative_path(CASE_PATH)
            ),
            "case_dataset_sha256": (
                sha256_file(CASE_PATH)
            ),
            "diagnostic_manifest": (
                relative_path(MANIFEST_PATH)
            ),
            "diagnostic_manifest_sha256": (
                sha256_file(MANIFEST_PATH)
            ),
            "frozen_config": (
                relative_path(
                    FROZEN_CONFIG_PATH
                )
            ),
            "frozen_config_sha256": (
                sha256_file(
                    FROZEN_CONFIG_PATH
                )
            ),
            "run_started_at": (
                started_at.isoformat()
            ),
            "run_completed_at": (
                completed_at.isoformat()
            ),
            "all_queries": (
                group_summary(records)
            ),
            "diagnostic_targets": (
                group_summary(
                    target_records
                )
            ),
            "controls": (
                group_summary(
                    control_records
                )
            ),
            "diagnostic_categories": (
                category_summaries
            ),
            "replay_path": (
                relative_path(
                    REPLAY_PATH
                )
            ),
            "replay_sha256": (
                replay_hash
            ),
        }

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

        REPLAY_TEMP_PATH.replace(
            REPLAY_PATH
        )

        SUMMARY_TEMP_PATH.replace(
            SUMMARY_PATH
        )

    except Exception:
        REPLAY_TEMP_PATH.unlink(
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
        "independent_query_count="
        f"{all_summary['query_count']}"
    )
    print(
        "runtime_completed_count="
        f"{all_summary['runtime_completed_count']}"
    )
    print(
        "all_query_target_hit_count="
        f"{all_summary['target_fact_hit_count']}"
    )
    print(
        "all_query_target_hit_rate="
        f"{all_summary['target_fact_hit_rate']:.4f}"
    )
    print(
        "base_target_hit_count="
        f"{all_summary['base_target_hit_count']}"
    )
    print(
        "expansion_recovered_count="
        f"{all_summary['expansion_recovered_count']}"
    )
    print(
        "all_diagnosis_counts="
        f"{all_summary['diagnosis_counts']}"
    )
    print("-" * 80)
    print(
        "diagnostic_target_count="
        f"{target_summary['query_count']}"
    )
    print(
        "diagnostic_target_hit_count="
        f"{target_summary['target_fact_hit_count']}"
    )
    print(
        "diagnostic_target_hit_rate="
        f"{target_summary['target_fact_hit_rate']:.4f}"
    )
    print(
        "diagnostic_target_diagnosis_counts="
        f"{target_summary['diagnosis_counts']}"
    )
    print(
        "control_query_count="
        f"{control_summary['query_count']}"
    )
    print(
        "control_target_hit_count="
        f"{control_summary['target_fact_hit_count']}"
    )
    print(
        "control_target_hit_rate="
        f"{control_summary['target_fact_hit_rate']:.4f}"
    )
    print("-" * 80)

    for category, values in (
        category_summaries.items()
    ):
        print(
            f"category={category}, "
            f"query_count={values['query_count']}, "
            f"hit_count="
            f"{values['target_fact_hit_count']}, "
            f"diagnoses="
            f"{values['diagnosis_counts']}"
        )

    print("-" * 80)
    print("DIAGNOSTIC TARGET DETAILS")

    for record in target_records:
        print(
            f"{record['case_id']}/"
            f"{record['query_id']}: "
            f"category="
            f"{record['diagnostic_category']}, "
            f"hit={record['target_fact_hit']}, "
            f"diagnosis="
            f"{record['diagnosis']}"
        )
        print(
            "  metric_id="
            f"{record['metric_id']}"
        )
        print(
            "  gold_pdf_page="
            f"{record['gold_evidence_pdf_page']}"
        )
        print(
            "  support_in_base="
            f"{record['target_support_in_base_context']}, "
            "support_in_full="
            f"{record['target_support_in_full_context']}, "
            "support_in_selected="
            f"{record['target_support_in_selected_context']}"
        )
        print(
            "  base_pdf_pages="
            f"{record['base_pdf_pages']}"
        )
        print(
            "  selected_pdf_pages="
            f"{record['selected_context_pdf_pages']}"
        )

    print("-" * 80)
    print(f"replay_path={REPLAY_PATH}")
    print(f"summary_path={SUMMARY_PATH}")
    print(
        "all_24_queries_executed_independently=true"
    )
    print(
        "frozen_configuration_preserved=true"
    )
    print(
        "independent_query_replay_finished=true"
    )


if __name__ == "__main__":
    main()