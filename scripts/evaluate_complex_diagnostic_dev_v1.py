from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import (
    asdict,
    dataclass,
    is_dataclass,
)
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from app.schemas.context_budget import (
    ContextBudgetPolicy,
)
from app.services.budgeted_context_oracle_retrieval_adapter import (
    BudgetedContextOracleRetrievalAdapter,
)
from app.services.chunk_dataset_source import (
    LoadedChunkDataset,
    load_chunk_dataset_source,
)
from app.services.complex_oracle_answer_generator_v2 import (
    ComplexOracleAnswerGeneratorV2,
)
from app.services.complex_oracle_calculator_adapter import (
    ComplexOracleCalculatorAdapter,
)
from app.services.complex_oracle_retrieval_runtime import (
    ComplexReportRetrievalRoute,
    build_default_complex_oracle_hit_provider,
)
from app.services.complex_plan_batch_runner import (
    ComplexPlanBatchRun,
    write_complex_plan_batch_results,
)
from app.services.complex_plan_eval_dataset import (
    load_complex_financial_eval_cases,
)
from app.services.complex_plan_eval_integrity import (
    validate_complex_plan_eval_integrity,
)
from app.services.complex_plan_oracle import (
    execute_gold_oracle_case,
)
from app.services.registry import RegistryBundle
from app.services.registry_loader import (
    load_registry_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMPLEX_ROOT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "complex_plan"
)

REGISTRY_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "registries"
)

CHUNK_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks"
)

BM25_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "indices"
    / "bm25"
)

VECTOR_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "indices"
    / "vector"
)

CASE_PATH = (
    COMPLEX_ROOT
    / "complex_plan_diagnostic_dev_v1.jsonl"
)

MANIFEST_PATH = (
    COMPLEX_ROOT
    / "complex_plan_diagnostic_dev_v1_manifest.json"
)

FROZEN_CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluations"
    / "complex_plan"
    / "complex_plan_dev_v2"
    / "context_budget_frozen_config_v1.json"
)

DEV_V2_PATH = (
    COMPLEX_ROOT
    / "complex_plan_dev_v2.jsonl"
)

TEST_V1_PATH = (
    COMPLEX_ROOT
    / "complex_plan_test_v1.jsonl"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluations"
    / "complex_plan"
    / "complex_plan_diagnostic_dev_v1"
)

RESULT_PATH = (
    OUTPUT_ROOT
    / (
        "diagnostic_baseline_"
        "top5_budgeted_context_v1.jsonl"
    )
)

AUDIT_PATH = (
    OUTPUT_ROOT
    / "diagnostic_baseline_query_audit_v1.jsonl"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "diagnostic_baseline_summary_v1.json"
)

RESULT_TEMP_PATH = (
    OUTPUT_ROOT
    / (
        "diagnostic_baseline_"
        "top5_budgeted_context_v1.tmp.jsonl"
    )
)

AUDIT_TEMP_PATH = (
    OUTPUT_ROOT
    / "diagnostic_baseline_query_audit_v1.tmp.jsonl"
)

SUMMARY_TEMP_PATH = (
    OUTPUT_ROOT
    / "diagnostic_baseline_summary_v1.tmp.json"
)

RUN_ID_PREFIX = (
    "complex_run_diagnostic_dev_v1_"
    "baseline_budgeted_context_v1"
)

EXPECTED_CASE_IDS = tuple(
    f"complex_{number:03d}"
    for number in range(31, 39)
)

EXPECTED_REPORT_IDS = (
    "gree_electric_2024",
    "haier_smart_home_2024",
    "hisense_home_2024",
    "midea_group_2024",
)

EXPECTED_RETRIEVER_ID = (
    "complex_hybrid_reranker_bge_fixed_v1_"
    "adjacent_page_context_v1_"
    "gated_lexical_adjacent_budget_v1_"
    "registry_context_fact_resolver_v1"
)


@dataclass(frozen=True, slots=True)
class PreflightContext:
    cases: tuple[Any, ...]
    registry_bundle: RegistryBundle
    routes: tuple[
        ComplexReportRetrievalRoute,
        ...,
    ]
    chunk_sources_by_report_id: dict[
        str,
        LoadedChunkDataset,
    ]
    diagnostic_manifest: dict[str, Any]
    frozen_config: dict[str, Any]
    targets_by_key: dict[
        tuple[str, str],
        dict[str, Any],
    ]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def canonical_sha256(
    value: object,
) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


def load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def relative_path(path: Path) -> str:
    try:
        value = path.relative_to(
            PROJECT_ROOT
        )
    except ValueError:
        value = path

    return str(value).replace(
        "\\",
        "/",
    )


def jsonable(value: object) -> object:
    if value is None:
        return None

    if hasattr(value, "model_dump"):
        return value.model_dump(
            mode="json"
        )

    if is_dataclass(value):
        return jsonable(
            asdict(value)
        )

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(key): jsonable(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (tuple, list, set),
    ):
        return [
            jsonable(item)
            for item in value
        ]

    return value


def load_registry() -> RegistryBundle:
    (
        bundle,
        _,
        _,
        _,
    ) = load_registry_bundle(
        companies_path=(
            REGISTRY_ROOT / "companies.yaml"
        ),
        reports_path=(
            REGISTRY_ROOT / "reports.yaml"
        ),
        metrics_path=(
            REGISTRY_ROOT / "metrics.yaml"
        ),
        evidences_path=(
            REGISTRY_ROOT / "evidences.yaml"
        ),
        financial_facts_path=(
            REGISTRY_ROOT
            / "financial_facts.yaml"
        ),
    )

    return bundle


def discover_routes(
    report_ids: tuple[str, ...],
) -> tuple[
    tuple[
        ComplexReportRetrievalRoute,
        ...,
    ],
    dict[str, LoadedChunkDataset],
]:
    bm25_by_report: dict[
        str,
        Path,
    ] = {}

    for manifest_path in BM25_ROOT.glob(
        "*/index_manifest.json"
    ):
        manifest = load_json(
            manifest_path
        )

        report_id = manifest["report_id"]

        if report_id not in report_ids:
            continue

        if report_id in bm25_by_report:
            raise RuntimeError(
                "duplicate_bm25_route="
                f"{report_id}"
            )

        bm25_by_report[
            report_id
        ] = manifest_path.parent

    missing_reports = sorted(
        set(report_ids)
        - set(bm25_by_report)
    )

    if missing_reports:
        raise RuntimeError(
            "missing_bm25_routes="
            f"{missing_reports}"
        )

    routes = []
    chunk_sources: dict[
        str,
        LoadedChunkDataset,
    ] = {}

    for report_id in report_ids:
        bm25_directory = (
            bm25_by_report[report_id]
        )

        bm25_manifest = load_json(
            bm25_directory
            / "index_manifest.json"
        )

        chunk_dataset_id = (
            bm25_manifest[
                "chunk_dataset_id"
            ]
        )

        chunk_directory = (
            CHUNK_ROOT
            / report_id
            / chunk_dataset_id
        )

        if not chunk_directory.is_dir():
            raise RuntimeError(
                "missing_chunk_dataset="
                f"{chunk_directory}"
            )

        source = load_chunk_dataset_source(
            chunk_directory
        )

        if (
            source.manifest.report_id
            != report_id
        ):
            raise RuntimeError(
                "chunk_report_mismatch="
                f"{report_id}"
            )

        routes.append(
            ComplexReportRetrievalRoute(
                report_id=report_id,
                chunk_dataset_directory=(
                    chunk_directory
                ),
                bm25_index_directory=(
                    bm25_directory
                ),
            )
        )

        chunk_sources[
            report_id
        ] = source

    return (
        tuple(routes),
        chunk_sources,
    )


def validate_output_state(
    *,
    execution_requested: bool,
) -> None:
    output_paths = (
        RESULT_PATH,
        AUDIT_PATH,
        SUMMARY_PATH,
        RESULT_TEMP_PATH,
        AUDIT_TEMP_PATH,
        SUMMARY_TEMP_PATH,
    )

    existing_paths = [
        path
        for path in output_paths
        if path.exists()
    ]

    if execution_requested and existing_paths:
        joined = ", ".join(
            str(path)
            for path in existing_paths
        )

        raise RuntimeError(
            "baseline_output_already_exists="
            f"{joined}"
        )


def run_preflight(
    *,
    execution_requested: bool,
) -> PreflightContext:
    required_files = (
        CASE_PATH,
        MANIFEST_PATH,
        FROZEN_CONFIG_PATH,
        DEV_V2_PATH,
        TEST_V1_PATH,
        REGISTRY_ROOT / "companies.yaml",
        REGISTRY_ROOT / "reports.yaml",
        REGISTRY_ROOT / "metrics.yaml",
        REGISTRY_ROOT / "evidences.yaml",
        (
            REGISTRY_ROOT
            / "financial_facts.yaml"
        ),
    )

    for path in required_files:
        if not path.is_file():
            raise RuntimeError(
                f"missing_required_file={path}"
            )

    for path in (
        CHUNK_ROOT,
        BM25_ROOT,
        VECTOR_ROOT,
    ):
        if not path.is_dir():
            raise RuntimeError(
                f"missing_required_directory={path}"
            )

    validate_output_state(
        execution_requested=(
            execution_requested
        )
    )

    diagnostic_manifest = load_json(
        MANIFEST_PATH
    )

    if diagnostic_manifest[
        "dataset_id"
    ] != "complex_plan_diagnostic_dev_v1":
        raise RuntimeError(
            "unexpected_diagnostic_dataset_id"
        )

    if diagnostic_manifest[
        "status"
    ] != "verified":
        raise RuntimeError(
            "diagnostic_dataset_not_verified"
        )

    if diagnostic_manifest[
        "manual_semantic_review_required"
    ] is not False:
        raise RuntimeError(
            "manual_review_not_completed"
        )

    if diagnostic_manifest[
        "dataset_sha256"
    ] != sha256_file(CASE_PATH):
        raise RuntimeError(
            "diagnostic_dataset_hash_mismatch"
        )

    frozen_inputs = diagnostic_manifest[
        "frozen_inputs"
    ]

    if frozen_inputs[
        "complex_plan_dev_v2_sha256"
    ] != sha256_file(DEV_V2_PATH):
        raise RuntimeError(
            "complex_plan_dev_v2_hash_changed"
        )

    if frozen_inputs[
        "complex_plan_test_v1_sha256"
    ] != sha256_file(TEST_V1_PATH):
        raise RuntimeError(
            "complex_plan_test_v1_hash_changed"
        )

    cases = (
        load_complex_financial_eval_cases(
            CASE_PATH
        )
    )

    actual_case_ids = tuple(
        case.case_id
        for case in cases
    )

    if actual_case_ids != EXPECTED_CASE_IDS:
        raise RuntimeError(
            "diagnostic_case_identity_mismatch"
        )

    if any(
        case.validation_status.value
        != "verified"
        for case in cases
    ):
        raise RuntimeError(
            "diagnostic_case_not_verified"
        )

    if any(
        case.validated_by
        != "manual_review"
        for case in cases
    ):
        raise RuntimeError(
            "unexpected_case_validator"
        )

    planned_query_count = sum(
        len(
            case.gold_rewrite.retrieval_queries
        )
        for case in cases
    )

    if planned_query_count != 24:
        raise RuntimeError(
            "planned_query_count_not_24="
            f"{planned_query_count}"
        )

    if any(
        case.gold_calculation_ids
        for case in cases
    ):
        raise RuntimeError(
            "diagnostic_calculation_reference_found"
        )

    registry_bundle = load_registry()

    validate_complex_plan_eval_integrity(
        cases=cases,
        calculations=[],
        registry_bundle=registry_bundle,
    )

    targets_by_key = {
        (
            item["case_id"],
            item["query_id"],
        ): item
        for item in diagnostic_manifest[
            "diagnostic_targets"
        ]
    }

    if len(targets_by_key) != 8:
        raise RuntimeError(
            "diagnostic_target_count_not_8"
        )

    query_by_key = {
        (
            case.case_id,
            query.query_id,
        ): query
        for case in cases
        for query
        in case.gold_rewrite.retrieval_queries
    }

    for key, target in (
        targets_by_key.items()
    ):
        query = query_by_key.get(key)

        if query is None:
            raise RuntimeError(
                f"missing_target_query={key}"
            )

        if (
            query.target_fact_id
            != target["target_fact_id"]
        ):
            raise RuntimeError(
                f"target_fact_mismatch={key}"
            )

        if (
            query.semantic_query
            != target["semantic_query"]
        ):
            raise RuntimeError(
                f"target_query_text_mismatch={key}"
            )

    frozen_config = load_json(
        FROZEN_CONFIG_PATH
    )

    if frozen_config["status"] != (
        "frozen_before_test"
    ):
        raise RuntimeError(
            "configuration_is_not_frozen"
        )

    frozen_parameters = (
        frozen_config[
            "frozen_parameters"
        ]
    )

    parameter_hash = canonical_sha256(
        frozen_parameters
    )

    if parameter_hash != frozen_config[
        "frozen_parameter_sha256"
    ]:
        raise RuntimeError(
            "frozen_parameter_hash_mismatch"
        )

    base_config = frozen_parameters[
        "base_retrieval"
    ]

    selection_config = frozen_parameters[
        "context_selection"
    ]

    if base_config["base_top_k"] != 5:
        raise RuntimeError(
            "base_top_k_not_5"
        )

    if (
        base_config[
            "dense_candidate_count"
        ]
        != 50
    ):
        raise RuntimeError(
            "dense_candidate_count_changed"
        )

    if (
        base_config[
            "bm25_candidate_count"
        ]
        != 50
    ):
        raise RuntimeError(
            "bm25_candidate_count_changed"
        )

    if (
        base_config[
            "rerank_candidate_count"
        ]
        != 50
    ):
        raise RuntimeError(
            "rerank_candidate_count_changed"
        )

    if (
        selection_config[
            "max_expanded_items"
        ]
        != 2
    ):
        raise RuntimeError(
            "expanded_item_budget_changed"
        )

    if (
        selection_config[
            "max_expanded_chars"
        ]
        != 1600
    ):
        raise RuntimeError(
            "expanded_char_budget_changed"
        )

    report_ids = tuple(
        sorted({
            report_id
            for case in cases
            for report_id in case.report_ids
        })
    )

    if report_ids != EXPECTED_REPORT_IDS:
        raise RuntimeError(
            "unexpected_report_set="
            f"{report_ids}"
        )

    (
        routes,
        chunk_sources,
    ) = discover_routes(
        report_ids
    )

    return PreflightContext(
        cases=cases,
        registry_bundle=registry_bundle,
        routes=routes,
        chunk_sources_by_report_id=(
            chunk_sources
        ),
        diagnostic_manifest=(
            diagnostic_manifest
        ),
        frozen_config=frozen_config,
        targets_by_key=targets_by_key,
    )


def print_preflight(
    context: PreflightContext,
) -> None:
    planned_query_count = sum(
        len(
            case.gold_rewrite.retrieval_queries
        )
        for case in context.cases
    )

    frozen_parameters = (
        context.frozen_config[
            "frozen_parameters"
        ]
    )

    base_config = frozen_parameters[
        "base_retrieval"
    ]

    selection_config = frozen_parameters[
        "context_selection"
    ]

    print("-" * 80)
    print("mode=diagnostic_baseline_preflight")
    print(f"case_count={len(context.cases)}")
    print(
        "case_ids="
        f"{[case.case_id for case in context.cases]}"
    )
    print(
        f"planned_query_count="
        f"{planned_query_count}"
    )
    print(
        "diagnostic_target_count="
        f"{len(context.targets_by_key)}"
    )
    print(
        "control_query_count="
        f"{planned_query_count - len(context.targets_by_key)}"
    )
    print(
        f"base_top_k="
        f"{base_config['base_top_k']}"
    )
    print(
        "dense_candidate_count="
        f"{base_config['dense_candidate_count']}"
    )
    print(
        "bm25_candidate_count="
        f"{base_config['bm25_candidate_count']}"
    )
    print(
        "rerank_candidate_count="
        f"{base_config['rerank_candidate_count']}"
    )
    print(
        "max_expanded_items="
        f"{selection_config['max_expanded_items']}"
    )
    print(
        "max_expanded_chars="
        f"{selection_config['max_expanded_chars']}"
    )
    print(
        "frozen_parameter_sha256="
        f"{context.frozen_config['frozen_parameter_sha256']}"
    )
    print(
        "gold_fact_ids_available_to_runtime=false"
    )
    print(
        "gold_evidence_ids_available_to_runtime=false"
    )
    print(
        "gold_pdf_pages_available_to_runtime=false"
    )
    print("query_specific_rules_allowed=false")
    print("parameter_retuning_allowed=false")
    print(
        "diagnostic_baseline_preflight_passed=true"
    )
    print("-" * 80)


def build_retriever(
    context: PreflightContext,
) -> BudgetedContextOracleRetrievalAdapter:
    frozen_parameters = (
        context.frozen_config[
            "frozen_parameters"
        ]
    )

    base_config = frozen_parameters[
        "base_retrieval"
    ]

    candidate_config = frozen_parameters[
        "context_candidate_generation"
    ]

    selection_config = frozen_parameters[
        "context_selection"
    ]

    resolution_config = frozen_parameters[
        "fact_resolution"
    ]

    print(
        "building_frozen_diagnostic_provider=true"
    )

    hit_provider = (
        build_default_complex_oracle_hit_provider(
            routes=context.routes,
            vector_index_output_root=(
                VECTOR_ROOT
            ),
            device="cpu",
            embedding_batch_size=16,
            reranker_batch_size=8,
            local_files_only=True,
            dense_candidate_count=(
                base_config[
                    "dense_candidate_count"
                ]
            ),
            bm25_candidate_count=(
                base_config[
                    "bm25_candidate_count"
                ]
            ),
            rerank_candidate_count=(
                base_config[
                    "rerank_candidate_count"
                ]
            ),
            return_count=(
                base_config["base_top_k"]
            ),
            rank_constant=(
                base_config["rank_constant"]
            ),
            embedding_model_revision=(
                base_config[
                    "embedding_model_revision"
                ]
            ),
            reranker_model_revision=(
                base_config[
                    "reranker_model_revision"
                ]
            ),
            show_progress_bar=True,
        )
    )

    policy = (
        ContextBudgetPolicy
        .model_validate(
            selection_config
        )
    )

    retriever = (
        BudgetedContextOracleRetrievalAdapter(
            registry_bundle=(
                context.registry_bundle
            ),
            hit_provider=hit_provider,
            chunk_sources_by_report_id=(
                context
                .chunk_sources_by_report_id
            ),
            resolver_version=(
                resolution_config[
                    "resolver_version"
                ]
            ),
            context_strategy_id=(
                candidate_config[
                    "strategy_id"
                ]
            ),
            policy=policy,
        )
    )

    if (
        retriever.retriever_id
        != EXPECTED_RETRIEVER_ID
    ):
        raise RuntimeError(
            "retriever_identity_mismatch="
            f"{retriever.retriever_id}"
        )

    return retriever


def index_by_query_id(
    values: tuple[Any, ...],
) -> dict[str, Any]:
    return {
        value.query_id: value
        for value in values
    }


def execute_cases(
    *,
    context: PreflightContext,
    retriever: (
        BudgetedContextOracleRetrievalAdapter
    ),
) -> tuple[
    ComplexPlanBatchRun,
    list[dict[str, Any]],
]:
    calculator = (
        ComplexOracleCalculatorAdapter(
            registry_bundle=(
                context.registry_bundle
            )
        )
    )

    generator = (
        ComplexOracleAnswerGeneratorV2(
            registry_bundle=(
                context.registry_bundle
            )
        )
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

    results = []
    audit_records: list[
        dict[str, Any]
    ] = []

    for case in context.cases:
        retriever.clear_audit_records()

        result = execute_gold_oracle_case(
            run_id=(
                f"{RUN_ID_PREFIX}_"
                f"{case.case_id}"
            ),
            case=case,
            retriever=retriever,
            calculator=calculator,
            generator=generator,
            top_k=top_k,
        )

        results.append(result)

        trace_by_query = {
            trace.query_id: trace
            for trace
            in result.retrieval_traces
        }

        expansion_by_query = (
            index_by_query_id(
                retriever
                .full_expansion_audit_records
            )
        )

        base_by_query = (
            index_by_query_id(
                retriever
                .base_resolution_audit_records
            )
        )

        selection_by_query = (
            index_by_query_id(
                retriever
                .budget_selection_audit_records
            )
        )

        final_by_query = (
            index_by_query_id(
                retriever
                .final_resolution_audit_records
            )
        )

        for query in (
            case.gold_rewrite.retrieval_queries
        ):
            key = (
                case.case_id,
                query.query_id,
            )

            diagnostic_target = (
                context.targets_by_key.get(
                    key
                )
            )

            trace = trace_by_query.get(
                query.query_id
            )

            expansion = (
                expansion_by_query.get(
                    query.query_id
                )
            )

            base_resolution = (
                base_by_query.get(
                    query.query_id
                )
            )

            selection = (
                selection_by_query.get(
                    query.query_id
                )
            )

            final_resolution = (
                final_by_query.get(
                    query.query_id
                )
            )

            actual_fact_ids = (
                tuple(
                    trace.retrieved_fact_ids
                )
                if trace is not None
                else ()
            )

            actual_evidence_ids = (
                tuple(
                    trace.retrieved_evidence_ids
                )
                if trace is not None
                else ()
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
                query.target_fact_id
                in actual_fact_ids
            )

            base_target_hit = (
                query.target_fact_id
                in base_fact_ids
            )

            final_target_hit = (
                query.target_fact_id
                in final_fact_ids
            )

            gate_decision = (
                selection.gate_decision
                if selection is not None
                else None
            )

            expansion_recovered = (
                gate_decision
                == "expansion_required"
                and not base_target_hit
                and final_target_hit
            )

            audit_records.append(
                {
                    "schema_version": 1,
                    "case_id": case.case_id,
                    "query_id": query.query_id,
                    "is_diagnostic_target": (
                        diagnostic_target
                        is not None
                    ),
                    "diagnostic_category": (
                        diagnostic_target[
                            "diagnostic_category"
                        ]
                        if diagnostic_target
                        is not None
                        else "control"
                    ),
                    "company_id": (
                        query.company_id
                    ),
                    "report_id": (
                        query.report_id
                    ),
                    "metric_id": (
                        query.metric_id
                    ),
                    "semantic_query": (
                        query.semantic_query
                    ),
                    "gold_target_fact_id": (
                        query.target_fact_id
                    ),
                    "gold_evidence_id": (
                        query.target_fact_id.replace(
                            "fact_",
                            "evidence_",
                            1,
                        )
                    ),
                    "gold_pdf_pages": list(
                        query.gold_pdf_pages
                    ),
                    "executed": (
                        trace is not None
                    ),
                    "retrieval_status": (
                        trace.status
                        if trace is not None
                        else "not_executed"
                    ),
                    "actual_fact_ids": list(
                        actual_fact_ids
                    ),
                    "actual_evidence_ids": list(
                        actual_evidence_ids
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
                    "expansion_recovered": (
                        expansion_recovered
                    ),
                    "selected_expanded_item_count": (
                        selection
                        .selected_expanded_item_count
                        if selection is not None
                        else 0
                    ),
                    "selected_expanded_char_count": (
                        selection
                        .selected_expanded_char_count
                        if selection is not None
                        else 0
                    ),
                    "error_message": (
                        trace.error_message
                        if trace is not None
                        else None
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

    return (
        ComplexPlanBatchRun(
            results=tuple(results)
        ),
        audit_records,
    )


def query_group_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(records)

    executed = sum(
        bool(record["executed"])
        for record in records
    )

    target_hits = sum(
        bool(record["target_fact_hit"])
        for record in records
    )

    base_hits = sum(
        bool(record["base_target_hit"])
        for record in records
    )

    expansion_required = sum(
        record["gate_decision"]
        == "expansion_required"
        for record in records
    )

    expansion_recovered = sum(
        bool(record["expansion_recovered"])
        for record in records
    )

    selected_items = sum(
        record[
            "selected_expanded_item_count"
        ]
        for record in records
    )

    selected_chars = sum(
        record[
            "selected_expanded_char_count"
        ]
        for record in records
    )

    return {
        "query_count": total,
        "executed_query_count": executed,
        "target_fact_hit_count": (
            target_hits
        ),
        "target_fact_hit_rate": (
            target_hits / total
            if total
            else 0.0
        ),
        "base_target_hit_count": (
            base_hits
        ),
        "expansion_required_count": (
            expansion_required
        ),
        "expansion_recovered_count": (
            expansion_recovered
        ),
        "selected_expanded_item_count": (
            selected_items
        ),
        "selected_expanded_char_count": (
            selected_chars
        ),
    }


def build_summary(
    *,
    context: PreflightContext,
    batch: ComplexPlanBatchRun,
    audit_records: list[dict[str, Any]],
    started_at: datetime,
    completed_at: datetime,
    result_sha256: str,
    audit_sha256: str,
) -> dict[str, Any]:
    target_records = [
        record
        for record in audit_records
        if record["is_diagnostic_target"]
    ]

    control_records = [
        record
        for record in audit_records
        if not record[
            "is_diagnostic_target"
        ]
    ]

    category_summaries = {}

    categories = sorted({
        record["diagnostic_category"]
        for record in target_records
    })

    for category in categories:
        category_records = [
            record
            for record in target_records
            if record[
                "diagnostic_category"
            ] == category
        ]

        category_summaries[
            category
        ] = query_group_summary(
            category_records
        )

    case_summaries = []

    for case in context.cases:
        result = next(
            item
            for item in batch.results
            if item.case_id == case.case_id
        )

        records = [
            record
            for record in audit_records
            if record["case_id"]
            == case.case_id
        ]

        target_record = next(
            (
                record
                for record in records
                if record[
                    "is_diagnostic_target"
                ]
            ),
            None,
        )

        case_summaries.append(
            {
                "case_id": case.case_id,
                "status": result.status,
                "planned_query_count": (
                    len(records)
                ),
                "executed_query_count": sum(
                    bool(record["executed"])
                    for record in records
                ),
                "target_fact_hit_count": sum(
                    bool(
                        record[
                            "target_fact_hit"
                        ]
                    )
                    for record in records
                ),
                "all_query_targets_hit": all(
                    bool(
                        record[
                            "target_fact_hit"
                        ]
                    )
                    for record in records
                ),
                "diagnostic_target_query_id": (
                    target_record[
                        "query_id"
                    ]
                    if target_record
                    is not None
                    else None
                ),
                "diagnostic_category": (
                    target_record[
                        "diagnostic_category"
                    ]
                    if target_record
                    is not None
                    else None
                ),
                "diagnostic_target_hit": (
                    bool(
                        target_record[
                            "target_fact_hit"
                        ]
                    )
                    if target_record
                    is not None
                    else False
                ),
                "error_stage": (
                    result.error_stage
                ),
                "error_message": (
                    result.error_message
                ),
            }
        )

    answer_exact_match_count = sum(
        (
            result.answer is not None
            and result.answer.answer_text
            == next(
                case.gold_answer.answer_text
                for case in context.cases
                if case.case_id
                == result.case_id
            )
        )
        for result in batch.results
    )

    gate_counts = Counter(
        record["gate_decision"]
        for record in audit_records
        if record["gate_decision"]
        is not None
    )

    return {
        "schema_version": 1,
        "experiment_id": (
            "complex_diagnostic_dev_v1_"
            "baseline_top5_budgeted_context_v1"
        ),
        "split": "diagnostic_dev",
        "formal_test": False,
        "experiment_role": (
            "development_failure_diagnosis"
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
        "frozen_parameter_sha256": (
            context.frozen_config[
                "frozen_parameter_sha256"
            ]
        ),
        "frozen_parameters": (
            context.frozen_config[
                "frozen_parameters"
            ]
        ),
        "run_started_at": (
            started_at.isoformat()
        ),
        "run_completed_at": (
            completed_at.isoformat()
        ),
        "case_count": batch.case_count,
        "completed_count": (
            batch.completed_count
        ),
        "failed_count": (
            batch.failed_count
        ),
        "refused_count": (
            batch.refused_count
        ),
        "all_completed": (
            batch.all_completed
        ),
        "completion_rate": (
            batch.completed_count
            / batch.case_count
        ),
        "all_queries": (
            query_group_summary(
                audit_records
            )
        ),
        "diagnostic_targets": (
            query_group_summary(
                target_records
            )
        ),
        "controls": (
            query_group_summary(
                control_records
            )
        ),
        "diagnostic_categories": (
            category_summaries
        ),
        "gate_decision_counts": dict(
            sorted(gate_counts.items())
        ),
        "answer_exact_match_count": (
            answer_exact_match_count
        ),
        "case_summaries": (
            case_summaries
        ),
        "retriever_ids": sorted({
            result.retriever_id
            for result in batch.results
        }),
        "generator_ids": sorted({
            result.generator_id
            for result in batch.results
        }),
        "top_k_values": sorted({
            trace.top_k
            for result in batch.results
            for trace
            in result.retrieval_traces
        }),
        "result_path": (
            relative_path(RESULT_PATH)
        ),
        "result_sha256": result_sha256,
        "audit_path": (
            relative_path(AUDIT_PATH)
        ),
        "audit_sha256": audit_sha256,
    }


def write_outputs(
    *,
    context: PreflightContext,
    batch: ComplexPlanBatchRun,
    audit_records: list[dict[str, Any]],
    started_at: datetime,
    completed_at: datetime,
) -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_text = "\n".join(
        json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for record in audit_records
    ) + "\n"

    try:
        write_complex_plan_batch_results(
            batch=batch,
            output_path=RESULT_TEMP_PATH,
        )

        with AUDIT_TEMP_PATH.open(
            "x",
            encoding="utf-8",
        ) as file:
            file.write(audit_text)

        result_hash = sha256_file(
            RESULT_TEMP_PATH
        )

        audit_hash = sha256_file(
            AUDIT_TEMP_PATH
        )

        summary = build_summary(
            context=context,
            batch=batch,
            audit_records=audit_records,
            started_at=started_at,
            completed_at=completed_at,
            result_sha256=result_hash,
            audit_sha256=audit_hash,
        )

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

        RESULT_TEMP_PATH.replace(
            RESULT_PATH
        )

        AUDIT_TEMP_PATH.replace(
            AUDIT_PATH
        )

        SUMMARY_TEMP_PATH.replace(
            SUMMARY_PATH
        )

        return summary

    except Exception:
        for path in (
            RESULT_TEMP_PATH,
            AUDIT_TEMP_PATH,
            SUMMARY_TEMP_PATH,
        ):
            path.unlink(
                missing_ok=True
            )

        raise


def print_results(
    *,
    context: PreflightContext,
    batch: ComplexPlanBatchRun,
    audit_records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    all_queries = summary[
        "all_queries"
    ]

    targets = summary[
        "diagnostic_targets"
    ]

    controls = summary[
        "controls"
    ]

    print("-" * 80)
    print(f"case_count={batch.case_count}")
    print(
        f"completed_count="
        f"{batch.completed_count}"
    )
    print(
        f"failed_count="
        f"{batch.failed_count}"
    )
    print(
        f"refused_count="
        f"{batch.refused_count}"
    )
    print(
        f"all_completed="
        f"{str(batch.all_completed).lower()}"
    )
    print(
        "completion_rate="
        f"{summary['completion_rate']:.4f}"
    )
    print(
        "planned_query_count="
        f"{all_queries['query_count']}"
    )
    print(
        "executed_query_count="
        f"{all_queries['executed_query_count']}"
    )
    print(
        "target_fact_hit_count="
        f"{all_queries['target_fact_hit_count']}"
    )
    print(
        "query_target_hit_rate="
        f"{all_queries['target_fact_hit_rate']:.4f}"
    )
    print(
        "base_target_hit_count="
        f"{all_queries['base_target_hit_count']}"
    )
    print(
        "expansion_required_count="
        f"{all_queries['expansion_required_count']}"
    )
    print(
        "expansion_recovered_count="
        f"{all_queries['expansion_recovered_count']}"
    )
    print("-" * 80)
    print(
        "diagnostic_target_query_count="
        f"{targets['query_count']}"
    )
    print(
        "diagnostic_target_hit_count="
        f"{targets['target_fact_hit_count']}"
    )
    print(
        "diagnostic_target_hit_rate="
        f"{targets['target_fact_hit_rate']:.4f}"
    )
    print(
        "diagnostic_target_base_hit_count="
        f"{targets['base_target_hit_count']}"
    )
    print(
        "diagnostic_target_expansion_recovered_count="
        f"{targets['expansion_recovered_count']}"
    )
    print(
        "control_query_count="
        f"{controls['query_count']}"
    )
    print(
        "control_target_hit_count="
        f"{controls['target_fact_hit_count']}"
    )
    print(
        "control_target_hit_rate="
        f"{controls['target_fact_hit_rate']:.4f}"
    )
    print("-" * 80)

    for category, values in summary[
        "diagnostic_categories"
    ].items():
        print(
            f"category={category}, "
            f"query_count={values['query_count']}, "
            f"hit_count="
            f"{values['target_fact_hit_count']}, "
            f"base_hit_count="
            f"{values['base_target_hit_count']}, "
            f"expansion_recovered="
            f"{values['expansion_recovered_count']}"
        )

    print("-" * 80)

    for case_summary in summary[
        "case_summaries"
    ]:
        print(
            f"{case_summary['case_id']}: "
            f"status={case_summary['status']}, "
            f"query_hits="
            f"{case_summary['target_fact_hit_count']}/"
            f"{case_summary['planned_query_count']}, "
            f"diagnostic_target="
            f"{case_summary['diagnostic_target_hit']}, "
            f"category="
            f"{case_summary['diagnostic_category']}"
        )

        if (
            case_summary["status"]
            != "completed"
        ):
            print(
                "  error_stage="
                f"{case_summary['error_stage']}"
            )
            print(
                "  error_message="
                f"{case_summary['error_message']}"
            )

    print("-" * 80)
    print("UNRESOLVED OR NOT EXECUTED QUERIES")

    unresolved_count = 0

    for record in audit_records:
        if record["target_fact_hit"]:
            continue

        unresolved_count += 1

        print(
            f"{record['case_id']}/"
            f"{record['query_id']}: "
            f"target="
            f"{record['is_diagnostic_target']}, "
            f"category="
            f"{record['diagnostic_category']}, "
            f"metric_id="
            f"{record['metric_id']}, "
            f"executed="
            f"{record['executed']}, "
            f"gate="
            f"{record['gate_decision']}"
        )

        print(
            "  expected_fact_id="
            f"{record['gold_target_fact_id']}"
        )

        print(
            "  actual_fact_ids="
            f"{record['actual_fact_ids']}"
        )

    print(
        f"unresolved_query_count="
        f"{unresolved_count}"
    )
    print("-" * 80)
    print(f"result_path={RESULT_PATH}")
    print(f"audit_path={AUDIT_PATH}")
    print(f"summary_path={SUMMARY_PATH}")
    print(
        "frozen_configuration_preserved=true"
    )
    print(
        "diagnostic_baseline_finished=true"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Run the real frozen diagnostic baseline. "
            "Without this option only preflight is run."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    context = run_preflight(
        execution_requested=args.execute
    )

    print_preflight(
        context
    )

    if not args.execute:
        print(
            "execution_requested=false"
        )
        return

    started_at = datetime.now(
        timezone.utc
    )

    retriever = build_retriever(
        context
    )

    (
        batch,
        audit_records,
    ) = execute_cases(
        context=context,
        retriever=retriever,
    )

    completed_at = datetime.now(
        timezone.utc
    )

    summary = write_outputs(
        context=context,
        batch=batch,
        audit_records=audit_records,
        started_at=started_at,
        completed_at=completed_at,
    )

    print_results(
        context=context,
        batch=batch,
        audit_records=audit_records,
        summary=summary,
    )


if __name__ == "__main__":
    main()