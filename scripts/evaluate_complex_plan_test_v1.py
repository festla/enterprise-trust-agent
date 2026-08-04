from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import (
    asdict,
    dataclass,
    is_dataclass,
)
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from app.schemas.complex_plan_eval import (  # noqa: E402
    ComplexFinancialEvalCase,
)
from app.schemas.context_budget import (  # noqa: E402
    ContextBudgetPolicy,
)
from app.services.budgeted_context_oracle_retrieval_adapter import (  # noqa: E402
    BudgetedContextOracleRetrievalAdapter,
)
from app.services.chunk_dataset_source import (  # noqa: E402
    LoadedChunkDataset,
    load_chunk_dataset_source,
)
from app.services.complex_oracle_answer_generator_v2 import (  # noqa: E402
    ComplexOracleAnswerGeneratorV2,
)
from app.services.complex_oracle_calculator_adapter import (  # noqa: E402
    ComplexOracleCalculatorAdapter,
)
from app.services.complex_oracle_retrieval_runtime import (  # noqa: E402
    ComplexReportRetrievalRoute,
    build_default_complex_oracle_hit_provider,
)
from app.services.complex_plan_batch_runner import (  # noqa: E402
    ComplexPlanBatchRun,
    write_complex_plan_batch_results,
)
from app.services.complex_plan_eval_dataset import (  # noqa: E402
    load_complex_financial_eval_cases,
)
from app.services.complex_plan_eval_integrity import (  # noqa: E402
    validate_complex_plan_eval_integrity,
)
from app.services.complex_plan_oracle import (  # noqa: E402
    execute_gold_oracle_case,
)
from app.services.derived_calculation_dataset import (  # noqa: E402
    load_derived_calculations,
)
from app.services.registry import (  # noqa: E402
    RegistryBundle,
)
from app.services.registry_loader import (  # noqa: E402
    load_registry_bundle,
)


CASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "complex_plan"
    / "complex_plan_test_v1.jsonl"
)

CALCULATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "complex_plan"
    / "gold_calculations_test_v1.jsonl"
)

TEST_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "complex_plan"
    / "complex_plan_test_v1_manifest.json"
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

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluations"
    / "complex_plan"
    / "complex_plan_test_v1"
)

RESULT_PATH = (
    OUTPUT_ROOT
    / "formal_test_top5_budgeted_context_v1.jsonl"
)

AUDIT_PATH = (
    OUTPUT_ROOT
    / "formal_test_context_audit_v1.jsonl"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "formal_test_summary_v1.json"
)

RESULT_TEMP_PATH = (
    OUTPUT_ROOT
    / "formal_test_top5_budgeted_context_v1.tmp.jsonl"
)

AUDIT_TEMP_PATH = (
    OUTPUT_ROOT
    / "formal_test_context_audit_v1.tmp.jsonl"
)

SUMMARY_TEMP_PATH = (
    OUTPUT_ROOT
    / "formal_test_summary_v1.tmp.json"
)

RUN_ID_PREFIX = (
    "complex_run_test_v1_frozen_"
    "top5_budgeted_context_v1"
)

EXPECTED_CASE_IDS = tuple(
    f"complex_{number:03d}"
    for number in range(21, 31)
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
    cases: tuple[
        ComplexFinancialEvalCase,
        ...,
    ]

    registry_bundle: RegistryBundle

    routes: tuple[
        ComplexReportRetrievalRoute,
        ...,
    ]

    chunk_sources_by_report_id: dict[
        str,
        LoadedChunkDataset,
    ]

    test_manifest: dict[str, Any]
    frozen_config: dict[str, Any]

    existing_output_paths: tuple[
        Path,
        ...,
    ]


def _sha256_bytes(
    value: bytes,
) -> str:
    return hashlib.sha256(
        value
    ).hexdigest()


def _sha256_file(
    path: Path,
) -> str:
    return _sha256_bytes(
        path.read_bytes()
    )


def _canonical_sha256(
    value: object,
) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return _sha256_bytes(
        payload
    )


def _load_json(
    path: Path,
) -> dict[str, Any]:
    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError(
            f"Cannot load JSON file: {path}"
        ) from exc


def _relative_path(
    path: Path,
) -> str:
    try:
        relative = path.relative_to(
            PROJECT_ROOT
        )
    except ValueError:
        relative = path

    return str(relative).replace(
        "\\",
        "/",
    )


def _jsonable(
    value: object,
) -> object:
    if value is None:
        return None

    if hasattr(value, "model_dump"):
        return value.model_dump(
            mode="json"
        )

    if is_dataclass(value):
        return asdict(value)

    return value


def _validate_required_files() -> None:
    required_files = (
        CASE_PATH,
        CALCULATION_PATH,
        TEST_MANIFEST_PATH,
        FROZEN_CONFIG_PATH,
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
                f"Missing required file: {path}"
            )

    required_directories = (
        CHUNK_ROOT,
        BM25_ROOT,
        VECTOR_ROOT,
    )

    for path in required_directories:
        if not path.is_dir():
            raise RuntimeError(
                f"Missing required directory: {path}"
            )


def _validate_output_state(
    *,
    execution_requested: bool,
) -> tuple[Path, ...]:
    final_paths = (
        RESULT_PATH,
        AUDIT_PATH,
        SUMMARY_PATH,
    )

    temporary_paths = (
        RESULT_TEMP_PATH,
        AUDIT_TEMP_PATH,
        SUMMARY_TEMP_PATH,
    )

    stale_temporary_paths = tuple(
        path
        for path in temporary_paths
        if path.exists()
    )

    if stale_temporary_paths:
        raise RuntimeError(
            "Stale temporary output exists: "
            + ", ".join(
                str(path)
                for path
                in stale_temporary_paths
            )
        )

    existing_final_paths = tuple(
        path
        for path in final_paths
        if path.exists()
    )

    if (
        existing_final_paths
        and len(existing_final_paths)
        != len(final_paths)
    ):
        raise RuntimeError(
            "Partial formal Test output exists: "
            + ", ".join(
                str(path)
                for path
                in existing_final_paths
            )
        )

    if (
        execution_requested
        and existing_final_paths
    ):
        raise RuntimeError(
            "Formal Test has already been "
            "executed; output overwrite is forbidden."
        )

    return existing_final_paths


def _validate_test_manifest(
    test_manifest: dict[str, Any],
) -> None:
    if (
        test_manifest.get("schema_version")
        != 1
    ):
        raise RuntimeError(
            "Unexpected Test manifest schema"
        )

    if (
        test_manifest.get("dataset_id")
        != "complex_plan_test_v1"
    ):
        raise RuntimeError(
            "Unexpected Test dataset_id"
        )

    if test_manifest.get("split") != "test":
        raise RuntimeError(
            "Manifest split must be test"
        )

    if (
        test_manifest.get("status")
        != "frozen"
    ):
        raise RuntimeError(
            "Test dataset is not frozen"
        )

    case_manifest = test_manifest[
        "case_file"
    ]

    calculation_manifest = test_manifest[
        "calculation_file"
    ]

    if (
        _sha256_file(CASE_PATH)
        != case_manifest["sha256"]
    ):
        raise RuntimeError(
            "Frozen Test Case hash mismatch"
        )

    if (
        _sha256_file(CALCULATION_PATH)
        != calculation_manifest["sha256"]
    ):
        raise RuntimeError(
            "Frozen calculation hash mismatch"
        )

    if case_manifest["case_count"] != 10:
        raise RuntimeError(
            "Frozen Test case_count must be 10"
        )

    if tuple(
        case_manifest["case_ids"]
    ) != EXPECTED_CASE_IDS:
        raise RuntimeError(
            "Frozen Test case_ids mismatch"
        )

    if (
        calculation_manifest[
            "calculation_count"
        ]
        != 7
    ):
        raise RuntimeError(
            "Frozen calculation_count must be 7"
        )

    policy = test_manifest[
        "evaluation_policy"
    ]

    forbidden_flags = (
        "allow_post_test_query_tuning",
        "allow_post_test_parameter_tuning",
        "allow_post_test_gold_edit",
    )

    for flag_name in forbidden_flags:
        if policy.get(flag_name) is not False:
            raise RuntimeError(
                "Frozen Test policy violation: "
                f"{flag_name}"
            )


def _validate_frozen_config(
    frozen_config: dict[str, Any],
) -> None:
    if (
        frozen_config.get("schema_version")
        != 1
    ):
        raise RuntimeError(
            "Unexpected frozen config schema"
        )

    if (
        frozen_config.get("status")
        != "frozen_before_test"
    ):
        raise RuntimeError(
            "Runtime configuration is not "
            "frozen_before_test"
        )

    frozen_parameters = (
        frozen_config["frozen_parameters"]
    )

    actual_parameter_sha256 = (
        _canonical_sha256(
            frozen_parameters
        )
    )

    expected_parameter_sha256 = (
        frozen_config[
            "frozen_parameter_sha256"
        ]
    )

    if (
        actual_parameter_sha256
        != expected_parameter_sha256
    ):
        raise RuntimeError(
            "Frozen parameter hash mismatch"
        )

    protocol = frozen_config[
        "test_protocol"
    ]

    if tuple(
        protocol["target_case_ids"]
    ) != EXPECTED_CASE_IDS:
        raise RuntimeError(
            "Frozen target_case_ids mismatch"
        )

    if (
        protocol[
            "parameter_retuning_allowed"
        ]
        is not False
    ):
        raise RuntimeError(
            "Frozen protocol unexpectedly "
            "allows parameter retuning"
        )

    if (
        protocol[
            "query_specific_rules_allowed"
        ]
        is not False
    ):
        raise RuntimeError(
            "Frozen protocol unexpectedly "
            "allows query-specific rules"
        )

    gold_runtime_flags = (
        "gold_evidence_ids_available_to_runtime",
        "gold_fact_ids_available_to_runtime",
        "gold_pdf_pages_available_to_runtime",
    )

    for flag_name in gold_runtime_flags:
        if protocol[flag_name] is not False:
            raise RuntimeError(
                "Gold data must not be available "
                f"to runtime: {flag_name}"
            )

    base_config = frozen_parameters[
        "base_retrieval"
    ]

    if (
        base_config["retriever_type"]
        != "hybrid_rrf_reranker"
    ):
        raise RuntimeError(
            "Unexpected retriever_type"
        )

    if base_config["base_top_k"] != 5:
        raise RuntimeError(
            "Frozen base_top_k must be 5"
        )

    if (
        base_config[
            "dense_candidate_count"
        ]
        != 50
    ):
        raise RuntimeError(
            "Frozen dense candidate count "
            "must be 50"
        )

    if (
        base_config[
            "bm25_candidate_count"
        ]
        != 50
    ):
        raise RuntimeError(
            "Frozen BM25 candidate count "
            "must be 50"
        )

    if (
        base_config[
            "rerank_candidate_count"
        ]
        != 50
    ):
        raise RuntimeError(
            "Frozen rerank candidate count "
            "must be 50"
        )

    if base_config["rank_constant"] != 60:
        raise RuntimeError(
            "Frozen RRF rank_constant must be 60"
        )

    if (
        base_config["embedding_model_name"]
        != "BAAI/bge-small-zh-v1.5"
    ):
        raise RuntimeError(
            "Unexpected embedding model"
        )

    if (
        base_config["reranker_model_name"]
        != "BAAI/bge-reranker-base"
    ):
        raise RuntimeError(
            "Unexpected reranker model"
        )

    candidate_config = (
        frozen_parameters[
            "context_candidate_generation"
        ]
    )

    if candidate_config["page_window"] != 1:
        raise RuntimeError(
            "Frozen page_window must be 1"
        )

    if (
        candidate_config["strategy_id"]
        != "adjacent_page_context_v1"
    ):
        raise RuntimeError(
            "Unexpected context strategy"
        )

    selection_config = (
        frozen_parameters[
            "context_selection"
        ]
    )

    if (
        selection_config[
            "expand_only_when_base_unresolved"
        ]
        is not True
    ):
        raise RuntimeError(
            "Context expansion gate changed"
        )

    if (
        selection_config[
            "max_expanded_items"
        ]
        != 2
    ):
        raise RuntimeError(
            "Frozen expanded item budget "
            "must be 2"
        )

    if (
        selection_config[
            "max_expanded_chars"
        ]
        != 1600
    ):
        raise RuntimeError(
            "Frozen expanded character budget "
            "must be 1600"
        )

    resolution_config = (
        frozen_parameters[
            "fact_resolution"
        ]
    )

    if (
        resolution_config[
            "requires_verified_evidence"
        ]
        is not True
        or resolution_config[
            "requires_verified_fact"
        ]
        is not True
    ):
        raise RuntimeError(
            "Runtime must require verified "
            "Evidence and FinancialFact"
        )


def _load_registry_bundle() -> RegistryBundle:
    (
        bundle,
        _,
        _,
        _,
    ) = load_registry_bundle(
        companies_path=(
            REGISTRY_ROOT
            / "companies.yaml"
        ),
        reports_path=(
            REGISTRY_ROOT
            / "reports.yaml"
        ),
        metrics_path=(
            REGISTRY_ROOT
            / "metrics.yaml"
        ),
        evidences_path=(
            REGISTRY_ROOT
            / "evidences.yaml"
        ),
        financial_facts_path=(
            REGISTRY_ROOT
            / "financial_facts.yaml"
        ),
    )

    return bundle


def _discover_routes(
    report_ids: tuple[str, ...],
) -> tuple[
    tuple[
        ComplexReportRetrievalRoute,
        ...,
    ],
    dict[str, LoadedChunkDataset],
]:
    bm25_directory_by_report_id: dict[
        str,
        Path,
    ] = {}

    for manifest_path in BM25_ROOT.glob(
        "*/index_manifest.json"
    ):
        manifest = _load_json(
            manifest_path
        )

        report_id = manifest[
            "report_id"
        ]

        if report_id not in report_ids:
            continue

        if report_id in (
            bm25_directory_by_report_id
        ):
            raise RuntimeError(
                "Duplicate BM25 route for "
                f"{report_id}"
            )

        bm25_directory_by_report_id[
            report_id
        ] = manifest_path.parent

    missing_report_ids = sorted(
        set(report_ids)
        - set(
            bm25_directory_by_report_id
        )
    )

    if missing_report_ids:
        raise RuntimeError(
            "Missing BM25 routes: "
            f"{missing_report_ids}"
        )

    routes: list[
        ComplexReportRetrievalRoute
    ] = []

    chunk_sources_by_report_id: dict[
        str,
        LoadedChunkDataset,
    ] = {}

    for report_id in report_ids:
        bm25_directory = (
            bm25_directory_by_report_id[
                report_id
            ]
        )

        bm25_manifest = _load_json(
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
                "Missing ChunkDataset: "
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
                "ChunkDataset report identity "
                f"mismatch: {report_id}"
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

        chunk_sources_by_report_id[
            report_id
        ] = source

    return (
        tuple(routes),
        chunk_sources_by_report_id,
    )


def run_preflight(
    *,
    execution_requested: bool,
) -> PreflightContext:
    _validate_required_files()

    existing_output_paths = (
        _validate_output_state(
            execution_requested=(
                execution_requested
            )
        )
    )

    test_manifest = _load_json(
        TEST_MANIFEST_PATH
    )

    frozen_config = _load_json(
        FROZEN_CONFIG_PATH
    )

    _validate_test_manifest(
        test_manifest
    )

    _validate_frozen_config(
        frozen_config
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
            "Test case identity mismatch"
        )

    if any(
        case.validation_status.value
        != "verified"
        for case in cases
    ):
        raise RuntimeError(
            "Every Test Case must be verified"
        )

    if any(
        case.validated_by
        != "manual_review"
        for case in cases
    ):
        raise RuntimeError(
            "Unexpected Test Case validator"
        )

    calculations = (
        load_derived_calculations(
            CALCULATION_PATH
        )
    )

    if len(calculations) != 7:
        raise RuntimeError(
            "Expected 7 Test calculations"
        )

    registry_bundle = (
        _load_registry_bundle()
    )

    validate_complex_plan_eval_integrity(
        cases=cases,
        calculations=calculations,
        registry_bundle=registry_bundle,
    )

    report_ids = tuple(
        sorted({
            report_id
            for case in cases
            for report_id
            in case.report_ids
        })
    )

    if report_ids != EXPECTED_REPORT_IDS:
        raise RuntimeError(
            "Unexpected Test report set"
        )

    (
        routes,
        chunk_sources_by_report_id,
    ) = _discover_routes(report_ids)

    return PreflightContext(
        cases=cases,
        registry_bundle=registry_bundle,
        routes=routes,
        chunk_sources_by_report_id=(
            chunk_sources_by_report_id
        ),
        test_manifest=test_manifest,
        frozen_config=frozen_config,
        existing_output_paths=(
            existing_output_paths
        ),
    )


def print_preflight_summary(
    context: PreflightContext,
) -> None:
    cases = context.cases

    status_counts = Counter(
        case.validation_status.value
        for case in cases
    )

    planned_query_count = sum(
        len(
            case.gold_rewrite
            .retrieval_queries
        )
        for case in cases
    )

    calculation_reference_count = sum(
        len(case.gold_calculation_ids)
        for case in cases
    )

    output_exists = bool(
        context.existing_output_paths
    )

    print("-" * 80)
    print("mode=preflight")
    print(f"case_count={len(cases)}")
    print(
        "case_ids="
        f"{[case.case_id for case in cases]}"
    )
    print(
        "status_counts="
        f"{dict(status_counts)}"
    )
    print(
        "planned_query_count="
        f"{planned_query_count}"
    )
    print(
        "calculation_reference_count="
        f"{calculation_reference_count}"
    )
    print(
        "report_ids="
        f"{list(EXPECTED_REPORT_IDS)}"
    )
    print(
        "frozen_case_sha256="
        f"{context.test_manifest['case_file']['sha256']}"
    )
    print(
        "frozen_calculation_sha256="
        f"{context.test_manifest['calculation_file']['sha256']}"
    )
    print(
        "frozen_parameter_sha256="
        f"{context.frozen_config['frozen_parameter_sha256']}"
    )
    print(
        f"formal_test_output_exists="
        f"{str(output_exists).lower()}"
    )
    print(
        "complex_plan_test_v1_"
        "preflight_passed=true"
    )
    print("-" * 80)


def _build_retriever(
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
        "building_frozen_real_provider=true"
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
                base_config[
                    "base_top_k"
                ]
            ),
            rank_constant=(
                base_config[
                    "rank_constant"
                ]
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
            "Retriever identity mismatch: "
            f"{retriever.retriever_id}"
        )

    return retriever


def _index_audit_by_query(
    values: tuple[object, ...],
) -> dict[str, object]:
    return {
        value.query_id: value
        for value in values
    }


def _execute_cases(
    *,
    context: PreflightContext,
    retriever: (
        BudgetedContextOracleRetrievalAdapter
    ),
) -> tuple[
    ComplexPlanBatchRun,
    list[dict[str, object]],
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
        dict[str, object]
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

        full_expansion_by_query = (
            _index_audit_by_query(
                retriever
                .full_expansion_audit_records
            )
        )

        base_resolution_by_query = (
            _index_audit_by_query(
                retriever
                .base_resolution_audit_records
            )
        )

        budget_selection_by_query = (
            _index_audit_by_query(
                retriever
                .budget_selection_audit_records
            )
        )

        final_resolution_by_query = (
            _index_audit_by_query(
                retriever
                .final_resolution_audit_records
            )
        )

        for query_item in (
            case.gold_rewrite
            .retrieval_queries
        ):
            query_id = (
                query_item.query_id
            )

            audit_records.append(
                {
                    "schema_version": 1,
                    "case_id": (
                        case.case_id
                    ),
                    "query_id": query_id,
                    "semantic_query": (
                        query_item
                        .semantic_query
                    ),
                    "retrieval_trace": (
                        _jsonable(
                            trace_by_query.get(
                                query_id
                            )
                        )
                    ),
                    "full_expansion": (
                        _jsonable(
                            full_expansion_by_query
                            .get(query_id)
                        )
                    ),
                    "base_resolution": (
                        _jsonable(
                            base_resolution_by_query
                            .get(query_id)
                        )
                    ),
                    "budget_selection": (
                        _jsonable(
                            budget_selection_by_query
                            .get(query_id)
                        )
                    ),
                    "final_resolution": (
                        _jsonable(
                            final_resolution_by_query
                            .get(query_id)
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


def _build_summary(
    *,
    context: PreflightContext,
    batch: ComplexPlanBatchRun,
    audit_records: list[
        dict[str, object]
    ],
    run_started_at: datetime,
    run_completed_at: datetime,
    result_sha256: str,
    audit_sha256: str,
) -> dict[str, object]:
    results = batch.results

    case_by_id = {
        case.case_id: case
        for case in context.cases
    }

    planned_query_count = sum(
        len(
            case.gold_rewrite
            .retrieval_queries
        )
        for case in context.cases
    )

    executed_query_count = sum(
        len(result.retrieval_traces)
        for result in results
    )

    resolved_query_count = sum(
        bool(trace.retrieved_fact_ids)
        for result in results
        for trace in result.retrieval_traces
    )

    calculation_trace_count = sum(
        len(result.calculation_traces)
        for result in results
    )

    completed_calculation_count = sum(
        trace.status == "completed"
        for result in results
        for trace
        in result.calculation_traces
    )

    answer_exact_match_count = 0
    fact_reference_exact_match_count = 0
    evidence_reference_exact_match_count = 0
    calculation_reference_exact_match_count = 0

    failure_records = []

    for result in results:
        case = case_by_id[
            result.case_id
        ]

        if result.answer is not None:
            answer_exact_match_count += (
                result.answer.answer_text
                == case.gold_answer.answer_text
            )

            fact_reference_exact_match_count += (
                tuple(
                    result.answer
                    .supporting_fact_ids
                )
                == tuple(
                    case.gold_answer
                    .supporting_fact_ids
                )
            )

            evidence_reference_exact_match_count += (
                tuple(
                    result.answer
                    .citation_evidence_ids
                )
                == tuple(
                    case.gold_answer
                    .evidence_ids
                )
            )

            calculation_reference_exact_match_count += (
                tuple(
                    result.answer
                    .supporting_calculation_ids
                )
                == tuple(
                    case.gold_answer
                    .supporting_calculation_ids
                )
            )

        if result.status != "completed":
            failure_records.append(
                {
                    "case_id": (
                        result.case_id
                    ),
                    "status": (
                        result.status
                    ),
                    "error_stage": (
                        result.error_stage
                    ),
                    "error_message": (
                        result.error_message
                    ),
                    "unresolved_query_ids": [
                        trace.query_id
                        for trace
                        in result.retrieval_traces
                        if not (
                            trace
                            .retrieved_fact_ids
                        )
                    ],
                }
            )

    budget_selections = [
        record["budget_selection"]
        for record in audit_records
        if record["budget_selection"]
        is not None
    ]

    base_resolved_query_count = sum(
        selection["gate_decision"]
        == "base_resolved"
        for selection in budget_selections
    )

    expansion_required_query_count = sum(
        selection["gate_decision"]
        == "expansion_required"
        for selection in budget_selections
    )

    expansion_recovered_query_count = 0

    for record in audit_records:
        selection = record[
            "budget_selection"
        ]

        resolution = record[
            "final_resolution"
        ]

        if (
            selection is not None
            and selection[
                "gate_decision"
            ]
            == "expansion_required"
            and resolution is not None
            and resolution["supports"]
        ):
            expansion_recovered_query_count += 1

    selected_expanded_item_count = sum(
        selection[
            "selected_expanded_item_count"
        ]
        for selection in budget_selections
    )

    selected_expanded_char_count = sum(
        selection[
            "selected_expanded_char_count"
        ]
        for selection in budget_selections
    )

    return {
        "schema_version": 1,
        "experiment_id": (
            "complex_plan_test_v1_"
            "frozen_top5_budgeted_context_v1"
        ),
        "split": "test",
        "formal_test": True,
        "post_test_retuning_allowed": False,
        "case_dataset": (
            _relative_path(CASE_PATH)
        ),
        "case_dataset_sha256": (
            _sha256_file(CASE_PATH)
        ),
        "calculation_dataset": (
            _relative_path(
                CALCULATION_PATH
            )
        ),
        "calculation_dataset_sha256": (
            _sha256_file(
                CALCULATION_PATH
            )
        ),
        "test_manifest_sha256": (
            _sha256_file(
                TEST_MANIFEST_PATH
            )
        ),
        "frozen_config": (
            _relative_path(
                FROZEN_CONFIG_PATH
            )
        ),
        "frozen_config_sha256": (
            _sha256_file(
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
            run_started_at.isoformat()
        ),
        "run_completed_at": (
            run_completed_at.isoformat()
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
        "planned_query_count": (
            planned_query_count
        ),
        "executed_query_count": (
            executed_query_count
        ),
        "resolved_query_count": (
            resolved_query_count
        ),
        "query_resolution_rate": (
            resolved_query_count
            / planned_query_count
        ),
        "calculation_trace_count": (
            calculation_trace_count
        ),
        "completed_calculation_count": (
            completed_calculation_count
        ),
        "answer_exact_match_count": (
            answer_exact_match_count
        ),
        "fact_reference_exact_match_count": (
            fact_reference_exact_match_count
        ),
        "evidence_reference_exact_match_count": (
            evidence_reference_exact_match_count
        ),
        "calculation_reference_exact_match_count": (
            calculation_reference_exact_match_count
        ),
        "base_resolved_query_count": (
            base_resolved_query_count
        ),
        "expansion_required_query_count": (
            expansion_required_query_count
        ),
        "expansion_recovered_query_count": (
            expansion_recovered_query_count
        ),
        "selected_expanded_item_count": (
            selected_expanded_item_count
        ),
        "selected_expanded_char_count": (
            selected_expanded_char_count
        ),
        "retriever_ids": sorted({
            result.retriever_id
            for result in results
        }),
        "generator_ids": sorted({
            result.generator_id
            for result in results
        }),
        "calculator_ids": sorted({
            result.calculator_id
            for result in results
            if result.calculator_id
            is not None
        }),
        "top_k_values": sorted({
            trace.top_k
            for result in results
            for trace
            in result.retrieval_traces
        }),
        "failure_records": (
            failure_records
        ),
        "result_path": (
            _relative_path(RESULT_PATH)
        ),
        "result_sha256": (
            result_sha256
        ),
        "audit_path": (
            _relative_path(AUDIT_PATH)
        ),
        "audit_sha256": (
            audit_sha256
        ),
    }


def _write_outputs(
    *,
    context: PreflightContext,
    batch: ComplexPlanBatchRun,
    audit_records: list[
        dict[str, object]
    ],
    run_started_at: datetime,
    run_completed_at: datetime,
) -> dict[str, object]:
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
            output_path=(
                RESULT_TEMP_PATH
            ),
        )

        with AUDIT_TEMP_PATH.open(
            "x",
            encoding="utf-8",
        ) as file:
            file.write(audit_text)

        result_sha256 = _sha256_file(
            RESULT_TEMP_PATH
        )

        audit_sha256 = _sha256_file(
            AUDIT_TEMP_PATH
        )

        summary = _build_summary(
            context=context,
            batch=batch,
            audit_records=audit_records,
            run_started_at=(
                run_started_at
            ),
            run_completed_at=(
                run_completed_at
            ),
            result_sha256=(
                result_sha256
            ),
            audit_sha256=(
                audit_sha256
            ),
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
            if path.exists():
                path.unlink()

        raise


def print_formal_result(
    *,
    context: PreflightContext,
    batch: ComplexPlanBatchRun,
    summary: dict[str, object],
) -> None:
    case_by_id = {
        case.case_id: case
        for case in context.cases
    }

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
        f"{summary['planned_query_count']}"
    )
    print(
        "executed_query_count="
        f"{summary['executed_query_count']}"
    )
    print(
        "resolved_query_count="
        f"{summary['resolved_query_count']}"
    )
    print(
        "calculation_trace_count="
        f"{summary['calculation_trace_count']}"
    )
    print(
        "completed_calculation_count="
        f"{summary['completed_calculation_count']}"
    )
    print(
        "answer_exact_match_count="
        f"{summary['answer_exact_match_count']}"
    )
    print(
        "base_resolved_query_count="
        f"{summary['base_resolved_query_count']}"
    )
    print(
        "expansion_required_query_count="
        f"{summary['expansion_required_query_count']}"
    )
    print(
        "expansion_recovered_query_count="
        f"{summary['expansion_recovered_query_count']}"
    )
    print(
        "selected_expanded_item_count="
        f"{summary['selected_expanded_item_count']}"
    )
    print(
        "selected_expanded_char_count="
        f"{summary['selected_expanded_char_count']}"
    )
    print("-" * 80)

    for result in batch.results:
        case = case_by_id[
            result.case_id
        ]

        planned_count = len(
            case.gold_rewrite
            .retrieval_queries
        )

        resolved_count = sum(
            bool(trace.retrieved_fact_ids)
            for trace
            in result.retrieval_traces
        )

        completed_calculations = sum(
            trace.status == "completed"
            for trace
            in result.calculation_traces
        )

        print(
            f"{result.case_id}: "
            f"status={result.status}, "
            f"retrieval="
            f"{resolved_count}/{planned_count}, "
            f"calculations="
            f"{completed_calculations}"
        )

        if result.answer is not None:
            print(
                "  answer="
                f"{result.answer.answer_text}"
            )
        else:
            print(
                "  error_stage="
                f"{result.error_stage}"
            )
            print(
                "  error_message="
                f"{result.error_message}"
            )

    print("-" * 80)
    print(
        f"result_path="
        f"{_relative_path(RESULT_PATH)}"
    )
    print(
        f"audit_path="
        f"{_relative_path(AUDIT_PATH)}"
    )
    print(
        f"summary_path="
        f"{_relative_path(SUMMARY_PATH)}"
    )
    print(
        "frozen_test_configuration_"
        "preserved=true"
    )
    print(
        "formal_test_run_finished=true"
    )


def execute_formal_test(
    context: PreflightContext,
) -> None:
    if context.existing_output_paths:
        raise RuntimeError(
            "Formal Test output already exists"
        )

    retriever = _build_retriever(
        context
    )

    run_started_at = (
        datetime.now(timezone.utc)
    )

    batch, audit_records = (
        _execute_cases(
            context=context,
            retriever=retriever,
        )
    )

    run_completed_at = (
        datetime.now(timezone.utc)
    )

    summary = _write_outputs(
        context=context,
        batch=batch,
        audit_records=audit_records,
        run_started_at=run_started_at,
        run_completed_at=run_completed_at,
    )

    print_formal_result(
        context=context,
        batch=batch,
        summary=summary,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen complex financial "
            "planning Test v1 evaluation."
        )
    )

    mode_group = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    mode_group.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate frozen data, configuration, "
            "registries and index routes without "
            "loading models or running Test."
        ),
    )

    mode_group.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Execute the one-shot frozen formal "
            "Test evaluation."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    context = run_preflight(
        execution_requested=args.execute
    )

    print_preflight_summary(
        context
    )

    if args.preflight_only:
        return

    execute_formal_test(
        context
    )


if __name__ == "__main__":
    main()