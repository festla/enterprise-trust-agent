from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from app.services.competition_dataset import (
    build_competition_question,
    load_competition_qa_excel,
)
from app.services.competition_excel_runtime import (
    load_competition_excel_runtime_resources,
)
from app.workflow.competition_agent_graph import (
    build_competition_agent_graph,
)
from app.workflow.competition_runtime import (
    load_competition_agent_model_resources_from_environment,
    load_competition_agent_runtime_resources,
)


QA_FILE = Path(
    "data/competition/private/qa/QA数据.xlsx"
)

SPLIT_FILE = Path(
    "data/competition/processed/"
    "competition_eval_split_v1.json"
)

ATTACHMENTS_ROOT = Path(
    "data/competition/private/attachments"
)

CORPUS_DIR = Path(
    "data/competition/processed/corpora/"
    "competition_corpus_8cbec682dfce4962"
)

INDEX_DIR = Path(
    "data/competition/processed/indexes/bm25/"
    "competition_bm25_index_07bc5262330bc2a5"
)

OUTPUT_FILE = Path(
    "data/competition/processed/eval/"
    "competition_unified_agent_dev_v1.json"
)


EXPECTED_CASES = 102
EXPECTED_TEXT_CASES = 69
EXPECTED_EXCEL_CASES = 33

EXPECTED_EXCEL_SOLVERS = {
    "lookup": 12,
    "compare": 11,
    "calculation": 10,
}

EXPECTED_TEXT_LLM_REQUESTS = (
    EXPECTED_TEXT_CASES * 2
)

EXPECTED_EXCEL_LLM_REQUESTS = 0

EXPECTED_TOTAL_LLM_REQUESTS = (
    EXPECTED_TEXT_LLM_REQUESTS
    + EXPECTED_EXCEL_LLM_REQUESTS
)


TEXT_TRACE = (
    "source_resolution",
    "retrieval",
    "evidence_assembly",
    "readiness",
    "sufficiency",
    "answer",
    "citation",
)

EXCEL_TRACE = (
    "source_resolution",
    "excel_solver",
    "citation",
)


def _request_count(
    provider,
) -> int:
    value = getattr(
        provider,
        "request_count",
        0,
    )

    return (
        value
        if isinstance(
            value,
            int,
        )
        else 0
    )


def _build_group_summary(
    rows: list[
        dict[str, object]
    ],
    field_name: str,
) -> dict[str, object]:
    groups: dict[
        str,
        list[
            dict[str, object]
        ],
    ] = {}

    for row in rows:
        key = str(
            row[field_name]
        )

        groups.setdefault(
            key,
            [],
        ).append(row)

    summary = {}

    for (
        key,
        group,
    ) in sorted(
        groups.items()
    ):
        count = len(group)

        answered = sum(
            int(
                row["status"]
                == "answered"
            )
            for row in group
        )

        correct = sum(
            int(
                bool(
                    row["correct"]
                )
            )
            for row in group
        )

        citation_valid = sum(
            int(
                bool(
                    row[
                        "citation_valid"
                    ]
                )
            )
            for row in group
        )

        trace_valid = sum(
            int(
                bool(
                    row[
                        "trace_valid"
                    ]
                )
            )
            for row in group
        )

        llm_requests = sum(
            int(
                row[
                    "llm_requests"
                ]
            )
            for row in group
        )

        summary[key] = {
            "case_count": count,
            "answered_count": (
                answered
            ),
            "correct_count": (
                correct
            ),
            "accuracy": (
                correct / count
                if count
                else 0.0
            ),
            "citation_valid_count": (
                citation_valid
            ),
            "trace_valid_count": (
                trace_valid
            ),
            "llm_requests": (
                llm_requests
            ),
        }

    return summary


def main() -> None:
    # ========================================================
    # Dataset / Frozen Split
    # ========================================================

    cases = tuple(
        load_competition_qa_excel(
            QA_FILE
        )
    )

    case_by_id = {
        case.case_id: case
        for case in cases
    }

    split = json.loads(
        SPLIT_FILE.read_text(
            encoding="utf-8"
        )
    )

    dev_case_ids = tuple(
        split["dev_case_ids"]
    )

    dev_cases = [
        case_by_id[case_id]
        for case_id
        in dev_case_ids
    ]

    dev_cases.sort(
        key=lambda case: (
            case.case_id
        )
    )

    # ========================================================
    # Frozen Split Invariants
    # ========================================================

    if (
        len(dev_cases)
        != EXPECTED_CASES
    ):
        raise RuntimeError(
            "Frozen Dev Case 数量"
            "发生变化："
            f"{len(dev_cases)}"
        )

    text_case_count = sum(
        int(
            case.source_type
            in {
                "word",
                "pdf",
            }
        )
        for case in dev_cases
    )

    excel_case_count = sum(
        int(
            case.source_type
            == "excel"
        )
        for case in dev_cases
    )

    if (
        text_case_count
        != EXPECTED_TEXT_CASES
    ):
        raise RuntimeError(
            "Frozen Text Dev Case "
            "数量发生变化："
            f"{text_case_count}"
        )

    if (
        excel_case_count
        != EXPECTED_EXCEL_CASES
    ):
        raise RuntimeError(
            "Frozen Excel Dev Case "
            "数量发生变化："
            f"{excel_case_count}"
        )

    # ========================================================
    # Process Resources
    # ========================================================

    runtime_resources = (
        load_competition_agent_runtime_resources(
            attachments_root=(
                ATTACHMENTS_ROOT
            ),
            corpus_directory=(
                CORPUS_DIR
            ),
            index_directory=(
                INDEX_DIR
            ),
        )
    )

    excel_resources = (
        load_competition_excel_runtime_resources(
            attachments_root=(
                ATTACHMENTS_ROOT
            )
        )
    )

    model_resources = (
        load_competition_agent_model_resources_from_environment()
    )

    graph = (
        build_competition_agent_graph(
            runtime_resources=(
                runtime_resources
            ),
            excel_resources=(
                excel_resources
            ),
            model_resources=(
                model_resources
            ),
        )
    )

    # ========================================================
    # Evaluation
    # ========================================================

    print()
    print(
        "===== Competition Unified "
        "Agent Frozen Dev ====="
    )

    print(
        "Cases:",
        len(dev_cases),
    )

    print(
        "Text:",
        text_case_count,
    )

    print(
        "Excel:",
        excel_case_count,
    )

    print(
        "Sufficiency provider:",
        (
            model_resources
            .sufficiency_provider
            .provider_id
        ),
    )

    print(
        "Answer provider:",
        (
            model_resources
            .answer_provider
            .provider_id
        ),
    )

    print()

    rows: list[
        dict[str, object]
    ] = []

    status_counts = Counter()
    solver_counts = Counter()
    error_counts = Counter()

    correct_count = 0
    citation_valid_count = 0
    trace_valid_count = 0

    text_llm_requests = 0
    excel_llm_requests = 0

    for (
        index,
        case,
    ) in enumerate(
        dev_cases,
        start=1,
    ):
        question = (
            build_competition_question(
                case
            )
        )

        before_sufficiency = (
            _request_count(
                model_resources
                .sufficiency_provider
            )
        )

        before_answer = (
            _request_count(
                model_resources
                .answer_provider
            )
        )

        try:
            # --------------------------------------------
            # Gold-free Agent execution
            # --------------------------------------------

            final_state = (
                graph.invoke(
                    {
                        "question": (
                            question
                        ),
                    }
                )
            )

            after_sufficiency = (
                _request_count(
                    model_resources
                    .sufficiency_provider
                )
            )

            after_answer = (
                _request_count(
                    model_resources
                    .answer_provider
                )
            )

            sufficiency_requests = (
                after_sufficiency
                - before_sufficiency
            )

            answer_requests = (
                after_answer
                - before_answer
            )

            llm_requests = (
                sufficiency_requests
                + answer_requests
            )

            if (
                case.source_type
                == "excel"
            ):
                excel_llm_requests += (
                    llm_requests
                )
            else:
                text_llm_requests += (
                    llm_requests
                )

            result = (
                final_state.get(
                    "result"
                )
            )

            if result is None:
                raise RuntimeError(
                    "Graph 返回后缺少 "
                    "final result"
                )

            status_counts[
                result.status
            ] += 1

            resolution = (
                final_state.get(
                    "source_resolution"
                )
            )

            evidence_bundle = (
                final_state.get(
                    "evidence_bundle"
                )
            )

            excel_runtime = (
                final_state.get(
                    "excel_runtime"
                )
            )

            trace = final_state.get(
                "trace",
                (),
            )

            trace_stages = tuple(
                event.stage
                for event
                in trace
            )

            expected_trace = (
                EXCEL_TRACE
                if (
                    case.source_type
                    == "excel"
                )
                else TEXT_TRACE
            )

            trace_valid = (
                trace_stages
                == expected_trace
            )

            if trace_valid:
                trace_valid_count += 1

            solver_type = None

            if (
                excel_runtime
                is not None
            ):
                solver_type = (
                    excel_runtime
                    .solver_type
                )

                solver_counts[
                    solver_type
                ] += 1

            predicted_option = None

            citation_ids: tuple[
                str,
                ...
            ] = ()

            answer_evidence_ids: tuple[
                str,
                ...
            ] = ()

            bound_evidence_ids: tuple[
                str,
                ...
            ] = ()

            citation_valid = False

            refusal_code = None
            failure_code = None

            if (
                result.status
                == "answered"
            ):
                predicted_option = (
                    result
                    .answer
                    .answer
                )

                citation_ids = (
                    result
                    .answer
                    .citation_ids
                )

                answer_evidence_ids = (
                    result
                    .answer
                    .evidence_ids
                )

                bound_evidence_ids = tuple(
                    citation.evidence_id
                    for citation
                    in (
                        result
                        .citations
                        .citations
                    )
                )

                citation_valid = (
                    bool(
                        citation_ids
                    )
                    and (
                        len(
                            citation_ids
                        )
                        == len(
                            answer_evidence_ids
                        )
                    )
                    and (
                        answer_evidence_ids
                        == bound_evidence_ids
                    )
                )

                if citation_valid:
                    citation_valid_count += 1

            elif (
                result.status
                == "refused"
            ):
                refusal_code = getattr(
                    result,
                    "refusal_code",
                    None,
                )

            elif (
                result.status
                == "failed"
            ):
                failure_code = getattr(
                    result,
                    "failure_code",
                    None,
                )

            # --------------------------------------------
            # Gold enters ONLY here
            # --------------------------------------------

            correct = (
                predicted_option
                == case.answer
            )

            if correct:
                correct_count += 1

            evidence_count = (
                len(
                    evidence_bundle
                    .evidences
                )
                if evidence_bundle
                is not None
                else 0
            )

            source_id = (
                resolution.source_id
                if resolution
                is not None
                else None
            )

            row = {
                "case_id": (
                    case.case_id
                ),
                "source_type": (
                    case.source_type
                ),
                "qa_type": (
                    case.qa_type
                ),
                "status": (
                    result.status
                ),
                "source_id": (
                    source_id
                ),
                "solver_type": (
                    solver_type
                ),
                "predicted_option": (
                    predicted_option
                ),
                "gold_option": (
                    case.answer
                ),
                "correct": (
                    correct
                ),
                "evidence_count": (
                    evidence_count
                ),
                "citation_ids": list(
                    citation_ids
                ),
                "answer_evidence_ids": list(
                    answer_evidence_ids
                ),
                "bound_evidence_ids": list(
                    bound_evidence_ids
                ),
                "citation_valid": (
                    citation_valid
                ),
                "trace_stages": list(
                    trace_stages
                ),
                "trace_valid": (
                    trace_valid
                ),
                "sufficiency_requests": (
                    sufficiency_requests
                ),
                "answer_requests": (
                    answer_requests
                ),
                "llm_requests": (
                    llm_requests
                ),
                "refusal_code": (
                    refusal_code
                ),
                "failure_code": (
                    failure_code
                ),
                "error_type": None,
                "error_message": None,
            }

            route_name = (
                "excel"
                if (
                    case.source_type
                    == "excel"
                )
                else "text"
            )

            print(
                f"[{index:03d}/"
                f"{len(dev_cases):03d}] "
                f"{case.case_id} "
                f"route={route_name} "
                f"type={case.qa_type} "
                f"status={result.status} "
                f"pred={predicted_option} "
                f"gold={case.answer} "
                f"correct={correct} "
                f"citation={citation_valid} "
                f"trace={trace_valid} "
                f"llm="
                f"{sufficiency_requests}"
                "+"
                f"{answer_requests}"
            )

        except Exception as exc:
            after_sufficiency = (
                _request_count(
                    model_resources
                    .sufficiency_provider
                )
            )

            after_answer = (
                _request_count(
                    model_resources
                    .answer_provider
                )
            )

            sufficiency_requests = (
                after_sufficiency
                - before_sufficiency
            )

            answer_requests = (
                after_answer
                - before_answer
            )

            llm_requests = (
                sufficiency_requests
                + answer_requests
            )

            if (
                case.source_type
                == "excel"
            ):
                excel_llm_requests += (
                    llm_requests
                )
            else:
                text_llm_requests += (
                    llm_requests
                )

            error_type = (
                type(exc).__name__
            )

            error_counts[
                error_type
            ] += 1

            status_counts[
                "exception"
            ] += 1

            row = {
                "case_id": (
                    case.case_id
                ),
                "source_type": (
                    case.source_type
                ),
                "qa_type": (
                    case.qa_type
                ),
                "status": (
                    "exception"
                ),
                "source_id": None,
                "solver_type": None,
                "predicted_option": None,
                "gold_option": (
                    case.answer
                ),
                "correct": False,
                "evidence_count": 0,
                "citation_ids": [],
                "answer_evidence_ids": [],
                "bound_evidence_ids": [],
                "citation_valid": False,
                "trace_stages": [],
                "trace_valid": False,
                "sufficiency_requests": (
                    sufficiency_requests
                ),
                "answer_requests": (
                    answer_requests
                ),
                "llm_requests": (
                    llm_requests
                ),
                "refusal_code": None,
                "failure_code": None,
                "error_type": (
                    error_type
                ),
                "error_message": (
                    str(exc)
                ),
            }

            print(
                f"[{index:03d}/"
                f"{len(dev_cases):03d}] "
                f"{case.case_id} "
                "EXCEPTION "
                f"{error_type}: "
                f"{exc}"
            )

        rows.append(row)

    # ========================================================
    # Summary
    # ========================================================

    answered_count = (
        status_counts[
            "answered"
        ]
    )

    total_llm_requests = (
        text_llm_requests
        + excel_llm_requests
    )

    overall_accuracy = (
        correct_count
        / EXPECTED_CASES
    )

    summary = {
        "case_count": (
            EXPECTED_CASES
        ),
        "text_case_count": (
            text_case_count
        ),
        "excel_case_count": (
            excel_case_count
        ),
        "answered_count": (
            answered_count
        ),
        "correct_count": (
            correct_count
        ),
        "accuracy": (
            overall_accuracy
        ),
        "citation_valid_count": (
            citation_valid_count
        ),
        "citation_valid_rate": (
            citation_valid_count
            / EXPECTED_CASES
        ),
        "trace_valid_count": (
            trace_valid_count
        ),
        "trace_valid_rate": (
            trace_valid_count
            / EXPECTED_CASES
        ),
        "text_llm_requests": (
            text_llm_requests
        ),
        "excel_llm_requests": (
            excel_llm_requests
        ),
        "total_llm_requests": (
            total_llm_requests
        ),
        "status_counts": dict(
            sorted(
                status_counts.items()
            )
        ),
        "solver_counts": dict(
            sorted(
                solver_counts.items()
            )
        ),
        "error_counts": dict(
            sorted(
                error_counts.items()
            )
        ),
        "excel_workbook_cache_size": (
            len(
                excel_resources
                .workbook_cache
            )
        ),
        "by_source_type": (
            _build_group_summary(
                rows,
                "source_type",
            )
        ),
        "by_qa_type": (
            _build_group_summary(
                rows,
                "qa_type",
            )
        ),
    }

    # ========================================================
    # Console Summary
    # ========================================================

    print()
    print(
        "===== Unified Agent Summary ====="
    )

    print(
        "Answered:",
        f"{answered_count}/"
        f"{EXPECTED_CASES}",
    )

    print(
        "Correct:",
        f"{correct_count}/"
        f"{EXPECTED_CASES}",
    )

    print(
        "Accuracy:",
        f"{overall_accuracy:.4f}",
    )

    print(
        "Citation valid:",
        f"{citation_valid_count}/"
        f"{EXPECTED_CASES}",
    )

    print(
        "Trace valid:",
        f"{trace_valid_count}/"
        f"{EXPECTED_CASES}",
    )

    print(
        "Status:",
        dict(
            status_counts
        ),
    )

    print(
        "Errors:",
        dict(
            error_counts
        ),
    )

    print()
    print(
        "LLM Requests:"
    )

    print(
        "  Text:",
        text_llm_requests,
        "/",
        EXPECTED_TEXT_LLM_REQUESTS,
    )

    print(
        "  Excel:",
        excel_llm_requests,
        "/",
        EXPECTED_EXCEL_LLM_REQUESTS,
    )

    print(
        "  Total:",
        total_llm_requests,
        "/",
        EXPECTED_TOTAL_LLM_REQUESTS,
    )

    print()
    print(
        "Excel Solvers:",
        dict(
            solver_counts
        ),
    )

    print(
        "Excel Workbook cache:",
        len(
            excel_resources
            .workbook_cache
        ),
    )

    print()
    print(
        "By source type:"
    )

    for (
        source_type,
        group,
    ) in (
        summary[
            "by_source_type"
        ].items()
    ):
        print(
            f"  {source_type}: "
            f"{group['correct_count']}/"
            f"{group['case_count']} "
            f"accuracy="
            f"{group['accuracy']:.4f} "
            f"llm="
            f"{group['llm_requests']}"
        )

    print()
    print(
        "By QA type:"
    )

    for (
        qa_type,
        group,
    ) in (
        summary[
            "by_qa_type"
        ].items()
    ):
        print(
            f"  {qa_type}: "
            f"{group['correct_count']}/"
            f"{group['case_count']} "
            f"accuracy="
            f"{group['accuracy']:.4f} "
            f"llm="
            f"{group['llm_requests']}"
        )

    # ========================================================
    # Failure Analysis Preview
    # ========================================================

    problematic = [
        row
        for row in rows
        if (
            not bool(
                row["correct"]
            )
            or not bool(
                row["citation_valid"]
            )
            or not bool(
                row["trace_valid"]
            )
            or (
                row["status"]
                != "answered"
            )
        )
    ]

    if problematic:
        print()
        print(
            "Problematic Cases:"
        )

        for row in problematic:
            print(
                " ",
                row["case_id"],
                "source=",
                row[
                    "source_type"
                ],
                "qa_type=",
                row[
                    "qa_type"
                ],
                "status=",
                row[
                    "status"
                ],
                "pred=",
                row[
                    "predicted_option"
                ],
                "gold=",
                row[
                    "gold_option"
                ],
                "citation=",
                row[
                    "citation_valid"
                ],
                "trace=",
                row[
                    "trace_valid"
                ],
                "llm=",
                row[
                    "llm_requests"
                ],
                "error=",
                row[
                    "error_type"
                ],
            )

    # ========================================================
    # Artifact
    # ========================================================

    payload = {
        "schema_version": 1,
        "evaluation_version": (
            "competition_unified_"
            "agent_dev_v1"
        ),
        "providers": {
            "sufficiency": (
                model_resources
                .sufficiency_provider
                .provider_id
            ),
            "answer": (
                model_resources
                .answer_provider
                .provider_id
            ),
        },
        "summary": (
            summary
        ),
        "cases": rows,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Saved:",
        OUTPUT_FILE,
    )

    # ========================================================
    # Strict Acceptance Gate
    # ========================================================

    if error_counts:
        raise RuntimeError(
            "Unified Agent Dev "
            "存在未处理异常"
        )

    if (
        answered_count
        != EXPECTED_CASES
    ):
        raise RuntimeError(
            "Unified Agent "
            "没有达到 102/102 answered"
        )

    if (
        correct_count
        != EXPECTED_CASES
    ):
        raise RuntimeError(
            "Unified Agent "
            "没有达到 102/102 correct"
        )

    if (
        citation_valid_count
        != EXPECTED_CASES
    ):
        raise RuntimeError(
            "Unified Agent Citation "
            "没有达到 102/102"
        )

    if (
        trace_valid_count
        != EXPECTED_CASES
    ):
        raise RuntimeError(
            "Unified Agent Route Trace "
            "没有达到 102/102"
        )

    if (
        text_llm_requests
        != EXPECTED_TEXT_LLM_REQUESTS
    ):
        raise RuntimeError(
            "Text Agent LLM 调用数量异常："
            f"{text_llm_requests} "
            f"!= "
            f"{EXPECTED_TEXT_LLM_REQUESTS}"
        )

    if (
        excel_llm_requests
        != 0
    ):
        raise RuntimeError(
            "Excel Agent 不应该调用 LLM："
            f"{excel_llm_requests}"
        )

    if (
        total_llm_requests
        != EXPECTED_TOTAL_LLM_REQUESTS
    ):
        raise RuntimeError(
            "Unified Agent 总 LLM "
            "调用数量异常："
            f"{total_llm_requests} "
            f"!= "
            f"{EXPECTED_TOTAL_LLM_REQUESTS}"
        )

    for (
        solver_type,
        expected_count,
    ) in (
        EXPECTED_EXCEL_SOLVERS.items()
    ):
        if (
            solver_counts[
                solver_type
            ]
            != expected_count
        ):
            raise RuntimeError(
                "Excel Solver 路由数量异常："
                f"{solver_type}="
                f"{solver_counts[solver_type]} "
                f"!= {expected_count}"
            )

    print()
    print(
        "UNIFIED AGENT DEV ACCEPTED"
    )


if __name__ == "__main__":
    main()