from __future__ import annotations

import argparse
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
from app.rag.bailian_answer_provider import (
    BailianAnswerProviderRequestError,
)

QA_FILE = Path(
    "data/competition/private/qa/QA数据.xlsx"
)

SPLIT_FILE = Path(
    "data/competition/processed/"
    "competition_eval_split_v1.json"
)

ATTACHMENTS_ROOT = Path(
    "data/competition/deploy/attachments"
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
    "competition_unified_agent_final_v1.json"
)

SCORE_FILE = Path(
    "data/competition/processed/eval/"
    "competition_unified_agent_final_score_v1.json"
)

EXPECTED_CASES = 198

RUNTIME_FREEZE_COMMIT = (
    "30b4d61e2790211f093a3068ae0ba44204154330"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--route",
        choices=(
            "all",
            "excel",
            "text",
        ),
        default="all",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--score",
        action="store_true",
    )

    return parser.parse_args()


def _request_count(
    provider,
) -> int:
    value = getattr(
        provider,
        "request_count",
        0,
    )

    if isinstance(value, int):
        return value

    return 0


def _load_dataset():
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

    test_case_ids = tuple(
        split["test_case_ids"]
    )

    if (
        len(test_case_ids)
        != EXPECTED_CASES
    ):
        raise RuntimeError(
            "Frozen Test Case 数量异常: "
            f"{len(test_case_ids)}"
        )

    test_cases = [
        case_by_id[case_id]
        for case_id
        in test_case_ids
    ]

    test_cases.sort(
        key=lambda case: case.case_id
    )

    return (
        case_by_id,
        test_case_ids,
        test_cases,
    )


def _save_payload(
    *,
    rows: list[dict[str, object]],
    sufficiency_provider_id: str,
    answer_provider_id: str,
) -> None:
    payload = {
        "schema_version": 1,
        "evaluation_version": (
            "competition_unified_"
            "agent_final_v1"
        ),
        "runtime_freeze_commit": (
            RUNTIME_FREEZE_COMMIT
        ),
        "providers": {
            "sufficiency": (
                sufficiency_provider_id
            ),
            "answer": (
                answer_provider_id
            ),
        },
        "completed_case_count": (
            len(rows)
        ),
        "cases": rows,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_file = (
        OUTPUT_FILE.with_suffix(
            ".json.tmp"
        )
    )

    temp_file.write_text(
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

    temp_file.replace(
        OUTPUT_FILE
    )


def _load_resume_rows(
    *,
    resume: bool,
    sufficiency_provider_id: str,
    answer_provider_id: str,
) -> list[dict[str, object]]:
    if not OUTPUT_FILE.exists():
        return []

    if not resume:
        raise RuntimeError(
            "Final artifact 已存在。"
            "如需继续，请使用 --resume；"
            "如需整轮重跑，请先手动删除 artifact。"
        )

    payload = json.loads(
        OUTPUT_FILE.read_text(
            encoding="utf-8"
        )
    )

    if (
        payload.get(
            "runtime_freeze_commit"
        )
        != RUNTIME_FREEZE_COMMIT
    ):
        raise RuntimeError(
            "Resume artifact 的 "
            "runtime freeze commit 不一致"
        )

    providers = payload.get(
        "providers",
        {},
    )

    if (
        providers.get(
            "sufficiency"
        )
        != sufficiency_provider_id
    ):
        raise RuntimeError(
            "Resume artifact 的 "
            "Sufficiency Provider 不一致"
        )

    if (
        providers.get(
            "answer"
        )
        != answer_provider_id
    ):
        raise RuntimeError(
            "Resume artifact 的 "
            "Answer Provider 不一致"
        )

    rows = list(
        payload.get(
            "cases",
            [],
        )
    )

    case_ids = [
        str(row["case_id"])
        for row in rows
    ]

    if (
        len(case_ids)
        != len(set(case_ids))
    ):
        raise RuntimeError(
            "Final artifact 存在重复 case_id"
        )

    return rows


def _citation_valid(
    result,
) -> tuple[
    bool,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    if (
        result.status
        != "answered"
    ):
        return (
            False,
            (),
            (),
            (),
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

    valid = (
        bool(citation_ids)
        and (
            len(citation_ids)
            == len(
                answer_evidence_ids
            )
        )
        and (
            answer_evidence_ids
            == bound_evidence_ids
        )
    )

    return (
        valid,
        citation_ids,
        answer_evidence_ids,
        bound_evidence_ids,
    )


def run_inference(
    *,
    route: str,
    resume: bool,
) -> None:
    (
        _,
        _,
        test_cases,
    ) = _load_dataset()

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

    sufficiency_provider_id = (
        model_resources
        .sufficiency_provider
        .provider_id
    )

    answer_provider_id = (
        model_resources
        .answer_provider
        .provider_id
    )

    print()
    print(
        "===== Competition Unified "
        "Agent FINAL ====="
    )
    print(
        "Runtime freeze:",
        RUNTIME_FREEZE_COMMIT,
    )
    print(
        "Sufficiency:",
        sufficiency_provider_id,
    )
    print(
        "Answer:",
        answer_provider_id,
    )
    print(
        "Route:",
        route,
    )
    print()

    if (
        "qwen3.8-max"
        not in sufficiency_provider_id
        or
        "qwen3.8-max"
        not in answer_provider_id
    ):
        raise RuntimeError(
            "FINAL evaluation 要求 "
            "Sufficiency 和 Answer "
            "都使用 qwen3.8-max"
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

    rows = _load_resume_rows(
        resume=resume,
        sufficiency_provider_id=(
            sufficiency_provider_id
        ),
        answer_provider_id=(
            answer_provider_id
        ),
    )

    completed_ids = {
        str(row["case_id"])
        for row in rows
    }

    selected_cases = []

    for case in test_cases:
        if (
            route == "excel"
            and case.source_type
            != "excel"
        ):
            continue

        if (
            route == "text"
            and case.source_type
            not in {
                "word",
                "pdf",
            }
        ):
            continue

        selected_cases.append(
            case
        )

    pending_cases = [
        case
        for case in selected_cases
        if case.case_id
        not in completed_ids
    ]

    print(
        "Already completed:",
        len(rows),
    )
    print(
        "Selected:",
        len(selected_cases),
    )
    print(
        "Pending:",
        len(pending_cases),
    )
    print()

    for (
        index,
        case,
    ) in enumerate(
        pending_cases,
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
            final_state = (
                graph.invoke(
                    {
                        "question": (
                            question
                        )
                    }
                )
            )

        except Exception as exc:
            # ================================================
            # Provider / network / quota failure
            #
            # 这类异常不是单个 held-out case 的业务失败。
            # 当前 case 不落盘，立即停止；
            # 恢复网络 / 额度后使用 --resume 继续。
            # ================================================

            if isinstance(
                exc,
                BailianAnswerProviderRequestError,
            ):
                print()
                print(
                    "FINAL INFERENCE PAUSED"
                )
                print(
                    "Case:",
                    case.case_id,
                )
                print(
                    "Provider request failed:",
                    str(exc),
                )
                print(
                    "This case was NOT saved."
                )
                print(
                    "Completed cases remain safe."
                )
                print(
                    "Resume with the same model "
                    "using --resume."
                )

                raise

            # ================================================
            # Final held-out evaluation:
            #
            # 单个 Case 的 Runtime Exception 属于该 Case
            # 的最终失败结果，不应该终止剩余 198 测试。
            #
            # 注意：
            # 这里只增强 evaluator 的 fault isolation，
            # 不修改 Agent / Solver 行为。
            # ================================================

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

            error_type = (
                type(exc).__name__
            )

            error_message = str(
                exc
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
                    "exception"
                ),
                "source_id": None,
                "solver_type": None,
                "predicted_option": None,
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
                    error_message
                ),
            }

            rows.append(
                row
            )

            _save_payload(
                rows=rows,
                sufficiency_provider_id=(
                    sufficiency_provider_id
                ),
                answer_provider_id=(
                    answer_provider_id
                ),
            )

            print(
                f"[{index:03d}/"
                f"{len(pending_cases):03d}] "
                f"{case.case_id} "
                f"source={case.source_type} "
                f"type={case.qa_type} "
                "status=exception "
                f"error={error_type} "
                f"llm="
                f"{sufficiency_requests}"
                "+"
                f"{answer_requests} "
                f"total_saved={len(rows)}"
            )

            continue

        # --------------------------------------------
        # Provider / network / quota failure
        #
        # 这类错误不是单个 held-out Case 的业务失败，
        # 不能写入 Final artifact。
        #
        # 停止整个 evaluation，保留之前逐题落盘结果；
        # 恢复网络或额度后使用 --resume 继续。
        # --------------------------------------------


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

        result = (
            final_state.get(
                "result"
            )
        )

        if result is None:
            raise RuntimeError(
                "Graph 返回后缺少 result"
            )

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
            for event in trace
        )

        expected_trace = (
            EXCEL_TRACE
            if case.source_type
            == "excel"
            else TEXT_TRACE
        )

        trace_valid = (
            trace_stages
            == expected_trace
        )

        (
            citation_valid,
            citation_ids,
            answer_evidence_ids,
            bound_evidence_ids,
        ) = _citation_valid(
            result
        )

        predicted_option = None
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

        solver_type = None

        if (
            excel_runtime
            is not None
        ):
            solver_type = (
                excel_runtime
                .solver_type
            )

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
        }

        rows.append(
            row
        )

        _save_payload(
            rows=rows,
            sufficiency_provider_id=(
                sufficiency_provider_id
            ),
            answer_provider_id=(
                answer_provider_id
            ),
        )

        print(
            f"[{index:03d}/"
            f"{len(pending_cases):03d}] "
            f"{case.case_id} "
            f"source={case.source_type} "
            f"type={case.qa_type} "
            f"status={result.status} "
            f"pred={predicted_option} "
            f"citation={citation_valid} "
            f"trace={trace_valid} "
            f"llm="
            f"{sufficiency_requests}"
            "+"
            f"{answer_requests} "
            f"total_saved={len(rows)}"
        )

    print()
    print(
        "FINAL INFERENCE ROUTE COMPLETE"
    )
    print(
        "Total saved:",
        len(rows),
    )
    print(
        "Saved:",
        OUTPUT_FILE,
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
        ).append(
            row
        )

    summary = {}

    for key, group in sorted(
        groups.items()
    ):
        count = len(group)

        correct = sum(
            int(
                bool(
                    row["correct"]
                )
            )
            for row in group
        )

        summary[key] = {
            "case_count": count,
            "correct_count": (
                correct
            ),
            "accuracy": (
                correct / count
                if count
                else 0.0
            ),
        }

    return summary


def score_final() -> None:
    (
        case_by_id,
        test_case_ids,
        _,
    ) = _load_dataset()

    if not OUTPUT_FILE.exists():
        raise RuntimeError(
            "Final inference artifact 不存在"
        )

    payload = json.loads(
        OUTPUT_FILE.read_text(
            encoding="utf-8"
        )
    )

    rows = list(
        payload.get(
            "cases",
            [],
        )
    )

    if (
        len(rows)
        != EXPECTED_CASES
    ):
        raise RuntimeError(
            "Final inference 尚未完成："
            f"{len(rows)}/{EXPECTED_CASES}"
        )

    row_by_id = {
        str(row["case_id"]): row
        for row in rows
    }

    if (
        set(row_by_id)
        != set(test_case_ids)
    ):
        raise RuntimeError(
            "Final artifact 与 Frozen Test Split 不一致"
        )

    scored_rows = []

    for case_id in test_case_ids:
        case = (
            case_by_id[
                case_id
            ]
        )

        row = dict(
            row_by_id[
                case_id
            ]
        )

        correct = (
            row.get(
                "predicted_option"
            )
            == case.answer
        )

        row[
            "gold_option"
        ] = case.answer

        row[
            "correct"
        ] = correct

        scored_rows.append(
            row
        )

    status_counts = Counter(
        str(row["status"])
        for row in scored_rows
    )

    solver_counts = Counter(
        str(row["solver_type"])
        for row in scored_rows
        if row.get(
            "solver_type"
        )
        is not None
    )

    correct_count = sum(
        int(
            bool(
                row["correct"]
            )
        )
        for row in scored_rows
    )

    citation_valid_count = sum(
        int(
            bool(
                row[
                    "citation_valid"
                ]
            )
        )
        for row in scored_rows
    )

    trace_valid_count = sum(
        int(
            bool(
                row[
                    "trace_valid"
                ]
            )
        )
        for row in scored_rows
    )

    total_llm_requests = sum(
        int(
            row[
                "llm_requests"
            ]
        )
        for row in scored_rows
    )

    text_llm_requests = sum(
        int(
            row[
                "llm_requests"
            ]
        )
        for row in scored_rows
        if row[
            "source_type"
        ]
        in {
            "word",
            "pdf",
        }
    )

    excel_llm_requests = sum(
        int(
            row[
                "llm_requests"
            ]
        )
        for row in scored_rows
        if row[
            "source_type"
        ]
        == "excel"
    )

    summary = {
        "case_count": (
            EXPECTED_CASES
        ),
        "answered_count": (
            status_counts[
                "answered"
            ]
        ),
        "refused_count": (
            status_counts[
                "refused"
            ]
        ),
        "failed_count": (
            status_counts[
                "failed"
            ]
            + status_counts[
                "exception"
            ]
        ),
        "exception_count": (
            status_counts[
                "exception"
            ]
        ),
        "correct_count": (
            correct_count
        ),
        "accuracy": (
            correct_count
            / EXPECTED_CASES
        ),
        "citation_valid_count": (
            citation_valid_count
        ),
        "trace_valid_count": (
            trace_valid_count
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
        "by_source_type": (
            _build_group_summary(
                scored_rows,
                "source_type",
            )
        ),
        "by_qa_type": (
            _build_group_summary(
                scored_rows,
                "qa_type",
            )
        ),
    }

    score_payload = {
        "schema_version": 1,
        "evaluation_version": (
            "competition_unified_"
            "agent_final_score_v1"
        ),
        "runtime_freeze_commit": (
            RUNTIME_FREEZE_COMMIT
        ),
        "providers": (
            payload.get(
                "providers"
            )
        ),
        "summary": (
            summary
        ),
        "cases": (
            scored_rows
        ),
    }

    SCORE_FILE.write_text(
        json.dumps(
            score_payload,
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
        "===== FINAL HELD-OUT SCORE ====="
    )

    print(
        "Cases:",
        EXPECTED_CASES,
    )

    print(
        "Answered:",
        status_counts[
            "answered"
        ],
    )

    print(
        "Refused:",
        status_counts[
            "refused"
        ],
    )

    failed_count = (
        status_counts[
            "failed"
        ]
        + status_counts[
            "exception"
        ]
    )

    print(
        "Failed:",
        failed_count,
    )

    print(
        "Exceptions:",
        status_counts[
            "exception"
        ],
    )

    print(
        "Correct:",
        f"{correct_count}/"
        f"{EXPECTED_CASES}",
    )

    print(
        "Accuracy:",
        f"{correct_count / EXPECTED_CASES:.4f}",
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

    print()
    print(
        "LLM Requests:"
    )

    print(
        "  Text:",
        text_llm_requests,
    )

    print(
        "  Excel:",
        excel_llm_requests,
    )

    print(
        "  Total:",
        total_llm_requests,
    )

    print()
    print(
        "By source type:"
    )

    for source_type, group in (
        summary[
            "by_source_type"
        ].items()
    ):
        print(
            f"  {source_type}: "
            f"{group['correct_count']}/"
            f"{group['case_count']} "
            f"accuracy="
            f"{group['accuracy']:.4f}"
        )

    print()
    print(
        "By QA type:"
    )

    for qa_type, group in (
        summary[
            "by_qa_type"
        ].items()
    ):
        print(
            f"  {qa_type}: "
            f"{group['correct_count']}/"
            f"{group['case_count']} "
            f"accuracy="
            f"{group['accuracy']:.4f}"
        )

    print()
    print(
        "Saved:",
        SCORE_FILE,
    )


def main() -> None:
    args = parse_args()

    if args.score:
        score_final()
        return

    run_inference(
        route=args.route,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()