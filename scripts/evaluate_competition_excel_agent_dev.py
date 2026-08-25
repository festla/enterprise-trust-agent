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
    CompetitionAgentModelResources,
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
    "competition_excel_agent_dev_v1.json"
)


EXPECTED_COUNTS = {
    "表格取数": 12,
    "表格比较": 11,
    "表格计算": 10,
}

EXPECTED_TOTAL = 33


class NeverCalledSufficiencyProvider:
    """
    Excel Agent V1 不应该调用 LLM Judge。
    """

    provider_id = (
        "eval:excel-never-called-sufficiency"
    )

    def assess(
        self,
        **kwargs,
    ):
        raise AssertionError(
            "Excel Agent 不应该调用 "
            "Sufficiency Provider"
        )


class NeverCalledAnswerProvider:
    """
    Excel Agent V1 不应该调用 LLM Answer。
    """

    provider_id = (
        "eval:excel-never-called-answer"
    )

    def generate(
        self,
        **kwargs,
    ):
        raise AssertionError(
            "Excel Agent 不应该调用 "
            "Answer Provider"
        )


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

    excel_cases = [
        case_by_id[case_id]
        for case_id
        in dev_case_ids
        if (
            case_by_id[
                case_id
            ].source_type
            == "excel"
        )
    ]

    excel_cases.sort(
        key=lambda case: (
            case.case_id
        )
    )

    # ========================================================
    # Frozen Split Invariants
    # ========================================================

    if (
        len(excel_cases)
        != EXPECTED_TOTAL
    ):
        raise RuntimeError(
            "Frozen Excel Dev Case "
            "数量发生变化："
            f"{len(excel_cases)}"
        )

    actual_counts = Counter(
        case.qa_type
        for case
        in excel_cases
    )

    for (
        qa_type,
        expected,
    ) in EXPECTED_COUNTS.items():
        if (
            actual_counts[
                qa_type
            ]
            != expected
        ):
            raise RuntimeError(
                f"{qa_type} 数量发生变化："
                f"{actual_counts[qa_type]}"
                f" != {expected}"
            )

    # ========================================================
    # Runtime Resources
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
        CompetitionAgentModelResources(
            sufficiency_provider=(
                NeverCalledSufficiencyProvider()
            ),
            answer_provider=(
                NeverCalledAnswerProvider()
            ),
        )
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
        "===== Competition Excel "
        "Agent Frozen Dev ====="
    )

    print(
        "Cases:",
        len(excel_cases),
    )

    print(
        "By QA type:",
        dict(actual_counts),
    )

    print()

    results: list[
        dict[str, object]
    ] = []

    correct_count = 0

    correct_by_type = Counter()

    status_counts = Counter()

    solver_counts = Counter()

    error_counts = Counter()

    citation_valid_count = 0

    trace_valid_count = 0

    expected_trace = (
        "source_resolution",
        "excel_solver",
        "citation",
    )

    for (
        index,
        case,
    ) in enumerate(
        excel_cases,
        start=1,
    ):
        question = (
            build_competition_question(
                case
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

            result = (
                final_state.get(
                    "result"
                )
            )

            if result is None:
                raise RuntimeError(
                    "Graph 返回后 "
                    "缺少 final result"
                )

            status_counts[
                result.status
            ] += 1

            excel_runtime = (
                final_state.get(
                    "excel_runtime"
                )
            )

            evidence_bundle = (
                final_state.get(
                    "evidence_bundle"
                )
            )

            resolution = (
                final_state.get(
                    "source_resolution"
                )
            )

            trace = (
                final_state.get(
                    "trace",
                    (),
                )
            )

            trace_stages = tuple(
                event.stage
                for event
                in trace
            )

            trace_valid = (
                trace_stages
                == expected_trace
            )

            if trace_valid:
                trace_valid_count += 1

            predicted_option = None

            citation_ids: tuple[
                str,
                ...
            ] = ()

            evidence_ids: tuple[
                str,
                ...
            ] = ()

            bound_evidence_ids: tuple[
                str,
                ...
            ] = ()

            citation_valid = False

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

                evidence_ids = (
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
                    bool(citation_ids)
                    and (
                        len(citation_ids)
                        == len(
                            evidence_ids
                        )
                    )
                    and (
                        evidence_ids
                        == bound_evidence_ids
                    )
                )

                if citation_valid:
                    citation_valid_count += 1

            # --------------------------------------------
            # Gold only enters here
            # --------------------------------------------

            correct = (
                predicted_option
                == case.answer
            )

            if correct:
                correct_count += 1

                correct_by_type[
                    case.qa_type
                ] += 1

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
                "evidence_ids": list(
                    evidence_ids
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
                "error_type": None,
                "error_message": None,
            }

            print(
                f"[{index:02d}/"
                f"{len(excel_cases):02d}] "
                f"{case.case_id} "
                f"{case.qa_type} "
                f"solver={solver_type} "
                f"status={result.status} "
                f"pred={predicted_option} "
                f"gold={case.answer} "
                f"correct={correct} "
                f"evidence={evidence_count} "
                f"citations="
                f"{len(citation_ids)} "
                f"trace={trace_valid}"
            )

        except Exception as exc:
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
                "evidence_ids": [],
                "bound_evidence_ids": [],
                "citation_valid": False,
                "trace_stages": [],
                "trace_valid": False,
                "error_type": (
                    error_type
                ),
                "error_message": (
                    str(exc)
                ),
            }

            print(
                f"[{index:02d}/"
                f"{len(excel_cases):02d}] "
                f"{case.case_id} "
                "EXCEPTION "
                f"{error_type}: "
                f"{exc}"
            )

        results.append(
            row
        )

    # ========================================================
    # Summary
    # ========================================================

    by_qa_type = {}

    for (
        qa_type,
        expected,
    ) in EXPECTED_COUNTS.items():
        subset = [
            row
            for row in results
            if (
                row["qa_type"]
                == qa_type
            )
        ]

        subset_correct = sum(
            int(
                bool(
                    row["correct"]
                )
            )
            for row in subset
        )

        by_qa_type[
            qa_type
        ] = {
            "case_count": (
                len(subset)
            ),
            "expected_count": (
                expected
            ),
            "correct_count": (
                subset_correct
            ),
            "accuracy": (
                subset_correct
                / len(subset)
                if subset
                else 0.0
            ),
        }

    case_count = len(
        excel_cases
    )

    accuracy = (
        correct_count
        / case_count
    )

    answered_count = (
        status_counts[
            "answered"
        ]
    )

    summary = {
        "case_count": (
            case_count
        ),
        "answered_count": (
            answered_count
        ),
        "correct_count": (
            correct_count
        ),
        "accuracy": (
            accuracy
        ),
        "citation_valid_count": (
            citation_valid_count
        ),
        "citation_valid_rate": (
            citation_valid_count
            / answered_count
            if answered_count
            else 0.0
        ),
        "trace_valid_count": (
            trace_valid_count
        ),
        "trace_valid_rate": (
            trace_valid_count
            / case_count
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
        "workbook_cache_size": len(
            excel_resources
            .workbook_cache
        ),
        "by_qa_type": (
            by_qa_type
        ),
    }

    print()
    print(
        "===== Excel Agent Summary ====="
    )

    print(
        "Answered:",
        f"{answered_count}/"
        f"{case_count}",
    )

    print(
        "Correct:",
        f"{correct_count}/"
        f"{case_count}",
    )

    print(
        "Accuracy:",
        f"{accuracy:.4f}",
    )

    print(
        "Citation valid:",
        f"{citation_valid_count}/"
        f"{answered_count}",
    )

    print(
        "Trace valid:",
        f"{trace_valid_count}/"
        f"{case_count}",
    )

    print(
        "Status:",
        dict(
            status_counts
        ),
    )

    print(
        "Solvers:",
        dict(
            solver_counts
        ),
    )

    print(
        "Errors:",
        dict(
            error_counts
        ),
    )

    print(
        "Workbook cache size:",
        len(
            excel_resources
            .workbook_cache
        ),
    )

    print()
    print(
        "By QA type:"
    )

    for (
        qa_type,
        group,
    ) in (
        by_qa_type.items()
    ):
        print(
            f"  {qa_type}: "
            f"{group['correct_count']}/"
            f"{group['case_count']} "
            f"accuracy="
            f"{group['accuracy']:.4f}"
        )

    problematic = [
        row
        for row in results
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
                row["qa_type"],
                "status=",
                row["status"],
                "pred=",
                row[
                    "predicted_option"
                ],
                "gold=",
                row[
                    "gold_option"
                ],
                "citation_valid=",
                row[
                    "citation_valid"
                ],
                "trace_valid=",
                row[
                    "trace_valid"
                ],
                "error=",
                row[
                    "error_type"
                ],
            )

    # ========================================================
    # Save Artifact
    # ========================================================

    payload = {
        "schema_version": 1,
        "evaluation_version": (
            "competition_excel_"
            "agent_dev_v1"
        ),
        "summary": summary,
        "cases": results,
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
            "Excel Agent Dev "
            "存在执行异常"
        )

    if (
        answered_count
        != EXPECTED_TOTAL
    ):
        raise RuntimeError(
            "Excel Agent "
            "没有 33/33 answered"
        )

    if (
        correct_count
        != EXPECTED_TOTAL
    ):
        raise RuntimeError(
            "Excel Agent "
            "没有保持 33/33"
        )

    if (
        citation_valid_count
        != EXPECTED_TOTAL
    ):
        raise RuntimeError(
            "Excel Agent Citation "
            "没有达到 33/33"
        )

    if (
        trace_valid_count
        != EXPECTED_TOTAL
    ):
        raise RuntimeError(
            "Excel Agent Route "
            "没有达到 33/33"
        )

    for (
        qa_type,
        expected,
    ) in EXPECTED_COUNTS.items():
        if (
            correct_by_type[
                qa_type
            ]
            != expected
        ):
            raise RuntimeError(
                f"{qa_type} "
                "Agent Regression："
                f"{correct_by_type[qa_type]}/"
                f"{expected}"
            )

    print()
    print(
        "EXCEL AGENT DEV ACCEPTED"
    )


if __name__ == "__main__":
    main()