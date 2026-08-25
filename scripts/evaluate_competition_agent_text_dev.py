from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from app.schemas.competition import (
    CompetitionQaCase,
    CompetitionQuestion,
)
from app.services.competition_dataset import (
    load_competition_qa_excel,
)
from app.workflow.competition_agent_graph import (
    build_competition_agent_graph,
)
from app.workflow.competition_runtime import (
    load_competition_agent_model_resources_from_environment,
    load_competition_agent_runtime_resources,
)
from scripts.evaluate_competition_bm25_dev import (
    load_dev_case_ids,
)


DEFAULT_QA_FILE = Path(
    "data/competition/private/qa/QA数据.xlsx"
)

DEFAULT_SPLIT_FILE = Path(
    "data/competition/processed/"
    "competition_eval_split_v1.json"
)

DEFAULT_ATTACHMENTS_ROOT = Path(
    "data/competition/private/attachments"
)

DEFAULT_CORPUS_DIR = Path(
    "data/competition/processed/corpora/"
    "competition_corpus_8cbec682dfce4962"
)

DEFAULT_INDEX_DIR = Path(
    "data/competition/processed/indexes/bm25/"
    "competition_bm25_index_07bc5262330bc2a5"
)

DEFAULT_OUTPUT_FILE = Path(
    "data/competition/processed/eval/"
    "competition_agent_text_dev_v1.json"
)

EXPECTED_TEXT_DEV_CASES = 69


def _build_question(
    case: CompetitionQaCase,
) -> CompetitionQuestion:
    """
    Gold-safe Runtime 输入。

    answer / answer_text / evidence
    均不进入 Agent Graph。
    """

    return CompetitionQuestion(
        case_id=case.case_id,
        source_type=case.source_type,
        qa_type=case.qa_type,
        question=case.question,
        option_a=case.option_a,
        option_b=case.option_b,
        option_c=case.option_c,
        option_d=case.option_d,
        source_title=case.source_title,
        file_label=case.file_label,
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
        if isinstance(value, int)
        else 0
    )


def _group_summary(
    cases: list[dict[str, object]],
    field_name: str,
) -> dict[str, object]:
    groups: dict[
        str,
        list[dict[str, object]],
    ] = {}

    for case in cases:
        value = str(
            case[field_name]
        )

        groups.setdefault(
            value,
            [],
        ).append(case)

    result: dict[str, object] = {}

    for (
        value,
        group,
    ) in sorted(
        groups.items()
    ):
        case_count = len(group)

        answered_count = sum(
            int(
                row["status"]
                == "answered"
            )
            for row in group
        )

        refused_count = sum(
            int(
                row["status"]
                == "refused"
            )
            for row in group
        )

        failed_count = sum(
            int(
                row["status"]
                in {
                    "failed",
                    "exception",
                }
            )
            for row in group
        )

        correct_count = sum(
            int(
                bool(
                    row["correct"]
                )
            )
            for row in group
        )

        result[value] = {
            "case_count": (
                case_count
            ),
            "answered_count": (
                answered_count
            ),
            "refused_count": (
                refused_count
            ),
            "failed_count": (
                failed_count
            ),
            "correct_count": (
                correct_count
            ),
            "accuracy": (
                correct_count
                / case_count
            ),
        }

    return result


def _build_summary(
    cases: list[dict[str, object]],
) -> dict[str, object]:
    case_count = len(cases)

    answered_count = sum(
        int(
            case["status"]
            == "answered"
        )
        for case in cases
    )

    refused_count = sum(
        int(
            case["status"]
            == "refused"
        )
        for case in cases
    )

    failed_count = sum(
        int(
            case["status"]
            == "failed"
        )
        for case in cases
    )

    exception_count = sum(
        int(
            case["status"]
            == "exception"
        )
        for case in cases
    )

    correct_count = sum(
        int(
            bool(
                case["correct"]
            )
        )
        for case in cases
    )

    citation_valid_count = sum(
        int(
            (
                case["status"]
                == "answered"
            )
            and bool(
                case["citation_ids"]
            )
        )
        for case in cases
    )

    total_evidence = sum(
        int(
            case[
                "evidence_count"
            ]
        )
        for case in cases
    )

    total_sufficiency_requests = sum(
        int(
            case[
                "sufficiency_requests"
            ]
        )
        for case in cases
    )

    total_answer_requests = sum(
        int(
            case[
                "answer_requests"
            ]
        )
        for case in cases
    )

    refusal_codes = Counter(
        str(
            case["refusal_code"]
        )
        for case in cases
        if (
            case["refusal_code"]
            is not None
        )
    )

    failure_codes = Counter(
        str(
            case["failure_code"]
        )
        for case in cases
        if (
            case["failure_code"]
            is not None
        )
    )

    return {
        "case_count": case_count,

        "answered_count": (
            answered_count
        ),

        "refused_count": (
            refused_count
        ),

        "failed_count": (
            failed_count
        ),

        "exception_count": (
            exception_count
        ),

        "correct_count": (
            correct_count
        ),

        # refused / failed /
        # exception 均按未答对计算。
        "accuracy": (
            correct_count
            / case_count
            if case_count
            else 0.0
        ),

        "answer_accuracy": (
            correct_count
            / answered_count
            if answered_count
            else 0.0
        ),

        # Agent 正常产生业务终态：
        # answered 或 refused。
        "execution_success_rate": (
            (
                answered_count
                + refused_count
            )
            / case_count
            if case_count
            else 0.0
        ),

        "system_failure_rate": (
            (
                failed_count
                + exception_count
            )
            / case_count
            if case_count
            else 0.0
        ),

        "citation_valid_count": (
            citation_valid_count
        ),

        "citation_valid_rate_among_answered": (
            citation_valid_count
            / answered_count
            if answered_count
            else 0.0
        ),

        "mean_evidence_count": (
            total_evidence
            / case_count
            if case_count
            else 0.0
        ),

        "sufficiency_requests": (
            total_sufficiency_requests
        ),

        "answer_requests": (
            total_answer_requests
        ),

        "total_llm_requests": (
            total_sufficiency_requests
            + total_answer_requests
        ),

        "refusal_codes": dict(
            sorted(
                refusal_codes.items()
            )
        ),

        "failure_codes": dict(
            sorted(
                failure_codes.items()
            )
        ),

        "by_source_type": (
            _group_summary(
                cases,
                "source_type",
            )
        ),

        "by_qa_type": (
            _group_summary(
                cases,
                "qa_type",
            )
        ),

        "by_difficulty": (
            _group_summary(
                cases,
                "difficulty",
            )
        ),
    }


def run(
    *,
    qa_file: Path,
    split_file: Path,
    attachments_root: Path,
    corpus_directory: Path,
    index_directory: Path,
    output_file: Path,
    limit: int | None,
    case_ids: tuple[
        str,
        ...
    ],
) -> None:
    all_cases = tuple(
        load_competition_qa_excel(
            qa_file
        )
    )

    case_by_id = {
        case.case_id: case
        for case in all_cases
    }

    dev_case_ids = (
        load_dev_case_ids(
            split_file
        )
    )

    text_cases = [
        case_by_id[case_id]
        for case_id in dev_case_ids
        if (
            case_by_id[
                case_id
            ].source_type
            in {
                "word",
                "pdf",
            }
        )
    ]

    text_cases.sort(
        key=lambda case: (
            case.case_id
        )
    )

    if (
        not case_ids
        and limit is None
        and len(text_cases)
        != EXPECTED_TEXT_DEV_CASES
    ):
        raise RuntimeError(
            "Frozen Dev Text Case "
            "数量发生变化："
            f"{len(text_cases)}"
        )

    if case_ids:
        requested = set(
            case_ids
        )

        valid_ids = {
            case.case_id
            for case in text_cases
        }

        unknown = (
            requested
            - valid_ids
        )

        if unknown:
            raise RuntimeError(
                "请求 Case 不属于 "
                "Frozen Text Dev："
                f"{tuple(sorted(unknown))}"
            )

        text_cases = [
            case
            for case in text_cases
            if case.case_id
            in requested
        ]

    if limit is not None:
        if limit < 1:
            raise ValueError(
                "limit 必须 >= 1"
            )

        text_cases = (
            text_cases[:limit]
        )

    #
    # Process startup：
    # 全部资源只加载一次。
    #
    runtime_resources = (
        load_competition_agent_runtime_resources(
            attachments_root=(
                attachments_root
            ),
            corpus_directory=(
                corpus_directory
            ),
            index_directory=(
                index_directory
            ),
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
            model_resources=(
                model_resources
            ),
        )
    )

    print()
    print(
        "===== Competition Agent "
        "Text Frozen Dev ====="
    )

    print(
        "Cases:",
        len(text_cases),
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

    results: list[
        dict[str, object]
    ] = []

    for (
        index,
        case,
    ) in enumerate(
        text_cases,
        start=1,
    ):
        question = (
            _build_question(
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
            final_state = graph.invoke(
                {
                    "question": (
                        question
                    ),
                }
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
                "difficulty": (
                    case.difficulty
                ),
                "gold_answer": (
                    case.answer
                ),
                "predicted_answer": None,
                "status": "exception",
                "correct": False,
                "refusal_code": None,
                "failure_stage": None,
                "failure_code": None,
                "retryable": None,
                "error_type": (
                    type(exc).__name__
                ),
                "error_message": (
                    str(exc)
                ),
                "source_id": None,
                "expanded_hit_count": 0,
                "evidence_count": 0,
                "evidence_chars": 0,
                "sufficiency_status": None,
                "judge_answer": None,
                "citation_ids": [],
                "trace": [],
                "sufficiency_requests": (
                    after_sufficiency
                    - before_sufficiency
                ),
                "answer_requests": (
                    after_answer
                    - before_answer
                ),
            }

            results.append(row)

            print(
                f"[{index:02d}/"
                f"{len(text_cases):02d}] "
                f"{case.case_id} "
                "EXCEPTION "
                f"{type(exc).__name__}"
            )

            continue

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

        result = final_state.get(
            "result"
        )

        if result is None:
            raise RuntimeError(
                "Graph 正常返回但 "
                "State 中没有 result："
                f"{case.case_id}"
            )

        resolution = final_state.get(
            "source_resolution"
        )

        retrieval = final_state.get(
            "retrieval"
        )

        evidence = final_state.get(
            "evidence_bundle"
        )

        sufficiency = final_state.get(
            "sufficiency"
        )

        trace = final_state.get(
            "trace",
            (),
        )

        source_id = (
            resolution.source_id
            if resolution is not None
            else None
        )

        expanded_hit_count = (
            len(
                retrieval
                .expanded_hits
            )
            if retrieval is not None
            else 0
        )

        evidence_count = (
            len(
                evidence.evidences
            )
            if evidence is not None
            else 0
        )

        evidence_chars = (
            sum(
                len(
                    item.raw_content
                )
                for item
                in evidence.evidences
            )
            if evidence is not None
            else 0
        )

        predicted_answer = None
        citation_ids: list[str] = []

        refusal_code = None
        failure_stage = None
        failure_code = None
        retryable = None

        error_type = None
        error_message = None

        if (
            result.status
            == "answered"
        ):
            predicted_answer = (
                result.answer.answer
            )

            citation_ids = list(
                result.answer
                .citation_ids
            )

        elif (
            result.status
            == "refused"
        ):
            refusal_code = (
                result.refusal_code
            )

        elif (
            result.status
            == "failed"
        ):
            failure_stage = (
                result.failure_stage
            )

            failure_code = (
                result.failure_code
            )

            retryable = (
                result.retryable
            )

            error_type = (
                result.error_type
            )

            error_message = (
                result.error_message
            )

        correct = (
            predicted_answer
            == case.answer
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
            "difficulty": (
                case.difficulty
            ),
            "gold_answer": (
                case.answer
            ),
            "predicted_answer": (
                predicted_answer
            ),
            "status": (
                result.status
            ),
            "correct": correct,
            "refusal_code": (
                refusal_code
            ),
            "failure_stage": (
                failure_stage
            ),
            "failure_code": (
                failure_code
            ),
            "retryable": (
                retryable
            ),
            "error_type": (
                error_type
            ),
            "error_message": (
                error_message
            ),
            "source_id": (
                source_id
            ),
            "expanded_hit_count": (
                expanded_hit_count
            ),
            "evidence_count": (
                evidence_count
            ),
            "evidence_chars": (
                evidence_chars
            ),
            "sufficiency_status": (
                sufficiency.status
                if sufficiency
                is not None
                else None
            ),
            "judge_answer": (
                sufficiency
                .supported_answer
                if sufficiency
                is not None
                else None
            ),
            "citation_ids": (
                citation_ids
            ),
            "trace": [
                event.model_dump(
                    mode="json"
                )
                for event in trace
            ],
            "sufficiency_requests": (
                after_sufficiency
                - before_sufficiency
            ),
            "answer_requests": (
                after_answer
                - before_answer
            ),
        }

        results.append(row)

        print(
            f"[{index:02d}/"
            f"{len(text_cases):02d}] "
            f"{case.case_id} "
            f"status={result.status} "
            f"pred={predicted_answer} "
            f"gold={case.answer} "
            f"correct={correct} "
            f"judge="
            f"{row['judge_answer']} "
            f"trace={len(trace)} "
            f"llm="
            f"{row['sufficiency_requests']}"
            "+"
            f"{row['answer_requests']}"
        )

    summary = (
        _build_summary(
            results
        )
    )

    print()
    print(
        "===== Agent Summary ====="
    )

    print(
        "Answered:",
        f"{summary['answered_count']}/"
        f"{summary['case_count']}",
    )

    print(
        "Refused:",
        summary[
            "refused_count"
        ],
    )

    print(
        "Failed:",
        summary[
            "failed_count"
        ],
    )

    print(
        "Exceptions:",
        summary[
            "exception_count"
        ],
    )

    print(
        "Correct:",
        f"{summary['correct_count']}/"
        f"{summary['case_count']}",
    )

    print(
        "Accuracy:",
        f"{summary['accuracy']:.4f}",
    )

    print(
        "Answer accuracy:",
        f"{summary['answer_accuracy']:.4f}",
    )

    print(
        "Execution success:",
        f"{summary['execution_success_rate']:.4f}",
    )

    print(
        "System failure:",
        f"{summary['system_failure_rate']:.4f}",
    )

    print(
        "Citation valid:",
        f"{summary['citation_valid_count']}/"
        f"{summary['answered_count']}",
    )

    print(
        "Mean evidence:",
        f"{summary['mean_evidence_count']:.2f}",
    )

    print(
        "LLM requests:",
        summary[
            "total_llm_requests"
        ],
    )

    print()
    print(
        "By QA type:"
    )

    for (
        qa_type,
        group,
    ) in summary[
        "by_qa_type"
    ].items():
        print(
            f"  {qa_type}: "
            f"correct="
            f"{group['correct_count']}/"
            f"{group['case_count']} "
            f"answered="
            f"{group['answered_count']} "
            f"refused="
            f"{group['refused_count']} "
            f"failed="
            f"{group['failed_count']}"
        )

    problematic = [
        row
        for row in results
        if not bool(
            row["correct"]
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
                "status=",
                row["status"],
                "gold=",
                row["gold_answer"],
                "pred=",
                row[
                    "predicted_answer"
                ],
                "refusal=",
                row[
                    "refusal_code"
                ],
                "failure=",
                row[
                    "failure_code"
                ],
            )

    payload = {
        "schema_version": 1,
        "evaluation_version": (
            "competition_agent_"
            "text_dev_v1"
        ),
        "provider": {
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
        "summary": summary,
        "cases": results,
    }

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file.write_text(
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
        output_file,
    )


def build_argument_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Competition 69-case "
            "Text Agent Frozen Dev E2E"
        )
    )

    parser.add_argument(
        "--qa",
        type=Path,
        default=DEFAULT_QA_FILE,
    )

    parser.add_argument(
        "--split",
        type=Path,
        default=DEFAULT_SPLIT_FILE,
    )

    parser.add_argument(
        "--attachments",
        type=Path,
        default=(
            DEFAULT_ATTACHMENTS_ROOT
        ),
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
    )

    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_DIR,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
    )

    return parser


def main() -> None:
    arguments = (
        build_argument_parser()
        .parse_args()
    )

    run(
        qa_file=arguments.qa,
        split_file=arguments.split,
        attachments_root=(
            arguments.attachments
        ),
        corpus_directory=(
            arguments.corpus
        ),
        index_directory=(
            arguments.index
        ),
        output_file=(
            arguments.output
        ),
        limit=arguments.limit,
        case_ids=tuple(
            arguments.case_id
        ),
    )


if __name__ == "__main__":
    main()