from __future__ import annotations

import shutil

from pathlib import Path

from app.services.runtime_eval import (
    build_runtime_eval_environment,
    evaluate_runtime_cases,
    load_runtime_eval_cases,
    write_runtime_eval_results,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

CASES_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "runtime"
    / "runtime_control_dev_v1.jsonl"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluations"
    / "runtime"
    / "runtime_control_dev_v1"
)

TRAJECTORY_ROOT = (
    OUTPUT_ROOT
    / "trajectories"
)

RESULTS_PATH = (
    OUTPUT_ROOT
    / "results.jsonl"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "summary.json"
)


def main(
) -> None:
    cases = (
        load_runtime_eval_cases(
            CASES_PATH
        )
    )

    # 同一份 dev eval 允许反复运行。
    # 只清理该版本自己的生成目录。
    shutil.rmtree(
        OUTPUT_ROOT,
        ignore_errors=True,
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    environment = (
        build_runtime_eval_environment(
            project_root=(
                PROJECT_ROOT
            ),
            trajectory_root=(
                TRAJECTORY_ROOT
            ),
        )
    )

    (
        results,
        summary,
    ) = evaluate_runtime_cases(
        environment=environment,
        cases=cases,
    )

    write_runtime_eval_results(
        results_path=(
            RESULTS_PATH
        ),
        summary_path=(
            SUMMARY_PATH
        ),
        results=results,
        summary=summary,
    )

    print(
        "\n===== Runtime Control Dev V1 ====="
    )

    print(
        f"cases: {summary.case_count}"
    )

    print(
        "passed: "
        f"{summary.passed_count}/"
        f"{summary.case_count}"
    )

    print(
        "Intent Accuracy: "
        f"{summary.intent_accuracy:.2%}"
    )

    print(
        "Argument Accuracy: "
        f"{summary.argument_accuracy:.2%}"
    )

    print(
        "Plan Accuracy: "
        f"{summary.plan_accuracy:.2%}"
    )

    print(
        "Tool Accuracy: "
        f"{summary.tool_accuracy:.2%}"
    )

    print(
        "Tool Sequence Accuracy: "
        f"{summary.tool_sequence_accuracy:.2%}"
    )

    print(
        "Termination Accuracy: "
        f"{summary.termination_accuracy:.2%}"
    )

    print(
        "Task Success Rate: "
        f"{summary.task_success_rate:.2%}"
    )

    print(
        "Replay Success: "
        f"{summary.replay_success_count}/"
        f"{summary.replay_applicable_count} "
        f"({summary.replay_success_rate:.2%})"
    )

    failed_cases = tuple(
        result
        for result in results
        if not result.case_pass
    )

    if failed_cases:
        print(
            "\n失败 Case："
        )

        for result in failed_cases:
            print(
                f"- {result.case_id}: "
                f"{result.question}"
            )

            print(
                "  intent="
                f"{result.actual_intent}"
            )

            print(
                "  actions="
                f"{result.actual_plan_actions}"
            )

            print(
                "  tools="
                f"{result.actual_tool_sequence}"
            )

            print(
                "  status="
                f"{result.actual_final_status}"
                "/"
                f"{result.actual_stop_reason}"
            )

            if (
                result.error_message
                is not None
            ):
                print(
                    "  error="
                    f"{result.error_message}"
                )

    print(
        f"\nresults={RESULTS_PATH}"
    )

    print(
        f"summary={SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()