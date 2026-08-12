from collections import Counter
from pathlib import Path

from app.services.runtime_eval import (
    build_runtime_control_dev_v1_cases,
    build_runtime_eval_environment,
    evaluate_runtime_cases,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


def test_runtime_control_dev_v1_has_50_cases(
) -> None:
    cases = (
        build_runtime_control_dev_v1_cases()
    )

    assert len(cases) == 50

    category_counts = Counter(
        case.category
        for case in cases
    )

    assert category_counts == {
        "financial_fact": 16,
        "financial_calculation": 8,
        "financial_comparison": 10,
        "document_evidence": 8,
        "clarification": 4,
        "unsupported": 4,
    }

    assert len(
        {
            case.case_id
            for case in cases
        }
    ) == 50

    replay_required_count = sum(
        case.replay_required
        for case in cases
    )

    # 42 completed + 4 unsupported refused。
    # 4 clarification 尚未终止，因此无最终 Trajectory。
    assert (
        replay_required_count
        == 46
    )


def test_runtime_control_dev_v1_executes_all_50_cases(
    tmp_path: Path,
) -> None:
    cases = (
        build_runtime_control_dev_v1_cases()
    )

    environment = (
        build_runtime_eval_environment(
            project_root=(
                PROJECT_ROOT
            ),
            trajectory_root=(
                tmp_path
                / "trajectories"
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

    assert (
        summary.case_count
        == 50
    )

    assert (
        summary.passed_count
        == 50
    )

    assert (
        summary.completed_count
        == 42
    )

    assert (
        summary.refused_count
        == 4
    )

    assert (
        summary.awaiting_human_count
        == 4
    )

    assert (
        summary.failed_count
        == 0
    )

    assert (
        summary.intent_accuracy
        == 1.0
    )

    assert (
        summary.argument_accuracy
        == 1.0
    )

    assert (
        summary.plan_accuracy
        == 1.0
    )

    assert (
        summary.tool_accuracy
        == 1.0
    )

    assert (
        summary.tool_sequence_accuracy
        == 1.0
    )

    assert (
        summary.termination_accuracy
        == 1.0
    )

    assert (
        summary.task_success_rate
        == 1.0
    )

    assert (
        summary.replay_applicable_count
        == 46
    )

    assert (
        summary.replay_success_count
        == 46
    )

    assert (
        summary.replay_success_rate
        == 1.0
    )

    failed_case_ids = tuple(
        result.case_id
        for result in results
        if not result.case_pass
    )

    assert failed_case_ids == ()