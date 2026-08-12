from __future__ import annotations

import ast
import json

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluations"
    / "runtime"
    / "runtime_control_dev_v1"
    / "summary.json"
)


DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "runtime"
    / "runtime_control_dev_v1.jsonl"
)


REQUIRED_FILES = (
    "app/services/agent_runtime.py",
    "app/services/runtime_graph.py",
    "app/services/runtime_plan_executor.py",
    "app/services/runtime_completion.py",
    "app/services/runtime_query_parser.py",
    "app/services/runtime_intent_router.py",
    "app/services/runtime_planner.py",
    "app/services/tool_registry.py",
    "app/services/checkpoint_store.py",
    "app/services/trajectory_store.py",
    "app/services/financial_data_tool.py",
    "app/services/document_retrieval_tool.py",
    "app/services/calculation_tool.py",
    "data/evaluation/runtime/runtime_control_dev_v1.jsonl",
    (
        "data/processed/evaluations/runtime/"
        "runtime_control_dev_v1/results.jsonl"
    ),
    (
        "data/processed/evaluations/runtime/"
        "runtime_control_dev_v1/summary.json"
    ),
)


PRODUCTION_RUNTIME_FILES = (
    "app/services/agent_runtime.py",
    "app/services/runtime_graph.py",
    "app/services/runtime_plan_executor.py",
    "app/services/runtime_completion.py",
    "app/services/runtime_query_parser.py",
    "app/services/runtime_intent_router.py",
    "app/services/runtime_planner.py",
    "app/services/financial_data_tool.py",
    "app/services/document_retrieval_tool.py",
    "app/services/calculation_tool.py",
)


FORBIDDEN_PRODUCTION_IMPORT_PARTS = (
    "gold_oracle",
    "runtime_eval",
)


@dataclass(
    frozen=True,
    slots=True,
)
class AcceptanceCheck:
    name: str

    passed: bool

    detail: str


def _load_summary(
) -> dict:
    if not SUMMARY_PATH.exists():
        return {}

    return json.loads(
        SUMMARY_PATH.read_text(
            encoding="utf-8"
        )
    )


def _count_jsonl_records(
    path: Path,
) -> int:
    if not path.exists():
        return 0

    count = 0

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                count += 1

    return count


def _collect_import_modules(
    path: Path,
) -> tuple[str, ...]:
    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        ),
        filename=str(path),
    )

    modules: list[str] = []

    for node in ast.walk(
        tree
    ):
        if isinstance(
            node,
            ast.Import,
        ):
            modules.extend(
                alias.name
                for alias
                in node.names
            )

        elif isinstance(
            node,
            ast.ImportFrom,
        ):
            if node.module:
                modules.append(
                    node.module
                )

    return tuple(modules)


def _check_required_files(
) -> AcceptanceCheck:
    missing = tuple(
        relative_path
        for relative_path
        in REQUIRED_FILES
        if not (
            PROJECT_ROOT
            / relative_path
        ).exists()
    )

    return AcceptanceCheck(
        name="required_runtime_artifacts",
        passed=not missing,
        detail=(
            "all required files exist"
            if not missing
            else f"missing={missing}"
        ),
    )


def _check_dataset(
) -> AcceptanceCheck:
    count = (
        _count_jsonl_records(
            DATASET_PATH
        )
    )

    return AcceptanceCheck(
        name="runtime_control_dataset",
        passed=count == 50,
        detail=(
            f"case_count={count}, "
            "expected=50"
        ),
    )


def _check_summary(
    summary: dict,
) -> AcceptanceCheck:
    if not summary:
        return AcceptanceCheck(
            name="runtime_control_summary",
            passed=False,
            detail="summary.json missing or empty",
        )

    required_values = {
        "case_count": 50,
        "passed_count": 50,
        "failed_count": 0,
        "intent_accuracy": 1.0,
        "argument_accuracy": 1.0,
        "plan_accuracy": 1.0,
        "tool_accuracy": 1.0,
        "tool_sequence_accuracy": 1.0,
        "termination_accuracy": 1.0,
        "task_success_rate": 1.0,
        "replay_applicable_count": 46,
        "replay_success_count": 46,
        "replay_success_rate": 1.0,
    }

    mismatches: list[str] = []

    for (
        key,
        expected,
    ) in required_values.items():
        actual = summary.get(
            key
        )

        if actual != expected:
            mismatches.append(
                f"{key}: "
                f"actual={actual!r}, "
                f"expected={expected!r}"
            )

    return AcceptanceCheck(
        name="runtime_control_summary",
        passed=not mismatches,
        detail=(
            "50/50 control cases passed"
            if not mismatches
            else "; ".join(
                mismatches
            )
        ),
    )


def _check_runtime_import_boundaries(
) -> AcceptanceCheck:
    violations: list[str] = []

    for relative_path in (
        PRODUCTION_RUNTIME_FILES
    ):
        path = (
            PROJECT_ROOT
            / relative_path
        )

        modules = (
            _collect_import_modules(
                path
            )
        )

        for module in modules:
            if any(
                forbidden
                in module
                for forbidden
                in (
                    FORBIDDEN_PRODUCTION_IMPORT_PARTS
                )
            ):
                violations.append(
                    f"{relative_path} "
                    f"imports {module}"
                )

    return AcceptanceCheck(
        name="production_eval_boundary",
        passed=not violations,
        detail=(
            "production runtime has no "
            "Gold/Runtime-Eval imports"
            if not violations
            else "; ".join(
                violations
            )
        ),
    )


def _check_langgraph(
) -> AcceptanceCheck:
    path = (
        PROJECT_ROOT
        / "app"
        / "services"
        / "runtime_graph.py"
    )

    if not path.exists():
        return AcceptanceCheck(
            name="langgraph_adapter",
            passed=False,
            detail="runtime_graph.py missing",
        )

    content = path.read_text(
        encoding="utf-8"
    )

    required_tokens = (
        "StateGraph",
        "interrupt",
        "compile",
    )

    missing = tuple(
        token
        for token
        in required_tokens
        if token not in content
    )

    return AcceptanceCheck(
        name="langgraph_adapter",
        passed=not missing,
        detail=(
            "StateGraph + interrupt available"
            if not missing
            else f"missing={missing}"
        ),
    )


def _check_recovery_api(
) -> AcceptanceCheck:
    path = (
        PROJECT_ROOT
        / "app"
        / "services"
        / "agent_runtime.py"
    )

    content = path.read_text(
        encoding="utf-8"
    )

    required_tokens = (
        "def resume(",
        "_continue_from_state",
        "_persist_checkpoint",
        "_save_trajectory",
    )

    missing = tuple(
        token
        for token
        in required_tokens
        if token not in content
    )

    return AcceptanceCheck(
        name="runtime_recovery",
        passed=not missing,
        detail=(
            "resume + checkpoint + trajectory available"
            if not missing
            else f"missing={missing}"
        ),
    )


def _check_readme(
) -> AcceptanceCheck:
    path = (
        PROJECT_ROOT
        / "README.md"
    )

    if not path.exists():
        return AcceptanceCheck(
            name="readme",
            passed=False,
            detail="README.md missing",
        )

    content = path.read_text(
        encoding="utf-8"
    ).strip()

    return AcceptanceCheck(
        name="readme",
        passed=len(content) > 200,
        detail=(
            f"README length={len(content)}"
        ),
    )


def main(
) -> None:
    summary = (
        _load_summary()
    )

    checks = (
        _check_required_files(),
        _check_dataset(),
        _check_summary(
            summary
        ),
        _check_runtime_import_boundaries(),
        _check_langgraph(),
        _check_recovery_api(),
        _check_readme(),
    )

    print(
        "\n===== Week 6 Acceptance ====="
    )

    for check in checks:
        marker = (
            "PASS"
            if check.passed
            else "FAIL"
        )

        print(
            f"[{marker}] "
            f"{check.name}: "
            f"{check.detail}"
        )

    passed_count = sum(
        check.passed
        for check in checks
    )

    print(
        "\nAcceptance: "
        f"{passed_count}/{len(checks)}"
    )

    if (
        passed_count
        != len(checks)
    ):
        raise SystemExit(1)

    print(
        "\nWEEK 6 ACCEPTED"
    )


if __name__ == "__main__":
    main()