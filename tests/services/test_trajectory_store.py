from __future__ import annotations

import json

from datetime import (
    datetime,
    timezone,
)
from pathlib import Path

import pytest

from app.schemas.agent_runtime import (
    AgentAnswer,
    AgentTrajectory,
    NodeSpan,
)
from app.schemas.tool_registry import (
    ToolCallTrace,
)
from app.services.trajectory_store import (
    CorruptTrajectoryError,
    IncompatibleTrajectoryVersionError,
    TrajectoryAlreadyExistsError,
    TrajectoryExportExistsError,
    TrajectoryNotFoundError,
    TrajectoryStore,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _build_tool_trace(
    *,
    attempt: int,
    status: str,
    error_type: str | None = None,
    error_message: str | None = None,
) -> ToolCallTrace:
    now = _now()

    return ToolCallTrace(
        tool_call_id=(
            f"toolcall_{attempt}"
        ),
        request_id="request_1",
        run_id="run_1",
        step_id="s1",
        tool_name="demo_tool",
        tool_version="1.0.0",
        argument_summary={
            "value": 4,
        },
        arguments_sha256="0" * 64,
        idempotency_key="1" * 64,
        attempt=attempt,
        status=status,
        started_at=now,
        completed_at=now,
        latency_ms=1.0,
        result_size_bytes=(
            0
            if status
            not in {
                "succeeded",
                "reused",
            }
            else 20
        ),
        error_type=error_type,
        error_message=error_message,
    )


def _build_node_span(
    *,
    span_id: str,
    node_name: str,
) -> NodeSpan:
    now = _now()

    return NodeSpan(
        span_id=span_id,
        node_name=node_name,
        attempt=1,
        status="completed",
        input_summary={
            "question": "营业收入是多少？",
        },
        output_summary={
            "status": "ok",
        },
        started_at=now,
        completed_at=now,
        latency_ms=1.0,
        checkpoint_revision=1,
        error_type=None,
        error_message=None,
    )


def _build_trajectory(
    *,
    run_id: str = "run_1",
) -> AgentTrajectory:
    now = _now()

    answer = AgentAnswer(
        answer_type="financial",
        answer_text=(
            "美的集团2024年营业收入"
            "为407,149,600,000元。"
        ),
        supporting_fact_ids=(
            "fact_midea_group_2024_revenue",
        ),
        supporting_calculation_ids=(
            "calculation_demo",
        ),
        citation_evidence_ids=(
            "evidence_midea_group_2024_revenue",
        ),
        document_citation_ids=(),
        confidence=1.0,
    )

    return AgentTrajectory(
        request_id="request_1",
        trace_id="trace_1",
        run_id=run_id,
        thread_id="thread_1",
        query=(
            "美的集团2024年营业收入是多少？"
        ),
        intent="financial_fact",
        planner_version="planner_v1",
        retriever_version="retriever_v1",
        calculator_version="calculator_v1",
        generator_version="generator_v1",
        prompt_version=None,
        prompt_sha256=None,
        model_name=None,
        parsed_query=None,
        runtime_plan=None,
        node_spans=(
            _build_node_span(
                span_id="span_1",
                node_name="parse_query",
            ),
            _build_node_span(
                span_id="span_2",
                node_name="execute_plan",
            ),
        ),
        tool_call_traces=(
            _build_tool_trace(
                attempt=1,
                status="retryable_error",
                error_type="RetryableToolError",
                error_message=(
                    "temporary failure"
                ),
            ),
            _build_tool_trace(
                attempt=2,
                status="succeeded",
            ),
        ),
        retrieval_traces=(),
        calculation_traces=(),
        retrieved_documents=(),
        resolved_fact_ids=(
            "fact_midea_group_2024_revenue",
        ),
        evidence_ids=(
            "evidence_midea_group_2024_revenue",
        ),
        calculation_ids=(
            "calculation_demo",
        ),
        citations=(),
        errors=(),
        answer=answer,
        input_tokens=0,
        output_tokens=0,
        estimated_cost=0.0,
        started_at=now,
        completed_at=now,
        latency_ms=10.0,
        final_status="completed",
        stop_reason="completed",
    )


def test_save_and_load_round_trip(
    tmp_path: Path,
) -> None:
    store = TrajectoryStore(
        tmp_path
    )

    trajectory = _build_trajectory()

    saved_path = store.save(
        trajectory
    )

    assert saved_path.exists()

    restored = store.load(
        trajectory.run_id
    )

    assert restored == trajectory


def test_store_rejects_overwrite(
    tmp_path: Path,
) -> None:
    store = TrajectoryStore(
        tmp_path
    )

    trajectory = _build_trajectory()

    store.save(trajectory)

    with pytest.raises(
        TrajectoryAlreadyExistsError,
        match="禁止覆盖",
    ):
        store.save(trajectory)


def test_load_missing_trajectory(
    tmp_path: Path,
) -> None:
    store = TrajectoryStore(
        tmp_path
    )

    with pytest.raises(
        TrajectoryNotFoundError,
        match="没有找到",
    ):
        store.load(
            "missing_run"
        )


def test_filename_is_windows_safe(
    tmp_path: Path,
) -> None:
    store = TrajectoryStore(
        tmp_path
    )

    trajectory = _build_trajectory(
        run_id="run:2026:08:07"
    )

    path = store.save(
        trajectory
    )

    assert ":" not in path.name

    restored = store.load(
        "run:2026:08:07"
    )

    assert (
        restored.run_id
        == "run:2026:08:07"
    )


def test_store_detects_corrupt_payload(
    tmp_path: Path,
) -> None:
    store = TrajectoryStore(
        tmp_path
    )

    trajectory = _build_trajectory()

    path = store.save(
        trajectory
    )

    envelope = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    envelope[
        "trajectory"
    ][
        "query"
    ] = "被篡改的问题"

    path.write_text(
        json.dumps(
            envelope,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        CorruptTrajectoryError,
        match="SHA-256",
    ):
        store.load(
            trajectory.run_id
        )


def test_store_rejects_incompatible_version(
    tmp_path: Path,
) -> None:
    store = TrajectoryStore(
        tmp_path
    )

    trajectory = _build_trajectory()

    path = store.save(
        trajectory
    )

    envelope = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    envelope["schema_version"] = 99

    path.write_text(
        json.dumps(
            envelope,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        IncompatibleTrajectoryVersionError,
        match="Trajectory Schema",
    ):
        store.load(
            trajectory.run_id
        )


def test_replay_contains_runtime_sequence(
    tmp_path: Path,
) -> None:
    store = TrajectoryStore(
        tmp_path
    )

    trajectory = _build_trajectory()

    store.save(trajectory)

    replay = store.replay(
        trajectory.run_id
    )

    assert replay.nodes == (
        "parse_query",
        "execute_plan",
    )

    assert replay.tools == (
        "demo_tool",
        "demo_tool",
    )

    assert replay.tool_arguments == (
        {
            "value": 4,
        },
        {
            "value": 4,
        },
    )

    assert replay.retries == (
        (
            "demo_tool:"
            "s1:"
            "attempt=2:"
            "status=succeeded"
        ),
    )


def test_replay_contains_failures_and_refs(
    tmp_path: Path,
) -> None:
    store = TrajectoryStore(
        tmp_path
    )

    trajectory = _build_trajectory()

    store.save(trajectory)

    replay = store.replay(
        trajectory.run_id
    )

    assert len(
        replay.failures
    ) == 1

    assert (
        "RetryableToolError"
        in replay.failures[0]
    )

    assert (
        replay.supporting_fact_ids
        == (
            "fact_midea_group_2024_revenue",
        )
    )

    assert replay.evidence_ids == (
        "evidence_midea_group_2024_revenue",
    )

    assert replay.calculation_ids == (
        "calculation_demo",
    )

    assert (
        replay.final_status
        == "completed"
    )

    assert (
        replay.stop_reason
        == "completed"
    )


def test_list_run_ids_is_sorted(
    tmp_path: Path,
) -> None:
    store = TrajectoryStore(
        tmp_path
    )

    store.save(
        _build_trajectory(
            run_id="run_z"
        )
    )

    store.save(
        _build_trajectory(
            run_id="run_a"
        )
    )

    assert store.list_run_ids() == (
        "run_a",
        "run_z",
    )


def test_export_all_writes_standard_jsonl(
    tmp_path: Path,
) -> None:
    store_directory = (
        tmp_path / "store"
    )

    store = TrajectoryStore(
        store_directory
    )

    store.save(
        _build_trajectory(
            run_id="run_b"
        )
    )

    store.save(
        _build_trajectory(
            run_id="run_a"
        )
    )

    export_path = (
        tmp_path
        / "export"
        / "trajectories.jsonl"
    )

    count = store.export_all(
        export_path
    )

    assert count == 2

    lines = (
        export_path
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    assert len(lines) == 2

    first_payload = json.loads(
        lines[0]
    )

    second_payload = json.loads(
        lines[1]
    )

    assert (
        first_payload["run_id"]
        == "run_a"
    )

    assert (
        second_payload["run_id"]
        == "run_b"
    )

    assert (
        "trajectory_sha256"
        not in first_payload
    )


def test_export_rejects_existing_target(
    tmp_path: Path,
) -> None:
    store = TrajectoryStore(
        tmp_path / "store"
    )

    store.save(
        _build_trajectory()
    )

    export_path = (
        tmp_path
        / "trajectories.jsonl"
    )

    store.export_all(
        export_path
    )

    with pytest.raises(
        TrajectoryExportExistsError,
        match="已经存在",
    ):
        store.export_all(
            export_path
        )