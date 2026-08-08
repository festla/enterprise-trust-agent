from __future__ import annotations

import time

from typing import Any

import pytest

from pydantic import (
    BaseModel,
    ConfigDict,
)

from app.schemas.tool_registry import (
    ToolDefinition,
)
from app.services.tool_registry import (
    DuplicateToolError,
    InMemoryToolResultCache,
    RetryableToolError,
    ToolExecutionFailedError,
    ToolNotFoundError,
    ToolRegistry,
    ToolResultTooLargeError,
    ToolSchemaMismatchError,
    ToolExecutor,
)


class DemoInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    value: int


class DemoOutput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    doubled: int


class OtherInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    text: str


def _build_definition(
    *,
    tool_name: str = "demo_tool",
    timeout_seconds: float = 1.0,
    max_retries: int = 1,
    idempotent: bool = True,
    max_result_bytes: int = 1024,
) -> ToolDefinition:
    return ToolDefinition(
        tool_name=tool_name,
        description="测试工具",
        version="1.0.0",
        input_schema=(
            DemoInput.model_json_schema()
        ),
        output_schema=(
            DemoOutput.model_json_schema()
        ),
        permission="execute_calculation",
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        idempotent=idempotent,
        max_result_bytes=(
            max_result_bytes
        ),
    )


def _success_handler(
    input_value: BaseModel,
) -> dict[str, Any]:
    assert isinstance(
        input_value,
        DemoInput,
    )

    return {
        "doubled": (
            input_value.value * 2
        ),
    }


def _build_registry(
    *,
    definition: (
        ToolDefinition | None
    ) = None,
    handler: Any = None,
) -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        definition=(
            definition
            or _build_definition()
        ),
        input_model=DemoInput,
        output_model=DemoOutput,
        handler=(
            handler
            or _success_handler
        ),
    )

    return registry


def _execute(
    executor: ToolExecutor,
    *,
    arguments: (
        dict[str, Any] | None
    ) = None,
):
    return executor.execute(
        tool_name="demo_tool",
        arguments=(
            arguments
            or {
                "value": 4,
            }
        ),
        request_id="request_1",
        run_id="run_1",
        step_id="s1",
        granted_permissions={
            "execute_calculation",
        },
    )


def test_registry_registers_and_gets_tool() -> None:
    registry = _build_registry()

    assert len(registry) == 1

    assert "demo_tool" in registry

    tool = registry.get(
        "demo_tool"
    )

    assert (
        tool.definition.tool_name
        == "demo_tool"
    )

    assert tuple(
        definition.tool_name
        for definition
        in registry.definitions()
    ) == (
        "demo_tool",
    )


def test_registry_rejects_duplicate_tool() -> None:
    registry = _build_registry()

    with pytest.raises(
        DuplicateToolError,
        match="已经注册",
    ):
        registry.register(
            definition=(
                _build_definition()
            ),
            input_model=DemoInput,
            output_model=DemoOutput,
            handler=_success_handler,
        )


def test_registry_rejects_schema_mismatch() -> None:
    registry = ToolRegistry()

    with pytest.raises(
        ToolSchemaMismatchError,
        match="input_schema",
    ):
        registry.register(
            definition=(
                _build_definition()
            ),
            input_model=OtherInput,
            output_model=DemoOutput,
            handler=_success_handler,
        )


def test_registry_rejects_unknown_tool() -> None:
    registry = ToolRegistry()

    with pytest.raises(
        ToolNotFoundError,
        match="工具不存在",
    ):
        registry.get(
            "missing_tool"
        )


def test_executor_runs_valid_tool() -> None:
    registry = _build_registry()

    executor = ToolExecutor(
        registry,
        retry_backoff_seconds=0,
    )

    result = _execute(
        executor
    )

    assert result.output == {
        "doubled": 8,
    }

    assert result.reused is False

    assert len(result.traces) == 1

    trace = result.traces[0]

    assert trace.status == "succeeded"
    assert trace.attempt == 1
    assert trace.result_size_bytes > 0

    assert len(
        trace.arguments_sha256
    ) == 64

    assert len(
        trace.idempotency_key
    ) == 64


def test_executor_rejects_missing_permission() -> None:
    registry = _build_registry()

    executor = ToolExecutor(
        registry,
        retry_backoff_seconds=0,
    )

    with pytest.raises(
        ToolExecutionFailedError,
        match="缺少工具权限",
    ) as exc_info:
        executor.execute(
            tool_name="demo_tool",
            arguments={
                "value": 4,
            },
            request_id="request_1",
            run_id="run_1",
            step_id="s1",
            granted_permissions=set(),
        )

    traces = exc_info.value.traces

    assert len(traces) == 1

    assert (
        traces[0].status
        == "permanent_error"
    )


def test_executor_rejects_invalid_input() -> None:
    registry = _build_registry()

    executor = ToolExecutor(
        registry,
        retry_backoff_seconds=0,
    )

    with pytest.raises(
        ToolExecutionFailedError,
        match="输入未通过 Schema",
    ) as exc_info:
        _execute(
            executor,
            arguments={
                "value": "not-an-int",
            },
        )

    assert len(
        exc_info.value.traces
    ) == 1

    assert (
        exc_info
        .value
        .traces[0]
        .status
        == "permanent_error"
    )


def test_executor_rejects_invalid_output() -> None:
    def bad_handler(
        input_value: BaseModel,
    ) -> dict[str, Any]:
        assert isinstance(
            input_value,
            DemoInput,
        )

        return {
            "wrong_field": 123,
        }

    registry = _build_registry(
        handler=bad_handler,
    )

    executor = ToolExecutor(
        registry,
        retry_backoff_seconds=0,
    )

    with pytest.raises(
        ToolExecutionFailedError,
        match="输出未通过 Schema",
    ) as exc_info:
        _execute(executor)

    trace = (
        exc_info.value.traces[0]
    )

    assert (
        trace.status
        == "permanent_error"
    )


def test_executor_retries_retryable_error() -> None:
    attempts = 0

    def flaky_handler(
        input_value: BaseModel,
    ) -> dict[str, Any]:
        nonlocal attempts

        assert isinstance(
            input_value,
            DemoInput,
        )

        attempts += 1

        if attempts == 1:
            raise RetryableToolError(
                "temporary failure"
            )

        return {
            "doubled": (
                input_value.value * 2
            ),
        }

    registry = _build_registry(
        handler=flaky_handler,
    )

    executor = ToolExecutor(
        registry,
        retry_backoff_seconds=0,
    )

    result = _execute(executor)

    assert attempts == 2

    assert tuple(
        trace.status
        for trace in result.traces
    ) == (
        "retryable_error",
        "succeeded",
    )

    assert tuple(
        trace.attempt
        for trace in result.traces
    ) == (
        1,
        2,
    )


def test_executor_stops_after_retry_budget() -> None:
    attempts = 0

    def always_fail(
        input_value: BaseModel,
    ) -> dict[str, Any]:
        nonlocal attempts

        attempts += 1

        raise RetryableToolError(
            "still unavailable"
        )

    registry = _build_registry(
        handler=always_fail,
    )

    executor = ToolExecutor(
        registry,
        retry_backoff_seconds=0,
    )

    with pytest.raises(
        ToolExecutionFailedError,
        match="可重试错误已耗尽",
    ) as exc_info:
        _execute(executor)

    assert attempts == 2

    assert tuple(
        trace.status
        for trace
        in exc_info.value.traces
    ) == (
        "retryable_error",
        "retryable_error",
    )


def test_executor_timeout_is_audited() -> None:
    def slow_handler(
        input_value: BaseModel,
    ) -> dict[str, Any]:
        time.sleep(0.03)

        assert isinstance(
            input_value,
            DemoInput,
        )

        return {
            "doubled": (
                input_value.value * 2
            ),
        }

    definition = _build_definition(
        timeout_seconds=0.005,
        max_retries=1,
    )

    registry = _build_registry(
        definition=definition,
        handler=slow_handler,
    )

    executor = ToolExecutor(
        registry,
        retry_backoff_seconds=0,
    )

    with pytest.raises(
        ToolExecutionFailedError,
        match="超时",
    ) as exc_info:
        _execute(executor)

    assert tuple(
        trace.status
        for trace
        in exc_info.value.traces
    ) == (
        "timed_out",
        "timed_out",
    )

    assert tuple(
        trace.attempt
        for trace
        in exc_info.value.traces
    ) == (
        1,
        2,
    )


def test_unexpected_exception_is_not_retried() -> None:
    attempts = 0

    def crash_handler(
        input_value: BaseModel,
    ) -> dict[str, Any]:
        nonlocal attempts

        attempts += 1

        raise RuntimeError(
            "unexpected crash"
        )

    registry = _build_registry(
        handler=crash_handler,
    )

    executor = ToolExecutor(
        registry,
        retry_backoff_seconds=0,
    )

    with pytest.raises(
        ToolExecutionFailedError,
        match="未分类异常",
    ) as exc_info:
        _execute(executor)

    assert attempts == 1

    assert len(
        exc_info.value.traces
    ) == 1

    assert (
        exc_info
        .value
        .traces[0]
        .status
        == "permanent_error"
    )


def test_executor_rejects_large_result() -> None:
    class LargeOutput(BaseModel):
        model_config = ConfigDict(
            extra="forbid",
            frozen=True,
        )

        text: str

    def large_handler(
        input_value: BaseModel,
    ) -> dict[str, Any]:
        return {
            "text": "x" * 1000,
        }

    definition = ToolDefinition(
        tool_name="large_tool",
        description="大结果测试",
        version="1.0.0",
        input_schema=(
            DemoInput.model_json_schema()
        ),
        output_schema=(
            LargeOutput.model_json_schema()
        ),
        permission="execute_calculation",
        timeout_seconds=1.0,
        max_retries=0,
        idempotent=True,
        max_result_bytes=128,
    )

    registry = ToolRegistry()

    registry.register(
        definition=definition,
        input_model=DemoInput,
        output_model=LargeOutput,
        handler=large_handler,
    )

    executor = ToolExecutor(
        registry,
        retry_backoff_seconds=0,
    )

    with pytest.raises(
        ToolExecutionFailedError,
        match="超过最大限制",
    ) as exc_info:
        executor.execute(
            tool_name="large_tool",
            arguments={
                "value": 1,
            },
            request_id="request_1",
            run_id="run_1",
            step_id="s1",
            granted_permissions={
                "execute_calculation",
            },
        )

    assert (
        exc_info
        .value
        .traces[0]
        .status
        == "permanent_error"
    )


def test_idempotent_tool_reuses_success_result() -> None:
    handler_calls = 0

    def counted_handler(
        input_value: BaseModel,
    ) -> dict[str, Any]:
        nonlocal handler_calls

        handler_calls += 1

        assert isinstance(
            input_value,
            DemoInput,
        )

        return {
            "doubled": (
                input_value.value * 2
            ),
        }

    registry = _build_registry(
        handler=counted_handler,
    )

    cache = InMemoryToolResultCache()

    executor = ToolExecutor(
        registry,
        result_cache=cache,
        retry_backoff_seconds=0,
    )

    first_result = _execute(
        executor
    )

    second_result = _execute(
        executor
    )

    assert handler_calls == 1

    assert first_result.reused is False

    assert second_result.reused is True

    assert (
        second_result.output
        == first_result.output
    )

    assert len(
        second_result.traces
    ) == 1

    assert (
        second_result
        .traces[0]
        .status
        == "reused"
    )


def test_non_idempotent_tool_is_never_cached() -> None:
    handler_calls = 0

    def counted_handler(
        input_value: BaseModel,
    ) -> dict[str, Any]:
        nonlocal handler_calls

        handler_calls += 1

        assert isinstance(
            input_value,
            DemoInput,
        )

        return {
            "doubled": (
                input_value.value * 2
            ),
        }

    definition = _build_definition(
        max_retries=0,
        idempotent=False,
    )

    registry = _build_registry(
        definition=definition,
        handler=counted_handler,
    )

    executor = ToolExecutor(
        registry,
        retry_backoff_seconds=0,
    )

    first_result = _execute(
        executor
    )

    second_result = _execute(
        executor
    )

    assert handler_calls == 2

    assert first_result.reused is False
    assert second_result.reused is False