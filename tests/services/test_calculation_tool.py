from __future__ import annotations

from decimal import Decimal

import pytest

from app.schemas.complex_plan_eval_result import (
    ComplexCalculationTrace,
)
from app.schemas.tool_registry import (
    ExecuteCalculationInput,
    ExecuteCalculationOutput,
)
from app.services.calculation_tool import (
    CalculationToolError,
    ExecuteCalculationTool,
    register_execute_calculation_tool,
)
from app.services.tool_registry import (
    ToolExecutor,
    ToolRegistry,
)


# ============================================================
# 6C 的单元测试不需要再次真正执行 Decimal
# 财务公式。
#
# 原因：
#
# DerivedCalculationService
# ComplexOracleCalculatorAdapter
#
# 已经分别有自己的测试。
#
# 这里测试的是 Runtime Tool Adapter。
# ============================================================


def _build_input(
) -> ExecuteCalculationInput:
    return ExecuteCalculationInput(
        calculation_id=(
            "calculation_midea_group_"
            "2024_gross_profit_margin"
        ),
        formula_id=(
            "gross_profit_margin_formula"
        ),
        input_fact_ids=(
            "fact_midea_group_2024_revenue",
            (
                "fact_midea_group_2024_"
                "operating_cost"
            ),
        ),
    )


def _build_completed_trace(
) -> ComplexCalculationTrace:
    return ComplexCalculationTrace(
        calculation_id=(
            "calculation_midea_group_"
            "2024_gross_profit_margin"
        ),
        metric_id=(
            "gross_profit_margin"
        ),
        formula_id=(
            "gross_profit_margin_formula"
        ),
        input_fact_ids=(
            "fact_midea_group_2024_revenue",
            (
                "fact_midea_group_2024_"
                "operating_cost"
            ),
        ),
        status="completed",
        result_value=Decimal(
            "20.7768"
        ),
        result_unit="percent",
        latency_ms=1.5,
        error_message=None,
    )


class FakeCalculationProvider:
    def __init__(
        self,
        trace: ComplexCalculationTrace,
    ) -> None:
        self.trace = trace

        self.calls: list[
            dict[str, object]
        ] = []

    def calculate(
        self,
        *,
        calculation_id: str,
        formula_id: str,
        input_fact_ids: tuple[
            str,
            ...
        ],
    ) -> ComplexCalculationTrace:
        self.calls.append(
            {
                "calculation_id": (
                    calculation_id
                ),
                "formula_id": (
                    formula_id
                ),
                "input_fact_ids": (
                    input_fact_ids
                ),
            }
        )

        return self.trace


# ============================================================
#
# 它验证最核心的 Adapter 行为：
#
# ExecuteCalculationInput
#            ↓
# CalculationProvider.calculate(...)
#            ↓
# ExecuteCalculationOutput
# ============================================================


def test_execute_calculation_forwards_request(
) -> None:
    input_value = _build_input()

    provider = FakeCalculationProvider(
        _build_completed_trace()
    )

    tool = ExecuteCalculationTool(
        calculation_provider=provider
    )

    output = tool.handle(
        input_value
    )

    assert isinstance(
        output,
        ExecuteCalculationOutput,
    )

    assert len(provider.calls) == 1

    assert provider.calls[0] == {
        "calculation_id": (
            input_value.calculation_id
        ),
        "formula_id": (
            input_value.formula_id
        ),
        "input_fact_ids": (
            input_value.input_fact_ids
        ),
    }

    assert (
        output.trace.status
        == "completed"
    )

    assert (
        output.trace.result_value
        == Decimal("20.7768")
    )

    assert (
        output.trace.result_unit
        == "percent"
    )


# ============================================================
# Domain Failure != Tool Infrastructure Failure
#
# Calculator 成功运行了，
# 但计算本身失败，例如：
#
# Fact 不存在
# Formula 不支持
# Fact 指标类型不匹配
#
# 此时我们仍然保留 failed CalculationTrace，
# 方便 Agent Runtime 做审计和后续恢复。
# ============================================================


def test_execute_calculation_preserves_failed_trace(
) -> None:
    input_value = _build_input()

    failed_trace = (
        ComplexCalculationTrace(
            calculation_id=(
                input_value
                .calculation_id
            ),
            metric_id=(
                "gross_profit_margin"
            ),
            formula_id=(
                input_value.formula_id
            ),
            input_fact_ids=(
                input_value
                .input_fact_ids
            ),
            status="failed",
            result_value=None,
            result_unit=None,
            latency_ms=1.0,
            error_message=(
                "FinancialFact 不存在"
            ),
        )
    )

    provider = FakeCalculationProvider(
        failed_trace
    )

    tool = ExecuteCalculationTool(
        calculation_provider=provider
    )

    output = tool.handle(
        input_value
    )

    assert (
        output.trace.status
        == "failed"
    )

    assert (
        output.trace.error_message
        == "FinancialFact 不存在"
    )


# ============================================================
# Runtime 不能盲目信任底层 Provider。
#
# 即使 Pydantic Trace 本身合法，
# 如果它不是“这次请求”的结果，
# Tool 也必须拒绝。
# ============================================================


def test_execute_calculation_rejects_mismatched_trace(
) -> None:
    input_value = _build_input()

    wrong_trace = (
        ComplexCalculationTrace(
            calculation_id=(
                "calculation_wrong_result"
            ),
            metric_id=(
                "gross_profit_margin"
            ),
            formula_id=(
                input_value.formula_id
            ),
            input_fact_ids=(
                input_value
                .input_fact_ids
            ),
            status="completed",
            result_value=Decimal(
                "20.7768"
            ),
            result_unit="percent",
            latency_ms=1.0,
        )
    )

    provider = FakeCalculationProvider(
        wrong_trace
    )

    tool = ExecuteCalculationTool(
        calculation_provider=provider
    )

    with pytest.raises(
        CalculationToolError,
        match="calculation_id",
    ):
        tool.handle(
            input_value
        )


def test_execute_calculation_definition(
) -> None:
    provider = FakeCalculationProvider(
        _build_completed_trace()
    )

    tool = ExecuteCalculationTool(
        calculation_provider=provider
    )

    definition = (
        tool.build_definition()
    )

    assert (
        definition.tool_name
        == "execute_calculation"
    )

    assert (
        definition.permission
        == "execute_calculation"
    )

    assert (
        definition.idempotent
        is True
    )

    assert (
        definition.max_retries
        == 0
    )

    assert (
        definition.input_schema
        == (
            ExecuteCalculationInput
            .model_json_schema()
        )
    )

    assert (
        definition.output_schema
        == (
            ExecuteCalculationOutput
            .model_json_schema()
        )
    )


# ============================================================
# 最终 Runtime 调用链：
#
# ToolExecutor
#      ↓
# permission check
#      ↓
# input schema validation
#      ↓
# idempotency cache
#      ↓
# ExecuteCalculationTool
#      ↓
# Calculator
#
# 第二次相同逻辑调用应该直接 reused，
# 所以 Provider 仍然只执行一次。
# ============================================================


def test_execute_calculation_runs_through_executor(
) -> None:
    input_value = _build_input()

    provider = FakeCalculationProvider(
        _build_completed_trace()
    )

    tool_registry = ToolRegistry()

    register_execute_calculation_tool(
        tool_registry=tool_registry,
        calculation_provider=provider,
    )

    executor = ToolExecutor(
        tool_registry,
        retry_backoff_seconds=0,
    )

    arguments = (
        input_value.model_dump(
            mode="json"
        )
    )

    first_result = (
        executor.execute(
            tool_name=(
                "execute_calculation"
            ),
            arguments=arguments,
            request_id="request_1",
            run_id="run_1",
            step_id="s1",
            granted_permissions={
                "execute_calculation",
            },
        )
    )

    assert (
        first_result.reused
        is False
    )

    assert (
        first_result.traces[0].status
        == "succeeded"
    )

    assert (
        first_result.output[
            "trace"
        ][
            "status"
        ]
        == "completed"
    )

    assert (
        Decimal(
            first_result.output[
                "trace"
            ][
                "result_value"
            ]
        )
        == Decimal("20.7768")
    )

    assert len(provider.calls) == 1

    second_result = (
        executor.execute(
            tool_name=(
                "execute_calculation"
            ),
            arguments=arguments,
            request_id="request_1",
            run_id="run_1",
            step_id="s1",
            granted_permissions={
                "execute_calculation",
            },
        )
    )

    assert (
        second_result.reused
        is True
    )

    assert (
        second_result.traces[0].status
        == "reused"
    )

    # ========================================================
    # 第二次调用命中 ToolResultCache，
    # 所以底层 Calculator 不应该再次执行。
    # ========================================================
    assert len(provider.calls) == 1