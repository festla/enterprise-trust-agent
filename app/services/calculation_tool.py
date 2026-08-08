from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from app.schemas.complex_plan_eval_result import (
    ComplexCalculationTrace,
)
from app.schemas.tool_registry import (
    ExecuteCalculationInput,
    ExecuteCalculationOutput,
    ToolDefinition,
)
from app.services.tool_registry import (
    ToolRegistry,
)


class CalculationToolError(
    ValueError
):
    """Runtime 确定性计算工具基础异常。"""


# ============================================================
# Tool 不应该知道底层 Calculator 的具体类名。
#
# 它只要求：
#
#     calculate(
#         calculation_id,
#         formula_id,
#         input_fact_ids,
#     )
#
# 这意味着：
#
# ExecuteCalculationTool
#         ↓
# CalculationProvider Protocol
#         ↓
# 真实 Registry Calculator
# / Fake Calculator
# / 后续其他确定性 Calculator
#
# 都可以接进来。
#
# 这就是“面向接口，而不是面向具体实现”。
# ============================================================


class CalculationProvider(
    Protocol
):
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
        """执行确定性财务计算。"""


@dataclass(
    frozen=True,
    slots=True,
)
class ExecuteCalculationTool:
    """把确定性 Calculator 包装成 Agent Runtime Tool。"""

    calculation_provider: (
        CalculationProvider
    )

    tool_name: str = (
        "execute_calculation"
    )

    tool_version: str = "1.0.0"

    def build_definition(
        self,
    ) -> ToolDefinition:
        return ToolDefinition(
            tool_name=self.tool_name,
            description=(
                "基于已核验 FinancialFact "
                "执行受支持的确定性财务计算。"
            ),
            version=self.tool_version,
            input_schema=(
                ExecuteCalculationInput
                .model_json_schema()
            ),
            output_schema=(
                ExecuteCalculationOutput
                .model_json_schema()
            ),
            permission=(
                "execute_calculation"
            ),

            # 当前 Calculator 是：
            #
            # Registry lookup
            # + Decimal deterministic calculation
            #
            # 正常情况下应该非常快。
            timeout_seconds=2.0,

            # =================================================
            # 【重点理解】
            #
            # 当前没有网络请求或类似 HTTP 503
            # 的临时故障，所以不自动 Retry。
            #
            # Calculator 自己返回的“公式不支持”
            # “Fact 不存在”等也是领域错误，
            # 重试没有意义。
            # =================================================
            max_retries=0,

            # =================================================
            # 财务计算没有副作用。
            #
            # 同一个：
            #
            # calculation_id
            # formula_id
            # input_fact_ids
            #
            # 可以安全复用结果。
            # =================================================
            idempotent=True,

            max_result_bytes=(
                64 * 1024
            ),
        )

    def handle(
        self,
        input_value: BaseModel,
    ) -> ExecuteCalculationOutput:
        if not isinstance(
            input_value,
            ExecuteCalculationInput,
        ):
            raise TypeError(
                "execute_calculation "
                "必须接收 "
                "ExecuteCalculationInput"
            )

        # ====================================================
        # Tool 自己不做：
        #
        # result = (a - b) / a
        #
        # 也不自己从 Registry 读取 value。
        #
        # 这些业务逻辑已经属于底层
        # deterministic calculator。
        #
        # Tool 的职责只是：
        #
        # Runtime Contract
        #       ↓
        # Calculator Contract
        # ====================================================

        trace = (
            self.calculation_provider
            .calculate(
                calculation_id=(
                    input_value
                    .calculation_id
                ),
                formula_id=(
                    input_value
                    .formula_id
                ),
                input_fact_ids=(
                    input_value
                    .input_fact_ids
                ),
            )
        )

        # ====================================================
        # Defense in depth：
        #
        # 即使 Calculator 是我们自己的实现，
        # Runtime 边界仍然要检查：
        #
        # 请求 calculation_A
        # 不能返回 calculation_B。
        #
        # 请求 input_fact_ids=(A, B)
        # Calculator 也不能偷偷改成 (B, A)。
        #
        # 对财务计算尤其重要，因为输入顺序
        # 会改变公式语义。
        # ====================================================

        self._validate_trace_matches_input(
            input_value=input_value,
            trace=trace,
        )

        return ExecuteCalculationOutput(
            trace=trace
        )

    @staticmethod
    def _validate_trace_matches_input(
        *,
        input_value: (
            ExecuteCalculationInput
        ),
        trace: ComplexCalculationTrace,
    ) -> None:
        if (
            trace.calculation_id
            != input_value.calculation_id
        ):
            raise CalculationToolError(
                "Calculator 返回的 "
                "calculation_id "
                "与请求不一致"
            )

        if (
            trace.formula_id
            != input_value.formula_id
        ):
            raise CalculationToolError(
                "Calculator 返回的 "
                "formula_id "
                "与请求不一致"
            )

        if (
            trace.input_fact_ids
            != input_value.input_fact_ids
        ):
            raise CalculationToolError(
                "Calculator 返回的 "
                "input_fact_ids "
                "与请求不一致"
            )


# ============================================================
# Runtime Factory 最后会：
#
# registry = ToolRegistry()
#
# register_query_financial_data_tool(...)
# register_retrieve_documents_tool(...)
# register_execute_calculation_tool(...)
#
# ============================================================


def register_execute_calculation_tool(
    *,
    tool_registry: ToolRegistry,
    calculation_provider: (
        CalculationProvider
    ),
) -> ExecuteCalculationTool:
    tool = ExecuteCalculationTool(
        calculation_provider=(
            calculation_provider
        )
    )

    tool_registry.register(
        definition=(
            tool.build_definition()
        ),
        input_model=(
            ExecuteCalculationInput
        ),
        output_model=(
            ExecuteCalculationOutput
        ),
        handler=tool.handle,
    )

    return tool