from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter

from app.schemas.complex_plan_eval_result import (
    ComplexCalculationTrace,
)
from app.services.derived_calculation_service import (
    build_current_ratio_calculation,
    build_debt_to_equity_ratio_calculation,
    build_effective_income_tax_rate_calculation,
    build_gross_profit_margin_calculation,
    build_operating_cash_flow_to_net_profit_ratio_calculation,
    build_selling_and_r_and_d_expense_ratio_calculation,
)
from app.services.registry import RegistryBundle


class ComplexOracleCalculatorAdapterError(
    ValueError
):
    """Oracle 确定性计算适配失败。"""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


_FORMULA_METRIC_IDS = {
    "gross_profit_margin_formula": (
        "gross_profit_margin"
    ),
    (
        "selling_and_r_and_d_expense_ratio_formula"
    ): (
        "selling_and_r_and_d_expense_ratio"
    ),
    (
        "operating_cash_flow_to_net_profit_ratio_formula"
    ): (
        "operating_cash_flow_to_net_profit_ratio"
    ),
    "current_ratio_formula": "current_ratio",
    "debt_to_equity_ratio_formula": (
        "debt_to_equity_ratio"
    ),
    "effective_income_tax_rate_formula": (
        "effective_income_tax_rate"
    ),
}


_FORMULA_INPUT_COUNTS = {
    "gross_profit_margin_formula": 2,
    (
        "selling_and_r_and_d_expense_ratio_formula"
    ): 3,
    (
        "operating_cash_flow_to_net_profit_ratio_formula"
    ): 2,
    "current_ratio_formula": 2,
    "debt_to_equity_ratio_formula": 2,
    "effective_income_tax_rate_formula": 2,
}

@dataclass(slots=True)
class ComplexOracleCalculatorAdapter:
    """把 Oracle Calculation 步骤接入确定性计算服务。"""

    registry_bundle: RegistryBundle

    calculator_id: str = (
        "deterministic_calculator_v1"
    )

    clock: Callable[
        [],
        datetime,
    ] = field(
        default=_utc_now,
        repr=False,
    )

    def __post_init__(self) -> None:
        normalized_calculator_id = (
            self.calculator_id.strip()
        )

        if not normalized_calculator_id:
            raise (
                ComplexOracleCalculatorAdapterError(
                    "calculator_id 不能为空"
                )
            )

        self.calculator_id = (
            normalized_calculator_id
        )

    def calculate(
        self,
        *,
        calculation_id: str,
        formula_id: str,
        input_fact_ids: tuple[str, ...],
    ) -> ComplexCalculationTrace:
        """根据实际检索得到的 Fact 执行确定性计算。"""

        if not input_fact_ids:
            raise (
                ComplexOracleCalculatorAdapterError(
                    "input_fact_ids 不能为空"
                )
            )

        if len(input_fact_ids) != len(
            set(input_fact_ids)
        ):
            raise (
                ComplexOracleCalculatorAdapterError(
                    "input_fact_ids "
                    "不能包含重复 ID"
                )
            )

        timer_start = perf_counter()

        metric_id = (
            _FORMULA_METRIC_IDS.get(
                formula_id,
                "unknown_metric",
            )
        )

        if formula_id not in _FORMULA_METRIC_IDS:
            return self._build_failed_trace(
                calculation_id=calculation_id,
                metric_id=metric_id,
                formula_id=formula_id,
                input_fact_ids=input_fact_ids,
                timer_start=timer_start,
                error_message=(
                    "当前 Calculator 不支持 "
                    f"formula_id：{formula_id}"
                ),
            )

        expected_input_count = (
            _FORMULA_INPUT_COUNTS[
                formula_id
            ]
        )

        if (
            len(input_fact_ids)
            != expected_input_count
        ):
            return self._build_failed_trace(
                calculation_id=calculation_id,
                metric_id=metric_id,
                formula_id=formula_id,
                input_fact_ids=input_fact_ids,
                timer_start=timer_start,
                error_message=(
                    f"{formula_id} 必须接收 "
                    f"{expected_input_count} 个输入 Fact"
                ),
            )

        try:
            input_facts = tuple(
                self.registry_bundle
                .financial_facts
                .require(fact_id)
                for fact_id in input_fact_ids
            )

            if (
                formula_id
                == "gross_profit_margin_formula"
            ):
                calculation = (
                    build_gross_profit_margin_calculation(
                        revenue_fact=input_facts[0],
                        operating_cost_fact=input_facts[1],
                        created_at=self.clock(),
                    )
                )

            elif (
                formula_id
                == (
                    "selling_and_r_and_d_"
                    "expense_ratio_formula"
                )
            ):
                calculation = (
                    build_selling_and_r_and_d_expense_ratio_calculation(
                        revenue_fact=input_facts[0],
                        selling_expenses_fact=(
                            input_facts[1]
                        ),
                        research_and_development_expenses_fact=(
                            input_facts[2]
                        ),
                        created_at=self.clock(),
                    )
                )

            elif (
                formula_id
                == (
                    "operating_cash_flow_to_"
                    "net_profit_ratio_formula"
                )
            ):
                calculation = (
                    build_operating_cash_flow_to_net_profit_ratio_calculation(
                        operating_cash_flow_fact=(
                            input_facts[0]
                        ),
                        net_profit_fact=input_facts[1],
                        created_at=self.clock(),
                    )
                )

            elif (
                formula_id
                == "current_ratio_formula"
            ):
                calculation = (
                    build_current_ratio_calculation(
                        current_assets_fact=(
                            input_facts[0]
                        ),
                        current_liabilities_fact=(
                            input_facts[1]
                        ),
                        created_at=self.clock(),
                    )
                )

            elif (
                formula_id
                == "debt_to_equity_ratio_formula"
            ):
                calculation = (
                    build_debt_to_equity_ratio_calculation(
                        total_liabilities_fact=(
                            input_facts[0]
                        ),
                        total_equity_fact=(
                            input_facts[1]
                        ),
                        created_at=self.clock(),
                    )
                )

            elif (
                formula_id
                == (
                    "effective_income_tax_rate_formula"
                )
            ):
                calculation = (
                    build_effective_income_tax_rate_calculation(
                        income_tax_expense_fact=(
                            input_facts[0]
                        ),
                        total_profit_fact=(
                            input_facts[1]
                        ),
                        created_at=self.clock(),
                    )
                )

            else:
                raise (
                    ComplexOracleCalculatorAdapterError(
                        "formula routing is missing: "
                        f"{formula_id}"
                    )
                )
            
            if (
                calculation.calculation_id
                != calculation_id
            ):
                raise (
                    ComplexOracleCalculatorAdapterError(
                        "确定性计算生成的 "
                        "calculation_id 与 Plan "
                        "不一致："
                        "expected="
                        f"{calculation_id}, "
                        "actual="
                        f"{calculation.calculation_id}"
                    )
                )

            if (
                calculation.formula_id
                != formula_id
            ):
                raise (
                    ComplexOracleCalculatorAdapterError(
                        "确定性计算返回了错误的 "
                        "formula_id"
                    )
                )

            if (
                tuple(
                    calculation.input_fact_ids
                )
                != input_fact_ids
            ):
                raise (
                    ComplexOracleCalculatorAdapterError(
                        "确定性计算改变了输入 "
                        "Fact 的顺序"
                    )
                )

            if (
                calculation.validation_status
                != "verified"
            ):
                raise (
                    ComplexOracleCalculatorAdapterError(
                        "确定性计算结果不是 "
                        "verified"
                    )
                )

            latency_ms = (
                perf_counter()
                - timer_start
            ) * 1000

            return ComplexCalculationTrace(
                calculation_id=(
                    calculation
                    .calculation_id
                ),
                metric_id=(
                    calculation.metric_id
                ),
                formula_id=(
                    calculation.formula_id
                ),
                input_fact_ids=tuple(
                    calculation
                    .input_fact_ids
                ),
                status="completed",
                result_value=(
                    calculation.result_value
                ),
                result_unit=(
                    calculation.result_unit
                ),
                latency_ms=latency_ms,
                error_message=None,
            )

        except Exception as exc:
            return self._build_failed_trace(
                calculation_id=calculation_id,
                metric_id=metric_id,
                formula_id=formula_id,
                input_fact_ids=input_fact_ids,
                timer_start=timer_start,
                error_message=str(exc),
            )

    def _build_failed_trace(
        self,
        *,
        calculation_id: str,
        metric_id: str,
        formula_id: str,
        input_fact_ids: tuple[str, ...],
        timer_start: float,
        error_message: str,
    ) -> ComplexCalculationTrace:
        latency_ms = (
            perf_counter()
            - timer_start
        ) * 1000

        return ComplexCalculationTrace(
            calculation_id=calculation_id,
            metric_id=metric_id,
            formula_id=formula_id,
            input_fact_ids=input_fact_ids,
            status="failed",
            result_value=None,
            result_unit=None,
            latency_ms=latency_ms,
            error_message=error_message,
        )