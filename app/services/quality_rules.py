from datetime import datetime
from decimal import Decimal

from app.schemas.enums import (
    PeriodType,
    Severity,
    StatementType,
    UnitCode,
    ValidationStatus,
)
from app.schemas.financial_fact import FinancialFact
from app.schemas.quality_signal import QualitySignal
from app.services.financial_calculator import (
    FinancialCalculationError,
    calculate_growth_rate,
)


class QualityRuleEvaluationError(ValueError):
    """财务事实不能用于目标经营质量规则。"""


def _validate_fact(
    fact: FinancialFact,
    *,
    expected_metric_id: str,
    expected_statement_type: StatementType,
) -> None:
    """检查参与质量规则的单条财务事实。"""

    if (
        fact.validation_status
        is not ValidationStatus.VERIFIED
    ):
        raise QualityRuleEvaluationError(
            "质量规则只能使用 verified FinancialFact"
        )

    if fact.metric_id != expected_metric_id:
        raise QualityRuleEvaluationError(
            "输入事实指标错误："
            f"预期 {expected_metric_id}，"
            f"实际 {fact.metric_id}"
        )

    if fact.statement_type is not expected_statement_type:
        raise QualityRuleEvaluationError(
            "输入事实所属报表类型不正确"
        )

    if fact.period_type is not PeriodType.DURATION:
        raise QualityRuleEvaluationError(
            "利润和经营现金流必须是 duration 事实"
        )

    if fact.normalized_unit is not UnitCode.CNY:
        raise QualityRuleEvaluationError(
            "输入事实必须归一化为 CNY"
        )

    if fact.currency != "CNY":
        raise QualityRuleEvaluationError(
            "输入事实 currency 必须为 CNY"
        )


def _validate_year_pair(
    current_fact: FinancialFact,
    previous_fact: FinancialFact,
) -> None:
    """检查同一指标的本期和上期事实。"""

    if current_fact.company_id != previous_fact.company_id:
        raise QualityRuleEvaluationError(
            "本期和上期事实必须属于同一公司"
        )

    if (
        current_fact.fiscal_year
        != previous_fact.fiscal_year + 1
    ):
        raise QualityRuleEvaluationError(
            "本期和上期事实必须是连续财务年度"
        )

    if (
        current_fact.statement_scope
        != previous_fact.statement_scope
    ):
        raise QualityRuleEvaluationError(
            "本期和上期事实必须使用相同报表口径"
        )


def _validate_cross_metric_context(
    current_profit: FinancialFact,
    current_cash_flow: FinancialFact,
    previous_profit: FinancialFact,
    previous_cash_flow: FinancialFact,
) -> None:
    """检查利润和现金流事实是否可以相互比较。"""

    if (
        current_profit.company_id
        != current_cash_flow.company_id
    ):
        raise QualityRuleEvaluationError(
            "利润与经营现金流必须属于同一公司"
        )

    if (
        previous_profit.company_id
        != previous_cash_flow.company_id
    ):
        raise QualityRuleEvaluationError(
            "上期利润与经营现金流必须属于同一公司"
        )

    if (
        current_profit.fiscal_year
        != current_cash_flow.fiscal_year
    ):
        raise QualityRuleEvaluationError(
            "本期利润与经营现金流必须属于同一年度"
        )

    if (
        previous_profit.fiscal_year
        != previous_cash_flow.fiscal_year
    ):
        raise QualityRuleEvaluationError(
            "上期利润与经营现金流必须属于同一年度"
        )

    if (
        current_profit.statement_scope
        != current_cash_flow.statement_scope
    ):
        raise QualityRuleEvaluationError(
            "利润与经营现金流必须使用相同报表口径"
        )


def _resolve_mismatch_severity(
    growth_gap: Decimal,
) -> Severity:
    """按照增长率差距确定信号严重程度。"""

    if growth_gap >= Decimal("20"):
        return Severity.HIGH

    if growth_gap >= Decimal("5"):
        return Severity.MEDIUM

    return Severity.LOW


def evaluate_profit_cash_flow_mismatch(
    *,
    current_profit_fact: FinancialFact,
    previous_profit_fact: FinancialFact,
    current_cash_flow_fact: FinancialFact,
    previous_cash_flow_fact: FinancialFact,
    created_at: datetime,
) -> QualitySignal | None:
    """识别利润增长但经营现金流下降的质量信号。"""

    _validate_fact(
        current_profit_fact,
        expected_metric_id=(
            "net_profit_attributable_to_parent"
        ),
        expected_statement_type=(
            StatementType.INCOME_STATEMENT
        ),
    )

    _validate_fact(
        previous_profit_fact,
        expected_metric_id=(
            "net_profit_attributable_to_parent"
        ),
        expected_statement_type=(
            StatementType.INCOME_STATEMENT
        ),
    )

    _validate_fact(
        current_cash_flow_fact,
        expected_metric_id=(
            "net_cash_flow_from_operating_activities"
        ),
        expected_statement_type=(
            StatementType.CASH_FLOW_STATEMENT
        ),
    )

    _validate_fact(
        previous_cash_flow_fact,
        expected_metric_id=(
            "net_cash_flow_from_operating_activities"
        ),
        expected_statement_type=(
            StatementType.CASH_FLOW_STATEMENT
        ),
    )

    _validate_year_pair(
        current_profit_fact,
        previous_profit_fact,
    )

    _validate_year_pair(
        current_cash_flow_fact,
        previous_cash_flow_fact,
    )

    _validate_cross_metric_context(
        current_profit_fact,
        current_cash_flow_fact,
        previous_profit_fact,
        previous_cash_flow_fact,
    )

    try:
        profit_growth = calculate_growth_rate(
            current_value=(
                current_profit_fact.normalized_value
            ),
            previous_value=(
                previous_profit_fact.normalized_value
            ),
        )

        cash_flow_growth = calculate_growth_rate(
            current_value=(
                current_cash_flow_fact.normalized_value
            ),
            previous_value=(
                previous_cash_flow_fact.normalized_value
            ),
        )
    except FinancialCalculationError as exc:
        raise QualityRuleEvaluationError(
            str(exc)
        ) from exc

    if not (
        profit_growth > 0
        and cash_flow_growth < 0
    ):
        return None

    growth_gap = (
        profit_growth - cash_flow_growth
    )

    severity = _resolve_mismatch_severity(
        growth_gap
    )

    fiscal_year = current_profit_fact.fiscal_year
    company_id = current_profit_fact.company_id

    return QualitySignal(
        signal_id=(
            f"signal_{company_id}_{fiscal_year}_"
            "profit_cash_flow_mismatch"
        ),
        rule_id="profit_cash_flow_mismatch",
        signal_type="profit_cash_flow_mismatch",
        company_id=company_id,
        fiscal_year=fiscal_year,
        severity=severity,
        title="利润增长与经营现金流下降背离",
        summary=(
            f"归母净利润同比增长 {profit_growth}%，"
            f"但经营现金流同比增长率为 "
            f"{cash_flow_growth}%，"
            f"二者相差 {growth_gap} 个百分点。"
        ),
        metric_values={
            "profit_growth_rate": profit_growth,
            "operating_cash_flow_growth_rate": (
                cash_flow_growth
            ),
            "growth_gap": growth_gap,
        },
        input_fact_ids=[
            current_profit_fact.fact_id,
            previous_profit_fact.fact_id,
            current_cash_flow_fact.fact_id,
            previous_cash_flow_fact.fact_id,
        ],
        rule_version="v1",
        validation_status=ValidationStatus.VERIFIED,
        validated_by="deterministic_quality_rule_v1",
        created_at=created_at,
    )