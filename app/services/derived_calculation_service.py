from datetime import datetime

from app.schemas.calculation import DerivedCalculation
from app.schemas.financial_fact import FinancialFact
from app.services.financial_calculator import (
    calculate_current_ratio,
    calculate_debt_to_equity_ratio,
    calculate_effective_income_tax_rate,
    calculate_gross_profit_margin,
    calculate_operating_cash_flow_to_net_profit_ratio,
    calculate_selling_and_r_and_d_expense_ratio,
)

class DerivedCalculationServiceError(ValueError):
    """输入事实不能用于目标派生计算。"""


def _validate_verified_fact(
    fact: FinancialFact,
) -> None:
    """参与可信计算的事实必须已经核验。"""

    if fact.validation_status.value != "verified":
        raise DerivedCalculationServiceError(
            "派生计算只能使用 verified FinancialFact"
        )


def _validate_same_fact_context(
    first: FinancialFact,
    second: FinancialFact,
) -> None:
    """两个输入事实必须属于同一报告和财务口径。"""

    if first.company_id != second.company_id:
        raise DerivedCalculationServiceError(
            "输入事实必须属于同一公司"
        )

    if first.report_id != second.report_id:
        raise DerivedCalculationServiceError(
            "输入事实必须来源于同一报告"
        )

    if first.fiscal_year != second.fiscal_year:
        raise DerivedCalculationServiceError(
            "输入事实必须属于同一财务年度"
        )

    if first.statement_scope != second.statement_scope:
        raise DerivedCalculationServiceError(
            "输入事实必须使用相同报表口径"
        )

    if first.period_type != second.period_type:
        raise DerivedCalculationServiceError(
            "输入事实必须使用相同期间类型"
        )

    if first.period_start != second.period_start:
        raise DerivedCalculationServiceError(
            "输入事实的 period_start 必须一致"
        )

    if first.period_end != second.period_end:
        raise DerivedCalculationServiceError(
            "输入事实的 period_end 必须一致"
        )

    if first.as_of_date != second.as_of_date:
        raise DerivedCalculationServiceError(
            "input facts must use the same as_of_date"
        )


def _validate_money_fact(
    fact: FinancialFact,
) -> None:
    """派生计算的金额输入必须归一化为人民币元。"""

    if fact.normalized_unit.value != "CNY":
        raise DerivedCalculationServiceError(
            "派生计算输入事实必须归一化为 CNY"
        )

    if fact.currency != "CNY":
        raise DerivedCalculationServiceError(
            "派生计算输入事实的 currency 必须为 CNY"
        )


def build_gross_profit_margin_calculation(
    *,
    revenue_fact: FinancialFact,
    operating_cost_fact: FinancialFact,
    created_at: datetime,
) -> DerivedCalculation:
    """根据营业收入和营业成本生成可信毛利率结果。"""

    _validate_verified_fact(revenue_fact)
    _validate_verified_fact(operating_cost_fact)

    if revenue_fact.metric_id != "revenue":
        raise DerivedCalculationServiceError(
            "revenue_fact.metric_id 必须为 revenue"
        )

    if operating_cost_fact.metric_id != "operating_cost":
        raise DerivedCalculationServiceError(
            "operating_cost_fact.metric_id "
            "必须为 operating_cost"
        )

    _validate_same_fact_context(
        revenue_fact,
        operating_cost_fact,
    )

    _validate_money_fact(revenue_fact)
    _validate_money_fact(operating_cost_fact)

    result = calculate_gross_profit_margin(
        revenue=revenue_fact.normalized_value,
        operating_cost=(
            operating_cost_fact.normalized_value
        ),
    )

    calculation_id = (
        "calculation_"
        f"{revenue_fact.company_id}_"
        f"{revenue_fact.fiscal_year}_"
        "gross_profit_margin"
    )

    return DerivedCalculation(
        calculation_id=calculation_id,
        metric_id="gross_profit_margin",
        formula_id="gross_profit_margin_formula",
        result_value=result,
        result_unit="percent",
        input_fact_ids=[
            revenue_fact.fact_id,
            operating_cost_fact.fact_id,
        ],
        calculation_version="v1",
        validation_status="verified",
        validated_by="deterministic_calculator_v1",
        created_at=created_at,
    )


def build_selling_and_r_and_d_expense_ratio_calculation(
    *,
    revenue_fact: FinancialFact,
    selling_expenses_fact: FinancialFact,
    research_and_development_expenses_fact: FinancialFact,
    created_at: datetime,
) -> DerivedCalculation:
    """生成销售费用与研发费用合计占收入比例。"""

    input_facts = (
        revenue_fact,
        selling_expenses_fact,
        research_and_development_expenses_fact,
    )

    for fact in input_facts:
        _validate_verified_fact(fact)
        _validate_money_fact(fact)

    if revenue_fact.metric_id != "revenue":
        raise DerivedCalculationServiceError(
            "revenue_fact.metric_id 必须为 revenue"
        )

    if (
        selling_expenses_fact.metric_id
        != "selling_expenses"
    ):
        raise DerivedCalculationServiceError(
            "selling_expenses_fact.metric_id "
            "必须为 selling_expenses"
        )

    if (
        research_and_development_expenses_fact
        .metric_id
        != "research_and_development_expenses"
    ):
        raise DerivedCalculationServiceError(
            "research_and_development_expenses_fact."
            "metric_id 必须为 "
            "research_and_development_expenses"
        )

    _validate_same_fact_context(
        revenue_fact,
        selling_expenses_fact,
    )

    _validate_same_fact_context(
        revenue_fact,
        research_and_development_expenses_fact,
    )

    result = (
        calculate_selling_and_r_and_d_expense_ratio(
            revenue=revenue_fact.normalized_value,
            selling_expenses=(
                selling_expenses_fact
                .normalized_value
            ),
            research_and_development_expenses=(
                research_and_development_expenses_fact
                .normalized_value
            ),
        )
    )

    calculation_id = (
        "calculation_"
        f"{revenue_fact.company_id}_"
        f"{revenue_fact.fiscal_year}_"
        "selling_and_r_and_d_expense_ratio"
    )

    return DerivedCalculation(
        calculation_id=calculation_id,
        metric_id=(
            "selling_and_r_and_d_expense_ratio"
        ),
        formula_id=(
            "selling_and_r_and_d_expense_ratio_formula"
        ),
        result_value=result,
        result_unit="percent",
        input_fact_ids=[
            revenue_fact.fact_id,
            selling_expenses_fact.fact_id,
            (
                research_and_development_expenses_fact
                .fact_id
            ),
        ],
        calculation_version="v1",
        validation_status="verified",
        validated_by="deterministic_calculator_v1",
        created_at=created_at,
    )


def build_operating_cash_flow_to_net_profit_ratio_calculation(
    *,
    operating_cash_flow_fact: FinancialFact,
    net_profit_fact: FinancialFact,
    created_at: datetime,
) -> DerivedCalculation:
    """生成经营活动现金流量净额与净利润比率。"""

    _validate_verified_fact(
        operating_cash_flow_fact
    )
    _validate_verified_fact(
        net_profit_fact
    )

    if (
        operating_cash_flow_fact.metric_id
        != "net_cash_flow_from_operating_activities"
    ):
        raise DerivedCalculationServiceError(
            "operating_cash_flow_fact.metric_id "
            "必须为 "
            "net_cash_flow_from_operating_activities"
        )

    if net_profit_fact.metric_id != "net_profit":
        raise DerivedCalculationServiceError(
            "net_profit_fact.metric_id "
            "必须为 net_profit"
        )

    _validate_same_fact_context(
        operating_cash_flow_fact,
        net_profit_fact,
    )

    _validate_money_fact(
        operating_cash_flow_fact
    )
    _validate_money_fact(
        net_profit_fact
    )

    result = (
        calculate_operating_cash_flow_to_net_profit_ratio(
            operating_cash_flow=(
                operating_cash_flow_fact
                .normalized_value
            ),
            net_profit=(
                net_profit_fact.normalized_value
            ),
        )
    )

    calculation_id = (
        "calculation_"
        f"{operating_cash_flow_fact.company_id}_"
        f"{operating_cash_flow_fact.fiscal_year}_"
        "operating_cash_flow_to_net_profit_ratio"
    )

    return DerivedCalculation(
        calculation_id=calculation_id,
        metric_id=(
            "operating_cash_flow_to_net_profit_ratio"
        ),
        formula_id=(
            "operating_cash_flow_to_net_profit_ratio_formula"
        ),
        result_value=result,
        result_unit="ratio",
        input_fact_ids=[
            operating_cash_flow_fact.fact_id,
            net_profit_fact.fact_id,
        ],
        calculation_version="v1",
        validation_status="verified",
        validated_by="deterministic_calculator_v1",
        created_at=created_at,
    )


def build_current_ratio_calculation(
    *,
    current_assets_fact: FinancialFact,
    current_liabilities_fact: FinancialFact,
    created_at: datetime,
) -> DerivedCalculation:
    """Build a verified current-ratio calculation."""

    input_facts = (
        current_assets_fact,
        current_liabilities_fact,
    )

    for fact in input_facts:
        _validate_verified_fact(fact)
        _validate_money_fact(fact)

    if current_assets_fact.metric_id != "current_assets":
        raise DerivedCalculationServiceError(
            "current_assets_fact.metric_id "
            "must be current_assets"
        )

    if (
        current_liabilities_fact.metric_id
        != "current_liabilities"
    ):
        raise DerivedCalculationServiceError(
            "current_liabilities_fact.metric_id "
            "must be current_liabilities"
        )

    _validate_same_fact_context(
        current_assets_fact,
        current_liabilities_fact,
    )

    result = calculate_current_ratio(
        current_assets=(
            current_assets_fact.normalized_value
        ),
        current_liabilities=(
            current_liabilities_fact.normalized_value
        ),
    )

    calculation_id = (
        "calculation_"
        f"{current_assets_fact.company_id}_"
        f"{current_assets_fact.fiscal_year}_"
        "current_ratio"
    )

    return DerivedCalculation(
        calculation_id=calculation_id,
        metric_id="current_ratio",
        formula_id="current_ratio_formula",
        result_value=result,
        result_unit="ratio",
        input_fact_ids=[
            current_assets_fact.fact_id,
            current_liabilities_fact.fact_id,
        ],
        calculation_version="v1",
        validation_status="verified",
        validated_by="deterministic_calculator_v1",
        created_at=created_at,
    )


def build_debt_to_equity_ratio_calculation(
    *,
    total_liabilities_fact: FinancialFact,
    total_equity_fact: FinancialFact,
    created_at: datetime,
) -> DerivedCalculation:
    """Build a verified debt-to-equity calculation."""

    input_facts = (
        total_liabilities_fact,
        total_equity_fact,
    )

    for fact in input_facts:
        _validate_verified_fact(fact)
        _validate_money_fact(fact)

    if (
        total_liabilities_fact.metric_id
        != "total_liabilities"
    ):
        raise DerivedCalculationServiceError(
            "total_liabilities_fact.metric_id "
            "must be total_liabilities"
        )

    if total_equity_fact.metric_id != "total_equity":
        raise DerivedCalculationServiceError(
            "total_equity_fact.metric_id "
            "must be total_equity"
        )

    _validate_same_fact_context(
        total_liabilities_fact,
        total_equity_fact,
    )

    result = calculate_debt_to_equity_ratio(
        total_liabilities=(
            total_liabilities_fact.normalized_value
        ),
        total_equity=(
            total_equity_fact.normalized_value
        ),
    )

    calculation_id = (
        "calculation_"
        f"{total_liabilities_fact.company_id}_"
        f"{total_liabilities_fact.fiscal_year}_"
        "debt_to_equity_ratio"
    )

    return DerivedCalculation(
        calculation_id=calculation_id,
        metric_id="debt_to_equity_ratio",
        formula_id="debt_to_equity_ratio_formula",
        result_value=result,
        result_unit="ratio",
        input_fact_ids=[
            total_liabilities_fact.fact_id,
            total_equity_fact.fact_id,
        ],
        calculation_version="v1",
        validation_status="verified",
        validated_by="deterministic_calculator_v1",
        created_at=created_at,
    )


def build_effective_income_tax_rate_calculation(
    *,
    income_tax_expense_fact: FinancialFact,
    total_profit_fact: FinancialFact,
    created_at: datetime,
) -> DerivedCalculation:
    """Build a verified effective-income-tax-rate calculation."""

    input_facts = (
        income_tax_expense_fact,
        total_profit_fact,
    )

    for fact in input_facts:
        _validate_verified_fact(fact)
        _validate_money_fact(fact)

    if (
        income_tax_expense_fact.metric_id
        != "income_tax_expense"
    ):
        raise DerivedCalculationServiceError(
            "income_tax_expense_fact.metric_id "
            "must be income_tax_expense"
        )

    if total_profit_fact.metric_id != "total_profit":
        raise DerivedCalculationServiceError(
            "total_profit_fact.metric_id "
            "must be total_profit"
        )

    _validate_same_fact_context(
        income_tax_expense_fact,
        total_profit_fact,
    )

    result = calculate_effective_income_tax_rate(
        income_tax_expense=(
            income_tax_expense_fact.normalized_value
        ),
        total_profit=(
            total_profit_fact.normalized_value
        ),
    )

    calculation_id = (
        "calculation_"
        f"{income_tax_expense_fact.company_id}_"
        f"{income_tax_expense_fact.fiscal_year}_"
        "effective_income_tax_rate"
    )

    return DerivedCalculation(
        calculation_id=calculation_id,
        metric_id="effective_income_tax_rate",
        formula_id=(
            "effective_income_tax_rate_formula"
        ),
        result_value=result,
        result_unit="percent",
        input_fact_ids=[
            income_tax_expense_fact.fact_id,
            total_profit_fact.fact_id,
        ],
        calculation_version="v1",
        validation_status="verified",
        validated_by="deterministic_calculator_v1",
        created_at=created_at,
    )