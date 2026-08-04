from decimal import Decimal, ROUND_HALF_UP


_DECIMAL_FOUR_PLACES = Decimal("0.0001")
_PERCENT_MULTIPLIER = Decimal("100")


class FinancialCalculationError(ValueError):
    """财务计算输入不满足业务约束。"""


def _round_to_four_places(
    value: Decimal,
) -> Decimal:
    """使用四舍五入保留四位小数。"""

    return value.quantize(
        _DECIMAL_FOUR_PLACES,
        rounding=ROUND_HALF_UP,
    )


def calculate_growth_rate(
    current_value: Decimal,
    previous_value: Decimal,
) -> Decimal:
    """计算同比增长率，结果单位为百分比。"""

    if previous_value <= 0:
        raise FinancialCalculationError(
            "同比增长率要求上期值大于 0"
        )

    growth_rate = (
        (current_value - previous_value)
        / previous_value
        * _PERCENT_MULTIPLIER
    )

    return _round_to_four_places(
        growth_rate
    )


def calculate_ratio(
    numerator: Decimal,
    denominator: Decimal,
) -> Decimal:
    """计算两个财务数值之间的比率。"""

    if denominator == 0:
        raise FinancialCalculationError(
            "比率计算的分母不能为 0"
        )

    ratio = numerator / denominator

    return _round_to_four_places(
        ratio
    )


def calculate_gross_profit_margin(
    revenue: Decimal,
    operating_cost: Decimal,
) -> Decimal:
    """计算毛利率，结果单位为百分比。"""

    if revenue <= 0:
        raise FinancialCalculationError(
            "毛利率计算要求营业收入大于 0"
        )

    if operating_cost < 0:
        raise FinancialCalculationError(
            "毛利率计算要求营业成本不能小于 0"
        )

    gross_profit_margin = (
        (revenue - operating_cost)
        / revenue
        * _PERCENT_MULTIPLIER
    )

    return _round_to_four_places(
        gross_profit_margin
    )


def calculate_selling_and_r_and_d_expense_ratio(
    *,
    revenue: Decimal,
    selling_expenses: Decimal,
    research_and_development_expenses: Decimal,
) -> Decimal:
    """计算销售费用与研发费用合计占营业收入比例。"""

    if revenue <= 0:
        raise FinancialCalculationError(
            "费用率计算要求营业收入大于 0"
        )

    if selling_expenses < 0:
        raise FinancialCalculationError(
            "费用率计算要求销售费用不能小于 0"
        )

    if research_and_development_expenses < 0:
        raise FinancialCalculationError(
            "费用率计算要求研发费用不能小于 0"
        )

    expense_ratio = (
        (
            selling_expenses
            + research_and_development_expenses
        )
        / revenue
        * _PERCENT_MULTIPLIER
    )

    return _round_to_four_places(
        expense_ratio
    )


def calculate_operating_cash_flow_to_net_profit_ratio(
    *,
    operating_cash_flow: Decimal,
    net_profit: Decimal,
) -> Decimal:
    """计算经营活动现金流量净额与净利润的比率。"""

    if net_profit == 0:
        raise FinancialCalculationError(
            "经营现金净流量与净利润比率要求"
            "净利润不能为 0"
        )

    ratio = operating_cash_flow / net_profit

    return _round_to_four_places(
        ratio
    )

def calculate_current_ratio(
    *,
    current_assets: Decimal,
    current_liabilities: Decimal,
) -> Decimal:
    """Calculate current assets divided by current liabilities."""

    if current_assets < 0:
        raise FinancialCalculationError(
            "current_assets must be >= 0"
        )

    if current_liabilities <= 0:
        raise FinancialCalculationError(
            "current_liabilities must be > 0"
        )

    result = (
        current_assets
        / current_liabilities
    )

    return _round_to_four_places(result)


def calculate_debt_to_equity_ratio(
    *,
    total_liabilities: Decimal,
    total_equity: Decimal,
) -> Decimal:
    """Calculate total liabilities divided by total equity."""

    if total_liabilities < 0:
        raise FinancialCalculationError(
            "total_liabilities must be >= 0"
        )

    if total_equity == 0:
        raise FinancialCalculationError(
            "total_equity must not be 0"
        )

    result = (
        total_liabilities
        / total_equity
    )

    return _round_to_four_places(result)


def calculate_effective_income_tax_rate(
    *,
    income_tax_expense: Decimal,
    total_profit: Decimal,
) -> Decimal:
    """Calculate income tax expense as a percent of total profit."""

    if total_profit == 0:
        raise FinancialCalculationError(
            "total_profit must not be 0"
        )

    result = (
        income_tax_expense
        / total_profit
        * _PERCENT_MULTIPLIER
    )

    return _round_to_four_places(result)