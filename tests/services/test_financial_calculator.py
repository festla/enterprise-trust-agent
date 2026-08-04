from decimal import Decimal

import pytest

from app.services.financial_calculator import (
    FinancialCalculationError,
    calculate_growth_rate,
    calculate_gross_profit_margin,
    calculate_operating_cash_flow_to_net_profit_ratio,
    calculate_ratio,
    calculate_selling_and_r_and_d_expense_ratio,
    calculate_current_ratio,
    calculate_debt_to_equity_ratio,
    calculate_effective_income_tax_rate,
)


def test_calculate_midea_revenue_growth_rate() -> None:
    """应正确计算美的集团营业收入同比增长率。"""

    result = calculate_growth_rate(
        current_value=Decimal("456451731000"),
        previous_value=Decimal("407149600000"),
    )

    assert result == Decimal("12.1091")


def test_calculate_negative_growth_rate() -> None:
    """本期值下降时应返回负增长率。"""

    result = calculate_growth_rate(
        current_value=Decimal("90"),
        previous_value=Decimal("100"),
    )

    assert result == Decimal("-10.0000")


@pytest.mark.parametrize(
    "previous_value",
    [
        Decimal("0"),
        Decimal("-100"),
    ],
)
def test_reject_invalid_growth_rate_base(
    previous_value: Decimal,
) -> None:
    """上期值不大于零时应拒绝计算同比增长率。"""

    with pytest.raises(
        FinancialCalculationError,
        match="上期值大于 0",
    ):
        calculate_growth_rate(
            current_value=Decimal("100"),
            previous_value=previous_value,
        )


def test_calculate_ratio_and_reject_zero_denominator() -> None:
    """应计算普通比率，并拒绝零分母。"""

    result = calculate_ratio(
        numerator=Decimal("3"),
        denominator=Decimal("2"),
    )

    assert result == Decimal("1.5000")

    with pytest.raises(
        FinancialCalculationError,
        match="分母不能为 0",
    ):
        calculate_ratio(
            numerator=Decimal("3"),
            denominator=Decimal("0"),
        )


def test_calculate_gree_gross_profit_margin() -> None:
    """应根据格力真实收入与成本计算毛利率。"""

    result = calculate_gross_profit_margin(
        revenue=Decimal("170447058533.57"),
        operating_cost=Decimal(
            "119641353216.21"
        ),
    )

    assert result == Decimal("29.8073")


def test_reject_invalid_gross_profit_margin_inputs() -> None:
    """收入无效或成本为负时应拒绝计算毛利率。"""

    with pytest.raises(
        FinancialCalculationError,
        match="营业收入大于 0",
    ):
        calculate_gross_profit_margin(
            revenue=Decimal("0"),
            operating_cost=Decimal("100"),
        )

    with pytest.raises(
        FinancialCalculationError,
        match="营业成本不能小于 0",
    ):
        calculate_gross_profit_margin(
            revenue=Decimal("100"),
            operating_cost=Decimal("-1"),
        )

def test_calculate_midea_selling_and_r_and_d_expense_ratio(
) -> None:
    """应正确计算美的销售与研发费用合计占收入比例。"""

    result = (
        calculate_selling_and_r_and_d_expense_ratio(
            revenue=Decimal(
                "407149600000"
            ),
            selling_expenses=Decimal(
                "38753649000"
            ),
            research_and_development_expenses=(
                Decimal(
                    "16232771000"
                )
            ),
        )
    )

    assert result == Decimal("13.5052")


def test_reject_invalid_expense_ratio_revenue(
) -> None:
    """营业收入不大于零时应拒绝计算费用率。"""

    with pytest.raises(
        FinancialCalculationError,
        match="营业收入大于 0",
    ):
        calculate_selling_and_r_and_d_expense_ratio(
            revenue=Decimal("0"),
            selling_expenses=Decimal("20"),
            research_and_development_expenses=(
                Decimal("10")
            ),
        )


@pytest.mark.parametrize(
    (
        "selling_expenses",
        "research_and_development_expenses",
        "expected_message",
    ),
    [
        (
            Decimal("-1"),
            Decimal("10"),
            "销售费用不能小于 0",
        ),
        (
            Decimal("10"),
            Decimal("-1"),
            "研发费用不能小于 0",
        ),
    ],
)
def test_reject_negative_expense_ratio_inputs(
    selling_expenses: Decimal,
    research_and_development_expenses: Decimal,
    expected_message: str,
) -> None:
    """销售费用和研发费用不能使用负数。"""

    with pytest.raises(
        FinancialCalculationError,
        match=expected_message,
    ):
        calculate_selling_and_r_and_d_expense_ratio(
            revenue=Decimal("100"),
            selling_expenses=selling_expenses,
            research_and_development_expenses=(
                research_and_development_expenses
            ),
        )


def test_calculate_haier_operating_cash_flow_to_net_profit_ratio(
) -> None:
    """应正确计算海尔经营现金净流量与净利润比率。"""

    result = (
        calculate_operating_cash_flow_to_net_profit_ratio(
            operating_cash_flow=Decimal(
                "26543081911.96"
            ),
            net_profit=Decimal(
                "19575612501.68"
            ),
        )
    )

    assert result == Decimal("1.3559")


def test_operating_cash_flow_ratio_allows_negative_cash_flow(
) -> None:
    """经营现金流为负时应保留负比率。"""

    result = (
        calculate_operating_cash_flow_to_net_profit_ratio(
            operating_cash_flow=Decimal("-50"),
            net_profit=Decimal("100"),
        )
    )

    assert result == Decimal("-0.5000")


def test_reject_zero_net_profit_for_cash_flow_ratio(
) -> None:
    """净利润为零时不能计算现金流与利润比率。"""

    with pytest.raises(
        FinancialCalculationError,
        match="净利润不能为 0",
    ):
        calculate_operating_cash_flow_to_net_profit_ratio(
            operating_cash_flow=Decimal("100"),
            net_profit=Decimal("0"),
        )

def test_calculate_current_ratio() -> None:
    result = calculate_current_ratio(
        current_assets=Decimal("1"),
        current_liabilities=Decimal("3"),
    )

    assert result == Decimal("0.3333")


@pytest.mark.parametrize(
    "current_liabilities",
    [
        Decimal("0"),
        Decimal("-1"),
    ],
)
def test_reject_invalid_current_liabilities(
    current_liabilities: Decimal,
) -> None:
    with pytest.raises(
        FinancialCalculationError,
        match="current_liabilities must be > 0",
    ):
        calculate_current_ratio(
            current_assets=Decimal("100"),
            current_liabilities=current_liabilities,
        )


def test_reject_negative_current_assets() -> None:
    with pytest.raises(
        FinancialCalculationError,
        match="current_assets must be >= 0",
    ):
        calculate_current_ratio(
            current_assets=Decimal("-1"),
            current_liabilities=Decimal("100"),
        )


def test_calculate_debt_to_equity_ratio() -> None:
    result = calculate_debt_to_equity_ratio(
        total_liabilities=Decimal("2"),
        total_equity=Decimal("3"),
    )

    assert result == Decimal("0.6667")

    negative_equity_result = (
        calculate_debt_to_equity_ratio(
            total_liabilities=Decimal("300"),
            total_equity=Decimal("-200"),
        )
    )

    assert negative_equity_result == Decimal(
        "-1.5000"
    )


@pytest.mark.parametrize(
    (
        "total_liabilities",
        "total_equity",
        "expected_message",
    ),
    [
        (
            Decimal("-1"),
            Decimal("100"),
            "total_liabilities must be >= 0",
        ),
        (
            Decimal("100"),
            Decimal("0"),
            "total_equity must not be 0",
        ),
    ],
)
def test_reject_invalid_debt_to_equity_inputs(
    total_liabilities: Decimal,
    total_equity: Decimal,
    expected_message: str,
) -> None:
    with pytest.raises(
        FinancialCalculationError,
        match=expected_message,
    ):
        calculate_debt_to_equity_ratio(
            total_liabilities=total_liabilities,
            total_equity=total_equity,
        )


def test_calculate_effective_income_tax_rate() -> None:
    result = calculate_effective_income_tax_rate(
        income_tax_expense=Decimal("1"),
        total_profit=Decimal("6"),
    )

    assert result == Decimal("16.6667")

    tax_benefit_result = (
        calculate_effective_income_tax_rate(
            income_tax_expense=Decimal("-5"),
            total_profit=Decimal("100"),
        )
    )

    assert tax_benefit_result == Decimal(
        "-5.0000"
    )


def test_reject_zero_total_profit() -> None:
    with pytest.raises(
        FinancialCalculationError,
        match="total_profit must not be 0",
    ):
        calculate_effective_income_tax_rate(
            income_tax_expense=Decimal("10"),
            total_profit=Decimal("0"),
        )