from datetime import (
    date,
    datetime,
    timedelta,
    timezone,
)
from decimal import Decimal

import pytest

from app.schemas.financial_fact import FinancialFact
from app.services.derived_calculation_service import (
    DerivedCalculationServiceError,
    build_current_ratio_calculation,
    build_debt_to_equity_ratio_calculation,
    build_effective_income_tax_rate_calculation,
    build_gross_profit_margin_calculation,
    build_operating_cash_flow_to_net_profit_ratio_calculation,
    build_selling_and_r_and_d_expense_ratio_calculation,
)

CHINA_TIMEZONE = timezone(
    timedelta(hours=8)
)

TEST_TIME = datetime(
    2026,
    7,
    24,
    12,
    0,
    tzinfo=CHINA_TIMEZONE,
)


def build_fact(
    *,
    fact_id: str,
    metric_id: str,
    raw_value: str,
    company_id: str = "gree_electric",
    report_id: str = "gree_electric_2025",
    fiscal_year: int = 2025,
    validation_status: str = "verified",
) -> FinancialFact:
    """构造用于派生计算测试的财务事实。"""

    validated_by = (
        "manual_review"
        if validation_status == "verified"
        else None
    )

    validated_at = (
        TEST_TIME
        if validation_status == "verified"
        else None
    )

    row_labels = {
        "revenue": "营业收入",
        "operating_cost": "营业成本",
        "selling_expenses": "销售费用",
        "research_and_development_expenses": (
            "研发费用"
        ),
        "net_cash_flow_from_operating_activities": (
            "经营活动产生的现金流量净额"
        ),
        "net_profit": "净利润",
    }

    statement_types = {
        "net_cash_flow_from_operating_activities": (
            "cash_flow_statement"
        ),
    }

    statement_type = statement_types.get(
        metric_id,
        "income_statement",
    )
    return FinancialFact(
        fact_id=fact_id,
        company_id=company_id,
        report_id=report_id,
        metric_id=metric_id,
        fiscal_year=fiscal_year,
        statement_type=statement_type,
        statement_scope="consolidated",
        period_type="duration",
        period_start=date(fiscal_year, 1, 1),
        period_end=date(fiscal_year, 12, 31),
        raw_value=Decimal(raw_value),
        raw_unit="CNY",
        unit_multiplier=Decimal("1"),
        normalized_value=Decimal(raw_value),
        normalized_unit="CNY",
        currency="CNY",
        table_name="合并利润表",
        row_label=row_labels.get(
            metric_id,
            "测试指标",
        ),
        column_label=f"{fiscal_year}年度",
        is_comparative_value=False,
        restatement_status="not_applicable",
        primary_evidence_id=(
            f"evidence_{fact_id}"
        ),
        validation_status=validation_status,
        validated_by=validated_by,
        validated_at=validated_at,
        source_version=(
            f"{company_id}_{fiscal_year}_v1"
        ),
        created_at=TEST_TIME,
        updated_at=TEST_TIME,
    )

def build_balance_sheet_fact(
    *,
    fact_id: str,
    metric_id: str,
    raw_value: str,
    fact_date: date = date(2024, 12, 31),
    company_id: str = "midea_group",
    report_id: str = "midea_group_2024",
    fiscal_year: int = 2024,
) -> FinancialFact:
    """Build a verified instant balance-sheet fact."""

    return FinancialFact(
        fact_id=fact_id,
        company_id=company_id,
        report_id=report_id,
        metric_id=metric_id,
        fiscal_year=fiscal_year,
        statement_type="balance_sheet",
        statement_scope="consolidated",
        period_type="instant",
        period_start=None,
        period_end=None,
        as_of_date=fact_date,
        raw_value=Decimal(raw_value),
        raw_unit="CNY",
        unit_multiplier=Decimal("1"),
        normalized_value=Decimal(raw_value),
        normalized_unit="CNY",
        currency="CNY",
        table_name="consolidated_balance_sheet",
        row_label=metric_id,
        column_label=f"{fiscal_year}-12-31",
        is_comparative_value=False,
        restatement_status="not_applicable",
        primary_evidence_id=(
            f"evidence_{fact_id}"
        ),
        validation_status="verified",
        validated_by="manual_review",
        validated_at=TEST_TIME,
        source_version=(
            f"{company_id}_{fiscal_year}_v1"
        ),
        created_at=TEST_TIME,
        updated_at=TEST_TIME,
    )

def test_build_gross_profit_margin_calculation() -> None:
    """应根据两条已核验事实生成可信毛利率。"""

    revenue = build_fact(
        fact_id="fact_gree_2025_revenue",
        metric_id="revenue",
        raw_value="170447058533.57",
    )

    operating_cost = build_fact(
        fact_id="fact_gree_2025_operating_cost",
        metric_id="operating_cost",
        raw_value="119641353216.21",
    )

    calculation = (
        build_gross_profit_margin_calculation(
            revenue_fact=revenue,
            operating_cost_fact=operating_cost,
            created_at=TEST_TIME,
        )
    )

    assert calculation.calculation_id == (
        "calculation_gree_electric_"
        "2025_gross_profit_margin"
    )

    assert calculation.metric_id == (
        "gross_profit_margin"
    )

    assert calculation.formula_id == (
        "gross_profit_margin_formula"
    )

    assert calculation.result_value == Decimal(
        "29.8073"
    )

    assert calculation.result_unit == "percent"

    assert calculation.input_fact_ids == [
        "fact_gree_2025_revenue",
        "fact_gree_2025_operating_cost",
    ]

    assert calculation.validation_status == "verified"
    assert calculation.validated_by == (
        "deterministic_calculator_v1"
    )


def test_reject_unverified_input_fact() -> None:
    """未核验事实不能参与可信派生计算。"""

    revenue = build_fact(
        fact_id="fact_gree_2025_revenue",
        metric_id="revenue",
        raw_value="170447058533.57",
        validation_status="pending",
    )

    operating_cost = build_fact(
        fact_id="fact_gree_2025_operating_cost",
        metric_id="operating_cost",
        raw_value="119641353216.21",
    )

    with pytest.raises(
        DerivedCalculationServiceError,
        match="只能使用 verified",
    ):
        build_gross_profit_margin_calculation(
            revenue_fact=revenue,
            operating_cost_fact=operating_cost,
            created_at=TEST_TIME,
        )


def test_reject_wrong_metric_role() -> None:
    """收入参数不能传入其他指标事实。"""

    invalid_revenue = build_fact(
        fact_id="fact_gree_2025_inventory",
        metric_id="inventory",
        raw_value="100",
    )

    operating_cost = build_fact(
        fact_id="fact_gree_2025_operating_cost",
        metric_id="operating_cost",
        raw_value="80",
    )

    with pytest.raises(
        DerivedCalculationServiceError,
        match="必须为 revenue",
    ):
        build_gross_profit_margin_calculation(
            revenue_fact=invalid_revenue,
            operating_cost_fact=operating_cost,
            created_at=TEST_TIME,
        )


def test_reject_facts_from_different_reports() -> None:
    """不同报告中的数值不能组成同一毛利率。"""

    revenue = build_fact(
        fact_id="fact_gree_2025_revenue",
        metric_id="revenue",
        raw_value="100",
    )

    operating_cost = build_fact(
        fact_id="fact_gree_2024_operating_cost",
        metric_id="operating_cost",
        raw_value="80",
        report_id="gree_electric_2024",
        fiscal_year=2024,
    )

    with pytest.raises(
        DerivedCalculationServiceError,
        match="同一报告",
    ):
        build_gross_profit_margin_calculation(
            revenue_fact=revenue,
            operating_cost_fact=operating_cost,
            created_at=TEST_TIME,
        )


def test_reject_facts_from_different_companies() -> None:
    """不同公司的数值不能组成同一毛利率。"""

    revenue = build_fact(
        fact_id="fact_gree_2025_revenue",
        metric_id="revenue",
        raw_value="100",
    )

    operating_cost = build_fact(
        fact_id="fact_midea_2025_operating_cost",
        metric_id="operating_cost",
        raw_value="80",
        company_id="midea_group",
        report_id="midea_group_2025",
    )

    with pytest.raises(
        DerivedCalculationServiceError,
        match="同一公司",
    ):
        build_gross_profit_margin_calculation(
            revenue_fact=revenue,
            operating_cost_fact=operating_cost,
            created_at=TEST_TIME,
        )


def test_build_selling_and_r_and_d_expense_ratio_calculation(
) -> None:
    """应根据三个已核验事实生成费用率计算。"""

    revenue = build_fact(
        fact_id="fact_midea_group_2024_revenue",
        metric_id="revenue",
        raw_value="407149600000",
        company_id="midea_group",
        report_id="midea_group_2024",
        fiscal_year=2024,
    )

    selling_expenses = build_fact(
        fact_id=(
            "fact_midea_group_2024_"
            "selling_expenses"
        ),
        metric_id="selling_expenses",
        raw_value="38753649000",
        company_id="midea_group",
        report_id="midea_group_2024",
        fiscal_year=2024,
    )

    research_and_development_expenses = (
        build_fact(
            fact_id=(
                "fact_midea_group_2024_"
                "research_and_development_expenses"
            ),
            metric_id=(
                "research_and_development_expenses"
            ),
            raw_value="16232771000",
            company_id="midea_group",
            report_id="midea_group_2024",
            fiscal_year=2024,
        )
    )

    calculation = (
        build_selling_and_r_and_d_expense_ratio_calculation(
            revenue_fact=revenue,
            selling_expenses_fact=selling_expenses,
            research_and_development_expenses_fact=(
                research_and_development_expenses
            ),
            created_at=TEST_TIME,
        )
    )

    assert calculation.calculation_id == (
        "calculation_midea_group_2024_"
        "selling_and_r_and_d_expense_ratio"
    )

    assert calculation.metric_id == (
        "selling_and_r_and_d_expense_ratio"
    )

    assert calculation.formula_id == (
        "selling_and_r_and_d_expense_ratio_formula"
    )

    assert calculation.result_value == Decimal(
        "13.5052"
    )

    assert calculation.result_unit == "percent"

    assert calculation.input_fact_ids == [
        "fact_midea_group_2024_revenue",
        (
            "fact_midea_group_2024_"
            "selling_expenses"
        ),
        (
            "fact_midea_group_2024_"
            "research_and_development_expenses"
        ),
    ]


def test_reject_wrong_expense_ratio_metric_role(
) -> None:
    """费用率输入必须使用正确的指标角色。"""

    revenue = build_fact(
        fact_id="fact_midea_group_2024_revenue",
        metric_id="revenue",
        raw_value="100",
        company_id="midea_group",
        report_id="midea_group_2024",
        fiscal_year=2024,
    )

    invalid_selling_expenses = build_fact(
        fact_id="fact_midea_group_2024_inventory",
        metric_id="inventory",
        raw_value="20",
        company_id="midea_group",
        report_id="midea_group_2024",
        fiscal_year=2024,
    )

    research_and_development_expenses = (
        build_fact(
            fact_id=(
                "fact_midea_group_2024_"
                "research_and_development_expenses"
            ),
            metric_id=(
                "research_and_development_expenses"
            ),
            raw_value="10",
            company_id="midea_group",
            report_id="midea_group_2024",
            fiscal_year=2024,
        )
    )

    with pytest.raises(
        DerivedCalculationServiceError,
        match="必须为 selling_expenses",
    ):
        build_selling_and_r_and_d_expense_ratio_calculation(
            revenue_fact=revenue,
            selling_expenses_fact=(
                invalid_selling_expenses
            ),
            research_and_development_expenses_fact=(
                research_and_development_expenses
            ),
            created_at=TEST_TIME,
        )


def test_build_operating_cash_flow_to_net_profit_ratio_calculation(
) -> None:
    """应根据现金流和净利润生成可信比率。"""

    operating_cash_flow = build_fact(
        fact_id=(
            "fact_haier_smart_home_2024_"
            "net_cash_flow_from_operating_activities"
        ),
        metric_id=(
            "net_cash_flow_from_operating_activities"
        ),
        raw_value="26543081911.96",
        company_id="haier_smart_home",
        report_id="haier_smart_home_2024",
        fiscal_year=2024,
    )

    net_profit = build_fact(
        fact_id=(
            "fact_haier_smart_home_2024_net_profit"
        ),
        metric_id="net_profit",
        raw_value="19575612501.68",
        company_id="haier_smart_home",
        report_id="haier_smart_home_2024",
        fiscal_year=2024,
    )

    calculation = (
        build_operating_cash_flow_to_net_profit_ratio_calculation(
            operating_cash_flow_fact=(
                operating_cash_flow
            ),
            net_profit_fact=net_profit,
            created_at=TEST_TIME,
        )
    )

    assert calculation.calculation_id == (
        "calculation_haier_smart_home_2024_"
        "operating_cash_flow_to_net_profit_ratio"
    )

    assert calculation.metric_id == (
        "operating_cash_flow_to_net_profit_ratio"
    )

    assert calculation.formula_id == (
        "operating_cash_flow_to_net_profit_ratio_formula"
    )

    assert calculation.result_value == Decimal(
        "1.3559"
    )

    assert calculation.result_unit == "ratio"

    assert calculation.input_fact_ids == [
        (
            "fact_haier_smart_home_2024_"
            "net_cash_flow_from_operating_activities"
        ),
        (
            "fact_haier_smart_home_2024_"
            "net_profit"
        ),
    ]


def test_reject_cash_profit_facts_from_different_reports(
) -> None:
    """现金流和净利润必须来自同一份报告。"""

    operating_cash_flow = build_fact(
        fact_id=(
            "fact_haier_smart_home_2024_"
            "net_cash_flow_from_operating_activities"
        ),
        metric_id=(
            "net_cash_flow_from_operating_activities"
        ),
        raw_value="100",
        company_id="haier_smart_home",
        report_id="haier_smart_home_2024",
        fiscal_year=2024,
    )

    net_profit = build_fact(
        fact_id=(
            "fact_haier_smart_home_2023_net_profit"
        ),
        metric_id="net_profit",
        raw_value="80",
        company_id="haier_smart_home",
        report_id="haier_smart_home_2023",
        fiscal_year=2023,
    )

    with pytest.raises(
        DerivedCalculationServiceError,
        match="同一报告",
    ):
        build_operating_cash_flow_to_net_profit_ratio_calculation(
            operating_cash_flow_fact=(
                operating_cash_flow
            ),
            net_profit_fact=net_profit,
            created_at=TEST_TIME,
        )


def test_build_current_ratio_calculation() -> None:
    current_assets = build_balance_sheet_fact(
        fact_id=(
            "fact_midea_group_2024_current_assets"
        ),
        metric_id="current_assets",
        raw_value="500",
    )

    current_liabilities = build_balance_sheet_fact(
        fact_id=(
            "fact_midea_group_2024_current_liabilities"
        ),
        metric_id="current_liabilities",
        raw_value="250",
    )

    calculation = build_current_ratio_calculation(
        current_assets_fact=current_assets,
        current_liabilities_fact=current_liabilities,
        created_at=TEST_TIME,
    )

    assert calculation.calculation_id == (
        "calculation_midea_group_2024_current_ratio"
    )
    assert calculation.metric_id == "current_ratio"
    assert calculation.formula_id == (
        "current_ratio_formula"
    )
    assert calculation.result_value == Decimal(
        "2.0000"
    )
    assert calculation.result_unit == "ratio"
    assert calculation.input_fact_ids == [
        "fact_midea_group_2024_current_assets",
        "fact_midea_group_2024_current_liabilities",
    ]
    assert calculation.validation_status == "verified"


def test_current_ratio_rejects_different_as_of_dates(
) -> None:
    current_assets = build_balance_sheet_fact(
        fact_id=(
            "fact_midea_group_2024_current_assets"
        ),
        metric_id="current_assets",
        raw_value="500",
        fact_date=date(2024, 12, 31),
    )

    current_liabilities = build_balance_sheet_fact(
        fact_id=(
            "fact_midea_group_2024_current_liabilities"
        ),
        metric_id="current_liabilities",
        raw_value="250",
        fact_date=date(2024, 12, 30),
    )

    with pytest.raises(
        DerivedCalculationServiceError,
        match="same as_of_date",
    ):
        build_current_ratio_calculation(
            current_assets_fact=current_assets,
            current_liabilities_fact=current_liabilities,
            created_at=TEST_TIME,
        )


def test_build_debt_to_equity_ratio_calculation(
) -> None:
    total_liabilities = build_balance_sheet_fact(
        fact_id=(
            "fact_midea_group_2024_total_liabilities"
        ),
        metric_id="total_liabilities",
        raw_value="300",
    )

    total_equity = build_balance_sheet_fact(
        fact_id=(
            "fact_midea_group_2024_total_equity"
        ),
        metric_id="total_equity",
        raw_value="200",
    )

    calculation = (
        build_debt_to_equity_ratio_calculation(
            total_liabilities_fact=total_liabilities,
            total_equity_fact=total_equity,
            created_at=TEST_TIME,
        )
    )

    assert calculation.calculation_id == (
        "calculation_midea_group_2024_"
        "debt_to_equity_ratio"
    )
    assert calculation.metric_id == (
        "debt_to_equity_ratio"
    )
    assert calculation.formula_id == (
        "debt_to_equity_ratio_formula"
    )
    assert calculation.result_value == Decimal(
        "1.5000"
    )
    assert calculation.result_unit == "ratio"
    assert calculation.input_fact_ids == [
        "fact_midea_group_2024_total_liabilities",
        "fact_midea_group_2024_total_equity",
    ]


def test_build_effective_income_tax_rate_calculation(
) -> None:
    income_tax_expense = build_fact(
        fact_id=(
            "fact_gree_electric_2025_"
            "income_tax_expense"
        ),
        metric_id="income_tax_expense",
        raw_value="10",
    )

    total_profit = build_fact(
        fact_id=(
            "fact_gree_electric_2025_total_profit"
        ),
        metric_id="total_profit",
        raw_value="40",
    )

    calculation = (
        build_effective_income_tax_rate_calculation(
            income_tax_expense_fact=(
                income_tax_expense
            ),
            total_profit_fact=total_profit,
            created_at=TEST_TIME,
        )
    )

    assert calculation.calculation_id == (
        "calculation_gree_electric_2025_"
        "effective_income_tax_rate"
    )
    assert calculation.metric_id == (
        "effective_income_tax_rate"
    )
    assert calculation.formula_id == (
        "effective_income_tax_rate_formula"
    )
    assert calculation.result_value == Decimal(
        "25.0000"
    )
    assert calculation.result_unit == "percent"
    assert calculation.input_fact_ids == [
        (
            "fact_gree_electric_2025_"
            "income_tax_expense"
        ),
        "fact_gree_electric_2025_total_profit",
    ]


def test_effective_tax_rate_rejects_wrong_metric_role(
) -> None:
    invalid_income_tax_expense = build_fact(
        fact_id="fact_gree_electric_2025_net_profit",
        metric_id="net_profit",
        raw_value="10",
    )

    total_profit = build_fact(
        fact_id=(
            "fact_gree_electric_2025_total_profit"
        ),
        metric_id="total_profit",
        raw_value="40",
    )

    with pytest.raises(
        DerivedCalculationServiceError,
        match="income_tax_expense_fact.metric_id",
    ):
        build_effective_income_tax_rate_calculation(
            income_tax_expense_fact=(
                invalid_income_tax_expense
            ),
            total_profit_fact=total_profit,
            created_at=TEST_TIME,
        )