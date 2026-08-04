from datetime import (
    date,
    datetime,
    timedelta,
    timezone,
)
from decimal import Decimal

import pytest

from app.schemas.financial_fact import FinancialFact
from app.services.quality_rules import (
    QualityRuleEvaluationError,
    evaluate_profit_cash_flow_mismatch,
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
    company_id: str,
    fiscal_year: int,
    metric_id: str,
    value: str,
    validation_status: str = "verified",
) -> FinancialFact:
    """构造经营质量规则测试使用的事实。"""

    statement_type = (
        "income_statement"
        if metric_id
        == "net_profit_attributable_to_parent"
        else "cash_flow_statement"
    )

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

    return FinancialFact(
        fact_id=fact_id,
        company_id=company_id,
        report_id=(
            f"{company_id}_{fiscal_year}"
        ),
        metric_id=metric_id,
        fiscal_year=fiscal_year,
        statement_type=statement_type,
        statement_scope="consolidated",
        period_type="duration",
        period_start=date(
            fiscal_year,
            1,
            1,
        ),
        period_end=date(
            fiscal_year,
            12,
            31,
        ),
        raw_value=Decimal(value),
        raw_unit="CNY",
        unit_multiplier=Decimal("1"),
        normalized_value=Decimal(value),
        normalized_unit="CNY",
        currency="CNY",
        table_name=(
            "合并利润表"
            if statement_type == "income_statement"
            else "合并现金流量表"
        ),
        row_label=metric_id,
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


def build_valid_inputs() -> dict[str, FinancialFact]:
    """构造利润增长、现金流下降的合法输入。"""

    return {
        "current_profit_fact": build_fact(
            fact_id="fact_demo_2025_profit",
            company_id="midea_group",
            fiscal_year=2025,
            metric_id=(
                "net_profit_attributable_to_parent"
            ),
            value="110",
        ),
        "previous_profit_fact": build_fact(
            fact_id="fact_demo_2024_profit",
            company_id="midea_group",
            fiscal_year=2024,
            metric_id=(
                "net_profit_attributable_to_parent"
            ),
            value="100",
        ),
        "current_cash_flow_fact": build_fact(
            fact_id="fact_demo_2025_cash_flow",
            company_id="midea_group",
            fiscal_year=2025,
            metric_id=(
                "net_cash_flow_from_operating_activities"
            ),
            value="90",
        ),
        "previous_cash_flow_fact": build_fact(
            fact_id="fact_demo_2024_cash_flow",
            company_id="midea_group",
            fiscal_year=2024,
            metric_id=(
                "net_cash_flow_from_operating_activities"
            ),
            value="100",
        ),
    }


def test_generate_profit_cash_flow_mismatch_signal() -> None:
    """利润增长、现金流下降时应生成质量信号。"""

    signal = evaluate_profit_cash_flow_mismatch(
        **build_valid_inputs(),
        created_at=TEST_TIME,
    )

    assert signal is not None

    assert signal.signal_type == (
        "profit_cash_flow_mismatch"
    )

    assert signal.company_id == "midea_group"
    assert signal.fiscal_year == 2025

    assert signal.metric_values[
        "profit_growth_rate"
    ] == Decimal("10.0000")

    assert signal.metric_values[
        "operating_cash_flow_growth_rate"
    ] == Decimal("-10.0000")

    assert signal.metric_values[
        "growth_gap"
    ] == Decimal("20.0000")

    assert signal.severity.value == "high"
    assert len(signal.input_fact_ids) == 4


def test_return_none_when_no_mismatch() -> None:
    """利润与现金流均增长时不应产生背离信号。"""

    inputs = build_valid_inputs()

    inputs["current_cash_flow_fact"] = build_fact(
        fact_id="fact_demo_2025_cash_flow",
        company_id="midea_group",
        fiscal_year=2025,
        metric_id=(
            "net_cash_flow_from_operating_activities"
        ),
        value="120",
    )

    signal = evaluate_profit_cash_flow_mismatch(
        **inputs,
        created_at=TEST_TIME,
    )

    assert signal is None


def test_reject_unverified_fact() -> None:
    """未核验事实不能参与质量分析。"""

    inputs = build_valid_inputs()

    inputs["current_profit_fact"] = build_fact(
        fact_id="fact_demo_2025_profit",
        company_id="midea_group",
        fiscal_year=2025,
        metric_id=(
            "net_profit_attributable_to_parent"
        ),
        value="110",
        validation_status="pending",
    )

    with pytest.raises(
        QualityRuleEvaluationError,
        match="只能使用 verified",
    ):
        evaluate_profit_cash_flow_mismatch(
            **inputs,
            created_at=TEST_TIME,
        )


def test_reject_non_consecutive_years() -> None:
    """质量规则不能比较不连续的年度。"""

    inputs = build_valid_inputs()

    inputs["previous_profit_fact"] = build_fact(
        fact_id="fact_demo_2023_profit",
        company_id="midea_group",
        fiscal_year=2023,
        metric_id=(
            "net_profit_attributable_to_parent"
        ),
        value="100",
    )

    with pytest.raises(
        QualityRuleEvaluationError,
        match="连续财务年度",
    ):
        evaluate_profit_cash_flow_mismatch(
            **inputs,
            created_at=TEST_TIME,
        )


def test_reject_different_companies() -> None:
    """不同公司的利润与现金流不能组合分析。"""

    inputs = build_valid_inputs()

    inputs["current_cash_flow_fact"] = build_fact(
        fact_id="fact_gree_2025_cash_flow",
        company_id="gree_electric",
        fiscal_year=2025,
        metric_id=(
            "net_cash_flow_from_operating_activities"
        ),
        value="90",
    )

    with pytest.raises(
        QualityRuleEvaluationError,
        match="同一公司",
    ):
        evaluate_profit_cash_flow_mismatch(
            **inputs,
            created_at=TEST_TIME,
        )
        