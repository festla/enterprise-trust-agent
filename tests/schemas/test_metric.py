from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.schemas.enums import (
    AliasMatchType,
    MetricOrigin,
    MetricValueType,
    PeriodType,
    StatementScope,
    StatementType,
    UnitCode,
)
from app.schemas.metric import FinancialMetric, MetricAlias


def build_valid_reported_metric_data() -> dict:
    """生成合法的直接披露指标数据。"""

    now = datetime.now(timezone.utc)

    return {
        "metric_id": "revenue",
        "display_name_cn": "营业收入",
        "display_name_en": "Revenue",
        "description": (
            "企业日常经营活动形成的收入，"
            "不等同于营业总收入"
        ),
        "metric_origin": "reported",
        "statement_type": "income_statement",
        "period_type": "duration",
        "default_unit": "CNY",
        "allowed_scopes": [
            "consolidated",
            "parent_company",
        ],
        "value_type": "decimal",
        "is_core_metric": True,
        "confusable_metric_ids": [
            "total_operating_revenue",
        ],
        "formula_id": None,
        "status": "active",
        "created_at": now,
        "updated_at": now,
    }


def test_create_valid_reported_metric() -> None:
    """合法直接披露指标应创建成功。"""

    metric = FinancialMetric(
        **build_valid_reported_metric_data()
    )

    assert metric.metric_id == "revenue"
    assert metric.metric_origin is MetricOrigin.REPORTED
    assert metric.statement_type is StatementType.INCOME_STATEMENT
    assert metric.period_type is PeriodType.DURATION
    assert metric.default_unit is UnitCode.CNY
    assert metric.value_type is MetricValueType.DECIMAL


def test_create_valid_derived_metric() -> None:
    """派生指标必须能够关联固定公式。"""

    data = build_valid_reported_metric_data()
    data.update(
        {
            "metric_id": "gross_profit_margin",
            "display_name_cn": "销售毛利率",
            "display_name_en": "Gross Profit Margin",
            "description": (
                "营业收入减营业成本后占营业收入的比例"
            ),
            "metric_origin": "derived",
            "statement_type": "other",
            "period_type": "duration",
            "default_unit": "percent",
            "allowed_scopes": ["consolidated"],
            "confusable_metric_ids": [],
            "formula_id": "gross_profit_margin_v1",
        }
    )

    metric = FinancialMetric(**data)

    assert metric.metric_origin is MetricOrigin.DERIVED
    assert metric.formula_id == "gross_profit_margin_v1"


def test_reject_reported_metric_with_formula() -> None:
    """直接披露指标不能填写派生公式。"""

    data = build_valid_reported_metric_data()
    data["formula_id"] = "unexpected_formula"

    with pytest.raises(ValidationError):
        FinancialMetric(**data)


def test_reject_derived_metric_without_formula() -> None:
    """派生指标必须填写 formula_id。"""

    data = build_valid_reported_metric_data()
    data["metric_origin"] = "derived"
    data["formula_id"] = None

    with pytest.raises(ValidationError):
        FinancialMetric(**data)


def test_reject_balance_sheet_duration_metric() -> None:
    """资产负债表指标必须属于时点指标。"""

    data = build_valid_reported_metric_data()
    data["metric_id"] = "inventory"
    data["display_name_cn"] = "存货"
    data["statement_type"] = "balance_sheet"
    data["period_type"] = "duration"
    data["confusable_metric_ids"] = []

    with pytest.raises(ValidationError):
        FinancialMetric(**data)


def test_reject_income_statement_instant_metric() -> None:
    """利润表指标必须属于期间指标。"""

    data = build_valid_reported_metric_data()
    data["period_type"] = "instant"

    with pytest.raises(ValidationError):
        FinancialMetric(**data)


def test_reject_empty_allowed_scopes() -> None:
    """指标至少需要一个允许口径。"""

    data = build_valid_reported_metric_data()
    data["allowed_scopes"] = []

    with pytest.raises(ValidationError):
        FinancialMetric(**data)


def test_reject_duplicate_allowed_scopes() -> None:
    """允许口径不能重复。"""

    data = build_valid_reported_metric_data()
    data["allowed_scopes"] = [
        "consolidated",
        "consolidated",
    ]

    with pytest.raises(ValidationError):
        FinancialMetric(**data)


def test_reject_unknown_allowed_scope() -> None:
    """标准指标定义不能将 unknown 作为允许口径。"""

    data = build_valid_reported_metric_data()
    data["allowed_scopes"] = ["unknown"]

    with pytest.raises(ValidationError):
        FinancialMetric(**data)


def test_reject_self_confusable_metric() -> None:
    """指标不能把自身标记为易混淆指标。"""

    data = build_valid_reported_metric_data()
    data["confusable_metric_ids"] = ["revenue"]

    with pytest.raises(ValidationError):
        FinancialMetric(**data)


def test_reject_invalid_confusable_metric_id() -> None:
    """易混淆指标 ID 必须符合统一命名规则。"""

    data = build_valid_reported_metric_data()
    data["confusable_metric_ids"] = ["Total Revenue"]

    with pytest.raises(ValidationError):
        FinancialMetric(**data)


def test_reject_metric_updated_before_created() -> None:
    """指标更新时间不能早于创建时间。"""

    now = datetime.now(timezone.utc)

    data = build_valid_reported_metric_data()
    data["created_at"] = now
    data["updated_at"] = now - timedelta(minutes=1)

    with pytest.raises(ValidationError):
        FinancialMetric(**data)


def test_create_valid_exact_alias() -> None:
    """精确名称别名应创建成功。"""

    alias = MetricAlias(
        alias_id="alias_revenue_001",
        metric_id="revenue",
        alias="营业收入",
        statement_type="income_statement",
        statement_scope=None,
        match_type="exact",
        priority=1,
        notes=None,
        status="active",
    )

    assert alias.metric_id == "revenue"
    assert alias.match_type is AliasMatchType.EXACT
    assert alias.statement_type is StatementType.INCOME_STATEMENT


def test_create_valid_regex_alias() -> None:
    """合法正则别名应通过校验。"""

    alias = MetricAlias(
        alias_id="alias_parent_profit_regex",
        metric_id="net_profit_attributable_to_parent",
        alias=r"归属于.*股东的净利润",
        statement_type="income_statement",
        statement_scope="consolidated",
        match_type="regex",
        priority=20,
        notes="兼容不同年报中的完整名称",
        status="active",
    )

    assert alias.match_type is AliasMatchType.REGEX
    assert alias.statement_scope is StatementScope.CONSOLIDATED


def test_reject_invalid_regex_alias() -> None:
    """无法编译的正则别名应被拒绝。"""

    with pytest.raises(ValidationError):
        MetricAlias(
            alias_id="alias_invalid_regex",
            metric_id="revenue",
            alias="(营业收入",
            statement_type="income_statement",
            statement_scope=None,
            match_type="regex",
            priority=10,
            notes=None,
            status="active",
        )