from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.schemas.calculation import DerivedCalculation


CHINA_TIMEZONE = timezone(
    timedelta(hours=8)
)


def build_valid_calculation(
    **overrides: object,
) -> dict[str, object]:
    """构造一条默认合法的派生计算结果。"""

    data: dict[str, object] = {
        "calculation_id": (
            "calculation_midea_2025_revenue_growth"
        ),
        "metric_id": "revenue_growth_rate",
        "formula_id": "growth_rate_formula",
        "result_value": Decimal("12.1091"),
        "result_unit": "percent",
        "input_fact_ids": [
            "fact_midea_group_2025_revenue",
            "fact_midea_group_2024_revenue",
        ],
        "calculation_version": "v1",
        "validation_status": "verified",
        "validated_by": "deterministic_calculator",
        "created_at": datetime(
            2026,
            7,
            24,
            12,
            0,
            tzinfo=CHINA_TIMEZONE,
        ),
    }

    data.update(overrides)

    return data


def test_create_valid_derived_calculation() -> None:
    """合法数据应创建派生计算结果。"""

    calculation = DerivedCalculation.model_validate(
        build_valid_calculation()
    )

    assert calculation.calculation_id == (
        "calculation_midea_2025_revenue_growth"
    )

    assert calculation.metric_id == (
        "revenue_growth_rate"
    )

    assert calculation.formula_id == (
        "growth_rate_formula"
    )

    assert calculation.result_value == Decimal(
        "12.1091"
    )

    assert calculation.result_unit == "percent"

    assert calculation.input_fact_ids == [
        "fact_midea_group_2025_revenue",
        "fact_midea_group_2024_revenue",
    ]

    assert calculation.validation_status == "verified"
    assert calculation.validated_by == (
        "deterministic_calculator"
    )


def test_reject_verified_calculation_without_validator() -> None:
    """verified 计算结果必须记录核验者。"""

    with pytest.raises(
        ValidationError,
        match="verified 计算结果必须填写 validated_by",
    ):
        DerivedCalculation.model_validate(
            build_valid_calculation(
                validated_by=None,
            )
        )


def test_reject_calculation_without_input_facts() -> None:
    """计算结果必须引用至少一条输入事实。"""

    with pytest.raises(ValidationError):
        DerivedCalculation.model_validate(
            build_valid_calculation(
                input_fact_ids=[],
            )
        )


def test_reject_naive_created_at() -> None:
    """created_at 不包含时区时应拒绝。"""

    with pytest.raises(
        ValidationError,
        match="datetime 必须包含时区信息",
    ):
        DerivedCalculation.model_validate(
            build_valid_calculation(
                created_at=datetime(
                    2026,
                    7,
                    24,
                    12,
                    0,
                ),
            )
        )


def test_reject_invalid_calculation_id() -> None:
    """calculation_id 必须符合小写下划线格式。"""

    with pytest.raises(ValidationError):
        DerivedCalculation.model_validate(
            build_valid_calculation(
                calculation_id=(
                    "Calculation-Midea-2025"
                ),
            )
        )


def test_reject_unknown_extra_field() -> None:
    """模型不应静默接受数据契约外的字段。"""

    with pytest.raises(ValidationError):
        DerivedCalculation.model_validate(
            build_valid_calculation(
                unknown_field="unexpected",
            )
        )