from datetime import datetime
from decimal import Decimal
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .enums import (
    Severity,
    ValidationStatus,
)


class QualitySignal(BaseModel):
    """由确定性规则生成的经营质量信号。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    signal_id: str = Field(
        min_length=1,
        pattern=r"^signal_[a-z0-9_]+$",
    )

    rule_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
    )

    signal_type: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
    )

    company_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
    )

    fiscal_year: int = Field(
        ge=2000,
        le=2100,
    )

    severity: Severity

    title: str = Field(
        min_length=1,
    )

    summary: str = Field(
        min_length=1,
    )

    metric_values: dict[str, Decimal] = Field(
        min_length=1,
    )

    input_fact_ids: list[str] = Field(
        min_length=1,
    )

    rule_version: str = Field(
        min_length=1,
    )

    validation_status: ValidationStatus = (
        ValidationStatus.PENDING
    )

    validated_by: str | None = Field(
        default=None,
        min_length=1,
    )

    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def validate_timezone(
        cls,
        value: datetime,
    ) -> datetime:
        """信号创建时间必须包含时区。"""

        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "datetime 必须包含时区信息"
            )

        return value

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        """检查质量信号的基本可信约束。"""

        if len(set(self.input_fact_ids)) != len(
            self.input_fact_ids
        ):
            raise ValueError(
                "input_fact_ids 不能包含重复事实"
            )

        if (
            self.validation_status
            is ValidationStatus.VERIFIED
            and self.validated_by is None
        ):
            raise ValueError(
                "verified 质量信号必须填写 validated_by"
            )

        return self