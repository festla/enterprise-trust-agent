from datetime import datetime
from decimal import Decimal
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class DerivedCalculation(BaseModel):
    """由已有事实计算得到的新指标结果。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    calculation_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
        min_length=1,
    )

    metric_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
        min_length=1,
    )

    formula_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
        min_length=1,
    )

    result_value: Decimal

    result_unit: str

    input_fact_ids: list[str] = Field(
        min_length=1,
    )

    calculation_version: str = Field(
        min_length=1,
    )

    validation_status: str

    validated_by: str | None = None

    created_at: datetime

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        """校验计算结果基本约束。"""

        if not self.input_fact_ids:
            raise ValueError(
                "计算结果必须包含输入事实"
            )

        if (
            self.validation_status == "verified"
            and self.validated_by is None
        ):
            raise ValueError(
                "verified 计算结果必须填写 validated_by"
            )

        if (
            self.created_at.tzinfo is None
            or self.created_at.utcoffset() is None
        ):
            raise ValueError(
                "datetime 必须包含时区信息"
            )

        return self