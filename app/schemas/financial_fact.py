from datetime import date, datetime
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
    EvidenceSupportType,
    PeriodType,
    RestatementStatus,
    StatementScope,
    StatementType,
    UnitCode,
    ValidationStatus,
)


MONEY_UNIT_MULTIPLIERS: dict[UnitCode, Decimal] = {
    UnitCode.CNY: Decimal("1"),
    UnitCode.CNY_THOUSAND: Decimal("1000"),
    UnitCode.CNY_TEN_THOUSAND: Decimal("10000"),
    UnitCode.CNY_MILLION: Decimal("1000000"),
    UnitCode.CNY_HUNDRED_MILLION: Decimal("100000000"),
}


class FinancialFact(BaseModel):
    """可查询、可计算、可追溯的结构化财务事实。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    fact_id: str = Field(
        min_length=1,
        pattern=r"^fact_[a-z0-9_]+$",
        description="财务事实唯一 ID",
    )

    company_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
        description="事实所属公司 ID",
    )

    report_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
        description="事实来源报告 ID",
    )

    metric_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
        description="标准财务指标 ID",
    )

    fiscal_year: int = Field(
        ge=2000,
        le=2100,
        description="该数值实际对应的财务年度",
    )

    statement_type: StatementType

    statement_scope: StatementScope

    period_type: PeriodType

    period_start: date | None = None

    period_end: date | None = None

    as_of_date: date | None = None

    raw_value: Decimal = Field(
        description="报告中披露的原始数值",
    )

    raw_unit: UnitCode

    unit_multiplier: Decimal = Field(
        gt=0,
        description="原始单位换算到标准单位的倍率",
    )

    normalized_value: Decimal = Field(
        description="单位归一化后的精确数值",
    )

    normalized_unit: UnitCode

    currency: str | None = Field(
        default=None,
        pattern=r"^[A-Z]{3}$",
        description="货币代码，金额指标通常为 CNY",
    )

    table_name: str = Field(
        min_length=1,
        description="原始报表或表格名称",
    )

    row_label: str = Field(
        min_length=1,
        description="原始指标行名称",
    )

    column_label: str = Field(
        min_length=1,
        description="原始年度列名称",
    )

    is_comparative_value: bool = False

    restatement_status: RestatementStatus

    primary_evidence_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
        description="直接支持该事实的主要证据 ID",
    )

    validation_status: ValidationStatus = (
        ValidationStatus.PENDING
    )

    validated_by: str | None = Field(
        default=None,
        min_length=1,
    )

    validated_at: datetime | None = None

    source_version: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
        description="来源文档版本 ID",
    )

    created_at: datetime

    updated_at: datetime

    @field_validator(
        "created_at",
        "updated_at",
        "validated_at",
    )
    @classmethod
    def validate_timezone(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        """非空时间必须包含时区信息。"""

        if value is None:
            return value

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "datetime 必须包含时区信息"
            )

        return value

    @model_validator(mode="after")
    def validate_fact_contract(self) -> Self:
        """检查财务事实的跨字段业务约束。"""

        self._validate_report_reference()
        self._validate_period_fields()
        self._validate_statement_period_type()
        self._validate_unit_normalization()
        self._validate_currency()
        self._validate_restatement_status()
        self._validate_verified_fact()
        self._validate_time_order()

        return self

    def _validate_report_reference(self) -> None:
        """检查 report_id 与公司、年度关系。"""

        try:
            report_company_id, report_year_text = (
                self.report_id.rsplit("_", maxsplit=1)
            )
            report_year = int(report_year_text)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "report_id 必须以四位报告年度结尾，"
                "例如 midea_2025"
            ) from exc

        if report_company_id != self.company_id:
            raise ValueError(
                "report_id 中的 company_id "
                "必须与 FinancialFact.company_id 一致"
            )

        if not self.is_comparative_value:
            if report_year != self.fiscal_year:
                raise ValueError(
                    "非比较列数值的 fiscal_year "
                    "必须与来源报告年度一致"
                )

        if self.is_comparative_value:
            if report_year < self.fiscal_year:
                raise ValueError(
                    "比较列数值的来源报告年度 "
                    "不能早于该数值对应的财务年度"
                )

    def _validate_period_fields(self) -> None:
        """检查时点指标和期间指标的日期字段。"""

        if self.period_type is PeriodType.INSTANT:
            if self.as_of_date is None:
                raise ValueError(
                    "instant 指标必须填写 as_of_date"
                )

            if (
                self.period_start is not None
                or self.period_end is not None
            ):
                raise ValueError(
                    "instant 指标不能填写 "
                    "period_start 或 period_end"
                )

        if self.period_type is PeriodType.DURATION:
            if (
                self.period_start is None
                or self.period_end is None
            ):
                raise ValueError(
                    "duration 指标必须同时填写 "
                    "period_start 和 period_end"
                )

            if self.as_of_date is not None:
                raise ValueError(
                    "duration 指标不能填写 as_of_date"
                )

            if self.period_end < self.period_start:
                raise ValueError(
                    "period_end 不能早于 period_start"
                )

    def _validate_statement_period_type(self) -> None:
        """检查报表类型与期间类型是否一致。"""

        if (
            self.statement_type
            is StatementType.BALANCE_SHEET
            and self.period_type
            is not PeriodType.INSTANT
        ):
            raise ValueError(
                "资产负债表事实必须是 instant 指标"
            )

        duration_statement_types = {
            StatementType.INCOME_STATEMENT,
            StatementType.CASH_FLOW_STATEMENT,
        }

        if (
            self.statement_type
            in duration_statement_types
            and self.period_type
            is not PeriodType.DURATION
        ):
            raise ValueError(
                "利润表和现金流量表事实 "
                "必须是 duration 指标"
            )

    def _validate_unit_normalization(self) -> None:
        """检查单位倍率和归一化结果。"""

        if (
            self.raw_unit is UnitCode.TEXT
            or self.normalized_unit is UnitCode.TEXT
        ):
            raise ValueError(
                "FinancialFact 当前只保存数值事实，"
                "不能使用 text 单位"
            )

        if self.raw_unit in MONEY_UNIT_MULTIPLIERS:
            expected_multiplier = (
                MONEY_UNIT_MULTIPLIERS[self.raw_unit]
            )

            if self.unit_multiplier != expected_multiplier:
                raise ValueError(
                    "金额原始单位与 unit_multiplier 不一致"
                )

            if self.normalized_unit is not UnitCode.CNY:
                raise ValueError(
                    "金额指标归一化后的单位必须为 CNY"
                )

        else:
            if self.unit_multiplier != Decimal("1"):
                raise ValueError(
                    "非金额单位的 unit_multiplier 必须为 1"
                )

            if self.normalized_unit is not self.raw_unit:
                raise ValueError(
                    "非金额单位归一化前后的单位必须一致"
                )

        expected_normalized_value = (
            self.raw_value * self.unit_multiplier
        )

        if self.normalized_value != expected_normalized_value:
            raise ValueError(
                "normalized_value 必须等于 "
                "raw_value × unit_multiplier"
            )

    def _validate_currency(self) -> None:
        """检查金额与货币代码的一致性。"""

        currency_units = {
            UnitCode.CNY,
            UnitCode.CNY_PER_SHARE,
        }

        if self.normalized_unit in currency_units:
            if self.currency != "CNY":
                raise ValueError(
                    "人民币金额或每股金额指标 "
                    "必须填写 currency='CNY'"
                )

        elif self.currency is not None:
            raise ValueError(
                "percent、ratio、count 等非货币指标 "
                "不能填写 currency"
            )

    def _validate_restatement_status(self) -> None:
        """检查比较列与重列状态。"""

        if not self.is_comparative_value:
            if (
                self.restatement_status
                is not RestatementStatus.NOT_APPLICABLE
            ):
                raise ValueError(
                    "当前期间数值的 restatement_status "
                    "必须为 not_applicable"
                )

        if self.is_comparative_value:
            if (
                self.restatement_status
                is RestatementStatus.NOT_APPLICABLE
            ):
                raise ValueError(
                    "比较列数值必须记录其重列状态"
                )

            if (
                self.validation_status
                is ValidationStatus.VERIFIED
                and self.restatement_status
                is RestatementStatus.UNKNOWN
            ):
                raise ValueError(
                    "verified 比较列数值不能保留 "
                    "unknown 重列状态"
                )

    def _validate_verified_fact(self) -> None:
        """已核验事实必须保留核验信息。"""

        if (
            self.validation_status
            is not ValidationStatus.VERIFIED
        ):
            return

        if self.statement_scope is StatementScope.UNKNOWN:
            raise ValueError(
                "verified 财务事实不能使用 unknown 口径"
            )

        if self.validated_by is None:
            raise ValueError(
                "verified 财务事实必须填写 validated_by"
            )

        if self.validated_at is None:
            raise ValueError(
                "verified 财务事实必须填写 validated_at"
            )

    def _validate_time_order(self) -> None:
        """检查创建、更新和核验时间顺序。"""

        if self.updated_at < self.created_at:
            raise ValueError(
                "updated_at 不能早于 created_at"
            )

        if (
            self.validated_at is not None
            and self.validated_at < self.created_at
        ):
            raise ValueError(
                "validated_at 不能早于 created_at"
            )


class FactEvidenceLink(BaseModel):
    """财务事实与来源证据之间的关联。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    fact_id: str = Field(
        min_length=1,
        pattern=r"^fact_[a-z0-9_]+$",
    )

    evidence_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
    )

    support_type: EvidenceSupportType

    notes: str | None = Field(
        default=None,
        min_length=1,
    )