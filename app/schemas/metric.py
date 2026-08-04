import re
from datetime import datetime
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .enums import (
    AliasMatchType,
    MetricOrigin,
    MetricValueType,
    PeriodType,
    RecordStatus,
    StatementScope,
    StatementType,
    UnitCode,
)


_ID_PATTERN = re.compile(r"^[a-z0-9_]+$")


class FinancialMetric(BaseModel):
    """财务指标的标准业务定义。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    metric_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
        description="财务指标的标准唯一 ID",
    )

    display_name_cn: str = Field(
        min_length=1,
        description="指标标准中文名称",
    )

    display_name_en: str | None = Field(
        default=None,
        description="指标英文名称",
    )

    description: str = Field(
        min_length=1,
        description="指标业务含义及口径说明",
    )

    metric_origin: MetricOrigin

    statement_type: StatementType

    period_type: PeriodType

    default_unit: UnitCode

    allowed_scopes: list[StatementScope] = Field(
        min_length=1,
        description="指标允许使用的报表口径",
    )

    value_type: MetricValueType

    is_core_metric: bool = False

    confusable_metric_ids: list[str] = Field(
        default_factory=list,
        description="容易与当前指标混淆的其他指标 ID",
    )

    formula_id: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9_]+$",
        description="派生指标关联的固定公式 ID",
    )

    status: RecordStatus = RecordStatus.ACTIVE

    created_at: datetime
    updated_at: datetime

    @field_validator("allowed_scopes")
    @classmethod
    def validate_allowed_scopes(
        cls,
        value: list[StatementScope],
    ) -> list[StatementScope]:
        """允许口径不能为空、不能重复，也不能包含 unknown。"""

        if not value:
            raise ValueError("allowed_scopes 不能为空")

        if len(value) != len(set(value)):
            raise ValueError("allowed_scopes 不能包含重复值")

        if StatementScope.UNKNOWN in value:
            raise ValueError(
                "FinancialMetric 的 allowed_scopes "
                "不能包含 unknown"
            )

        return value

    @field_validator("confusable_metric_ids")
    @classmethod
    def validate_confusable_metric_ids(
        cls,
        value: list[str],
    ) -> list[str]:
        """易混淆指标 ID 必须合法且不能重复。"""

        if len(value) != len(set(value)):
            raise ValueError(
                "confusable_metric_ids 不能包含重复值"
            )

        for metric_id in value:
            if _ID_PATTERN.fullmatch(metric_id) is None:
                raise ValueError(
                    "confusable_metric_ids 中的指标 ID "
                    "只能包含小写字母、数字和下划线"
                )

        return value

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_timezone(cls, value: datetime) -> datetime:
        """时间必须包含时区信息。"""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("datetime 必须包含时区信息")

        return value

    @model_validator(mode="after")
    def validate_metric_contract(self) -> Self:
        """检查财务指标的跨字段业务约束。"""

        self._validate_period_type()
        self._validate_formula()
        self._validate_value_type_and_unit()
        self._validate_confusable_metrics()
        self._validate_time_order()

        return self

    def _validate_period_type(self) -> None:
        """检查报表类型与期间类型是否一致。"""

        if (
            self.statement_type is StatementType.BALANCE_SHEET
            and self.period_type is not PeriodType.INSTANT
        ):
            raise ValueError(
                "资产负债表指标的 period_type 必须为 instant"
            )

        duration_statement_types = {
            StatementType.INCOME_STATEMENT,
            StatementType.CASH_FLOW_STATEMENT,
        }

        if (
            self.statement_type in duration_statement_types
            and self.period_type is not PeriodType.DURATION
        ):
            raise ValueError(
                "利润表和现金流量表指标的 "
                "period_type 必须为 duration"
            )

    def _validate_formula(self) -> None:
        """检查直接披露指标和派生指标的公式约束。"""

        if (
            self.metric_origin is MetricOrigin.REPORTED
            and self.formula_id is not None
        ):
            raise ValueError(
                "reported 指标不能填写 formula_id"
            )

        if (
            self.metric_origin is MetricOrigin.DERIVED
            and self.formula_id is None
        ):
            raise ValueError(
                "derived 指标必须填写 formula_id"
            )

    def _validate_value_type_and_unit(self) -> None:
        """检查指标值类型与单位是否基本一致。"""

        if (
            self.value_type is MetricValueType.TEXT
            and self.default_unit is not UnitCode.TEXT
        ):
            raise ValueError(
                "文本指标的 default_unit 必须为 text"
            )

        if (
            self.value_type is not MetricValueType.TEXT
            and self.default_unit is UnitCode.TEXT
        ):
            raise ValueError(
                "非文本指标不能使用 text 单位"
            )

    def _validate_confusable_metrics(self) -> None:
        """当前指标不能把自己列为易混淆指标。"""

        if self.metric_id in self.confusable_metric_ids:
            raise ValueError(
                "metric_id 不能出现在自己的 "
                "confusable_metric_ids 中"
            )

    def _validate_time_order(self) -> None:
        """更新时间不能早于创建时间。"""

        if self.updated_at < self.created_at:
            raise ValueError(
                "updated_at 不能早于 created_at"
            )


class MetricAlias(BaseModel):
    """年报原始指标名称与标准指标的映射关系。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    alias_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
        description="指标别名记录唯一 ID",
    )

    metric_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
        description="别名对应的标准指标 ID",
    )

    alias: str = Field(
        min_length=1,
        description="年报原始名称或用户常用说法",
    )

    statement_type: StatementType | None = None

    statement_scope: StatementScope | None = None

    match_type: AliasMatchType

    priority: int = Field(
        default=100,
        ge=1,
        description="匹配优先级，数值越小越优先",
    )

    notes: str | None = None

    status: RecordStatus = RecordStatus.ACTIVE

    @model_validator(mode="after")
    def validate_alias_contract(self) -> Self:
        """检查不同匹配方式对应的约束。"""

        if self.match_type is AliasMatchType.REGEX:
            try:
                re.compile(self.alias)
            except re.error as exc:
                raise ValueError(
                    f"alias 不是合法正则表达式：{exc}"
                ) from exc

        if (
            self.statement_scope is StatementScope.UNKNOWN
        ):
            raise ValueError(
                "MetricAlias 的 statement_scope "
                "不能显式设置为 unknown"
            )

        return self