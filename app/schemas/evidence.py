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
    AttributionType,
    EvidenceType,
    StatementScope,
    StatementType,
    ValidationStatus,
)


_PRINTED_PAGE_PATTERN = re.compile(r"^\d+(?:-\d+)?$")


class SourceEvidence(BaseModel):
    """支持财务事实或分析结论的原始来源证据。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    evidence_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
        description="来源证据唯一 ID",
    )

    report_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
        description="证据所属报告 ID",
    )

    document_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
        description="证据所属实际文档版本 ID",
    )

    page_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
        description="证据所在页面 ID",
    )

    chunk_id: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9_]+$",
        description="证据对应的检索 Chunk ID",
    )

    evidence_type: EvidenceType

    attribution_type: AttributionType

    statement_type: StatementType | None = None

    statement_scope: StatementScope | None = None

    section_title: str | None = Field(
        default=None,
        min_length=1,
    )

    subsection_title: str | None = Field(
        default=None,
        min_length=1,
    )

    table_name: str | None = Field(
        default=None,
        min_length=1,
    )

    row_label: str | None = Field(
        default=None,
        min_length=1,
    )

    column_label: str | None = Field(
        default=None,
        min_length=1,
    )

    printed_page: int | str | None = Field(
        default=None,
        description=(
            "报告印刷页码，可为单页整数或 48-49 形式的页码范围"
        ),
    )

    pdf_page: int = Field(
        ge=1,
        description="PDF 文件页码，使用从 1 开始的页码",
    )

    evidence_text: str = Field(
        min_length=1,
        description="能够直接支持事实或结论的证据内容",
    )

    cell_value: str | None = Field(
        default=None,
        min_length=1,
        description="表格单元格中的原始值",
    )

    source_hash: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="证据内容的 SHA-256 哈希",
    )

    validation_status: ValidationStatus = (
        ValidationStatus.PENDING
    )

    validated_by: str | None = Field(
        default=None,
        min_length=1,
        description="证据核验人或核验规则标识",
    )

    created_at: datetime

    @field_validator("printed_page")
    @classmethod
    def validate_printed_page(
        cls,
        value: int | str | None,
    ) -> int | str | None:
        """印刷页码必须为正整数或合法页码范围。"""

        if value is None:
            return value

        if isinstance(value, int):
            if value < 1:
                raise ValueError(
                    "printed_page 必须大于等于 1"
                )

            return value

        if _PRINTED_PAGE_PATTERN.fullmatch(value) is None:
            raise ValueError(
                "字符串 printed_page 必须使用 "
                "'48' 或 '48-49' 格式"
            )

        if "-" in value:
            start_text, end_text = value.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)

            if end < start:
                raise ValueError(
                    "printed_page 页码范围结束值 "
                    "不能小于开始值"
                )

        return value

    @field_validator("created_at")
    @classmethod
    def validate_timezone(
        cls,
        value: datetime,
    ) -> datetime:
        """时间必须包含时区信息。"""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "datetime 必须包含时区信息"
            )

        return value

    @model_validator(mode="after")
    def validate_evidence_contract(self) -> Self:
        """检查不同证据类型对应的跨字段约束。"""

        self._validate_attribution_type()
        self._validate_statement_scope()
        self._validate_table_evidence()
        self._validate_narrative_evidence()
        self._validate_verified_evidence()

        return self

    def _validate_attribution_type(self) -> None:
        """SourceEvidence 只能保存原始披露证据。"""

        allowed_attributions = {
            AttributionType.REPORT_DISCLOSURE,
            AttributionType.MANAGEMENT_STATEMENT,
        }

        if self.attribution_type not in allowed_attributions:
            raise ValueError(
                "SourceEvidence 只能使用 "
                "report_disclosure 或 management_statement"
            )

        if (
            self.evidence_type
            is EvidenceType.MANAGEMENT_STATEMENT
            and self.attribution_type
            is not AttributionType.MANAGEMENT_STATEMENT
        ):
            raise ValueError(
                "management_statement 类型证据的 "
                "attribution_type 必须为 management_statement"
            )

        if (
            self.evidence_type
            is not EvidenceType.MANAGEMENT_STATEMENT
            and self.attribution_type
            is AttributionType.MANAGEMENT_STATEMENT
        ):
            raise ValueError(
                "只有 management_statement 类型证据 "
                "可以使用 management_statement 归属"
            )

    def _validate_statement_scope(self) -> None:
        """证据口径不能显式设置为 unknown。"""

        if self.statement_scope is StatementScope.UNKNOWN:
            raise ValueError(
                "SourceEvidence 的 statement_scope "
                "不能显式设置为 unknown"
            )

    def _validate_table_evidence(self) -> None:
        """表格单元格证据必须保留完整定位信息。"""

        table_cell_types = {
            EvidenceType.FINANCIAL_STATEMENT_CELL,
            EvidenceType.FINANCIAL_SUMMARY_TABLE,
        }

        if self.evidence_type not in table_cell_types:
            return

        required_fields = {
            "statement_type": self.statement_type,
            "statement_scope": self.statement_scope,
            "table_name": self.table_name,
            "row_label": self.row_label,
            "column_label": self.column_label,
            "cell_value": self.cell_value,
        }

        missing_fields = [
            field_name
            for field_name, field_value in required_fields.items()
            if field_value is None
        ]

        if missing_fields:
            missing_text = ", ".join(missing_fields)

            raise ValueError(
                "财务表格证据缺少必要字段："
                f"{missing_text}"
            )

    def _validate_narrative_evidence(self) -> None:
        """叙述性证据必须能够定位到章节。"""

        narrative_types = {
            EvidenceType.MANAGEMENT_STATEMENT,
            EvidenceType.FINANCIAL_NOTE,
            EvidenceType.RISK_DISCLOSURE,
            EvidenceType.PARAGRAPH,
        }

        if (
            self.evidence_type in narrative_types
            and self.section_title is None
        ):
            raise ValueError(
                "叙述性证据必须填写 section_title"
            )

    def _validate_verified_evidence(self) -> None:
        """已核验证据必须记录核验人和双页码。"""

        if (
            self.validation_status
            is not ValidationStatus.VERIFIED
        ):
            return

        if self.validated_by is None:
            raise ValueError(
                "verified 证据必须填写 validated_by"
            )

        if self.printed_page is None:
            raise ValueError(
                "verified 证据必须填写 printed_page"
            )