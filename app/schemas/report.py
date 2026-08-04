from datetime import date, datetime
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .enums import (
    DocumentQualityGrade,
    PageMappingRuleType,
    RecordStatus,
    ReportType,
    Severity,
    ValidationStatus,
)


class Report(BaseModel):
    """某家公司某一财务年度的业务报告。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    report_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
        description="报告唯一 ID，MVP 使用 company_id_year 格式",
    )

    company_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
        description="报告所属公司的标准 ID",
    )

    fiscal_year: int = Field(
        ge=2000,
        le=2100,
        description="报告对应的财务年度",
    )

    report_type: ReportType = ReportType.ANNUAL_REPORT

    title: str = Field(
        min_length=1,
        description="报告原公告标题",
    )

    publication_date: date

    source_name: str = Field(
        min_length=1,
        description="报告来源名称",
    )

    source_uri: str | None = None

    quality_grade: DocumentQualityGrade

    citation_risk: Severity

    expected_pdf_page_count: int | None = Field(
        default=None,
        ge=1,
        description=(
            "人工核验的预期 PDF 总页数；"
            "文档接入时与解析器读取结果进行比较"
        ),
    )

    active_document_id: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9_]+$",
        description="当前有效的实际文档版本 ID",
    )

    status: RecordStatus = RecordStatus.ACTIVE

    notes: str | None = None

    created_at: datetime
    updated_at: datetime

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_timezone(cls, value: datetime) -> datetime:
        """时间必须包含时区信息。"""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("datetime 必须包含时区信息")

        return value

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        """检查报告 ID 和时间关系。"""

        expected_report_id = f"{self.company_id}_{self.fiscal_year}"

        if self.report_id != expected_report_id:
            raise ValueError(
                "report_id 必须等于 "
                f"'{expected_report_id}'，"
                "即 company_id 与 fiscal_year 的组合"
            )

        if self.updated_at < self.created_at:
            raise ValueError("updated_at 不能早于 created_at")

        return self


class PageMappingSegment(BaseModel):
    """一段印刷页码与 PDF 页码的映射规则。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    mapping_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
        description="页码映射规则唯一 ID",
    )

    report_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
        description="所属报告 ID",
    )

    printed_page_start: int | None = Field(
        default=None,
        ge=1,
        description="映射区间的起始印刷页码",
    )

    printed_page_end: int | None = Field(
        default=None,
        ge=1,
        description="映射区间的结束印刷页码",
    )

    pdf_page_start: int = Field(
        ge=1,
        description="映射区间的起始 PDF 页码，使用 1-based 页码",
    )

    pdf_page_end: int = Field(
        ge=1,
        description="映射区间的结束 PDF 页码，使用 1-based 页码",
    )

    offset: int | None = Field(
        default=None,
        description="PDF 页码减去印刷页码得到的偏移量",
    )

    rule_type: PageMappingRuleType

    notes: str | None = None

    validation_status: ValidationStatus = ValidationStatus.PENDING

    @model_validator(mode="after")
    def validate_mapping(self) -> Self:
        """检查页码区间和映射规则是否一致。"""

        has_printed_start = self.printed_page_start is not None
        has_printed_end = self.printed_page_end is not None

        if has_printed_start != has_printed_end:
            raise ValueError(
                "printed_page_start 和 printed_page_end "
                "必须同时填写或同时为空"
            )

        if self.pdf_page_end < self.pdf_page_start:
            raise ValueError(
                "pdf_page_end 不能小于 pdf_page_start"
            )

        if (
            self.printed_page_start is not None
            and self.printed_page_end is not None
            and self.printed_page_end < self.printed_page_start
        ):
            raise ValueError(
                "printed_page_end 不能小于 printed_page_start"
            )

        if (
            self.validation_status is ValidationStatus.VERIFIED
            and (
                self.printed_page_start is None
                or self.printed_page_end is None
            )
        ):
            raise ValueError(
                "verified 页码映射必须包含完整的印刷页码区间"
            )

        if self.rule_type is PageMappingRuleType.IDENTITY:
            self._validate_identity_mapping()

        elif self.rule_type is PageMappingRuleType.OFFSET:
            self._validate_offset_mapping()

        elif self.rule_type is PageMappingRuleType.CUSTOM:
            self._validate_custom_mapping()

        return self

    def _validate_identity_mapping(self) -> None:
        """检查印刷页码和 PDF 页码完全相同的情况。"""

        if (
            self.printed_page_start is None
            or self.printed_page_end is None
        ):
            raise ValueError(
                "identity 映射必须填写完整的印刷页码区间"
            )

        if self.offset not in (None, 0):
            raise ValueError(
                "identity 映射的 offset 必须为空或 0"
            )

        if (
            self.pdf_page_start != self.printed_page_start
            or self.pdf_page_end != self.printed_page_end
        ):
            raise ValueError(
                "identity 映射要求 PDF 页码与印刷页码完全一致"
            )

    def _validate_offset_mapping(self) -> None:
        """检查固定偏移量映射。"""

        if (
            self.printed_page_start is None
            or self.printed_page_end is None
        ):
            raise ValueError(
                "offset 映射必须填写完整的印刷页码区间"
            )

        if self.offset is None:
            raise ValueError(
                "offset 映射必须填写 offset"
            )

        expected_pdf_start = self.printed_page_start + self.offset
        expected_pdf_end = self.printed_page_end + self.offset

        if (
            self.pdf_page_start != expected_pdf_start
            or self.pdf_page_end != expected_pdf_end
        ):
            raise ValueError(
                "PDF 页码区间与印刷页码区间及 offset 不一致"
            )

    def _validate_custom_mapping(self) -> None:
        """检查自定义映射。"""

        if not self.notes:
            raise ValueError(
                "custom 映射必须在 notes 中说明特殊映射规则"
            )