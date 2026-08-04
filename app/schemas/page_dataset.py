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
    PageContentType,
    PageParseStatus,
)


class PageDatasetManifest(BaseModel):
    """一份完整页面 JSONL 数据集的审计记录。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    schema_version: int = Field(
        default=1,
        ge=1,
        le=1,
    )

    dataset_id: str = Field(
        min_length=1,
        pattern=(
            r"^page_dataset_[a-z0-9_]+_"
            r"[0-9a-f]{24}$"
        ),
    )

    document_id: str = Field(
        pattern=(
            r"^doc_[a-z0-9_]+_"
            r"[0-9a-f]{24}$"
        ),
    )

    report_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
    )

    source_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    mapping_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    pages_jsonl_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    page_schema_version: int = Field(
        default=1,
        ge=1,
        le=1,
    )

    parser_name: str = Field(
        min_length=1,
    )

    parser_version: str = Field(
        min_length=1,
    )

    normalizer_version: str = Field(
        min_length=1,
    )

    classifier_version: str = Field(
        min_length=1,
    )

    total_pdf_pages: int = Field(
        ge=1,
    )

    page_record_count: int = Field(
        ge=1,
    )

    mapped_page_count: int = Field(
        ge=0,
    )

    unmapped_page_count: int = Field(
        ge=0,
    )

    raw_char_count_total: int = Field(
        ge=0,
    )

    normalized_char_count_total: int = Field(
        ge=0,
    )

    content_type_counts: dict[
        PageContentType,
        int,
    ]

    parse_status_counts: dict[
        PageParseStatus,
        int,
    ]

    quality_gate_passed: bool

    quality_gate_errors: tuple[str, ...] = ()

    quality_warnings: tuple[str, ...] = ()

    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def validate_timezone(
        cls,
        value: datetime,
    ) -> datetime:
        """创建时间必须包含时区。"""

        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "datetime 必须包含时区信息"
            )

        return value

    @model_validator(mode="after")
    def validate_dataset_manifest(self) -> Self:
        """检查数据集统计和质量状态。"""

        if not self.document_id.startswith(
            f"doc_{self.report_id}_"
        ):
            raise ValueError(
                "document_id 必须属于 report_id"
            )

        if not self.dataset_id.startswith(
            f"page_dataset_{self.report_id}_"
        ):
            raise ValueError(
                "dataset_id 必须属于 report_id"
            )

        if (
            self.page_record_count
            != self.total_pdf_pages
        ):
            raise ValueError(
                "页面记录数必须等于 PDF 总页数"
            )

        if (
            self.mapped_page_count
            + self.unmapped_page_count
            != self.page_record_count
        ):
            raise ValueError(
                "mapped 与 unmapped 页数之和 "
                "必须等于页面记录数"
            )

        expected_content_types = set(
            PageContentType
        )

        if (
            set(self.content_type_counts)
            != expected_content_types
        ):
            raise ValueError(
                "content_type_counts 必须包含 "
                "全部页面内容类型"
            )

        expected_parse_statuses = set(
            PageParseStatus
        )

        if (
            set(self.parse_status_counts)
            != expected_parse_statuses
        ):
            raise ValueError(
                "parse_status_counts 必须包含 "
                "全部解析状态"
            )

        if any(
            count < 0
            for count
            in self.content_type_counts.values()
        ):
            raise ValueError(
                "内容类型计数不能为负数"
            )

        if any(
            count < 0
            for count
            in self.parse_status_counts.values()
        ):
            raise ValueError(
                "解析状态计数不能为负数"
            )

        if (
            sum(self.content_type_counts.values())
            != self.page_record_count
        ):
            raise ValueError(
                "内容类型计数之和必须等于页面记录数"
            )

        if (
            sum(self.parse_status_counts.values())
            != self.page_record_count
        ):
            raise ValueError(
                "解析状态计数之和必须等于页面记录数"
            )

        parse_error_count = (
            self.parse_status_counts[
                PageParseStatus.PARSE_ERROR
            ]
        )

        unknown_count = (
            self.content_type_counts[
                PageContentType.UNKNOWN
            ]
        )

        if parse_error_count != unknown_count:
            raise ValueError(
                "parse_error 页数必须等于 "
                "unknown 内容页数"
            )

        expected_quality_passed = (
            len(self.quality_gate_errors) == 0
        )

        if (
            self.quality_gate_passed
            != expected_quality_passed
        ):
            raise ValueError(
                "quality_gate_passed 必须与 "
                "quality_gate_errors 保持一致"
            )

        if (
            self.quality_gate_passed
            and parse_error_count > 0
        ):
            raise ValueError(
                "存在页面解析错误时，"
                "质量门禁不能通过"
            )

        return self