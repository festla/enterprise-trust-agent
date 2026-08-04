from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from datetime import datetime

from pydantic import field_validator

from .enums import (
    PageContentType,
    PageMappingStatus,
    PageParseStatus,
)


_DOCUMENT_ID_PATTERN = (
    r"^doc_[a-z0-9_]+_[0-9a-f]{24}$"
)

_PAGE_ID_PATTERN = (
    r"^doc_[a-z0-9_]+_[0-9a-f]{24}"
    r"_page_[0-9]{4,}$"
)


class PageMappingResult(BaseModel):
    """一个实际 PDF 页面及其印刷页码映射结果。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    page_id: str = Field(
        min_length=1,
        pattern=_PAGE_ID_PATTERN,
        description=(
            "与具体 document_id 和 PDF 页码绑定的稳定页面 ID"
        ),
    )

    document_id: str = Field(
        min_length=1,
        pattern=_DOCUMENT_ID_PATTERN,
        description="页面所属的实际 PDF 文档版本 ID",
    )

    report_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
        description="页面所属的业务报告 ID",
    )

    pdf_page: int = Field(
        ge=1,
        description="PDF 页码，使用从 1 开始的页码",
    )

    printed_page: int | None = Field(
        default=None,
        ge=1,
        description=(
            "报告印刷页码；封面等页面可以为空"
        ),
    )

    mapping_status: PageMappingStatus

    mapping_id: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9_]+$",
        description=(
            "实际命中的 PageMappingSegment ID"
        ),
    )

    @model_validator(mode="after")
    def validate_mapping_result(self) -> Self:
        """检查页面身份和映射状态是否一致。"""

        expected_document_prefix = (
            f"doc_{self.report_id}_"
        )

        if not self.document_id.startswith(
            expected_document_prefix
        ):
            raise ValueError(
                "document_id 必须属于 report_id"
            )

        expected_page_id = (
            f"{self.document_id}_page_"
            f"{self.pdf_page:04d}"
        )

        if self.page_id != expected_page_id:
            raise ValueError(
                "page_id 必须由 document_id "
                "和 pdf_page 生成"
            )

        if (
            self.mapping_status
            is PageMappingStatus.MAPPED
        ):
            if self.printed_page is None:
                raise ValueError(
                    "mapped 页面必须填写 printed_page"
                )

            if self.mapping_id is None:
                raise ValueError(
                    "mapped 页面必须填写 mapping_id"
                )

        if (
            self.mapping_status
            is PageMappingStatus.UNMAPPED
        ):
            if self.printed_page is not None:
                raise ValueError(
                    "unmapped 页面不能填写 printed_page"
                )

            if self.mapping_id is not None:
                raise ValueError(
                    "unmapped 页面不能填写 mapping_id"
                )

        return self


class ParsedPage(PageMappingResult):
    """从具体 PDF 文件中提取出的页面级结构化记录。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=False,
    )

    schema_version: int = Field(
        default=1,
        ge=1,
        le=1,
        description="ParsedPage 数据结构版本",
    )

    raw_text: str = Field(
        description=(
            "PDF 解析器直接返回的页面原始文本；"
            "不得用清洗结果覆盖"
        ),
    )

    normalized_text: str = Field(
        description="经过低风险规范化的页面文本",
    )

    raw_char_count: int = Field(
        ge=0,
        description="raw_text 的字符数量",
    )

    normalized_char_count: int = Field(
        ge=0,
        description="normalized_text 的字符数量",
    )

    text_block_count: int = Field(
        ge=0,
        description="PyMuPDF 识别出的文本块数量",
    )

    image_block_count: int = Field(
        ge=0,
        description="PyMuPDF 页面字典中的图片块数量",
    )

    embedded_image_count: int = Field(
        ge=0,
        description="页面引用的嵌入图片数量",
    )

    max_image_area_ratio: float = Field(
        ge=0,
        le=1,
        description=(
            "页面中单张图片占页面面积的最大比例"
        ),
    )

    content_type: PageContentType

    parse_status: PageParseStatus

    parse_error: str | None = Field(
        default=None,
        description="页面解析失败时的异常摘要",
    )

    parser_name: str = Field(
        min_length=1,
        description="页面文本解析器名称",
    )

    parser_version: str = Field(
        min_length=1,
        description="页面文本解析器版本",
    )

    parsed_at: datetime

    @field_validator("parsed_at")
    @classmethod
    def validate_parsed_at_timezone(
        cls,
        value: datetime,
    ) -> datetime:
        """页面解析时间必须包含时区。"""

        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "datetime 必须包含时区信息"
            )

        return value

    @model_validator(mode="after")
    def validate_parsed_page(self) -> Self:
        """检查文本、诊断信息与解析状态。"""

        if self.raw_char_count != len(self.raw_text):
            raise ValueError(
                "raw_char_count 必须等于 raw_text 长度"
            )

        if (
            self.normalized_char_count
            != len(self.normalized_text)
        ):
            raise ValueError(
                "normalized_char_count 必须等于 "
                "normalized_text 长度"
            )

        has_normalized_text = bool(
            self.normalized_text
        )

        has_image = (
            self.image_block_count > 0
            or self.embedded_image_count > 0
            or self.max_image_area_ratio > 0
        )

        if (
            self.parse_status
            is PageParseStatus.SUCCESS
        ):
            self._validate_success_page(
                has_normalized_text=has_normalized_text,
                has_image=has_image,
            )

        if (
            self.parse_status
            is PageParseStatus.PARSE_ERROR
        ):
            self._validate_error_page()

        return self

    def _validate_success_page(
        self,
        *,
        has_normalized_text: bool,
        has_image: bool,
    ) -> None:
        """检查成功解析页面的内容分类。"""

        if self.parse_error is not None:
            raise ValueError(
                "success 页面不能填写 parse_error"
            )

        if self.content_type is PageContentType.UNKNOWN:
            raise ValueError(
                "success 页面不能使用 unknown 内容类型"
            )

        if self.content_type in {
            PageContentType.TEXT,
            PageContentType.MIXED,
        }:
            if not has_normalized_text:
                raise ValueError(
                    "text 或 mixed 页面必须包含规范化文本"
                )

        if self.content_type in {
            PageContentType.EMPTY,
            PageContentType.SCANNED,
        }:
            if has_normalized_text:
                raise ValueError(
                    "empty 或 scanned 页面不能包含规范化文本"
                )

        if self.content_type is PageContentType.EMPTY:
            if has_image:
                raise ValueError(
                    "empty 页面不能包含图片信号"
                )

        if self.content_type is PageContentType.SCANNED:
            if not has_image:
                raise ValueError(
                    "scanned 页面必须包含图片信号"
                )

        if self.content_type is PageContentType.MIXED:
            if not has_image:
                raise ValueError(
                    "mixed 页面必须包含图片信号"
                )

    def _validate_error_page(self) -> None:
        """检查解析错误页面。"""

        if (
            self.parse_error is None
            or not self.parse_error.strip()
        ):
            raise ValueError(
                "parse_error 页面必须填写错误信息"
            )

        if self.content_type is not PageContentType.UNKNOWN:
            raise ValueError(
                "parse_error 页面的 content_type "
                "必须为 unknown"
            )

        if self.raw_text or self.normalized_text:
            raise ValueError(
                "parse_error 页面不能保存不完整文本"
            )

        diagnostic_values = (
            self.raw_char_count,
            self.normalized_char_count,
            self.text_block_count,
            self.image_block_count,
            self.embedded_image_count,
        )

        if any(value != 0 for value in diagnostic_values):
            raise ValueError(
                "parse_error 页面的字符和块统计必须为 0"
            )

        if self.max_image_area_ratio != 0:
            raise ValueError(
                "parse_error 页面的图片面积比例必须为 0"
            )


class PageMappingAudit(BaseModel):
    """一个文档版本的整份页码映射审计结果。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    document_id: str = Field(
        pattern=_DOCUMENT_ID_PATTERN,
    )

    report_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
    )

    total_pdf_pages: int = Field(
        ge=1,
    )

    mapped_page_count: int = Field(
        ge=0,
    )

    unmapped_pdf_pages: tuple[int, ...] = ()

    duplicate_printed_pages: dict[
        int,
        tuple[int, ...],
    ] = Field(
        default_factory=dict,
        description=(
            "同一个印刷页码对应的多个 PDF 页码；"
            "不一定是错误"
        ),
    )

    @model_validator(mode="after")
    def validate_audit(self) -> Self:
        """检查审计统计是否内部一致。"""

        expected_document_prefix = (
            f"doc_{self.report_id}_"
        )

        if not self.document_id.startswith(
            expected_document_prefix
        ):
            raise ValueError(
                "document_id 必须属于 report_id"
            )

        normalized_unmapped_pages = tuple(
            sorted(set(self.unmapped_pdf_pages))
        )

        if (
            self.unmapped_pdf_pages
            != normalized_unmapped_pages
        ):
            raise ValueError(
                "unmapped_pdf_pages 必须升序且不能重复"
            )

        for pdf_page in self.unmapped_pdf_pages:
            if not 1 <= pdf_page <= self.total_pdf_pages:
                raise ValueError(
                    "unmapped PDF 页码超出文档范围"
                )

        if (
            self.mapped_page_count
            + len(self.unmapped_pdf_pages)
            != self.total_pdf_pages
        ):
            raise ValueError(
                "已映射页数与未映射页数之和 "
                "必须等于 PDF 总页数"
            )

        for (
            printed_page,
            pdf_pages,
        ) in self.duplicate_printed_pages.items():
            if printed_page < 1:
                raise ValueError(
                    "重复印刷页码必须大于等于 1"
                )

            normalized_pdf_pages = tuple(
                sorted(set(pdf_pages))
            )

            if pdf_pages != normalized_pdf_pages:
                raise ValueError(
                    "重复印刷页对应的 PDF 页码 "
                    "必须升序且不能重复"
                )

            if len(pdf_pages) < 2:
                raise ValueError(
                    "重复印刷页必须至少对应两个 PDF 页面"
                )

            if any(
                pdf_page < 1
                or pdf_page > self.total_pdf_pages
                for pdf_page in pdf_pages
            ):
                raise ValueError(
                    "重复印刷页对应的 PDF 页码 "
                    "超出文档范围"
                )

        return self