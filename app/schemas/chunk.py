from __future__ import annotations

import hashlib
from typing import (
    Annotated,
    Literal,
    Self
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from .enums import (
    ChunkStrategy,
    PageContentType,
    PageMappingStatus,
    PageParseStatus,
    ReportType,
)


_DOCUMENT_ID_PATTERN = (
    r"^doc_[a-z0-9_]+_[0-9a-f]{24}$"
)

_PAGE_ID_PATTERN = (
    r"^doc_[a-z0-9_]+_[0-9a-f]{24}"
    r"_page_[0-9]{4,}$"
)

_PAGE_DATASET_ID_PATTERN = (
    r"^page_dataset_[a-z0-9_]+_"
    r"[0-9a-f]{24}$"
)

_CHUNK_DATASET_ID_PATTERN = (
    r"^chunk_dataset_[a-z0-9_]+_"
    r"[0-9a-f]{24}$"
)

_CHUNK_ID_PATTERN = (
    r"^chunk_[a-z0-9_]+_[0-9a-f]{24}$"
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class FixedLengthChunkingConfig(BaseModel):
    """固定字符长度切分策略的确定性配置。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    strategy: Literal[
        ChunkStrategy.FIXED_LENGTH
    ] = ChunkStrategy.FIXED_LENGTH

    chunker_name: Literal[
        "fixed_length"
    ] = "fixed_length"

    chunker_version: str = Field(
        default="fixed_length_chunker_v1",
        min_length=1,
    )

    source_text_field: Literal[
        "normalized_text"
    ] = "normalized_text"

    max_chars: int = Field(
        default=800,
        ge=1,
    )

    overlap_chars: int = Field(
        default=120,
        ge=0,
    )

    include_content_types: tuple[
        PageContentType,
        ...,
    ] = (
        PageContentType.TEXT,
        PageContentType.MIXED,
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """检查固定长度切分参数。"""

        if (
            self.strategy
            is not ChunkStrategy.FIXED_LENGTH
        ):
            raise ValueError(
                "FixedLengthChunkingConfig 的"
                "strategy 必须为 fixed_length"
            )

        if (
            self.overlap_chars
            >= self.max_chars
        ):
            raise ValueError(
                "overlap_chars 必须小于 max_chars"
            )

        if not self.include_content_types:
            raise ValueError(
                "include_content_types 不能为空"
            )

        if (
            len(set(self.include_content_types))
            != len(self.include_content_types)
        ):
            raise ValueError(
                "include_content_types " 
                "不能包含重复值"
            )

        allowed_types = {
            PageContentType.TEXT,
            PageContentType.MIXED,
        }

        if not set(
            self.include_content_types
        ).issubset(allowed_types):
            raise ValueError(
                "固定长度切分当前只允许 "
                "text 和 mixed 页面"
            )

        return self

    @property    # 把一个方法变成“像属性一样访问”的计算结果
    def step_chars(self) -> int:
        """相邻 Chunk 起点间的字符步长。"""

        return (
            self.max_chars
            - self.overlap_chars
        )


class ParagraphChunkingConfig(BaseModel):
    """基于空白行识别段落并进行组合切分。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    strategy: Literal[
        ChunkStrategy.PARAGRAPH
    ] = ChunkStrategy.PARAGRAPH

    chunker_name: Literal[
        "paragraph"
    ] = "paragraph"

    chunker_version: str = Field(
        default="paragraph_chunker_v1",
        min_length=1,
    )

    source_text_field: Literal[
        "normalized_text"
    ] = "normalized_text"

    paragraph_boundary_rule: Literal[
        "blank_line"
    ] = "blank_line"

    max_chars: int = Field(
        default=800,
        ge=1,
    )

    overlap_paragraphs: int = Field(
        default=1,
        ge=0,
        description=(
            "相邻 Chunk 重叠的完整段落数量"
        ),
    )

    long_paragraph_overlap_chars: int = Field(
        default=120,
        ge=0,
        description=(
            "单个段落超过 max_chars 时，"
            "字符级回退切分使用的 overlap"
        ),
    )

    include_content_types: tuple[
        PageContentType,
        ...,
    ] = (
        PageContentType.TEXT,
        PageContentType.MIXED,
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """检查段落切分参数。"""

        if (
            self.long_paragraph_overlap_chars
            >= self.max_chars
        ):
            raise ValueError(
                "long_paragraph_overlap_chars "
                "必须小于 max_chars"
            )

        if not self.include_content_types:
            raise ValueError(
                "include_content_types 不能为空"
            )

        if (
            len(set(self.include_content_types))
            != len(self.include_content_types)
        ):
            raise ValueError(
                "include_content_types "
                "不能包含重复值"
            )

        allowed_types = {
            PageContentType.TEXT,
            PageContentType.MIXED,
        }

        if not set(
            self.include_content_types
        ).issubset(allowed_types):
            raise ValueError(
                "段落切分当前只允许 "
                "text 和 mixed 页面"
            )

        return self

    @property
    def long_paragraph_step_chars(
        self,
    ) -> int:
        """超长段落回退切分时的字符步长。"""

        return (
            self.max_chars
            - self.long_paragraph_overlap_chars
        )


class SectionParagraphChunkingConfig(BaseModel):
    """段落切分并附加规则化章节上下文。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    strategy: Literal[
        ChunkStrategy.SECTION_PARAGRAPH
    ] = ChunkStrategy.SECTION_PARAGRAPH

    chunker_name: Literal[
        "section_paragraph"
    ] = "section_paragraph"

    chunker_version: str = Field(
        default=(
            "section_paragraph_chunker_v1"
        ),
        min_length=1,
    )

    heading_detector_version: str = Field(
        default=(
            "annual_report_heading_detector_v1"
        ),
        min_length=1,
    )

    source_text_field: Literal[
        "normalized_text"
    ] = "normalized_text"

    paragraph_boundary_rule: Literal[
        "blank_line"
    ] = "blank_line"

    max_chars: int = Field(
        default=800,
        ge=1,
    )

    overlap_paragraphs: int = Field(
        default=1,
        ge=0,
    )

    long_paragraph_overlap_chars: int = Field(
        default=120,
        ge=0,
    )

    max_heading_chars: int = Field(
        default=50,
        ge=4,
        le=200,
    )

    inherit_section_across_pages: bool = True

    include_content_types: tuple[
        PageContentType,
        ...,
    ] = (
        PageContentType.TEXT,
        PageContentType.MIXED,
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if (
            self.long_paragraph_overlap_chars
            >= self.max_chars
        ):
            raise ValueError(
                "long_paragraph_overlap_chars "
                "必须小于 max_chars"
            )

        if not self.include_content_types:
            raise ValueError(
                "include_content_types 不能为空"
            )

        if (
            len(set(self.include_content_types))
            != len(self.include_content_types)
        ):
            raise ValueError(
                "include_content_types "
                "不能包含重复值"
            )

        allowed_types = {
            PageContentType.TEXT,
            PageContentType.MIXED,
        }

        if not set(
            self.include_content_types
        ).issubset(allowed_types):
            raise ValueError(
                "章节段落切分只允许 "
                "text 和 mixed 页面"
            )

        return self


ChunkingConfig = Annotated[
    FixedLengthChunkingConfig
    | ParagraphChunkingConfig
    | SectionParagraphChunkingConfig,
    Field(discriminator="strategy"),
]


class Chunk(BaseModel):
    """可追溯到单页字符区间的检索文本单元。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=False,
    )

    schema_version: int = Field(
        default=1,
        ge=1,
        le=1,
    )

    chunk_id: str = Field(
        pattern=_CHUNK_ID_PATTERN,
    )

    chunk_dataset_id: str = Field(
        pattern=_CHUNK_DATASET_ID_PATTERN,
    )

    page_dataset_id: str = Field(
        pattern=_PAGE_DATASET_ID_PATTERN,
    )

    company_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    report_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    fiscal_year: int = Field(
        ge=2000,
        le=2100,
    )

    report_type: ReportType

    document_id: str = Field(
        pattern=_DOCUMENT_ID_PATTERN,
    )

    page_id: str = Field(
        pattern=_PAGE_ID_PATTERN,
    )

    pdf_page: int = Field(
        ge=1,
    )

    printed_page: int | None = Field(
        default=None,
        ge=1,
    )

    mapping_status: PageMappingStatus
    content_type: PageContentType
    parse_status: PageParseStatus

    chunk_index: int = Field(
        ge=0,
    )

    strategy: ChunkStrategy

    chunker_version: str = Field(
        min_length=1,
    )

    source_text_field: Literal[
        "normalized_text"
    ]

    source_start_char: int = Field(
        ge=0,
    )

    source_end_char: int = Field(
        ge=1,
    )

    text: str = Field(
        min_length=1,
    )

    char_count: int = Field(
        ge=1,
    )

    text_sha256: str = Field(
        pattern=_SHA256_PATTERN,
    )

    paragraph_start_index: int | None = Field(
        default=None,
        ge=0,
    )

    paragraph_end_index: int | None = Field(
        default=None,
        ge=0,
    )

    section_path: tuple[str, ...] = ()

    section_source_page_id: str | None = Field(
        default=None,
        pattern=_PAGE_ID_PATTERN,
    )

    section_inherited: bool = False

    @model_validator(mode="after")
    def validate_chunk(self) -> Self:
        """检查 Chunk 身份、来源和文本边界。"""

        if not self.document_id.startswith(
            f"doc_{self.report_id}_"
        ):
            raise ValueError(
                "document_id 必须属于 report_id"
            )

        if not self.page_dataset_id.startswith(
            f"page_dataset_{self.report_id}_"
        ):
            raise ValueError(
                "page_dataset_id 必须属于 report_id"
            )

        if not self.chunk_dataset_id.startswith(
            f"chunk_dataset_{self.report_id}_"
        ):
            raise ValueError(
                "chunk_dataset_id 必须属于 report_id"
            )

        if not self.chunk_id.startswith(
            f"chunk_{self.report_id}_"
        ):
            raise ValueError(
                "chunk_id 必须属于 report_id"
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
                    "mapped Chunk 必须填写 "
                    "printed_page"
                )

        elif self.printed_page is not None:
            raise ValueError(
                "unmapped Chunk 不能填写 "
                "printed_page"
            )

        if (
            self.parse_status
            is not PageParseStatus.SUCCESS
        ):
            raise ValueError(
                "Chunk 只能来自成功解析的页面"
            )

        if self.content_type not in {
            PageContentType.TEXT,
            PageContentType.MIXED,
        }:
            raise ValueError(
                "Chunk 只能来自 text 或 mixed 页面"
            )

        if (
            self.source_end_char
            <= self.source_start_char
        ):
            raise ValueError(
                "source_end_char 必须大于 "
                "source_start_char"
            )

        boundary_length = (
            self.source_end_char
            - self.source_start_char
        )

        if boundary_length != self.char_count:
            raise ValueError(
                "字符边界长度必须等于 char_count"
            )

        if len(self.text) != self.char_count:
            raise ValueError(
                "char_count 必须等于 text 长度"
            )

        expected_text_sha256 = hashlib.sha256(
            self.text.encode("utf-8")
        ).hexdigest()

        if (
            self.text_sha256
            != expected_text_sha256
        ):
            raise ValueError(
                "text_sha256 与 Chunk 文本不一致"
            )

        if (
            self.strategy
            is ChunkStrategy.FIXED_LENGTH
        ):
            fixed_length_metadata = (
                self.paragraph_start_index,
                self.paragraph_end_index,
                self.section_source_page_id,
            )

            if any(
                value is not None
                for value
                in fixed_length_metadata
            ):
                raise ValueError(
                    "fixed_length Chunk 不能填写 "
                    "段落或章节来源"
                )

            if (
                self.section_path
                or self.section_inherited
            ):
                raise ValueError(
                    "fixed_length Chunk 不能填写 "
                    "章节上下文"
                )

        if (
            self.strategy
            is ChunkStrategy.PARAGRAPH
        ):
            if (
                self.paragraph_start_index
                is None
                or self.paragraph_end_index
                is None
            ):
                raise ValueError(
                    "paragraph Chunk 必须填写 "
                    "段落起止索引"
                )

            if (
                self.paragraph_end_index
                < self.paragraph_start_index
            ):
                raise ValueError(
                    "paragraph_end_index 不能小于 "
                    "paragraph_start_index"
                )

            if (
                self.section_path
                or self.section_source_page_id
                is not None
                or self.section_inherited
            ):
                raise ValueError(
                    "paragraph Chunk 当前不能填写 "
                    "章节上下文"
                )


        if (
            self.strategy
            is ChunkStrategy.SECTION_PARAGRAPH
        ):
            if (
                self.paragraph_start_index
                is None
                or self.paragraph_end_index
                is None
            ):
                raise ValueError(
                    "section_paragraph Chunk "
                    "必须填写段落起止索引"
                )

            if (
                self.paragraph_end_index
                < self.paragraph_start_index
            ):
                raise ValueError(
                    "paragraph_end_index 不能小于 "
                    "paragraph_start_index"
                )

            if self.section_path:
                if (
                    self.section_source_page_id
                    is None
                ):
                    raise ValueError(
                        "存在 section_path 时必须填写 "
                        "section_source_page_id"
                    )

                if (
                    self.section_inherited
                    and self.section_source_page_id
                    == self.page_id
                ):
                    raise ValueError(
                        "继承章节的来源页面不能是 "
                        "当前页面"
                    )

                if (
                    not self.section_inherited
                    and self.section_source_page_id
                    != self.page_id
                ):
                    raise ValueError(
                        "非继承章节必须来自当前页面"
                    )

            else:
                if (
                    self.section_source_page_id
                    is not None
                    or self.section_inherited
                ):
                    raise ValueError(
                        "没有 section_path 时不能填写 "
                        "章节来源或继承状态"
                    )

        return self