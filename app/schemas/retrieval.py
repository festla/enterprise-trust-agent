from __future__ import annotations

from typing import Self, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .chunk import Chunk
from .enums import (
    ChunkStrategy,
    PageMappingStatus,
    ReportType,
    StatementScope,
    StatementType
)


class RetrievalFilter(BaseModel):
    """精确检索前应用的结构化元数据过滤条件。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    company_ids: tuple[str, ...] = ()
    report_ids: tuple[str, ...] = ()
    fiscal_years: tuple[int, ...] = ()
    report_types: tuple[ReportType, ...] = ()
    document_ids: tuple[str, ...] = ()
    page_ids: tuple[str, ...] = ()
    pdf_pages: tuple[int, ...] = ()

    @field_validator(
        "company_ids",
        "report_ids",
        "document_ids",
        "page_ids",
    )
    @classmethod
    def normalize_string_values(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        """字符串过滤值必须去重并保持稳定顺序。"""

        if any(not item for item in value):
            raise ValueError(
                "过滤条件不能包含空字符串"
            )

        return tuple(
            sorted(set(value))
        )

    @field_validator("fiscal_years")
    @classmethod
    def normalize_fiscal_years(
        cls,
        value: tuple[int, ...],
    ) -> tuple[int, ...]:
        if any(
            year < 2000 or year > 2100
            for year in value
        ):
            raise ValueError(
                "fiscal_year 超出支持范围"
            )

        return tuple(
            sorted(set(value))
        )

    @field_validator("pdf_pages")
    @classmethod
    def normalize_pdf_pages(
        cls,
        value: tuple[int, ...],
    ) -> tuple[int, ...]:
        if any(page < 1 for page in value):
            raise ValueError(
                "PDF 页码必须大于等于 1"
            )

        return tuple(
            sorted(set(value))
        )

    @field_validator("report_types")
    @classmethod
    def normalize_report_types(
        cls,
        value: tuple[
            ReportType,
            ...,
        ],
    ) -> tuple[ReportType, ...]:
        return tuple(
            sorted(
                set(value),
                key=lambda item: item.value,
            )
        )


class RetrievalHit(BaseModel):
    """一次检索返回的一个可追溯 Chunk。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=False,
        allow_inf_nan=False,
    )

    rank: int = Field(
        ge=1,
    )

    retriever_type: Literal[
        "dense",
        "bm25",
    ] = "dense"

    score_type: Literal[
        "cosine_similarity",
        "bm25",
    ] = "cosine_similarity"

    score: float = Field(
        description=(
            "检索器原始分数；具体含义由 "
            "retriever_type 和 score_type 决定"
        ),
    )

    chunk_id: str
    chunk_dataset_id: str

    company_id: str
    report_id: str

    fiscal_year: int = Field(
        ge=2000,
        le=2100,
    )

    report_type: ReportType

    document_id: str
    page_id: str

    pdf_page: int = Field(
        ge=1,
    )

    printed_page: int | None = Field(
        default=None,
        ge=1,
    )

    mapping_status: PageMappingStatus

    chunk_index: int = Field(
        ge=0,
    )

    strategy: ChunkStrategy

    source_start_char: int = Field(
        ge=0,
    )

    source_end_char: int = Field(
        ge=1,
    )

    section_path: tuple[str, ...] = ()

    text: str = Field(
        min_length=1,
    )

    @classmethod
    def from_chunk(
        cls,
        *,
        rank: int,
        score: float,
        chunk: Chunk,
        retriever_type: Literal[
            "dense",
            "bm25",
        ] = "dense",
        score_type: Literal[
            "cosine_similarity",
            "bm25",
        ] = "cosine_similarity",
    ) -> RetrievalHit:
        """由经过校验的 Chunk 构造检索结果。"""

        return cls(
            rank=rank,
            retriever_type=retriever_type,
            score_type=score_type,
            score=score,
            chunk_id=chunk.chunk_id,
            chunk_dataset_id=(
                chunk.chunk_dataset_id
            ),
            company_id=chunk.company_id,
            report_id=chunk.report_id,
            fiscal_year=chunk.fiscal_year,
            report_type=chunk.report_type,
            document_id=chunk.document_id,
            page_id=chunk.page_id,
            pdf_page=chunk.pdf_page,
            printed_page=chunk.printed_page,
            mapping_status=(
                chunk.mapping_status
            ),
            chunk_index=chunk.chunk_index,
            strategy=chunk.strategy,
            source_start_char=(
                chunk.source_start_char
            ),
            source_end_char=(
                chunk.source_end_char
            ),
            section_path=chunk.section_path,
            text=chunk.text,
        )

    
    @model_validator(mode="after")
    def validate_source_identity(
        self,
    ) -> Self:
        """检查检索结果中的页面来源。"""

        if self.retriever_type == "dense":
            if (
                self.score_type
                != "cosine_similarity"
            ):
                raise ValueError(
                    "Dense 结果必须使用 "
                    "cosine_similarity 分数"
                )

            if not -1 <= self.score <= 1:
                raise ValueError(
                    "余弦相似度必须位于 [-1, 1]"
                )

        elif self.retriever_type == "bm25":
            if self.score_type != "bm25":
                raise ValueError(
                    "BM25 结果必须使用 bm25 分数"
                )

            if self.score < 0:
                raise ValueError(
                    "BM25 分数不能小于 0"
                )

        if not self.document_id.startswith(
            f"doc_{self.report_id}_"
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
            and self.printed_page is None
        ):
            raise ValueError(
                "mapped 结果必须包含 printed_page"
            )

        if (
            self.mapping_status
            is PageMappingStatus.UNMAPPED
            and self.printed_page is not None
        ):
            raise ValueError(
                "unmapped 结果不能包含 printed_page"
            )

        if (
            self.source_end_char
            <= self.source_start_char
        ):
            raise ValueError(
                "来源字符区间无效"
            )

        return self


class RetrievalQueryPlan(BaseModel):
    """一次可审计的检索查询计划。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )

    schema_version: int = Field(
        default=1,
        ge=1,
        le=1,
    )

    original_query: str = Field(
        min_length=1,
    )

    semantic_query: str = Field(
        min_length=1,
        description=(
            "实际送入 Embedding 或词法检索器的查询"
        ),
    )

    filters: RetrievalFilter = Field(
        default_factory=RetrievalFilter,
    )

    intent_type: Literal[
        "financial_fact"
    ] = "financial_fact"

    metric_name: str = Field(
        min_length=1,
    )

    fiscal_year: int = Field(
        ge=2000,
        le=2100,
    )

    statement_type: StatementType

    statement_scope: StatementScope

    @field_validator(
        "original_query",
        "semantic_query",
        "metric_name",
        mode="before",
    )
    @classmethod
    def normalize_required_text(
        cls,
        value: object,
    ) -> object:
        """规范检索文本，不允许空白字符串。"""

        if not isinstance(value, str):
            return value

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "检索文本不能为空"
            )

        return normalized