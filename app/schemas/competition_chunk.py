from __future__ import annotations

import hashlib
from typing import (
    Literal,
    Self,
)
from dataclasses import dataclass
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.schemas.competition_evidence import (
    CompetitionKnowledgeSource,
)
from app.schemas.competition_text import (
    CompetitionTextSourceType,
)


CompetitionChunkType = Literal[
    "text",
    "table",
]


class CompetitionChunkSourceSpan(
    BaseModel
):
    """
    一个 Retrieval Chunk 对应的
    原始 TextBlock 字符区间。

    复用旧 Chunking 中：
        source_start_char
        source_end_char
    的可追溯思想。

    一个 Chunk 可以来自多个 TextBlock，
    因此这里使用 source_spans。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    block_id: str = Field(
        min_length=1,
    )

    block_index: int = Field(
        ge=0,
    )

    start_char: int = Field(
        ge=0,
    )

    end_char: int = Field(
        ge=1,
    )

    @model_validator(
        mode="after"
    )
    def validate_span(
        self,
    ) -> Self:
        if (
            self.end_char
            <= self.start_char
        ):
            raise ValueError(
                "end_char 必须大于 "
                "start_char"
            )

        return self


class CompetitionTextChunk(
    BaseModel
):
    """
    Competition 文本知识库最终用于检索的 Chunk。

    Parser Block:
        忠实表示原文结构。

    Retrieval Chunk:
        面向检索组织内容，
        但必须能够回溯到 Parser Block。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )

    schema_version: Literal[1] = 1

    chunk_id: str = Field(
        min_length=1,
    )

    source_id: str = Field(
        min_length=1,
    )

    doc_id: str = Field(
        min_length=1,
    )

    source_type: (
        CompetitionTextSourceType
    )

    chunk_index: int = Field(
        ge=0,
    )

    chunk_type: CompetitionChunkType

    # ========================================================
    # Source Provenance
    # ========================================================

    source_spans: tuple[
        CompetitionChunkSourceSpan,
        ...
    ] = Field(
        min_length=1,
    )

    # ========================================================
    # Retrieval Text
    # ========================================================

    text: str = Field(
        min_length=1,
    )

    char_count: int = Field(
        ge=1,
    )

    text_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    # ========================================================
    # Regulatory Structure
    #
    # section_path:
    #   第一章 总则
    #   第二节 风险管理
    #
    # article:
    #   第十二条
    #
    # item_path:
    #   （一）
    #   1.
    # ========================================================

    section_path: tuple[
        str,
        ...
    ] = ()

    article: str | None = None

    item_path: tuple[
        str,
        ...
    ] = ()

    # 当前 Chunk 是否直接包含 Article 标题，
    # 还是从前文继承 Article 上下文。
    article_inherited: bool = False

    # ========================================================
    # PDF Location
    # ========================================================

    page_start: int | None = Field(
        default=None,
        ge=1,
    )

    page_end: int | None = Field(
        default=None,
        ge=1,
    )

    # ========================================================
    # Word Location
    # ========================================================

    paragraph_start_index: (
        int | None
    ) = Field(
        default=None,
        ge=0,
    )

    paragraph_end_index: (
        int | None
    ) = Field(
        default=None,
        ge=0,
    )

    # ========================================================
    # Word Table
    # ========================================================

    table_index: int | None = Field(
        default=None,
        ge=0,
    )

    # 如果一个大表以后按行切分，
    # 保留原表中的行范围。
    #
    # 使用 0-based index。
    table_row_start: int | None = Field(
        default=None,
        ge=0,
    )

    table_row_end: int | None = Field(
        default=None,
        ge=0,
    )

    table_rows: tuple[
        tuple[
            str,
            ...
        ],
        ...
    ] = ()

    @model_validator(
        mode="after"
    )
    def validate_chunk(
        self,
    ) -> Self:
        # ====================================================
        # Text Identity
        # ====================================================

        if (
            len(self.text)
            != self.char_count
        ):
            raise ValueError(
                "char_count 必须等于 "
                "text 长度"
            )

        expected_hash = (
            hashlib.sha256(
                self.text.encode(
                    "utf-8"
                )
            ).hexdigest()
        )

        if (
            self.text_sha256
            != expected_hash
        ):
            raise ValueError(
                "text_sha256 与 "
                "Chunk 文本不一致"
            )

        # ====================================================
        # Source Span Order
        # ====================================================

        block_indexes = [
            span.block_index
            for span
            in self.source_spans
        ]

        if (
            block_indexes
            != sorted(
                block_indexes
            )
        ):
            raise ValueError(
                "source_spans 必须按照 "
                "原始 Block 顺序排列"
            )

        # ====================================================
        # PDF
        # ====================================================

        if self.source_type == "pdf":
            if (
                self.page_start is None
                or self.page_end is None
            ):
                raise ValueError(
                    "PDF Chunk 必须提供 "
                    "page_start/page_end"
                )

            if (
                self.page_end
                < self.page_start
            ):
                raise ValueError(
                    "page_end 不能小于 "
                    "page_start"
                )

            if (
                self.paragraph_start_index
                is not None
                or self.paragraph_end_index
                is not None
                or self.table_index
                is not None
            ):
                raise ValueError(
                    "PDF Chunk 不能包含 "
                    "Word 位置信息"
                )

        # ====================================================
        # Word
        # ====================================================

        if self.source_type == "word":
            if (
                self.page_start is not None
                or self.page_end is not None
            ):
                raise ValueError(
                    "Word Chunk 当前不使用 "
                    "page_start/page_end"
                )

        # ====================================================
        # Text Chunk
        # ====================================================

        if self.chunk_type == "text":
            if self.table_rows:
                raise ValueError(
                    "text Chunk 不能包含 "
                    "table_rows"
                )

            if (
                self.table_index is not None
                or self.table_row_start
                is not None
                or self.table_row_end
                is not None
            ):
                raise ValueError(
                    "text Chunk 不能包含 "
                    "table 位置信息"
                )

        # ====================================================
        # Table Chunk
        # ====================================================

        if self.chunk_type == "table":
            if self.source_type != "word":
                raise ValueError(
                    "当前 table Chunk "
                    "只支持 Word"
                )

            if self.table_index is None:
                raise ValueError(
                    "table Chunk 必须提供 "
                    "table_index"
                )

            if not self.table_rows:
                raise ValueError(
                    "table Chunk 必须保留 "
                    "table_rows"
                )

            if (
                self.table_row_start is None
                or self.table_row_end is None
            ):
                raise ValueError(
                    "table Chunk 必须提供 "
                    "table_row_start/"
                    "table_row_end"
                )

            if (
                self.table_row_end
                < self.table_row_start
            ):
                raise ValueError(
                    "table_row_end 不能小于 "
                    "table_row_start"
                )

            # 一个 Table Chunk 当前只能来源
            # 于一个原始 Table Block。
            if (
                len(
                    self.source_spans
                )
                != 1
            ):
                raise ValueError(
                    "table Chunk 当前必须只"
                    "来源于一个 Table Block"
                )

        return self

from typing import Literal


@dataclass(
    frozen=True,
    slots=True,
)
class CompetitionTableChunk:
    """
    Competition 表格 Chunk。

    表格证据必须保留：

    1. 原始来源
    2. 表格位置
    3. 结构上下文
    4. 可检索文本

    不直接把 table 转字符串丢给模型。
    """

    chunk_id: str

    source_id: str

    doc_id: str

    source_type: Literal[
        "word",
        "pdf",
    ]

    chunk_index: int

    chunk_type: Literal[
        "table",
    ]

    # =====================
    # 原始来源
    # =====================

    block_id: str

    block_index: int

    table_index: int


    # =====================
    # Table Context
    # =====================

    section_path: tuple[str, ...]

    article: str | None

    item_path: tuple[str, ...]


    # =====================
    # Table Metadata
    # =====================

    title: str | None

    unit: str | None

    frequency: str | None

    purpose: str | None

    content: str | None

    scope: str | None

    format: str | None


    # =====================
    # Table Content
    # =====================

    markdown_table: str

    text: str

    char_count: int

    text_sha256: str


    # =====================
    # Table Size
    # =====================

    rows: int

    cols: int


class CompetitionChunkDocument(
    BaseModel
):
    """
    一个 CompetitionTextDocument
    完成 Chunking 后的结果。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    source: CompetitionKnowledgeSource

    chunks: tuple[
        CompetitionTextChunk,
        ...
    ] = Field(
        min_length=1,
    )

    @model_validator(
        mode="after"
    )
    def validate_document(
        self,
    ) -> Self:
        chunk_ids = [
            chunk.chunk_id
            for chunk
            in self.chunks
        ]

        if (
            len(chunk_ids)
            != len(
                set(chunk_ids)
            )
        ):
            raise ValueError(
                "chunk_id 必须唯一"
            )

        actual_indexes = [
            chunk.chunk_index
            for chunk
            in self.chunks
        ]

        expected_indexes = list(
            range(
                len(self.chunks)
            )
        )

        if (
            actual_indexes
            != expected_indexes
        ):
            raise ValueError(
                "chunk_index 必须从 0 "
                "开始连续递增"
            )

        for chunk in self.chunks:
            if (
                chunk.source_id
                != self.source.source_id
            ):
                raise ValueError(
                    "Chunk source_id 与 "
                    "KnowledgeSource 不一致"
                )

            if (
                chunk.doc_id
                != self.source.doc_id
            ):
                raise ValueError(
                    "Chunk doc_id 与 "
                    "KnowledgeSource 不一致"
                )

            if (
                chunk.source_type
                != self.source.source_type
            ):
                raise ValueError(
                    "Chunk source_type 与 "
                    "KnowledgeSource 不一致"
                )

        return self