from __future__ import annotations

from typing import (
    Literal,
    Self,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.schemas.competition_evidence import (
    CompetitionKnowledgeSource,
)


CompetitionTextSourceType = Literal[
    "pdf",
    "word",
]

CompetitionTextBlockType = Literal[
    "page_text",
    "paragraph",
    "table",
]


class CompetitionTextBlock(
    BaseModel
):
    """
    PDF / Word Parser 输出的统一最小结构单元。

    注意：
    这还不是最终 Retrieval Chunk。

    Parser:
        Source -> TextBlock

    Chunker:
        TextBlock -> Retrieval Chunk
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )

    schema_version: Literal[1] = 1

    block_id: str = Field(
        min_length=1,
    )

    source_id: str = Field(
        min_length=1,
    )

    doc_id: str = Field(
        min_length=1,
    )

    source_type: CompetitionTextSourceType

    block_index: int = Field(
        ge=0,
    )

    block_type: CompetitionTextBlockType

    text: str = Field(
        min_length=1,
    )

    # ========================================================
    # PDF Location
    # ========================================================

    page: int | None = Field(
        default=None,
        ge=1,
    )

    # ========================================================
    # Word Location
    # ========================================================

    paragraph_index: int | None = Field(
        default=None,
        ge=0,
    )

    table_index: int | None = Field(
        default=None,
        ge=0,
    )

    # ========================================================
    # Table Structure
    #
    # Retrieval 时仍然使用 text，
    # 但原始表格结构不能丢。
    # ========================================================

    table_rows: tuple[
        tuple[
            str,
            ...
        ],
        ...
    ] = ()

    # 后续 Regulatory Chunker 填充。
    section_path: tuple[
        str,
        ...
    ] = ()

    @model_validator(
        mode="after"
    )
    def validate_block_location(
        self,
    ) -> Self:
        # ====================================================
        # PDF
        # ====================================================

        if self.block_type == "page_text":
            if self.source_type != "pdf":
                raise ValueError(
                    "page_text 只能来自 PDF"
                )

            if self.page is None:
                raise ValueError(
                    "PDF page_text 必须提供 page"
                )

            if (
                self.paragraph_index
                is not None
                or self.table_index
                is not None
            ):
                raise ValueError(
                    "PDF page_text 不能包含 "
                    "Word paragraph/table index"
                )

            if self.table_rows:
                raise ValueError(
                    "PDF page_text 不能包含 "
                    "table_rows"
                )

            return self

        # ====================================================
        # Word Paragraph
        # ====================================================

        if self.block_type == "paragraph":
            if self.source_type != "word":
                raise ValueError(
                    "paragraph Block "
                    "当前只允许来自 Word"
                )

            if self.paragraph_index is None:
                raise ValueError(
                    "Word paragraph 必须提供 "
                    "paragraph_index"
                )

            if self.page is not None:
                raise ValueError(
                    "Word paragraph 不使用 page"
                )

            if self.table_index is not None:
                raise ValueError(
                    "paragraph 不能提供 "
                    "table_index"
                )

            if self.table_rows:
                raise ValueError(
                    "paragraph 不能包含 "
                    "table_rows"
                )

            return self

        # ====================================================
        # Word Table
        # ====================================================

        if self.block_type == "table":
            if self.source_type != "word":
                raise ValueError(
                    "table Block "
                    "当前只允许来自 Word"
                )

            if self.table_index is None:
                raise ValueError(
                    "Word table 必须提供 "
                    "table_index"
                )

            if self.page is not None:
                raise ValueError(
                    "Word table 不使用 page"
                )

            if (
                self.paragraph_index
                is not None
            ):
                raise ValueError(
                    "table 不能提供 "
                    "paragraph_index"
                )

            if not self.table_rows:
                raise ValueError(
                    "Word table 必须保留 "
                    "table_rows"
                )

            if any(
                not row
                for row
                in self.table_rows
            ):
                raise ValueError(
                    "table_rows "
                    "不能包含空行"
                )

            return self

        raise ValueError(
            "未知 block_type"
        )


class CompetitionTextDocument(
    BaseModel
):
    """
    一个已经完成结构化解析的 PDF / Word 文档。

    KnowledgeSource:
        负责文件身份和版本。

    TextBlock:
        负责文件内部内容和位置。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    source: CompetitionKnowledgeSource

    blocks: tuple[
        CompetitionTextBlock,
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
        if (
            self.source.source_type
            not in {
                "pdf",
                "word",
            }
        ):
            raise ValueError(
                "CompetitionTextDocument "
                "只支持 PDF / Word"
            )

        block_ids = [
            block.block_id
            for block
            in self.blocks
        ]

        if (
            len(block_ids)
            != len(set(block_ids))
        ):
            raise ValueError(
                "block_id 必须唯一"
            )

        expected_indexes = list(
            range(
                len(self.blocks)
            )
        )

        actual_indexes = [
            block.block_index
            for block
            in self.blocks
        ]

        if (
            actual_indexes
            != expected_indexes
        ):
            raise ValueError(
                "block_index 必须从 0 "
                "开始连续递增"
            )

        for block in self.blocks:
            if (
                block.source_id
                != self.source.source_id
            ):
                raise ValueError(
                    "Block source_id 与 "
                    "KnowledgeSource 不一致"
                )

            if (
                block.doc_id
                != self.source.doc_id
            ):
                raise ValueError(
                    "Block doc_id 与 "
                    "KnowledgeSource 不一致"
                )

            if (
                block.source_type
                != self.source.source_type
            ):
                raise ValueError(
                    "Block source_type 与 "
                    "KnowledgeSource 不一致"
                )

        return self