from __future__ import annotations

import hashlib

from app.schemas.competition_chunk import (
    CompetitionChunkSourceSpan,
    CompetitionTextChunk,
)
from app.schemas.competition_text import (
    CompetitionTextBlock,
)
from app.services.competition_table_context import (
    CompetitionTableContext,
)


DEFAULT_TABLE_MAX_CHARS = 900


class CompetitionTableChunkError(
    RuntimeError
):
    """Competition 表格 Chunk 构建异常。"""


TableRows = tuple[
    tuple[str, ...],
    ...,
]

TableRowSlice = tuple[
    int,
    int,
    TableRows,
]


def _sha256_text(
    text: str,
) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def _build_chunk_id(
    *,
    doc_id: str,
    chunk_index: int,
) -> str:
    return (
        f"chunk:{doc_id}:"
        f"{chunk_index:05d}"
    )


def _render_table_rows(
    rows: TableRows,
) -> str:
    """
    将二维表格转换为稳定的检索文本。

    列之间使用制表符，
    行之间使用换行符。
    原始二维结构仍保存在 table_rows。
    """

    return "\n".join(
        "\t".join(row)
        for row in rows
    )


def _split_table_rows(
    rows: TableRows,
    *,
    max_chars: int,
) -> tuple[
    TableRowSlice,
    ...,
]:
    """
    只允许在完整行之间切分。

    max_chars 是软上限：
    如果单独一行已经超过 max_chars，
    仍然保留整行，不拆分单元格。
    """

    if max_chars <= 0:
        raise ValueError(
            "max_chars 必须大于 0"
        )

    if not rows:
        raise CompetitionTableChunkError(
            "表格没有可切分的行"
        )

    result: list[
        TableRowSlice
    ] = []

    current_rows: list[
        tuple[str, ...]
    ] = []

    current_start = 0

    for row_index, row in enumerate(
        rows
    ):
        candidate_rows = tuple(
            current_rows
            + [row]
        )

        candidate_text = (
            _render_table_rows(
                candidate_rows
            )
        )

        current_text = (
            _render_table_rows(
                tuple(current_rows)
            )
            if current_rows
            else ""
        )

        row_text = (
            _render_table_rows(
                (row,)
            )
        )

        should_flush = (
            bool(current_rows)
            and bool(current_text.strip())
            and bool(row_text.strip())
            and len(candidate_text)
            > max_chars
        )

        if should_flush:
            result.append(
                (
                    current_start,
                    row_index - 1,
                    tuple(current_rows),
                )
            )

            current_rows = [row]
            current_start = row_index

        else:
            current_rows.append(
                row
            )

    if current_rows:
        current_text = (
            _render_table_rows(
                tuple(current_rows)
            )
        )

        # 尾部如果只有空白行，
        # 将其合并回前一个连续行段，
        # 避免产生无文本 Chunk。
        if (
            not current_text.strip()
            and result
        ):
            (
                previous_start,
                _,
                previous_rows,
            ) = result.pop()

            merged_rows = (
                previous_rows
                + tuple(current_rows)
            )

            result.append(
                (
                    previous_start,
                    len(rows) - 1,
                    merged_rows,
                )
            )

        else:
            result.append(
                (
                    current_start,
                    len(rows) - 1,
                    tuple(current_rows),
                )
            )

    return tuple(result)


def _build_table_chunk(
    *,
    table_block: CompetitionTextBlock,
    context: CompetitionTableContext,
    chunk_index: int,
    row_start: int,
    row_end: int,
    rows: TableRows,
) -> CompetitionTextChunk:
    if table_block.table_index is None:
        raise CompetitionTableChunkError(
            "Word table Block 缺少 "
            "table_index"
        )

    text = _render_table_rows(
        rows
    )

    if not text.strip():
        raise CompetitionTableChunkError(
            "Table Chunk 没有可检索文本"
        )

    return CompetitionTextChunk(
        chunk_id=_build_chunk_id(
            doc_id=table_block.doc_id,
            chunk_index=chunk_index,
        ),
        source_id=(
            table_block.source_id
        ),
        doc_id=table_block.doc_id,
        source_type="word",
        chunk_index=chunk_index,
        chunk_type="table",
        source_spans=(
            CompetitionChunkSourceSpan(
                block_id=(
                    table_block.block_id
                ),
                block_index=(
                    table_block.block_index
                ),
                start_char=0,
                end_char=len(
                    table_block.text
                ),
            ),
        ),
        text=text,
        char_count=len(text),
        text_sha256=(
            _sha256_text(text)
        ),
        section_path=(
            context.section_path
        ),
        article=context.article,
        item_path=context.item_path,

        # 表格自身不包含 Article 行，
        # 因此存在 Article 时属于继承上下文。
        article_inherited=(
            context.article is not None
        ),
        table_index=(
            table_block.table_index
        ),
        table_row_start=row_start,
        table_row_end=row_end,
        table_rows=rows,
        table_title=context.title,
        table_unit=context.unit,
        table_frequency=(
            context.frequency
        ),
        table_purpose=(
            context.purpose
        ),
        table_content=context.content,
        table_scope=context.scope,
        table_format=(
            context.format or "table"
        ),
    )


def build_competition_table_chunks(
    *,
    table_block: CompetitionTextBlock,
    context: CompetitionTableContext,
    start_chunk_index: int = 0,
    max_chars: int = (
        DEFAULT_TABLE_MAX_CHARS
    ),
) -> tuple[
    CompetitionTextChunk,
    ...,
]:
    """
    将一个 Word Table Block 构造成
    一个或多个统一 CompetitionTextChunk。

    保证：

    1. 只在完整行之间切分；
    2. table_rows 不丢失、不重复；
    3. 行范围使用原表 0-based 索引；
    4. 所有 Chunk 保留原 Table Block 来源；
    5. 单行超过上限时不破坏单元格。
    """

    if start_chunk_index < 0:
        raise ValueError(
            "start_chunk_index "
            "不能小于 0"
        )

    if (
        table_block.block_type
        != "table"
    ):
        raise CompetitionTableChunkError(
            "table_block 必须是 "
            "table 类型"
        )

    if (
        table_block.source_type
        != "word"
    ):
        raise CompetitionTableChunkError(
            "当前只支持 Word table"
        )

    row_slices = _split_table_rows(
        table_block.table_rows,
        max_chars=max_chars,
    )

    chunks = []

    for offset, (
        row_start,
        row_end,
        rows,
    ) in enumerate(row_slices):
        chunks.append(
            _build_table_chunk(
                table_block=table_block,
                context=context,
                chunk_index=(
                    start_chunk_index
                    + offset
                ),
                row_start=row_start,
                row_end=row_end,
                rows=rows,
            )
        )

    return tuple(chunks)