from __future__ import annotations

import hashlib

import pytest

from app.schemas.competition_text import (
    CompetitionTextBlock,
)
from app.services.competition_table_chunker import (
    CompetitionTableChunkError,
    build_competition_table_chunks,
)
from app.services.competition_table_context import (
    CompetitionTableContext,
)


def _render_rows(
    rows: tuple[
        tuple[str, ...],
        ...,
    ],
) -> str:
    return "\n".join(
        "\t".join(row)
        for row in rows
    )


def _table_block(
    rows: tuple[
        tuple[str, ...],
        ...,
    ],
) -> CompetitionTextBlock:
    text = _render_rows(rows)

    return CompetitionTextBlock(
        block_id=(
            "block:"
            "doc_src_table_test_"
            "0123456789abcdef01234567:"
            "00003"
        ),
        source_id="src_table_test",
        doc_id=(
            "doc_src_table_test_"
            "0123456789abcdef01234567"
        ),
        source_type="word",
        block_index=3,
        block_type="table",
        text=text,
        table_index=2,
        table_rows=rows,
    )


def _table_context(
) -> CompetitionTableContext:
    return CompetitionTableContext(
        section_path=(
            "资本管理",
        ),
        article="第二十条",
        item_path=(
            "二、披露表格",
            "（一）资本充足率",
        ),
        title="资本充足率监管要求",
        unit="单位：百分比",
        frequency="季度",
        purpose="披露资本充足率",
        content="资本监管指标",
        scope="集团口径",
        format="table",
    )


def test_build_table_chunks_preserves_context_and_provenance(
) -> None:
    rows = (
        (
            "指标",
            "2024Q1",
        ),
        (
            "资本充足率",
            "10.5%",
        ),
    )

    block = _table_block(
        rows
    )

    chunks = (
        build_competition_table_chunks(
            table_block=block,
            context=_table_context(),
            start_chunk_index=7,
        )
    )

    assert len(chunks) == 1

    chunk = chunks[0]

    assert (
        chunk.chunk_id
        == (
            f"chunk:{block.doc_id}:"
            "00007"
        )
    )

    assert chunk.chunk_index == 7
    assert chunk.chunk_type == "table"

    assert chunk.table_index == 2
    assert chunk.table_row_start == 0
    assert chunk.table_row_end == 1
    assert chunk.table_rows == rows

    assert (
        chunk.table_title
        == "资本充足率监管要求"
    )

    assert (
        chunk.table_unit
        == "单位：百分比"
    )

    assert (
        chunk.section_path
        == ("资本管理",)
    )

    assert (
        chunk.article
        == "第二十条"
    )

    assert (
        chunk.article_inherited
        is True
    )

    assert len(
        chunk.source_spans
    ) == 1

    span = chunk.source_spans[0]

    assert span.block_id == block.block_id
    assert span.block_index == 3
    assert span.start_char == 0
    assert span.end_char == len(
        block.text
    )

    assert (
        chunk.char_count
        == len(chunk.text)
    )

    assert (
        chunk.text_sha256
        == hashlib.sha256(
            chunk.text.encode(
                "utf-8"
            )
        ).hexdigest()
    )


def test_build_table_chunks_splits_between_rows_without_loss(
) -> None:
    rows = (
        (
            "指标",
            "数值",
        ),
        (
            "资本充足率",
            "10.5%",
        ),
        (
            "一级资本充足率",
            "8.5%",
        ),
        (
            "核心一级资本充足率",
            "7.5%",
        ),
    )

    chunks = (
        build_competition_table_chunks(
            table_block=(
                _table_block(rows)
            ),
            context=_table_context(),
            start_chunk_index=10,
            max_chars=22,
        )
    )

    assert len(chunks) > 1

    restored_rows = tuple(
        row
        for chunk in chunks
        for row in chunk.table_rows
    )

    assert restored_rows == rows

    expected_row_start = 0

    for offset, chunk in enumerate(
        chunks
    ):
        assert (
            chunk.chunk_index
            == 10 + offset
        )

        assert (
            chunk.table_row_start
            == expected_row_start
        )

        assert (
            chunk.table_row_end
            is not None
        )

        expected_row_start = (
            chunk.table_row_end + 1
        )

        assert (
            chunk.char_count <= 22
            or len(chunk.table_rows)
            == 1
        )

    assert expected_row_start == len(
        rows
    )


def test_build_table_chunks_keeps_oversized_single_row(
) -> None:
    rows = (
        (
            "超长字段" * 20,
            "监管值",
        ),
    )

    chunks = (
        build_competition_table_chunks(
            table_block=(
                _table_block(rows)
            ),
            context=_table_context(),
            max_chars=10,
        )
    )

    assert len(chunks) == 1
    assert chunks[0].table_rows == rows
    assert chunks[0].char_count > 10


def test_build_table_chunks_rejects_non_table_block(
) -> None:
    block = CompetitionTextBlock(
        block_id="block:test:00000",
        source_id="src_test",
        doc_id="doc_test",
        source_type="word",
        block_index=0,
        block_type="paragraph",
        text="普通段落",
        paragraph_index=0,
        style_name="Normal",
    )

    with pytest.raises(
        CompetitionTableChunkError
    ):
        build_competition_table_chunks(
            table_block=block,
            context=_table_context(),
        )