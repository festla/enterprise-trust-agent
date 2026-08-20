from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from app.schemas.competition_chunk import (
    CompetitionChunkDocument,
    CompetitionChunkSourceSpan,
    CompetitionTextChunk,
    CompetitionTableChunk
)
from app.schemas.competition_evidence import (
    CompetitionKnowledgeSource,
)


def _sha(
    text: str,
) -> str:
    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def _pdf_source(
) -> CompetitionKnowledgeSource:
    return CompetitionKnowledgeSource(
        source_id=(
            "src_0123456789abcdef"
        ),
        doc_id=(
            "doc_src_0123456789abcdef_"
            "0123456789abcdef01234567"
        ),
        title="测试监管 PDF",
        source_type="pdf",
        relative_path="test.pdf",
        sha256="0" * 64,
    )


def _word_source(
) -> CompetitionKnowledgeSource:
    return CompetitionKnowledgeSource(
        source_id=(
            "src_abcdef0123456789"
        ),
        doc_id=(
            "doc_src_abcdef0123456789_"
            "0123456789abcdef01234567"
        ),
        title="测试监管 Word",
        source_type="word",
        relative_path="test.docx",
        sha256="1" * 64,
    )


def test_pdf_text_chunk_preserves_article_and_span(
) -> None:
    source = _pdf_source()

    text = (
        "第十二条 商业银行应当"
        "建立完善的风险管理制度。"
    )

    chunk = CompetitionTextChunk(
        chunk_id="chunk:test:00000",
        source_id=source.source_id,
        doc_id=source.doc_id,
        source_type="pdf",
        chunk_index=0,
        chunk_type="text",
        source_spans=(
            CompetitionChunkSourceSpan(
                block_id="block:test:00000",
                block_index=0,
                start_char=20,
                end_char=(
                    20 + len(text)
                ),
            ),
        ),
        text=text,
        char_count=len(text),
        text_sha256=_sha(
            text
        ),
        section_path=(
            "第二章 风险管理",
        ),
        article="第十二条",
        article_inherited=False,
        page_start=3,
        page_end=3,
    )

    assert (
        chunk.article
        == "第十二条"
    )

    assert (
        chunk.section_path
        == (
            "第二章 风险管理",
        )
    )

    assert (
        chunk.source_spans[0]
        .start_char
        == 20
    )


def test_word_text_chunk_can_span_paragraphs(
) -> None:
    source = _word_source()

    text = (
        "第十三条 银行业金融机构"
        "应建立内部控制制度。\n"
        "内部控制应覆盖各业务流程。"
    )

    chunk = CompetitionTextChunk(
        chunk_id="chunk:test:00000",
        source_id=source.source_id,
        doc_id=source.doc_id,
        source_type="word",
        chunk_index=0,
        chunk_type="text",
        source_spans=(
            CompetitionChunkSourceSpan(
                block_id="block:test:00010",
                block_index=10,
                start_char=0,
                end_char=20,
            ),
            CompetitionChunkSourceSpan(
                block_id="block:test:00011",
                block_index=11,
                start_char=0,
                end_char=15,
            ),
        ),
        text=text,
        char_count=len(text),
        text_sha256=_sha(
            text
        ),
        section_path=(
            "第三章 内部控制",
        ),
        article="第十三条",
        article_inherited=False,
        paragraph_start_index=20,
        paragraph_end_index=21,
    )

    assert (
        len(
            chunk.source_spans
        )
        == 2
    )

    assert (
        chunk.paragraph_start_index
        == 20
    )

    assert (
        chunk.paragraph_end_index
        == 21
    )


def test_word_table_chunk_preserves_rows(
) -> None:
    source = _word_source()

    rows = (
        (
            "指标",
            "监管要求",
        ),
        (
            "资本充足率",
            "不得低于规定标准",
        ),
    )

    text = (
        "指标\t监管要求\n"
        "资本充足率\t不得低于规定标准"
    )

    chunk = CompetitionTextChunk(
        chunk_id="chunk:test:00000",
        source_id=source.source_id,
        doc_id=source.doc_id,
        source_type="word",
        chunk_index=0,
        chunk_type="table",
        source_spans=(
            CompetitionChunkSourceSpan(
                block_id="block:test:00020",
                block_index=20,
                start_char=0,
                end_char=len(
                    text
                ),
            ),
        ),
        text=text,
        char_count=len(text),
        text_sha256=_sha(
            text
        ),
        section_path=(
            "第四章 监管指标",
        ),
        article="第二十条",
        article_inherited=True,
        table_index=0,
        table_row_start=0,
        table_row_end=1,
        table_rows=rows,
    )

    assert (
        chunk.chunk_type
        == "table"
    )

    assert (
        chunk.table_rows[1][0]
        == "资本充足率"
    )

    assert (
        chunk.article_inherited
        is True
    )


def test_chunk_rejects_invalid_hash(
) -> None:
    source = _pdf_source()

    with pytest.raises(
        ValidationError
    ):
        CompetitionTextChunk(
            chunk_id="chunk:test",
            source_id=source.source_id,
            doc_id=source.doc_id,
            source_type="pdf",
            chunk_index=0,
            chunk_type="text",
            source_spans=(
                CompetitionChunkSourceSpan(
                    block_id="block:test",
                    block_index=0,
                    start_char=0,
                    end_char=4,
                ),
            ),
            text="测试正文",
            char_count=4,
            text_sha256="0" * 64,
            page_start=1,
            page_end=1,
        )


def test_chunk_document_rejects_source_mismatch(
) -> None:
    source = _pdf_source()

    text = "测试正文"

    chunk = CompetitionTextChunk(
        chunk_id="chunk:test:00000",
        source_id=(
            "src_deadbeefdeadbeef"
        ),
        doc_id=source.doc_id,
        source_type="pdf",
        chunk_index=0,
        chunk_type="text",
        source_spans=(
            CompetitionChunkSourceSpan(
                block_id="block:test",
                block_index=0,
                start_char=0,
                end_char=len(
                    text
                ),
            ),
        ),
        text=text,
        char_count=len(text),
        text_sha256=_sha(
            text
        ),
        page_start=1,
        page_end=1,
    )

    with pytest.raises(
        ValidationError
    ):
        CompetitionChunkDocument(
            source=source,
            chunks=(
                chunk,
            ),
        )

def test_competition_table_chunk_schema():

    chunk = CompetitionTableChunk(
        chunk_id="chunk:test:00001",
        source_id="src:test",
        doc_id="doc:test",
        source_type="word",
        chunk_index=0,
        chunk_type="table",

        block_id="block:test",
        block_index=1,
        table_index=0,

        section_path=(
            "资本管理",
        ),

        article=None,

        item_path=(
            "资本充足率",
        ),

        title="资本充足率表",

        unit="百分比",

        frequency="季度",

        purpose="披露资本指标",

        content="资本充足率",

        scope="集团",

        format="表格",

        markdown_table=(
            "|指标|数值|\n"
            "|---|---|\n"
            "|资本充足率|10|"
        ),

        text="资本充足率表",

        char_count=6,

        text_sha256="a"*64,

        rows=2,

        cols=2,
    )


    assert chunk.chunk_type == "table"

    assert chunk.rows == 2

    assert (
        chunk.section_path
        ==
        ("资本管理",)
    )