import pytest

from pydantic import (
    ValidationError,
)

from app.schemas.competition_evidence import (
    CompetitionKnowledgeSource,
)
from app.schemas.competition_text import (
    CompetitionTextBlock,
    CompetitionTextDocument,
)


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
        sha256=(
            "0" * 64
        ),
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
        sha256=(
            "1" * 64
        ),
    )


def test_pdf_page_text_document(
) -> None:
    source = _pdf_source()

    block = CompetitionTextBlock(
        block_id=(
            "block:"
            "src_0123456789abcdef:"
            "00000"
        ),
        source_id=source.source_id,
        doc_id=source.doc_id,
        source_type="pdf",
        block_index=0,
        block_type="page_text",
        text=(
            "第一条 商业银行应当"
            "建立完善的风险管理制度。"
        ),
        page=1,
    )

    document = (
        CompetitionTextDocument(
            source=source,
            blocks=(
                block,
            ),
        )
    )

    assert (
        document.blocks[0].page
        == 1
    )

    assert (
        document.blocks[0]
        .block_type
        == "page_text"
    )


def test_word_document_preserves_paragraph_and_table(
) -> None:
    source = _word_source()

    paragraph = (
        CompetitionTextBlock(
            block_id=(
                "block:"
                "src_abcdef0123456789:"
                "00000"
            ),
            source_id=source.source_id,
            doc_id=source.doc_id,
            source_type="word",
            block_index=0,
            block_type="paragraph",
            text="第二章 风险管理",
            paragraph_index=0,
        )
    )

    table = (
        CompetitionTextBlock(
            block_id=(
                "block:"
                "src_abcdef0123456789:"
                "00001"
            ),
            source_id=source.source_id,
            doc_id=source.doc_id,
            source_type="word",
            block_index=1,
            block_type="table",
            text=(
                "指标\t监管要求\n"
                "资本充足率\t不得低于规定标准"
            ),
            table_index=0,
            table_rows=(
                (
                    "指标",
                    "监管要求",
                ),
                (
                    "资本充足率",
                    "不得低于规定标准",
                ),
            ),
        )
    )

    document = (
        CompetitionTextDocument(
            source=source,
            blocks=(
                paragraph,
                table,
            ),
        )
    )

    assert (
        len(document.blocks)
        == 2
    )

    assert (
        document.blocks[0]
        .block_type
        == "paragraph"
    )

    assert (
        document.blocks[1]
        .block_type
        == "table"
    )

    assert (
        document.blocks[1]
        .table_rows[1][0]
        == "资本充足率"
    )


def test_word_table_requires_table_rows(
) -> None:
    source = _word_source()

    with pytest.raises(
        ValidationError
    ):
        CompetitionTextBlock(
            block_id="block:test",
            source_id=source.source_id,
            doc_id=source.doc_id,
            source_type="word",
            block_index=0,
            block_type="table",
            text="测试表格",
            table_index=0,
            table_rows=(),
        )


def test_document_rejects_source_identity_mismatch(
) -> None:
    source = _pdf_source()

    block = CompetitionTextBlock(
        block_id="block:test",
        source_id=(
            "src_deadbeefdeadbeef"
        ),
        doc_id=source.doc_id,
        source_type="pdf",
        block_index=0,
        block_type="page_text",
        text="测试正文",
        page=1,
    )

    with pytest.raises(
        ValidationError
    ):
        CompetitionTextDocument(
            source=source,
            blocks=(
                block,
            ),
        )