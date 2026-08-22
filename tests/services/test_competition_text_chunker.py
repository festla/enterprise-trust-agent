from __future__ import annotations

from app.schemas.competition_evidence import (
    CompetitionKnowledgeSource,
)
from app.schemas.competition_text import (
    CompetitionTextBlock,
    CompetitionTextDocument,
)
from app.services.competition_text_chunker import (
    build_competition_text_chunks,
)


def _word_source(
) -> CompetitionKnowledgeSource:
    return CompetitionKnowledgeSource(
        source_id=(
            "src_word_01234567"
        ),
        doc_id=(
            "doc_src_word_01234567_"
            "0123456789abcdef01234567"
        ),
        title="测试 Word",
        source_type="word",
        relative_path="test.docx",
        sha256="1" * 64,
    )


def _pdf_source(
) -> CompetitionKnowledgeSource:
    return CompetitionKnowledgeSource(
        source_id=(
            "src_pdf_012345678"
        ),
        doc_id=(
            "doc_src_pdf_012345678_"
            "0123456789abcdef01234567"
        ),
        title="测试 PDF",
        source_type="pdf",
        relative_path="test.pdf",
        sha256="2" * 64,
    )


def _word_block(
    *,
    source: CompetitionKnowledgeSource,
    block_index: int,
    paragraph_index: int,
    text: str,
    outline_level: int | None = None,
) -> CompetitionTextBlock:
    return CompetitionTextBlock(
        block_id=(
            f"block:{source.doc_id}:"
            f"{block_index:05d}"
        ),
        source_id=(
            source.source_id
        ),
        doc_id=(
            source.doc_id
        ),
        source_type="word",
        block_index=(
            block_index
        ),
        block_type="paragraph",
        text=text,
        paragraph_index=(
            paragraph_index
        ),
        style_name="Normal",
        outline_level=(
            outline_level
        ),
    )

def _word_table_block(
    *,
    source: CompetitionKnowledgeSource,
    block_index: int,
    table_index: int,
    rows: tuple[
        tuple[str, ...],
        ...,
    ],
) -> CompetitionTextBlock:
    text = "\n".join(
        "\t".join(row)
        for row in rows
    )

    return CompetitionTextBlock(
        block_id=(
            f"block:{source.doc_id}:"
            f"{block_index:05d}"
        ),
        source_id=source.source_id,
        doc_id=source.doc_id,
        source_type="word",
        block_index=block_index,
        block_type="table",
        text=text,
        table_index=table_index,
        table_rows=rows,
    )

def test_word_chunks_inherit_outline_and_numbered_context(
) -> None:
    source = _word_source()

    document = (
        CompetitionTextDocument(
            source=source,
            blocks=(
                _word_block(
                    source=source,
                    block_index=0,
                    paragraph_index=0,
                    text="信用风险",
                    outline_level=0,
                ),
                _word_block(
                    source=source,
                    block_index=1,
                    paragraph_index=1,
                    text="二、披露表格",
                ),
                _word_block(
                    source=source,
                    block_index=2,
                    paragraph_index=2,
                    text=(
                        "（一）表格 CR1："
                        "资产质量"
                    ),
                ),
                _word_block(
                    source=source,
                    block_index=3,
                    paragraph_index=3,
                    text=(
                        "目的：披露商业银行"
                        "资产质量信息。"
                    ),
                ),
                _word_block(
                    source=source,
                    block_index=4,
                    paragraph_index=4,
                    text=(
                        "适用范围："
                        "国内系统重要性银行。"
                    ),
                ),
            ),
        )
    )

    chunks = (
        build_competition_text_chunks(
            document,
            max_chars=500,
        )
    )

    # 一、 / （一）分别改变 item_path，
    # 所以形成独立结构单元。
    assert len(
        chunks
    ) == 2

    first = chunks[0]

    second = chunks[1]

    assert (
        first.section_path
        == (
            "信用风险",
        )
    )

    assert (
        first.item_path
        == (
            "二、披露表格",
        )
    )

    assert (
        first.text
        == "二、披露表格"
    )

    assert (
        second.section_path
        == (
            "信用风险",
        )
    )

    assert (
        second.item_path
        == (
            "二、披露表格",
            "（一）表格 CR1：资产质量",
        )
    )

    assert (
        "目的："
        in second.text
    )

    assert (
        "适用范围："
        in second.text
    )


def test_same_context_paragraphs_are_merged(
) -> None:
    source = _word_source()

    document = (
        CompetitionTextDocument(
            source=source,
            blocks=(
                _word_block(
                    source=source,
                    block_index=0,
                    paragraph_index=0,
                    text="风险管理要求如下。",
                ),
                _word_block(
                    source=source,
                    block_index=1,
                    paragraph_index=1,
                    text="商业银行应建立制度。",
                ),
                _word_block(
                    source=source,
                    block_index=2,
                    paragraph_index=2,
                    text="制度应覆盖全部业务。",
                ),
            ),
        )
    )

    chunks = (
        build_competition_text_chunks(
            document,
            max_chars=500,
        )
    )

    assert len(
        chunks
    ) == 1

    assert (
        chunks[0].text
        == (
            "风险管理要求如下。\n"
            "商业银行应建立制度。\n"
            "制度应覆盖全部业务。"
        )
    )

    assert (
        len(
            chunks[0]
            .source_spans
        )
        == 3
    )

    assert (
        chunks[0]
        .paragraph_start_index
        == 0
    )

    assert (
        chunks[0]
        .paragraph_end_index
        == 2
    )


def test_article_and_following_body_share_chunk(
) -> None:
    source = _word_source()

    document = (
        CompetitionTextDocument(
            source=source,
            blocks=(
                _word_block(
                    source=source,
                    block_index=0,
                    paragraph_index=0,
                    text=(
                        "第十二条 "
                        "商业银行应建立制度。"
                    ),
                ),
                _word_block(
                    source=source,
                    block_index=1,
                    paragraph_index=1,
                    text=(
                        "制度应覆盖全部业务。"
                    ),
                ),
            ),
        )
    )

    chunks = (
        build_competition_text_chunks(
            document,
            max_chars=500,
        )
    )

    assert len(
        chunks
    ) == 1

    chunk = chunks[0]

    assert (
        chunk.article
        == "第十二条"
    )

    # 当前 Chunk 自己包含第十二条，
    # 所以不是 inherited。
    assert (
        chunk.article_inherited
        is False
    )

    assert (
        "制度应覆盖全部业务"
        in chunk.text
    )


def test_pdf_lines_preserve_source_offsets(
) -> None:
    source = _pdf_source()

    page_text = (
        "信用风险\n"
        "一、披露内容\n"
        "商业银行应披露信用风险信息。\n"
        "商业银行还应说明重大变化。"
    )

    block = CompetitionTextBlock(
        block_id=(
            f"block:{source.doc_id}:"
            "00000"
        ),
        source_id=source.source_id,
        doc_id=source.doc_id,
        source_type="pdf",
        block_index=0,
        block_type="page_text",
        text=page_text,
        page=3,
    )

    document = (
        CompetitionTextDocument(
            source=source,
            blocks=(
                block,
            ),
        )
    )

    chunks = (
        build_competition_text_chunks(
            document,
            max_chars=500,
        )
    )

    # PDF 没有 Word outline，
    # "信用风险" 本身只是普通正文。
    #
    # "一、披露内容" 会开启 item_l1，
    # 因此这里至少会出现两个 Context。
    assert len(
        chunks
    ) == 2

    second = chunks[1]

    assert (
        second.page_start
        == 3
    )

    assert (
        second.page_end
        == 3
    )

    assert (
        second.item_path
        == (
            "一、披露内容",
        )
    )

    for span in (
        second.source_spans
    ):
        recovered = page_text[
            span.start_char:
            span.end_char
        ]

        assert recovered.strip()


def test_max_chars_splits_long_text(
) -> None:
    source = _word_source()

    long_text = (
        "商业银行应建立风险管理制度。"
        * 30
    )

    document = (
        CompetitionTextDocument(
            source=source,
            blocks=(
                _word_block(
                    source=source,
                    block_index=0,
                    paragraph_index=0,
                    text=long_text,
                ),
            ),
        )
    )

    chunks = (
        build_competition_text_chunks(
            document,
            max_chars=100,
        )
    )

    assert len(
        chunks
    ) > 1

    assert all(
        chunk.char_count
        <= 100
        for chunk
        in chunks
    )

    assert all(
        len(
            chunk.source_spans
        )
        == 1
        for chunk
        in chunks
    )


def test_table_flushes_text_buffer(
) -> None:
    source = _word_source()

    table_text = (
        "指标\t要求\n"
        "资本充足率\t不得低于标准"
    )

    document = (
        CompetitionTextDocument(
            source=source,
            blocks=(
                _word_block(
                    source=source,
                    block_index=0,
                    paragraph_index=0,
                    text="表格前正文。",
                ),
                CompetitionTextBlock(
                    block_id=(
                        f"block:"
                        f"{source.doc_id}:"
                        "00001"
                    ),
                    source_id=(
                        source.source_id
                    ),
                    doc_id=(
                        source.doc_id
                    ),
                    source_type="word",
                    block_index=1,
                    block_type="table",
                    text=table_text,
                    table_index=0,
                    table_rows=(
                        (
                            "指标",
                            "要求",
                        ),
                        (
                            "资本充足率",
                            "不得低于标准",
                        ),
                    ),
                ),
                _word_block(
                    source=source,
                    block_index=2,
                    paragraph_index=1,
                    text="表格后填写说明。",
                ),
            ),
        )
    )

    chunks = (
        build_competition_text_chunks(
            document,
            max_chars=500,
        )
    )

    assert len(
        chunks
    ) == 3

    assert [
        chunk.chunk_type
        for chunk in chunks
    ] == [
        "text",
        "table",
        "text",
    ]

    assert [
        chunk.chunk_index
        for chunk in chunks
    ] == [
        0,
        1,
        2,
    ]

    assert (
        chunks[0].text
        == "表格前正文。"
    )

    table_chunk = chunks[1]

    assert (
        table_chunk.text
        == table_text
    )

    assert (
        table_chunk.table_index
        == 0
    )

    assert (
        table_chunk.table_row_start
        == 0
    )

    assert (
        table_chunk.table_row_end
        == 1
    )

    assert (
        table_chunk.table_rows
        == (
            (
                "指标",
                "要求",
            ),
            (
                "资本充足率",
                "不得低于标准",
            ),
        )
    )

    assert (
        table_chunk.table_format
        == "table"
    )

    assert (
        chunks[2].text
        == "表格后填写说明。"
    )

    assert len(
        {
            chunk.chunk_id
            for chunk in chunks
        }
    ) == 3


def test_table_inherits_regulatory_context_and_nearby_metadata(
) -> None:
    source = _word_source()

    table_rows = (
        (
            "指标",
            "数值",
        ),
        (
            "资本充足率",
            "10.5%",
        ),
    )

    table_text = (
        "指标\t数值\n"
        "资本充足率\t10.5%"
    )

    document = (
        CompetitionTextDocument(
            source=source,
            blocks=(
                _word_block(
                    source=source,
                    block_index=0,
                    paragraph_index=0,
                    text="资本管理",
                    outline_level=0,
                ),
                _word_block(
                    source=source,
                    block_index=1,
                    paragraph_index=1,
                    text="二、披露表格",
                ),
                _word_block(
                    source=source,
                    block_index=2,
                    paragraph_index=2,
                    text="单位：百分比",
                ),
                CompetitionTextBlock(
                    block_id=(
                        f"block:"
                        f"{source.doc_id}:"
                        "00003"
                    ),
                    source_id=(
                        source.source_id
                    ),
                    doc_id=(
                        source.doc_id
                    ),
                    source_type="word",
                    block_index=3,
                    block_type="table",
                    text=table_text,
                    table_index=0,
                    table_rows=table_rows,
                ),
                _word_block(
                    source=source,
                    block_index=4,
                    paragraph_index=3,
                    text="填写说明。",
                ),
            ),
        )
    )

    chunks = (
        build_competition_text_chunks(
            document,
            max_chars=500,
        )
    )

    assert [
        chunk.chunk_type
        for chunk in chunks
    ] == [
        "text",
        "table",
        "text",
    ]

    table_chunk = chunks[1]

    assert (
        table_chunk.section_path
        == ("资本管理",)
    )

    assert (
        table_chunk.item_path
        == ("二、披露表格",)
    )

    assert (
        table_chunk.table_unit
        == "单位：百分比"
    )

    assert (
        table_chunk.table_rows
        == table_rows
    )

    assert [
        chunk.chunk_index
        for chunk in chunks
    ] == [
        0,
        1,
        2,
    ]

    assert len(
        {
            chunk.chunk_id
            for chunk in chunks
        }
    ) == len(chunks)

def test_instruction_table_inherits_main_table_title(
) -> None:
    source = _word_source()

    first_title = (
        "（一）表格KM1："
        "监管并表关键审慎监管指标"
    )

    second_title = (
        "（二）表格OVA："
        "风险管理定性信息"
    )

    document = CompetitionTextDocument(
        source=source,
        blocks=(
            _word_block(
                source=source,
                block_index=0,
                paragraph_index=0,
                text="二、披露表格",
            ),
            _word_block(
                source=source,
                block_index=1,
                paragraph_index=1,
                text=first_title,
            ),
            _word_table_block(
                source=source,
                block_index=2,
                table_index=0,
                rows=(
                    (
                        "目的：披露关键审慎监管指标。",
                        "",
                    ),
                    (
                        "指标",
                        "数值",
                    ),
                ),
            ),
            _word_block(
                source=source,
                block_index=3,
                paragraph_index=2,
                text="填写说明",
                outline_level=1,
            ),
            _word_block(
                source=source,
                block_index=4,
                paragraph_index=3,
                text="1.定义",
            ),
            _word_table_block(
                source=source,
                block_index=5,
                table_index=1,
                rows=(
                    (
                        "行号",
                        "说明",
                    ),
                    (
                        "1行",
                        "监管指标说明。",
                    ),
                ),
            ),
            _word_block(
                source=source,
                block_index=6,
                paragraph_index=4,
                text=second_title,
            ),
            _word_table_block(
                source=source,
                block_index=7,
                table_index=2,
                rows=(
                    (
                        "目的：披露风险管理定性信息。",
                        "",
                    ),
                    (
                        "项目",
                        "内容",
                    ),
                ),
            ),
        ),
    )

    chunks = build_competition_text_chunks(
        document,
        max_chars=500,
    )

    table_chunks = [
        chunk
        for chunk in chunks
        if chunk.chunk_type == "table"
    ]

    assert [
        chunk.table_title
        for chunk in table_chunks
    ] == [
        first_title,
        first_title,
        second_title,
    ]

    # “填写说明”清除了法规 item_path，
    # 但不应清除表格标题继承关系。
    assert (
        table_chunks[1].item_path
        ==
        ("1.定义",)
    )