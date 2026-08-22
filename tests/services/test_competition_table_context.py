from app.schemas.competition_text import (
    CompetitionTextBlock,
)

from app.services.competition_table_context import (
    build_table_context,
    extract_explicit_table_title,
)

from app.services.competition_regulatory_context import (
    CompetitionRegulatoryContext,
)


def test_build_table_context():

    block = CompetitionTextBlock(

        block_id="table:1",

        source_id="src",

        doc_id="doc",

        source_type="word",

        block_index=1,

        block_type="table",

        text="表CR3：资本充足率",

        table_index=0,

        table_rows=[
            [
                "指标",
                "2024Q1",
            ],
            [
                "资本充足率",
                "10.5%",
            ],
        ],

    )


    context = (
        CompetitionRegulatoryContext(
            section_path=(
                "资本管理",
            ),

            article=None,

            item_path=(
                "二、资本披露",
            ),

        )
    )


    result = (
        build_table_context(
            table_block=block,
            context=context,
            nearby_text=(
                "单位：百分比",
                "频率：季度",
                "目的：披露资本情况",
            ),
        )
    )


    assert (
        result.section_path
        ==
        (
            "资本管理",
        )
    )


    assert (
        result.item_path
        ==
        (
            "二、资本披露",
        )
    )


    assert (
        result.title
        ==
        "表CR3：资本充足率"
    )


    assert (
        result.unit
        ==
        "单位：百分比"
    )


    assert (
        result.frequency
        ==
        "季度"
    )


    assert (
        result.purpose
        ==
        "目的：披露资本情况"
    )

def test_extracts_explicit_metadata_from_table_cells(
) -> None:
    rows = [
        [
            "目的：披露商业银行关键审慎监管指标。",
            "",
            "",
        ],
        [
            "适用范围：国内系统重要性银行。",
            "",
            "",
        ],
        [
            "内容：资本充足率、杠杆率和流动性风险指标。",
            "",
            "",
        ],
        [
            "频率：半年，并在发生重大变化时及时更新。",
            "",
            "",
        ],
        [
            "格式：固定。如有其他分类，可适当增加行。",
            "",
            "",
        ],
        [
            "指标",
            "本期",
            "上期",
        ],
    ]

    text = "\n".join(
        "\t".join(row).strip()
        for row in rows
    )

    block = CompetitionTextBlock(
        block_id="table:metadata",
        source_id="src",
        doc_id="doc",
        source_type="word",
        block_index=1,
        block_type="table",
        text=text,
        table_index=0,
        table_rows=rows,
    )

    context = CompetitionRegulatoryContext(
        section_path=("披露表格",),
        article=None,
        item_path=(
            "（一）表格KM1：监管并表关键审慎监管指标",
        ),
    )

    result = build_table_context(
        table_block=block,
        context=context,
    )

    assert result.purpose == (
        "目的：披露商业银行关键审慎监管指标。"
    )

    assert result.scope == (
        "适用范围：国内系统重要性银行。"
    )

    assert result.content == (
        "内容：资本充足率、杠杆率和流动性风险指标。"
    )

    assert result.frequency == "半年"
    assert result.format == "固定"

    assert result.unit is None

    # 不能把整张表复制进purpose。
    assert result.purpose != text
    assert len(result.purpose) < 100


def test_does_not_infer_metadata_from_unlabeled_values(
) -> None:
    rows = [
        [
            "序号",
            "披露内容",
            "表格",
            "类型",
            "频率",
        ],
        [
            "1",
            "风险管理和计量单位说明",
            "KM1",
            "固定",
            "季度",
        ],
        [
            "2",
            "风险管理定性信息",
            "OVA",
            "可变",
            "年度",
        ],
    ]

    text = "\n".join(
        "\t".join(row)
        for row in rows
    )

    block = CompetitionTextBlock(
        block_id="table:overview",
        source_id="src",
        doc_id="doc",
        source_type="word",
        block_index=1,
        block_type="table",
        text=text,
        table_index=0,
        table_rows=rows,
    )

    context = CompetitionRegulatoryContext(
        section_path=("总体要求",),
        article=None,
        item_path=(
            "六、国内系统重要性银行披露概览",
        ),
    )

    result = build_table_context(
        table_block=block,
        context=context,
    )

    # 表中虽然出现了“季度、年度、单位”，
    # 但没有明确的“标签：值”结构。
    assert result.frequency is None
    assert result.unit is None
    assert result.purpose is None
    assert result.content is None
    assert result.scope is None
    assert result.format == "table"


def test_extracts_metadata_from_nearby_labeled_paragraphs(
) -> None:
    rows = [
        [
            "指标",
            "数值",
        ],
        [
            "资本充足率",
            "10.5%",
        ],
    ]

    text = "\n".join(
        "\t".join(row)
        for row in rows
    )

    block = CompetitionTextBlock(
        block_id="table:nearby",
        source_id="src",
        doc_id="doc",
        source_type="word",
        block_index=1,
        block_type="table",
        text=text,
        table_index=0,
        table_rows=rows,
    )

    context = CompetitionRegulatoryContext(
        section_path=("资本管理",),
        article=None,
        item_path=(),
    )

    result = build_table_context(
        table_block=block,
        context=context,
        nearby_text=(
            "单位：百分比",
            "频率：季度。",
            "普通正文中包含年度和单位两个词。",
        ),
    )

    assert result.unit == "单位：百分比"
    assert result.frequency == "季度"
    assert result.purpose is None


def test_extracts_only_explicit_table_titles(
) -> None:
    assert (
        extract_explicit_table_title(
            "（一）表格KM1：监管并表关键审慎监管指标"
        )
        ==
        "（一）表格KM1：监管并表关键审慎监管指标"
    )

    assert (
        extract_explicit_table_title(
            "表CR3：资本充足率"
        )
        ==
        "表CR3：资本充足率"
    )

    assert (
        extract_explicit_table_title(
            "六、国内系统重要性银行披露概览"
        )
        is None
    )

    assert (
        extract_explicit_table_title(
            "本段正文提到了表格KM1：但不是标题"
        )
        is None
    )