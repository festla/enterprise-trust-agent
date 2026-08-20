from app.schemas.competition_text import (
    CompetitionTextBlock,
)

from app.services.competition_table_context import (
    build_table_context,
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