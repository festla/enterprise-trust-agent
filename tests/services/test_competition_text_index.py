from app.services.competition_text_index import (
    build_competition_text_index,
)

from app.schemas.competition_chunk import (
    CompetitionTextChunk,
)


def build_chunk(
    chunk_id="c1",
):

    return CompetitionTextChunk(

        chunk_id=chunk_id,

        source_id="source1",
        doc_id="doc1",

        source_type="word",

        chunk_index=0,

        chunk_type="text",

        source_spans=(
            {
                "block_id": "block1",
                "block_index": 0,
                "start_char": 0,
                "end_char": 8,
            },
        ),

        text="保险责任相关内容",

        char_count=8,

        text_sha256=(
            "b413303e761be581432e972b80a63c36a8b9c0bdd8e8618639cf6e0bdb887bf9"
        ),

        section_path=(
            "第一章",
        ),

        article="",

        item_path=(),

        page_start=None,
        page_end=None,

        table_index=None,
    )


def test_build_index():

    chunks = [
        build_chunk("c1"),
        build_chunk("c2"),
    ]


    index = build_competition_text_index(
        chunks
    )


    assert len(index) == 2

    assert index.items[0].chunk_id == "c1"



def test_index_metadata_keep():

    index = build_competition_text_index(
        [
            build_chunk()
        ]
    )


    item = index.items[0]


    assert (
        item.metadata["section_path"]
        ==
        ("第一章",)
    )


def test_empty_chunks():

    index = build_competition_text_index(
        []
    )


    assert len(index) == 0