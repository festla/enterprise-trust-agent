from app.schemas.competition_text_index import (
    CompetitionTextIndex,
    CompetitionTextIndexItem,
)


def test_index_item_creation():

    item = CompetitionTextIndexItem(
        chunk_id="chunk_001",
        source_id="source_001",
        doc_id="doc_001",
        text="test text",
        chunk_type="paragraph",
    )

    assert item.chunk_id == "chunk_001"
    assert item.text == "test text"



def test_index_get():

    index = CompetitionTextIndex(
        items=[
            CompetitionTextIndexItem(
                chunk_id="c1",
                source_id="s1",
                doc_id="d1",
                text="hello",
                chunk_type="paragraph",
            )
        ]
    )


    result = index.get(
        "c1"
    )

    assert result is not None
    assert result.text == "hello"


def test_index_length():

    index = CompetitionTextIndex(
        items=[]
    )

    assert len(index) == 0