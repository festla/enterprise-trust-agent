from __future__ import annotations

from collections.abc import Iterable

from app.schemas.competition_chunk import (
    CompetitionTextChunk,
)

from app.schemas.competition_text_index import (
    CompetitionTextIndex,
    CompetitionTextIndexItem,
)


def build_competition_text_index(
    chunks: Iterable[CompetitionTextChunk],
) -> CompetitionTextIndex:
    """
    CompetitionTextChunk
            |
            v
    CompetitionTextIndex
    """

    items: list[CompetitionTextIndexItem] = []


    for chunk in chunks:

        metadata = {
            "section_path": chunk.section_path,
            "article": chunk.article,
            "item_path": chunk.item_path,
            "source_type": chunk.source_type,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "table_index": chunk.table_index,
        }


        items.append(
            CompetitionTextIndexItem(
                chunk_id=chunk.chunk_id,
                source_id=chunk.source_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                chunk_type=chunk.chunk_type,
                metadata=metadata,
            )
        )


    return CompetitionTextIndex(
        items=items
    )