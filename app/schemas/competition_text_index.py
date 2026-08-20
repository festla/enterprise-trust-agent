from __future__ import annotations

from pydantic import BaseModel, Field


class CompetitionTextIndexItem(BaseModel):
    """
    单个可检索文本单元。

    从 CompetitionTextChunk 转换而来。
    """

    chunk_id: str

    source_id: str

    doc_id: str

    text: str

    chunk_type: str

    metadata: dict[str, object] = Field(
        default_factory=dict
    )


class CompetitionTextIndex(BaseModel):
    """
    文档检索索引。

    Step4.1:
        Chunk -> Index
    """

    items: list[CompetitionTextIndexItem]

    def __len__(self) -> int:
        return len(self.items)

    def get(
        self,
        chunk_id: str,
    ) -> CompetitionTextIndexItem | None:

        for item in self.items:
            if item.chunk_id == chunk_id:
                return item

        return None