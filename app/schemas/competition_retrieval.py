from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from app.schemas.competition_chunk import (
    CompetitionChunkType,
    CompetitionTextChunk,
)
from app.schemas.competition_text import (
    CompetitionTextSourceType,
)


class CompetitionRetrievalFilter(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    source_ids: tuple[str, ...] = ()
    doc_ids: tuple[str, ...] = ()

    source_types: tuple[
        CompetitionTextSourceType,
        ...,
    ] = ()

    chunk_types: tuple[
        CompetitionChunkType,
        ...,
    ] = ()

    @field_validator(
        "source_ids",
        "doc_ids",
    )
    @classmethod
    def validate_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(
            not item.strip()
            for item in value
        ):
            raise ValueError(
                "检索过滤 ID 不能为空"
            )

        if (
            len(value)
            != len(set(value))
        ):
            raise ValueError(
                "检索过滤 ID 不能重复"
            )

        return tuple(
            sorted(value)
        )

    @field_validator(
        "source_types",
        "chunk_types",
    )
    @classmethod
    def validate_categories(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if (
            len(value)
            != len(set(value))
        ):
            raise ValueError(
                "检索过滤类型不能重复"
            )

        return tuple(
            sorted(value)
        )


class CompetitionBM25Hit(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    schema_version: Literal[1] = 1

    rank: int = Field(
        ge=1,
    )

    score: float = Field(
        gt=0,
    )

    retriever_type: Literal[
        "bm25"
    ] = "bm25"

    score_type: Literal[
        "bm25"
    ] = "bm25"

    chunk: CompetitionTextChunk

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def source_id(self) -> str:
        return self.chunk.source_id

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

    @property
    def chunk_type(
        self,
    ) -> CompetitionChunkType:
        return self.chunk.chunk_type