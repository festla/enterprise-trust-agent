from __future__ import annotations

from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.schemas.competition_chunk import (
    CompetitionChunkType,
    CompetitionTextChunk,
)


CompetitionContextRelation = Literal[
    "seed",
    "previous_neighbor",
    "next_neighbor",
    "item_parent",
    "item_child",
]

CompetitionItemHierarchyTriggerMode = Literal[
    "seed_only",
    "seed_and_adjacent",
]


class CompetitionContextExpansionConfig(
    BaseModel
):
    """
    Competition Retrieval Context Expansion V1。

    支持：

    1. Retrieval Seed
    2. Previous / Next 邻接扩展
    3. Item Parent / Child 结构扩展

    hierarchy trigger mode:

        seed_only
            只有原始 Retrieval Seed
            可以触发 Item Hierarchy。

        seed_and_adjacent
            Seed 以及由该 Seed 的邻接窗口
            带入的 Chunk 都可以触发一次
            Item Hierarchy。

    注意：

    Item Hierarchy 新扩出的 Chunk
    不会继续触发下一轮结构扩展，
    因此不会形成递归闭包。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    expansion_method: Literal[
        "structured_context_expansion"
    ] = "structured_context_expansion"

    expansion_version: Literal[
        "competition_context_expansion_v1"
    ] = "competition_context_expansion_v1"

    seed_candidate_count: int = Field(
        default=10,
        ge=1,
    )

    previous_window: int = Field(
        default=1,
        ge=0,
    )

    next_window: int = Field(
        default=1,
        ge=0,
    )

    enable_item_hierarchy: bool = False

    enable_item_parent: bool = True

    enable_item_child: bool = True

    max_item_depth_delta: int = Field(
        default=1,
        ge=1,
    )

    item_hierarchy_trigger_mode: (
        CompetitionItemHierarchyTriggerMode
    ) = "seed_only"

    @model_validator(
        mode="after"
    )
    def validate_hierarchy_config(
        self,
    ) -> Self:
        if (
            self.enable_item_hierarchy
            and not self.enable_item_parent
            and not self.enable_item_child
        ):
            raise ValueError(
                "启用 Item Hierarchy 时，"
                "至少需要启用 parent 或 child"
            )

        return self


class CompetitionContextExpansionOrigin(
    BaseModel
):
    """
    一个 Expanded Chunk 为什么被加入 Context。

    seed_chunk_id:
        原始 Retrieval Seed。

    seed_rank:
        原始 Retrieval Seed Rank。

    trigger_chunk_id:
        实际触发当前扩展关系的 Chunk。

        对 seed / adjacent：
            trigger == seed。

        对传播式 hierarchy：
            trigger 可以是 Seed 的 ±1 Neighbor。

    relation:
        当前 Expanded Chunk 相对于
        trigger_chunk_id 的关系。

    distance:
        adjacent:
            chunk_index 距离。

        hierarchy:
            item_path 深度差。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    seed_chunk_id: str = Field(
        min_length=1,
    )

    seed_rank: int = Field(
        ge=1,
    )

    trigger_chunk_id: str = Field(
        min_length=1,
    )

    relation: CompetitionContextRelation

    distance: int = Field(
        ge=0,
    )

    @model_validator(
        mode="after"
    )
    def validate_origin(
        self,
    ) -> Self:
        if self.relation == "seed":
            if self.distance != 0:
                raise ValueError(
                    "seed relation 的 distance "
                    "必须等于 0"
                )

            if (
                self.trigger_chunk_id
                != self.seed_chunk_id
            ):
                raise ValueError(
                    "seed relation 的 trigger_chunk_id "
                    "必须等于 seed_chunk_id"
                )

        else:
            if self.distance < 1:
                raise ValueError(
                    "非 seed relation 的 distance "
                    "必须大于等于 1"
                )

        if self.relation in {
            "previous_neighbor",
            "next_neighbor",
        }:
            if (
                self.trigger_chunk_id
                != self.seed_chunk_id
            ):
                raise ValueError(
                    "邻接扩展的 trigger_chunk_id "
                    "必须等于 seed_chunk_id"
                )

        return self


class CompetitionContextExpandedHit(
    BaseModel
):
    """
    Context Expansion 后的唯一 Chunk。

    effective_seed_rank：

        最早能够把该 Chunk 带入 Context
        的 Retrieval Seed Rank。

    即使 Chunk 是：

        Seed
            ↓
        Previous Neighbor
            ↓
        Item Child

    其 effective_seed_rank 仍然继承
    原始 Retrieval Seed Rank。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    rank: int = Field(
        ge=1,
    )

    effective_seed_rank: int = Field(
        ge=1,
    )

    retriever_type: Literal[
        "bm25_context_expansion"
    ] = "bm25_context_expansion"

    origins: tuple[
        CompetitionContextExpansionOrigin,
        ...,
    ] = Field(
        min_length=1,
    )

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

    @property
    def is_seed(self) -> bool:
        return any(
            origin.relation == "seed"
            for origin in self.origins
        )

    @model_validator(
        mode="after"
    )
    def validate_expanded_hit(
        self,
    ) -> Self:
        expected_effective_rank = min(
            origin.seed_rank
            for origin in self.origins
        )

        if (
            self.effective_seed_rank
            != expected_effective_rank
        ):
            raise ValueError(
                "effective_seed_rank 必须等于"
                "最小 origin seed_rank"
            )

        origin_keys = tuple(
            (
                origin.seed_chunk_id,
                origin.seed_rank,
                origin.trigger_chunk_id,
                origin.relation,
                origin.distance,
            )
            for origin in self.origins
        )

        if (
            len(origin_keys)
            != len(set(origin_keys))
        ):
            raise ValueError(
                "origins 不能包含重复来源"
            )

        for origin in self.origins:
            if (
                origin.relation == "seed"
                and origin.seed_chunk_id
                != self.chunk_id
            ):
                raise ValueError(
                    "seed origin 的 seed_chunk_id "
                    "必须等于当前 chunk_id"
                )

        return self