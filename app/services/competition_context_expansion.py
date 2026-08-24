from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.schemas.competition_chunk import (
    CompetitionTextChunk,
)
from app.schemas.competition_context_expansion import (
    CompetitionContextExpandedHit,
    CompetitionContextExpansionConfig,
    CompetitionContextExpansionOrigin,
    CompetitionContextRelation,
)
from app.schemas.competition_gold import (
    CompetitionGoldFactRecord,
)
from app.schemas.competition_option_anchored_merge import (
    CompetitionBM25OptionAnchoredMergeHit,
)
from app.services.competition_gold_retrieval_eval import (
    CompetitionGoldChunkMatch,
    CompetitionGoldFactRetrievalResult,
)


class CompetitionContextExpansionError(
    ValueError
):
    """Context Expansion 基础异常。"""


class EmptyCompetitionContextSeedError(
    CompetitionContextExpansionError
):
    """没有 Retrieval Seed。"""


class InvalidCompetitionContextSeedError(
    CompetitionContextExpansionError
):
    """Seed 排名或身份无效。"""


class DuplicateCompetitionContextCorpusChunkError(
    CompetitionContextExpansionError
):
    """Corpus Chunk 身份冲突。"""


class CompetitionContextSeedCorpusMismatchError(
    CompetitionContextExpansionError
):
    """Seed Chunk 与 Corpus 中 Chunk 不一致。"""


@dataclass(slots=True)
class _ExpandedRecord:
    chunk: CompetitionTextChunk

    origins: list[
        CompetitionContextExpansionOrigin
    ]


@dataclass(frozen=True, slots=True)
class _HierarchyTrigger:
    """
    一个 Item Hierarchy Trigger。

    seed_chunk_id / seed_rank：
        原始 Retrieval Seed。

    trigger_chunk：
        真正用于匹配 item_path 的 Chunk。
    """

    seed_chunk_id: str
    seed_rank: int
    trigger_chunk: CompetitionTextChunk


def _chunk_identity(
    chunk: CompetitionTextChunk,
) -> dict[str, object]:
    return chunk.model_dump(
        mode="json"
    )


def _validate_seed_hits(
    seed_hits: Sequence[
        CompetitionBM25OptionAnchoredMergeHit
    ],
) -> None:
    if not seed_hits:
        raise EmptyCompetitionContextSeedError(
            "Context Expansion 至少需要"
            "一个 Retrieval Seed"
        )

    ranks = tuple(
        hit.rank
        for hit in seed_hits
    )

    expected_ranks = tuple(
        range(
            1,
            len(seed_hits) + 1,
        )
    )

    if ranks != expected_ranks:
        raise InvalidCompetitionContextSeedError(
            "Retrieval Seed rank 必须"
            "从 1 开始连续递增"
        )

    chunk_ids = tuple(
        hit.chunk_id
        for hit in seed_hits
    )

    if (
        len(chunk_ids)
        != len(set(chunk_ids))
    ):
        raise InvalidCompetitionContextSeedError(
            "Retrieval Seed 包含重复 chunk_id"
        )


def _build_corpus_indexes(
    chunks: Sequence[
        CompetitionTextChunk
    ],
) -> tuple[
    dict[
        str,
        CompetitionTextChunk,
    ],
    dict[
        tuple[str, int],
        CompetitionTextChunk,
    ],
    dict[
        str,
        tuple[
            CompetitionTextChunk,
            ...,
        ],
    ],
]:
    chunks_by_id: dict[
        str,
        CompetitionTextChunk,
    ] = {}

    chunks_by_source_index: dict[
        tuple[str, int],
        CompetitionTextChunk,
    ] = {}

    chunks_by_source_lists: dict[
        str,
        list[
            CompetitionTextChunk
        ],
    ] = {}

    for chunk in chunks:
        if chunk.chunk_id in chunks_by_id:
            raise (
                DuplicateCompetitionContextCorpusChunkError(
                    "Corpus 包含重复 chunk_id："
                    f"{chunk.chunk_id}"
                )
            )

        source_index_key = (
            chunk.source_id,
            chunk.chunk_index,
        )

        if (
            source_index_key
            in chunks_by_source_index
        ):
            raise (
                DuplicateCompetitionContextCorpusChunkError(
                    "Corpus 包含重复的 "
                    "source_id/chunk_index："
                    f"{source_index_key}"
                )
            )

        chunks_by_id[
            chunk.chunk_id
        ] = chunk

        chunks_by_source_index[
            source_index_key
        ] = chunk

        chunks_by_source_lists.setdefault(
            chunk.source_id,
            [],
        ).append(
            chunk
        )

    chunks_by_source = {
        source_id: tuple(
            sorted(
                source_chunks,
                key=lambda chunk: (
                    chunk.chunk_index,
                    chunk.chunk_id,
                ),
            )
        )
        for (
            source_id,
            source_chunks,
        ) in chunks_by_source_lists.items()
    }

    return (
        chunks_by_id,
        chunks_by_source_index,
        chunks_by_source,
    )


def _relation_priority(
    relation: CompetitionContextRelation,
) -> int:
    priorities = {
        "seed": 0,
        "previous_neighbor": 1,
        "next_neighbor": 2,
        "item_parent": 3,
        "item_child": 4,
    }

    return priorities[
        relation
    ]


def _append_origin(
    records: dict[
        str,
        _ExpandedRecord,
    ],
    *,
    chunk: CompetitionTextChunk,
    origin: CompetitionContextExpansionOrigin,
) -> None:
    record = records.get(
        chunk.chunk_id
    )

    if record is None:
        records[
            chunk.chunk_id
        ] = _ExpandedRecord(
            chunk=chunk,
            origins=[origin],
        )

        return

    if (
        _chunk_identity(record.chunk)
        != _chunk_identity(chunk)
    ):
        raise (
            DuplicateCompetitionContextCorpusChunkError(
                "相同 chunk_id 对应不同 Chunk 内容："
                f"{chunk.chunk_id}"
            )
        )

    origin_key = (
        origin.seed_chunk_id,
        origin.seed_rank,
        origin.trigger_chunk_id,
        origin.relation,
        origin.distance,
    )

    existing_keys = {
        (
            item.seed_chunk_id,
            item.seed_rank,
            item.trigger_chunk_id,
            item.relation,
            item.distance,
        )
        for item in record.origins
    }

    if (
        origin_key
        not in existing_keys
    ):
        record.origins.append(
            origin
        )


def _is_strict_item_prefix(
    prefix: tuple[str, ...],
    value: tuple[str, ...],
) -> bool:
    if not prefix:
        return False

    if (
        len(prefix)
        >= len(value)
    ):
        return False

    return (
        value[:len(prefix)]
        == prefix
    )


def _item_hierarchy_relation(
    *,
    trigger: CompetitionTextChunk,
    candidate: CompetitionTextChunk,
    max_depth_delta: int,
) -> tuple[
    CompetitionContextRelation,
    int,
] | None:
    """
    Candidate 相对于 Trigger 的结构关系。

    Candidate Path 是 Trigger Path 前缀：
        Candidate = item_parent

    Trigger Path 是 Candidate Path 前缀：
        Candidate = item_child
    """

    if (
        trigger.source_id
        != candidate.source_id
    ):
        return None

    if (
        trigger.chunk_id
        == candidate.chunk_id
    ):
        return None

    trigger_path = (
        trigger.item_path
    )

    candidate_path = (
        candidate.item_path
    )

    if (
        not trigger_path
        or not candidate_path
    ):
        return None

    if _is_strict_item_prefix(
        candidate_path,
        trigger_path,
    ):
        depth_delta = (
            len(trigger_path)
            - len(candidate_path)
        )

        if (
            depth_delta
            <= max_depth_delta
        ):
            return (
                "item_parent",
                depth_delta,
            )

        return None

    if _is_strict_item_prefix(
        trigger_path,
        candidate_path,
    ):
        depth_delta = (
            len(candidate_path)
            - len(trigger_path)
        )

        if (
            depth_delta
            <= max_depth_delta
        ):
            return (
                "item_child",
                depth_delta,
            )

        return None

    return None


def _append_adjacent_context(
    *,
    seed: CompetitionBM25OptionAnchoredMergeHit,
    records: dict[
        str,
        _ExpandedRecord,
    ],
    chunks_by_source_index: dict[
        tuple[str, int],
        CompetitionTextChunk,
    ],
    config: CompetitionContextExpansionConfig,
) -> tuple[
    CompetitionTextChunk,
    ...,
]:
    """
    添加邻接 Context。

    返回实际找到的邻接 Chunk，
    用于 seed_and_adjacent hierarchy trigger。
    """

    neighbors: list[
        CompetitionTextChunk
    ] = []

    for distance in range(
        1,
        config.previous_window + 1,
    ):
        neighbor = (
            chunks_by_source_index.get(
                (
                    seed.source_id,
                    (
                        seed.chunk.chunk_index
                        - distance
                    ),
                )
            )
        )

        if neighbor is None:
            continue

        _append_origin(
            records,
            chunk=neighbor,
            origin=(
                CompetitionContextExpansionOrigin(
                    seed_chunk_id=(
                        seed.chunk_id
                    ),
                    seed_rank=(
                        seed.rank
                    ),
                    trigger_chunk_id=(
                        seed.chunk_id
                    ),
                    relation=(
                        "previous_neighbor"
                    ),
                    distance=distance,
                )
            ),
        )

        neighbors.append(
            neighbor
        )

    for distance in range(
        1,
        config.next_window + 1,
    ):
        neighbor = (
            chunks_by_source_index.get(
                (
                    seed.source_id,
                    (
                        seed.chunk.chunk_index
                        + distance
                    ),
                )
            )
        )

        if neighbor is None:
            continue

        _append_origin(
            records,
            chunk=neighbor,
            origin=(
                CompetitionContextExpansionOrigin(
                    seed_chunk_id=(
                        seed.chunk_id
                    ),
                    seed_rank=(
                        seed.rank
                    ),
                    trigger_chunk_id=(
                        seed.chunk_id
                    ),
                    relation=(
                        "next_neighbor"
                    ),
                    distance=distance,
                )
            ),
        )

        neighbors.append(
            neighbor
        )

    return tuple(
        neighbors
    )


def _build_hierarchy_triggers(
    *,
    seed: CompetitionBM25OptionAnchoredMergeHit,
    adjacent_chunks: Sequence[
        CompetitionTextChunk
    ],
    config: CompetitionContextExpansionConfig,
) -> tuple[
    _HierarchyTrigger,
    ...,
]:
    if (
        not config.enable_item_hierarchy
    ):
        return ()

    trigger_chunks: list[
        CompetitionTextChunk
    ] = [
        seed.chunk
    ]

    if (
        config.item_hierarchy_trigger_mode
        == "seed_and_adjacent"
    ):
        trigger_chunks.extend(
            adjacent_chunks
        )

    unique_chunks: dict[
        str,
        CompetitionTextChunk,
    ] = {}

    for chunk in trigger_chunks:
        unique_chunks.setdefault(
            chunk.chunk_id,
            chunk,
        )

    return tuple(
        _HierarchyTrigger(
            seed_chunk_id=(
                seed.chunk_id
            ),
            seed_rank=seed.rank,
            trigger_chunk=chunk,
        )
        for chunk in (
            unique_chunks.values()
        )
    )


def _expand_item_hierarchy(
    *,
    hierarchy_trigger: _HierarchyTrigger,
    records: dict[
        str,
        _ExpandedRecord,
    ],
    source_chunks: Sequence[
        CompetitionTextChunk
    ],
    config: CompetitionContextExpansionConfig,
) -> None:
    """
    从一个 Trigger 做一次 Item hierarchy expansion。

    注意：

    这里只扫描 Trigger -> Candidate。

    Candidate 不会重新成为 Trigger，
    因此不会递归。
    """

    trigger_chunk = (
        hierarchy_trigger
        .trigger_chunk
    )

    if (
        not trigger_chunk.item_path
    ):
        return

    for candidate in source_chunks:
        relation = (
            _item_hierarchy_relation(
                trigger=trigger_chunk,
                candidate=candidate,
                max_depth_delta=(
                    config
                    .max_item_depth_delta
                ),
            )
        )

        if relation is None:
            continue

        (
            relation_name,
            depth_delta,
        ) = relation

        if (
            relation_name == "item_parent"
            and not config.enable_item_parent
        ):
            continue

        if (
            relation_name == "item_child"
            and not config.enable_item_child
        ):
            continue

        _append_origin(
            records,
            chunk=candidate,
            origin=(
                CompetitionContextExpansionOrigin(
                    seed_chunk_id=(
                        hierarchy_trigger
                        .seed_chunk_id
                    ),
                    seed_rank=(
                        hierarchy_trigger
                        .seed_rank
                    ),
                    trigger_chunk_id=(
                        trigger_chunk
                        .chunk_id
                    ),
                    relation=(
                        relation_name
                    ),
                    distance=(
                        depth_delta
                    ),
                )
            ),
        )


def expand_competition_retrieval_context(
    *,
    seed_hits: Sequence[
        CompetitionBM25OptionAnchoredMergeHit
    ],
    corpus_chunks: Sequence[
        CompetitionTextChunk
    ],
    config: (
        CompetitionContextExpansionConfig
        | None
    ) = None,
) -> tuple[
    CompetitionContextExpandedHit,
    ...,
]:
    """
    Competition Structured Context Expansion。

    每个 Retrieval Seed：

        Seed
          ↓
        ±N Adjacent
          ↓
        Optional One-hop Item Hierarchy

    hierarchy trigger：

        seed_only
            仅 Seed。

        seed_and_adjacent
            Seed + Adjacent。

    Item 扩展得到的新 Chunk
    不会继续递归触发 hierarchy。
    """

    _validate_seed_hits(
        seed_hits
    )

    active_config = (
        config
        if config is not None
        else CompetitionContextExpansionConfig()
    )

    (
        chunks_by_id,
        chunks_by_source_index,
        chunks_by_source,
    ) = _build_corpus_indexes(
        corpus_chunks
    )

    active_seeds = tuple(
        seed_hits[
            :active_config.seed_candidate_count
        ]
    )

    records: dict[
        str,
        _ExpandedRecord,
    ] = {}

    for seed in active_seeds:
        corpus_seed = (
            chunks_by_id.get(
                seed.chunk_id
            )
        )

        if corpus_seed is None:
            raise (
                CompetitionContextSeedCorpusMismatchError(
                    "Retrieval Seed 不存在于 Corpus："
                    f"{seed.chunk_id}"
                )
            )

        if (
            _chunk_identity(
                corpus_seed
            )
            != _chunk_identity(
                seed.chunk
            )
        ):
            raise (
                CompetitionContextSeedCorpusMismatchError(
                    "Retrieval Seed 与 Corpus "
                    "Chunk 内容不一致："
                    f"{seed.chunk_id}"
                )
            )

        #
        # 1. Seed
        #
        _append_origin(
            records,
            chunk=corpus_seed,
            origin=(
                CompetitionContextExpansionOrigin(
                    seed_chunk_id=(
                        seed.chunk_id
                    ),
                    seed_rank=(
                        seed.rank
                    ),
                    trigger_chunk_id=(
                        seed.chunk_id
                    ),
                    relation="seed",
                    distance=0,
                )
            ),
        )

        #
        # 2. Adjacent
        #
        adjacent_chunks = (
            _append_adjacent_context(
                seed=seed,
                records=records,
                chunks_by_source_index=(
                    chunks_by_source_index
                ),
                config=(
                    active_config
                ),
            )
        )

        #
        # 3. Build bounded hierarchy triggers
        #
        hierarchy_triggers = (
            _build_hierarchy_triggers(
                seed=seed,
                adjacent_chunks=(
                    adjacent_chunks
                ),
                config=(
                    active_config
                ),
            )
        )

        #
        # 4. One-hop hierarchy
        #
        source_chunks = (
            chunks_by_source.get(
                seed.source_id,
                (),
            )
        )

        for hierarchy_trigger in (
            hierarchy_triggers
        ):
            _expand_item_hierarchy(
                hierarchy_trigger=(
                    hierarchy_trigger
                ),
                records=records,
                source_chunks=(
                    source_chunks
                ),
                config=(
                    active_config
                ),
            )

    prepared: list[
        tuple[
            CompetitionTextChunk,
            tuple[
                CompetitionContextExpansionOrigin,
                ...,
            ],
            int,
        ]
    ] = []

    for record in records.values():
        origins = tuple(
            sorted(
                record.origins,
                key=lambda origin: (
                    origin.seed_rank,
                    _relation_priority(
                        origin.relation
                    ),
                    origin.distance,
                    origin.trigger_chunk_id,
                    origin.seed_chunk_id,
                ),
            )
        )

        effective_seed_rank = min(
            origin.seed_rank
            for origin in origins
        )

        prepared.append(
            (
                record.chunk,
                origins,
                effective_seed_rank,
            )
        )

    def sort_key(
        item: tuple[
            CompetitionTextChunk,
            tuple[
                CompetitionContextExpansionOrigin,
                ...,
            ],
            int,
        ],
    ) -> tuple[
        int,
        int,
        int,
        str,
        int,
        str,
    ]:
        (
            chunk,
            origins,
            effective_seed_rank,
        ) = item

        best_origin = min(
            origins,
            key=lambda origin: (
                origin.seed_rank,
                _relation_priority(
                    origin.relation
                ),
                origin.distance,
                origin.trigger_chunk_id,
                origin.seed_chunk_id,
            ),
        )

        return (
            effective_seed_rank,
            _relation_priority(
                best_origin.relation
            ),
            best_origin.distance,
            chunk.source_id,
            chunk.chunk_index,
            chunk.chunk_id,
        )

    ordered = tuple(
        sorted(
            prepared,
            key=sort_key,
        )
    )

    return tuple(
        CompetitionContextExpandedHit(
            rank=rank,
            effective_seed_rank=(
                effective_seed_rank
            ),
            origins=origins,
            chunk=chunk,
        )
        for (
            rank,
            (
                chunk,
                origins,
                effective_seed_rank,
            ),
        ) in enumerate(
            ordered,
            start=1,
        )
    )


def evaluate_competition_gold_fact_context_expanded(
    *,
    record: CompetitionGoldFactRecord,
    hits: Sequence[
        CompetitionContextExpandedHit
    ],
) -> CompetitionGoldFactRetrievalResult:
    if (
        record.review_status
        != "confirmed"
    ):
        raise ValueError(
            "Gold 检索评测只接受 confirmed 记录"
        )

    if (
        record.evidence_mode
        is None
    ):
        raise ValueError(
            "confirmed Gold 记录缺少 evidence_mode"
        )

    chunk_ids = tuple(
        hit.chunk_id
        for hit in hits
    )

    if (
        len(chunk_ids)
        != len(set(chunk_ids))
    ):
        raise ValueError(
            "Context Expansion 包含重复 chunk_id"
        )

    effective_rank_by_chunk_id = {
        hit.chunk_id: (
            hit.effective_seed_rank
        )
        for hit in hits
    }

    chunk_matches = tuple(
        CompetitionGoldChunkMatch(
            chunk_id=(
                reference.chunk_id
            ),
            role=reference.role,
            rank=(
                effective_rank_by_chunk_id.get(
                    reference.chunk_id
                )
            ),
        )
        for reference
        in record.gold_chunks
    )

    return (
        CompetitionGoldFactRetrievalResult(
            case_id=(
                record.case_id
            ),
            fact_index=(
                record.fact_index
            ),
            fact=record.fact,
            expected_source_id=(
                record.expected_source_id
            ),
            evidence_mode=(
                record.evidence_mode
            ),
            retrieved_count=(
                len(hits)
            ),
            chunk_matches=(
                chunk_matches
            ),
        )
    )