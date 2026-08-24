from __future__ import annotations

import hashlib

from app.schemas.competition_chunk import (
    CompetitionChunkSourceSpan,
    CompetitionTextChunk,
)
from app.schemas.competition_context_expansion import (
    CompetitionContextExpansionConfig,
)
from app.schemas.competition_gold import (
    CompetitionGoldChunkReference,
    CompetitionGoldFactRecord,
)
from app.schemas.competition_option_anchored_merge import (
    CompetitionBM25OptionAnchoredMergeHit,
)
from app.services.competition_context_expansion import (
    evaluate_competition_gold_fact_context_expanded,
    expand_competition_retrieval_context,
)


def _chunk(
    *,
    source_number: int,
    chunk_index: int,
    text: str,
    item_path: tuple[
        str,
        ...,
    ] = (),
) -> CompetitionTextChunk:
    source_id = (
        f"src_{source_number:016x}"
    )

    doc_id = (
        f"doc_{source_id}_"
        "0123456789abcdef01234567"
    )

    return CompetitionTextChunk(
        chunk_id=(
            f"chunk:{source_id}:"
            f"{chunk_index:05d}"
        ),
        source_id=source_id,
        doc_id=doc_id,
        source_type="word",
        chunk_index=chunk_index,
        chunk_type="text",
        source_spans=(
            CompetitionChunkSourceSpan(
                block_id=(
                    f"block:{doc_id}:"
                    f"{chunk_index:05d}"
                ),
                block_index=(
                    chunk_index
                ),
                start_char=0,
                end_char=len(
                    text
                ),
            ),
        ),
        text=text,
        char_count=len(
            text
        ),
        text_sha256=(
            hashlib.sha256(
                text.encode(
                    "utf-8"
                )
            ).hexdigest()
        ),
        item_path=item_path,
        paragraph_start_index=(
            chunk_index
        ),
        paragraph_end_index=(
            chunk_index
        ),
    )


def _seed(
    *,
    chunk: CompetitionTextChunk,
    rank: int = 1,
) -> CompetitionBM25OptionAnchoredMergeHit:
    return (
        CompetitionBM25OptionAnchoredMergeHit(
            rank=rank,
            score=(
                2.0
                + 1.0 / rank
            ),
            layer="option_anchor",
            question_only_rank=None,
            question_with_options_rank=(
                rank
            ),
            chunk=chunk,
        )
    )


def test_seed_only_does_not_propagate_from_neighbor(
) -> None:
    seed_chunk = _chunk(
        source_number=1,
        chunk_index=0,
        text="retrieval seed",
        item_path=(),
    )

    parent = _chunk(
        source_number=1,
        chunk_index=1,
        text="parent context",
        item_path=(
            "一、披露要求",
        ),
    )

    child = _chunk(
        source_number=1,
        chunk_index=8,
        text="direct child",
        item_path=(
            "一、披露要求",
            "（一）具体要求",
        ),
    )

    result = (
        expand_competition_retrieval_context(
            seed_hits=(
                _seed(
                    chunk=seed_chunk,
                ),
            ),
            corpus_chunks=(
                seed_chunk,
                parent,
                child,
            ),
            config=(
                CompetitionContextExpansionConfig(
                    previous_window=0,
                    next_window=1,
                    enable_item_hierarchy=True,
                    max_item_depth_delta=1,
                    item_hierarchy_trigger_mode=(
                        "seed_only"
                    ),
                )
            ),
        )
    )

    result_ids = {
        hit.chunk_id
        for hit in result
    }

    assert (
        parent.chunk_id
        in result_ids
    )

    assert (
        child.chunk_id
        not in result_ids
    )


def test_adjacent_neighbor_can_trigger_item_child(
) -> None:
    seed_chunk = _chunk(
        source_number=2,
        chunk_index=0,
        text="retrieval seed",
        item_path=(),
    )

    parent = _chunk(
        source_number=2,
        chunk_index=1,
        text="parent context",
        item_path=(
            "一、披露要求",
        ),
    )

    child = _chunk(
        source_number=2,
        chunk_index=8,
        text="direct child",
        item_path=(
            "一、披露要求",
            "（一）具体要求",
        ),
    )

    result = (
        expand_competition_retrieval_context(
            seed_hits=(
                _seed(
                    chunk=seed_chunk,
                ),
            ),
            corpus_chunks=(
                seed_chunk,
                parent,
                child,
            ),
            config=(
                CompetitionContextExpansionConfig(
                    previous_window=0,
                    next_window=1,
                    enable_item_hierarchy=True,
                    max_item_depth_delta=1,
                    item_hierarchy_trigger_mode=(
                        "seed_and_adjacent"
                    ),
                )
            ),
        )
    )

    by_id = {
        hit.chunk_id: hit
        for hit in result
    }

    assert (
        child.chunk_id
        in by_id
    )

    child_hit = (
        by_id[
            child.chunk_id
        ]
    )

    assert (
        child_hit.effective_seed_rank
        == 1
    )

    hierarchy_origins = tuple(
        origin
        for origin
        in child_hit.origins
        if (
            origin.relation
            == "item_child"
        )
    )

    assert len(
        hierarchy_origins
    ) == 1

    origin = (
        hierarchy_origins[0]
    )

    assert (
        origin.seed_chunk_id
        == seed_chunk.chunk_id
    )

    assert (
        origin.trigger_chunk_id
        == parent.chunk_id
    )


def test_propagation_can_complete_parent_child_gold(
) -> None:
    seed_chunk = _chunk(
        source_number=3,
        chunk_index=0,
        text="retrieval seed",
    )

    parent = _chunk(
        source_number=3,
        chunk_index=1,
        text="parent",
        item_path=(
            "一、风险管理",
        ),
    )

    direct = _chunk(
        source_number=3,
        chunk_index=9,
        text="direct",
        item_path=(
            "一、风险管理",
            "（一）信用风险",
        ),
    )

    expanded = (
        expand_competition_retrieval_context(
            seed_hits=(
                _seed(
                    chunk=seed_chunk,
                ),
            ),
            corpus_chunks=(
                seed_chunk,
                parent,
                direct,
            ),
            config=(
                CompetitionContextExpansionConfig(
                    previous_window=0,
                    next_window=1,
                    enable_item_hierarchy=True,
                    max_item_depth_delta=1,
                    item_hierarchy_trigger_mode=(
                        "seed_and_adjacent"
                    ),
                )
            ),
        )
    )

    record = (
        CompetitionGoldFactRecord(
            case_id="Q001",
            fact_index=0,
            fact="测试事实",
            expected_source_id=(
                seed_chunk.source_id
            ),
            review_status=(
                "confirmed"
            ),
            evidence_mode=(
                "parent_child"
            ),
            gold_chunks=(
                CompetitionGoldChunkReference(
                    chunk_id=(
                        direct.chunk_id
                    ),
                    role="direct",
                ),
                CompetitionGoldChunkReference(
                    chunk_id=(
                        parent.chunk_id
                    ),
                    role=(
                        "parent_context"
                    ),
                ),
            ),
        )
    )

    result = (
        evaluate_competition_gold_fact_context_expanded(
            record=record,
            hits=expanded,
        )
    )

    assert (
        result.direct_hit_at(1)
    )

    assert (
        result.complete_gold_hit_at(1)
    )

    assert (
        result.gold_chunk_recall_at(1)
        == 1.0
    )


def test_hierarchy_result_does_not_recursively_trigger(
) -> None:
    seed_chunk = _chunk(
        source_number=4,
        chunk_index=0,
        text="seed",
    )

    level_one = _chunk(
        source_number=4,
        chunk_index=1,
        text="level one",
        item_path=(
            "一、要求",
        ),
    )

    level_two = _chunk(
        source_number=4,
        chunk_index=5,
        text="level two",
        item_path=(
            "一、要求",
            "（一）要求",
        ),
    )

    level_three = _chunk(
        source_number=4,
        chunk_index=9,
        text="level three",
        item_path=(
            "一、要求",
            "（一）要求",
            "1.要求",
        ),
    )

    result = (
        expand_competition_retrieval_context(
            seed_hits=(
                _seed(
                    chunk=seed_chunk,
                ),
            ),
            corpus_chunks=(
                seed_chunk,
                level_one,
                level_two,
                level_three,
            ),
            config=(
                CompetitionContextExpansionConfig(
                    previous_window=0,
                    next_window=1,
                    enable_item_hierarchy=True,
                    max_item_depth_delta=1,
                    item_hierarchy_trigger_mode=(
                        "seed_and_adjacent"
                    ),
                )
            ),
        )
    )

    ids = {
        hit.chunk_id
        for hit in result
    }

    #
    # level_one 是 seed 的邻居，
    # 所以能触发一次 hierarchy，
    # 得到 level_two。
    #
    assert (
        level_two.chunk_id
        in ids
    )

    #
    # level_two 是 hierarchy 扩展结果，
    # 不允许再递归触发 level_three。
    #
    assert (
        level_three.chunk_id
        not in ids
    )