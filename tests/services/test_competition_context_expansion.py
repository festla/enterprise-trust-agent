from __future__ import annotations

import hashlib

import pytest

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
    CompetitionContextSeedCorpusMismatchError,
    EmptyCompetitionContextSeedError,
    InvalidCompetitionContextSeedError,
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

    text_sha256 = hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()

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
            text_sha256
        ),
        item_path=item_path,
        paragraph_start_index=(
            chunk_index
        ),
        paragraph_end_index=(
            chunk_index
        ),
    )


def _seed_hit(
    *,
    chunk: CompetitionTextChunk,
    rank: int,
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
            question_with_options_rank=rank,
            chunk=chunk,
        )
    )


def test_expands_previous_and_next_neighbor(
) -> None:
    first = _chunk(
        source_number=1,
        chunk_index=0,
        text="parent",
    )

    middle = _chunk(
        source_number=1,
        chunk_index=1,
        text="direct",
    )

    last = _chunk(
        source_number=1,
        chunk_index=2,
        text="supporting",
    )

    result = (
        expand_competition_retrieval_context(
            seed_hits=(
                _seed_hit(
                    chunk=middle,
                    rank=1,
                ),
            ),
            corpus_chunks=(
                first,
                middle,
                last,
            ),
        )
    )

    assert {
        hit.chunk_id
        for hit in result
    } == {
        first.chunk_id,
        middle.chunk_id,
        last.chunk_id,
    }

    assert all(
        hit.effective_seed_rank
        == 1
        for hit in result
    )


def test_item_parent_is_expanded(
) -> None:
    parent = _chunk(
        source_number=2,
        chunk_index=0,
        text="parent",
        item_path=(
            "一、风险管理",
        ),
    )

    child = _chunk(
        source_number=2,
        chunk_index=5,
        text="child",
        item_path=(
            "一、风险管理",
            "（一）信用风险",
        ),
    )

    result = (
        expand_competition_retrieval_context(
            seed_hits=(
                _seed_hit(
                    chunk=child,
                    rank=1,
                ),
            ),
            corpus_chunks=(
                parent,
                child,
            ),
            config=(
                CompetitionContextExpansionConfig(
                    previous_window=0,
                    next_window=0,
                    enable_item_hierarchy=True,
                    max_item_depth_delta=1,
                )
            ),
        )
    )

    by_id = {
        hit.chunk_id: hit
        for hit in result
    }

    assert (
        parent.chunk_id
        in by_id
    )

    parent_hit = (
        by_id[parent.chunk_id]
    )

    assert (
        parent_hit
        .effective_seed_rank
        == 1
    )

    assert any(
        origin.relation
        == "item_parent"
        for origin
        in parent_hit.origins
    )


def test_item_child_is_expanded(
) -> None:
    parent = _chunk(
        source_number=3,
        chunk_index=0,
        text="parent",
        item_path=(
            "一、风险管理",
        ),
    )

    child = _chunk(
        source_number=3,
        chunk_index=7,
        text="child",
        item_path=(
            "一、风险管理",
            "（一）信用风险",
        ),
    )

    result = (
        expand_competition_retrieval_context(
            seed_hits=(
                _seed_hit(
                    chunk=parent,
                    rank=2,
                ).model_copy(
                    update={
                        "rank": 1,
                    }
                ),
            ),
            corpus_chunks=(
                parent,
                child,
            ),
            config=(
                CompetitionContextExpansionConfig(
                    previous_window=0,
                    next_window=0,
                    enable_item_hierarchy=True,
                    max_item_depth_delta=1,
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

    assert any(
        origin.relation
        == "item_child"
        for origin
        in by_id[
            child.chunk_id
        ].origins
    )


def test_item_hierarchy_requires_strict_prefix(
) -> None:
    first = _chunk(
        source_number=4,
        chunk_index=0,
        text="first",
        item_path=(
            "一、风险管理",
            "（一）信用风险",
        ),
    )

    sibling = _chunk(
        source_number=4,
        chunk_index=10,
        text="sibling",
        item_path=(
            "一、风险管理",
            "（二）市场风险",
        ),
    )

    result = (
        expand_competition_retrieval_context(
            seed_hits=(
                _seed_hit(
                    chunk=first,
                    rank=1,
                ),
            ),
            corpus_chunks=(
                first,
                sibling,
            ),
            config=(
                CompetitionContextExpansionConfig(
                    previous_window=0,
                    next_window=0,
                    enable_item_hierarchy=True,
                )
            ),
        )
    )

    assert tuple(
        hit.chunk_id
        for hit in result
    ) == (
        first.chunk_id,
    )


def test_item_hierarchy_respects_depth_delta(
) -> None:
    root = _chunk(
        source_number=5,
        chunk_index=0,
        text="root",
        item_path=(
            "一、风险管理",
        ),
    )

    child = _chunk(
        source_number=5,
        chunk_index=5,
        text="child",
        item_path=(
            "一、风险管理",
            "（一）信用风险",
        ),
    )

    grandchild = _chunk(
        source_number=5,
        chunk_index=10,
        text="grandchild",
        item_path=(
            "一、风险管理",
            "（一）信用风险",
            "1.定义",
        ),
    )

    result = (
        expand_competition_retrieval_context(
            seed_hits=(
                _seed_hit(
                    chunk=root,
                    rank=1,
                ),
            ),
            corpus_chunks=(
                root,
                child,
                grandchild,
            ),
            config=(
                CompetitionContextExpansionConfig(
                    previous_window=0,
                    next_window=0,
                    enable_item_hierarchy=True,
                    max_item_depth_delta=1,
                )
            ),
        )
    )

    ids = {
        hit.chunk_id
        for hit in result
    }

    assert (
        child.chunk_id
        in ids
    )

    assert (
        grandchild.chunk_id
        not in ids
    )


def test_depth_two_can_be_enabled_explicitly(
) -> None:
    root = _chunk(
        source_number=6,
        chunk_index=0,
        text="root",
        item_path=(
            "一、风险管理",
        ),
    )

    grandchild = _chunk(
        source_number=6,
        chunk_index=10,
        text="grandchild",
        item_path=(
            "一、风险管理",
            "（一）信用风险",
            "1.定义",
        ),
    )

    result = (
        expand_competition_retrieval_context(
            seed_hits=(
                _seed_hit(
                    chunk=root,
                    rank=1,
                ),
            ),
            corpus_chunks=(
                root,
                grandchild,
            ),
            config=(
                CompetitionContextExpansionConfig(
                    previous_window=0,
                    next_window=0,
                    enable_item_hierarchy=True,
                    max_item_depth_delta=2,
                )
            ),
        )
    )

    assert {
        hit.chunk_id
        for hit in result
    } == {
        root.chunk_id,
        grandchild.chunk_id,
    }


def test_item_hierarchy_does_not_cross_source(
) -> None:
    parent = _chunk(
        source_number=7,
        chunk_index=0,
        text="parent",
        item_path=(
            "一、风险管理",
        ),
    )

    child_other_source = _chunk(
        source_number=8,
        chunk_index=5,
        text="child",
        item_path=(
            "一、风险管理",
            "（一）信用风险",
        ),
    )

    result = (
        expand_competition_retrieval_context(
            seed_hits=(
                _seed_hit(
                    chunk=parent,
                    rank=1,
                ),
            ),
            corpus_chunks=(
                parent,
                child_other_source,
            ),
            config=(
                CompetitionContextExpansionConfig(
                    previous_window=0,
                    next_window=0,
                    enable_item_hierarchy=True,
                )
            ),
        )
    )

    assert tuple(
        hit.chunk_id
        for hit in result
    ) == (
        parent.chunk_id,
    )


def test_empty_item_path_does_not_expand(
) -> None:
    first = _chunk(
        source_number=9,
        chunk_index=0,
        text="first",
    )

    second = _chunk(
        source_number=9,
        chunk_index=10,
        text="second",
    )

    result = (
        expand_competition_retrieval_context(
            seed_hits=(
                _seed_hit(
                    chunk=first,
                    rank=1,
                ),
            ),
            corpus_chunks=(
                first,
                second,
            ),
            config=(
                CompetitionContextExpansionConfig(
                    previous_window=0,
                    next_window=0,
                    enable_item_hierarchy=True,
                )
            ),
        )
    )

    assert tuple(
        hit.chunk_id
        for hit in result
    ) == (
        first.chunk_id,
    )


def test_deduplicates_multiple_expansion_origins(
) -> None:
    parent = _chunk(
        source_number=10,
        chunk_index=0,
        text="parent",
        item_path=(
            "一、风险管理",
        ),
    )

    child = _chunk(
        source_number=10,
        chunk_index=1,
        text="child",
        item_path=(
            "一、风险管理",
            "（一）信用风险",
        ),
    )

    result = (
        expand_competition_retrieval_context(
            seed_hits=(
                _seed_hit(
                    chunk=child,
                    rank=1,
                ),
            ),
            corpus_chunks=(
                parent,
                child,
            ),
            config=(
                CompetitionContextExpansionConfig(
                    previous_window=1,
                    next_window=1,
                    enable_item_hierarchy=True,
                )
            ),
        )
    )

    parent_hit = next(
        hit
        for hit in result
        if (
            hit.chunk_id
            == parent.chunk_id
        )
    )

    relations = {
        origin.relation
        for origin
        in parent_hit.origins
    }

    assert (
        "previous_neighbor"
        in relations
    )

    assert (
        "item_parent"
        in relations
    )


def test_parent_child_gold_complete_via_item_hierarchy(
) -> None:
    parent = _chunk(
        source_number=11,
        chunk_index=0,
        text="parent context",
        item_path=(
            "一、披露要求",
        ),
    )

    direct = _chunk(
        source_number=11,
        chunk_index=8,
        text="direct evidence",
        item_path=(
            "一、披露要求",
            "（一）具体要求",
        ),
    )

    expanded = (
        expand_competition_retrieval_context(
            seed_hits=(
                _seed_hit(
                    chunk=parent,
                    rank=1,
                ),
            ),
            corpus_chunks=(
                parent,
                direct,
            ),
            config=(
                CompetitionContextExpansionConfig(
                    previous_window=0,
                    next_window=0,
                    enable_item_hierarchy=True,
                    max_item_depth_delta=1,
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
                parent.source_id
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


def test_seed_candidate_count_limits_expansion(
) -> None:
    first = _chunk(
        source_number=12,
        chunk_index=0,
        text="first",
    )

    second = _chunk(
        source_number=12,
        chunk_index=1,
        text="second",
    )

    third = _chunk(
        source_number=12,
        chunk_index=2,
        text="third",
    )

    result = (
        expand_competition_retrieval_context(
            seed_hits=(
                _seed_hit(
                    chunk=first,
                    rank=1,
                ),
                _seed_hit(
                    chunk=third,
                    rank=2,
                ),
            ),
            corpus_chunks=(
                first,
                second,
                third,
            ),
            config=(
                CompetitionContextExpansionConfig(
                    seed_candidate_count=1,
                    previous_window=0,
                    next_window=1,
                )
            ),
        )
    )

    assert {
        hit.chunk_id
        for hit in result
    } == {
        first.chunk_id,
        second.chunk_id,
    }


def test_rejects_empty_and_invalid_seed_rank(
) -> None:
    chunk = _chunk(
        source_number=13,
        chunk_index=0,
        text="seed",
    )

    with pytest.raises(
        EmptyCompetitionContextSeedError,
        match="至少需要",
    ):
        expand_competition_retrieval_context(
            seed_hits=(),
            corpus_chunks=(
                chunk,
            ),
        )

    invalid = (
        _seed_hit(
            chunk=chunk,
            rank=1,
        ).model_copy(
            update={
                "rank": 2,
            }
        )
    )

    with pytest.raises(
        InvalidCompetitionContextSeedError,
        match="连续递增",
    ):
        expand_competition_retrieval_context(
            seed_hits=(
                invalid,
            ),
            corpus_chunks=(
                chunk,
            ),
        )


def test_rejects_seed_corpus_identity_mismatch(
) -> None:
    corpus_chunk = _chunk(
        source_number=14,
        chunk_index=0,
        text="original",
    )

    changed_text = (
        "changed"
    )

    changed_chunk = (
        corpus_chunk.model_copy(
            update={
                "text": changed_text,
                "char_count": len(
                    changed_text
                ),
                "text_sha256": (
                    hashlib.sha256(
                        changed_text.encode(
                            "utf-8"
                        )
                    ).hexdigest()
                ),
            }
        )
    )

    seed = _seed_hit(
        chunk=changed_chunk,
        rank=1,
    )

    with pytest.raises(
        CompetitionContextSeedCorpusMismatchError,
        match="内容不一致",
    ):
        expand_competition_retrieval_context(
            seed_hits=(
                seed,
            ),
            corpus_chunks=(
                corpus_chunk,
            ),
        )