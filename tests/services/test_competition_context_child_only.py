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
from app.schemas.competition_option_anchored_merge import (
    CompetitionBM25OptionAnchoredMergeHit,
)
from app.services.competition_context_expansion import (
    expand_competition_retrieval_context,
)


def _chunk(
    *,
    chunk_index: int,
    text: str,
    item_path: tuple[str, ...],
) -> CompetitionTextChunk:
    source_id = "source_child_only"

    doc_id = "doc_child_only"

    return CompetitionTextChunk(
        chunk_id=(
            f"chunk:{doc_id}:"
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
                block_index=chunk_index,
                start_char=0,
                end_char=len(text),
            ),
        ),
        text=text,
        char_count=len(text),
        text_sha256=(
            hashlib.sha256(
                text.encode("utf-8")
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
    chunk: CompetitionTextChunk,
) -> CompetitionBM25OptionAnchoredMergeHit:
    return (
        CompetitionBM25OptionAnchoredMergeHit(
            rank=1,
            score=3.0,
            layer="option_anchor",
            question_only_rank=None,
            question_with_options_rank=1,
            chunk=chunk,
        )
    )


def test_child_only_expands_child_but_not_parent(
) -> None:
    root = _chunk(
        chunk_index=0,
        text="root",
        item_path=(
            "一、风险管理",
        ),
    )

    child = _chunk(
        chunk_index=5,
        text="child",
        item_path=(
            "一、风险管理",
            "（一）信用风险",
        ),
    )

    grandchild = _chunk(
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
                _seed(root),
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
                    enable_item_parent=False,
                    enable_item_child=True,
                    max_item_depth_delta=2,
                )
            ),
        )
    )

    by_id = {
        hit.chunk_id: hit
        for hit in result
    }

    assert root.chunk_id in by_id
    assert child.chunk_id in by_id
    assert grandchild.chunk_id in by_id

    assert any(
        origin.relation == "item_child"
        for origin
        in by_id[
            child.chunk_id
        ].origins
    )

    assert any(
        origin.relation == "item_child"
        for origin
        in by_id[
            grandchild.chunk_id
        ].origins
    )


def test_child_only_does_not_expand_parent(
) -> None:
    parent = _chunk(
        chunk_index=0,
        text="parent",
        item_path=(
            "一、风险管理",
        ),
    )

    child = _chunk(
        chunk_index=10,
        text="child",
        item_path=(
            "一、风险管理",
            "（一）信用风险",
        ),
    )

    result = (
        expand_competition_retrieval_context(
            seed_hits=(
                _seed(child),
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
                    enable_item_parent=False,
                    enable_item_child=True,
                    max_item_depth_delta=2,
                )
            ),
        )
    )

    ids = {
        hit.chunk_id
        for hit in result
    }

    assert child.chunk_id in ids
    assert parent.chunk_id not in ids


def test_parent_direction_can_still_be_enabled(
) -> None:
    parent = _chunk(
        chunk_index=0,
        text="parent",
        item_path=(
            "一、风险管理",
        ),
    )

    child = _chunk(
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
                _seed(child),
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
                    enable_item_parent=True,
                    enable_item_child=False,
                    max_item_depth_delta=1,
                )
            ),
        )
    )

    ids = {
        hit.chunk_id
        for hit in result
    }

    assert child.chunk_id in ids
    assert parent.chunk_id in ids


def test_hierarchy_cannot_disable_both_directions(
) -> None:
    with pytest.raises(
        ValueError,
        match="至少需要",
    ):
        CompetitionContextExpansionConfig(
            enable_item_hierarchy=True,
            enable_item_parent=False,
            enable_item_child=False,
        )