from __future__ import annotations

import re
from collections.abc import Sequence

from app.schemas.complex_plan_eval_result import (
    ComplexRetrievalQueryOutput,
)
from app.schemas.context_budget import (
    BudgetedContextSelection,
    ContextBudgetPolicy,
    ContextCandidateAudit,
)
from app.schemas.context_expansion import (
    AdjacentPageContextExpansion,
    ContextExpansionItem,
)
from app.services.context_fact_resolver import (
    ContextFactResolution,
)


_TOKEN_PATTERN = re.compile(
    r"[A-Za-z_]+|[\u4e00-\u9fff]{2,}"
)


class ContextBudgetingError(ValueError):
    """Budgeted context selection failed."""


def build_base_only_context(
    expansion: AdjacentPageContextExpansion,
) -> AdjacentPageContextExpansion:
    """
    Build a context containing only the original reranked
    Top-k items.

    Adjacent candidates are not included in the returned
    context.
    """

    return _build_selected_context(
        source=expansion,
        selected_expanded_items=(),
        duplicate_chunk_count=0,
    )


def select_budgeted_adjacent_context(
    *,
    query: ComplexRetrievalQueryOutput,
    full_expansion: (
        AdjacentPageContextExpansion
    ),
    base_resolution: ContextFactResolution,
    policy: ContextBudgetPolicy | None = None,
    metric_hints: Sequence[str] = (),
) -> BudgetedContextSelection:
    """
    Apply gated lexical adjacent-page selection.

    Gate:
        If the original Top-k already resolves a Fact,
        do not select any expanded chunks.

    Ranking:
        1. query token hit count;
        2. query bigram overlap count;
        3. query bigram recall;
        4. better anchor retrieval rank;
        5. shorter chunk;
        6. stable chunk_id tie-breaker.

    Budget:
        Select candidates in ranking order until either
        the item budget or character budget is reached.
    """

    actual_policy = (
        policy
        if policy is not None
        else ContextBudgetPolicy()
    )

    normalized_metric_hints = tuple(
        dict.fromkeys(
            hint.strip()
            for hint in metric_hints
            if hint.strip()
        )
    )

    _validate_input_identity(
        query=query,
        full_expansion=(
            full_expansion
        ),
        base_resolution=(
            base_resolution
        ),
    )

    base_context = build_base_only_context(
        full_expansion
    )

    if base_resolution.fact_ids:
        return BudgetedContextSelection(
            schema_version=1,
            policy=actual_policy,
            query_id=query.query_id,
            gate_decision=(
                "base_resolved"
            ),
            base_fact_ids=(
                base_resolution.fact_ids
            ),
            candidate_item_count=0,
            candidate_char_count=0,
            candidates=(),
            selected_expanded_chunk_ids=(),
            dropped_expanded_chunk_ids=(),
            selected_expanded_item_count=0,
            selected_expanded_char_count=0,
            selected_context=base_context,
        )

    expanded_candidates = tuple(
        item
        for item in full_expansion.items
        if (
            item.origin != "retrieved"
        )
    )

    ranked_candidates = sorted(
        expanded_candidates,
        key=lambda item: _candidate_sort_key(
            query_text=query.semantic_query,
            candidate_text=item.text,
            metric_hints=(
                normalized_metric_hints
            ),
            anchor_retrieval_rank=(
                item.anchor_retrieval_rank
            ),
            text_char_count=(
                item.text_char_count
            ),
            chunk_id=item.chunk_id,
        ),
    )

    selected_items: list[
        ContextExpansionItem
    ] = []

    candidate_audits: list[
        ContextCandidateAudit
    ] = []

    selected_char_count = 0

    for selection_rank, item in enumerate(
        ranked_candidates,
        start=1,
    ):
        (
            token_hit_count,
            bigram_overlap_count,
            query_bigram_recall,
        ) = _score_candidate(
            query_text=query.semantic_query,
            candidate_text=item.text,
        )

        (
            metric_exact_match,
            metric_bigram_overlap_count,
            metric_bigram_recall,
        ) = _score_metric_candidate(
            metric_hints=(
                normalized_metric_hints
            ),
            candidate_text=item.text,
        )

        if (
            len(selected_items)
            >= actual_policy
            .max_expanded_items
        ):
            decision = "item_budget"

        elif (
            selected_char_count
            + item.text_char_count
            > actual_policy
            .max_expanded_chars
        ):
            decision = "char_budget"

        else:
            decision = "selected"

            selected_items.append(item)

            selected_char_count += (
                item.text_char_count
            )

        candidate_audits.append(
            ContextCandidateAudit(
                selection_rank=(
                    selection_rank
                ),
                chunk_id=item.chunk_id,
                pdf_page=item.pdf_page,
                anchor_chunk_id=(
                    item.anchor_chunk_id
                ),
                anchor_retrieval_rank=(
                    item
                    .anchor_retrieval_rank
                ),
                text_char_count=(
                    item.text_char_count
                ),
                token_hit_count=(
                    token_hit_count
                ),
                bigram_overlap_count=(
                    bigram_overlap_count
                ),
                query_bigram_recall=(
                    query_bigram_recall
                ),
                metric_exact_match=(
                    metric_exact_match
                ),
                metric_bigram_overlap_count=(
                    metric_bigram_overlap_count
                ),
                metric_bigram_recall=(
                    metric_bigram_recall
                ),
                decision=decision,
            )
        )

    selected_context = (
        _build_selected_context(
            source=full_expansion,
            selected_expanded_items=(
                tuple(selected_items)
            ),
            duplicate_chunk_count=(
                full_expansion
                .duplicate_chunk_count
            ),
        )
    )

    selected_chunk_ids = tuple(
        candidate.chunk_id
        for candidate
        in candidate_audits
        if candidate.decision
        == "selected"
    )

    dropped_chunk_ids = tuple(
        candidate.chunk_id
        for candidate
        in candidate_audits
        if candidate.decision
        != "selected"
    )

    return BudgetedContextSelection(
        schema_version=1,
        policy=actual_policy,
        query_id=query.query_id,
        gate_decision=(
            "expansion_required"
        ),
        base_fact_ids=(),
        candidate_item_count=len(
            candidate_audits
        ),
        candidate_char_count=sum(
            candidate.text_char_count
            for candidate
            in candidate_audits
        ),
        candidates=tuple(
            candidate_audits
        ),
        selected_expanded_chunk_ids=(
            selected_chunk_ids
        ),
        dropped_expanded_chunk_ids=(
            dropped_chunk_ids
        ),
        selected_expanded_item_count=(
            len(selected_chunk_ids)
        ),
        selected_expanded_char_count=(
            selected_char_count
        ),
        selected_context=(
            selected_context
        ),
    )


def score_lexical_context_candidate(
    *,
    query_text: str,
    candidate_text: str,
) -> tuple[
    int,
    int,
    float,
]:
    """
    Public deterministic scoring helper.

    Returns:
        token_hit_count,
        bigram_overlap_count,
        query_bigram_recall
    """

    if not query_text.strip():
        raise ContextBudgetingError(
            "query_text cannot be empty"
        )

    if not candidate_text:
        raise ContextBudgetingError(
            "candidate_text cannot be empty"
        )

    return _score_candidate(
        query_text=query_text,
        candidate_text=candidate_text,
    )


def _score_candidate(
    *,
    query_text: str,
    candidate_text: str,
) -> tuple[
    int,
    int,
    float,
]:
    normalized_candidate_text = (
        candidate_text.lower()
    )

    query_tokens = _extract_tokens(
        query_text
    )

    token_hit_count = sum(
        token
        in normalized_candidate_text
        for token in query_tokens
    )

    query_bigrams = _extract_bigrams(
        query_text
    )

    candidate_bigrams = (
        _extract_bigrams(
            candidate_text
        )
    )

    bigram_overlap_count = len(
        query_bigrams.intersection(
            candidate_bigrams
        )
    )

    query_bigram_recall = (
        bigram_overlap_count
        / len(query_bigrams)
        if query_bigrams
        else 0.0
    )

    return (
        token_hit_count,
        bigram_overlap_count,
        query_bigram_recall,
    )


def _score_metric_candidate(
    *,
    metric_hints: Sequence[str],
    candidate_text: str,
) -> tuple[
    int,
    int,
    float,
]:
    if not metric_hints:
        return 0, 0, 0.0

    normalized_candidate = "".join(
        character.lower()
        for character in candidate_text
        if character.isalnum()
    )

    candidate_bigrams = _extract_bigrams(
        candidate_text
    )

    exact_match = 0
    max_overlap = 0
    max_recall = 0.0

    for hint in metric_hints:
        normalized_hint = "".join(
            character.lower()
            for character in hint
            if character.isalnum()
        )

        if (
            normalized_hint
            and normalized_hint
            in normalized_candidate
        ):
            exact_match = 1

        hint_bigrams = _extract_bigrams(
            hint
        )

        overlap = len(
            hint_bigrams.intersection(
                candidate_bigrams
            )
        )

        recall = (
            overlap / len(hint_bigrams)
            if hint_bigrams
            else 0.0
        )

        max_overlap = max(
            max_overlap,
            overlap,
        )

        max_recall = max(
            max_recall,
            recall,
        )

    return (
        exact_match,
        max_overlap,
        max_recall,
    )


def _candidate_sort_key(
    *,
    query_text: str,
    candidate_text: str,
    metric_hints: Sequence[str],
    anchor_retrieval_rank: int,
    text_char_count: int,
    chunk_id: str,
) -> tuple[object, ...]:
    (
        token_hit_count,
        bigram_overlap_count,
        query_bigram_recall,
    ) = _score_candidate(
        query_text=query_text,
        candidate_text=candidate_text,
    )

    if not metric_hints:
        return (
            -token_hit_count,
            -bigram_overlap_count,
            -query_bigram_recall,
            anchor_retrieval_rank,
            text_char_count,
            chunk_id,
        )

    (
        metric_exact_match,
        metric_bigram_overlap_count,
        metric_bigram_recall,
    ) = _score_metric_candidate(
        metric_hints=metric_hints,
        candidate_text=candidate_text,
    )

    return (
        -metric_exact_match,
        -metric_bigram_recall,
        -metric_bigram_overlap_count,
        -bigram_overlap_count,
        -query_bigram_recall,
        -token_hit_count,
        anchor_retrieval_rank,
        text_char_count,
        chunk_id,
    )


def _extract_tokens(
    text: str,
) -> tuple[str, ...]:
    return tuple(
        token.lower()
        for token
        in _TOKEN_PATTERN.findall(text)
        if len(token) >= 2
    )


def _extract_bigrams(
    text: str,
) -> set[str]:
    compact_text = "".join(
        character.lower()
        for character in text
        if (
            character.isalnum()
            and not character.isdigit()
        )
    )

    if len(compact_text) < 2:
        return set()

    return {
        compact_text[index:index + 2]
        for index in range(
            len(compact_text) - 1
        )
    }


def _validate_input_identity(
    *,
    query: ComplexRetrievalQueryOutput,
    full_expansion: (
        AdjacentPageContextExpansion
    ),
    base_resolution: ContextFactResolution,
) -> None:
    compared_fields = (
        "query_id",
        "semantic_query",
        "company_id",
        "report_id",
        "fiscal_year",
        "report_type",
    )

    for field_name in compared_fields:
        query_value = getattr(
            query,
            field_name,
        )

        expansion_value = getattr(
            full_expansion,
            field_name,
        )

        if query_value != expansion_value:
            raise ContextBudgetingError(
                "query and full expansion "
                "identity mismatch: "
                f"{field_name}"
            )

    if (
        base_resolution.query_id
        != query.query_id
    ):
        raise ContextBudgetingError(
            "base resolution query_id does "
            "not match query"
        )

    invalid_base_supports = tuple(
        support
        for support
        in base_resolution.supports
        if (
            support.supporting_origin
            != "retrieved"
        )
    )

    if invalid_base_supports:
        raise ContextBudgetingError(
            "base_resolution can only contain "
            "supports from retrieved items"
        )


def _build_selected_context(
    *,
    source: AdjacentPageContextExpansion,
    selected_expanded_items: tuple[
        ContextExpansionItem,
        ...,
    ],
    duplicate_chunk_count: int,
) -> AdjacentPageContextExpansion:
    base_items = tuple(
        item
        for item in source.items
        if item.origin == "retrieved"
    )

    ordered_items: list[
        ContextExpansionItem
    ] = []

    for context_order, item in enumerate(
        (
            *base_items,
            *selected_expanded_items,
        ),
        start=1,
    ):
        item_values = item.model_dump(
            mode="python"
        )

        item_values[
            "context_order"
        ] = context_order

        ordered_items.append(
            ContextExpansionItem(
                **item_values
            )
        )

    items = tuple(ordered_items)

    selected_expanded_chunk_ids = tuple(
        item.chunk_id
        for item in items
        if (
            item.origin != "retrieved"
        )
    )

    base_chunk_ids = tuple(
        item.chunk_id
        for item in items
        if item.origin == "retrieved"
    )

    base_char_count = sum(
        item.text_char_count
        for item in items
        if item.origin == "retrieved"
    )

    expanded_char_count = sum(
        item.text_char_count
        for item in items
        if (
            item.origin != "retrieved"
        )
    )

    return AdjacentPageContextExpansion(
        schema_version=(
            source.schema_version
        ),
        strategy_id=(
            source.strategy_id
        ),
        query_id=source.query_id,
        original_query=(
            source.original_query
        ),
        semantic_query=(
            source.semantic_query
        ),
        company_id=source.company_id,
        report_id=source.report_id,
        fiscal_year=source.fiscal_year,
        report_type=source.report_type,
        document_id=source.document_id,
        base_top_k=source.base_top_k,
        page_window=source.page_window,
        items=items,
        base_chunk_ids=base_chunk_ids,
        expanded_chunk_ids=(
            selected_expanded_chunk_ids
        ),
        used_chunk_ids=tuple(
            item.chunk_id
            for item in items
        ),
        base_item_count=len(
            base_chunk_ids
        ),
        expanded_item_count=len(
            selected_expanded_chunk_ids
        ),
        total_item_count=len(items),
        duplicate_chunk_count=(
            duplicate_chunk_count
        ),
        base_char_count=(
            base_char_count
        ),
        expanded_char_count=(
            expanded_char_count
        ),
        total_char_count=(
            base_char_count
            + expanded_char_count
        ),
    )
