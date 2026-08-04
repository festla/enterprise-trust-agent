from __future__ import annotations

import pytest

from app.schemas.context_budget import (
    ContextBudgetPolicy,
)
from app.services.budgeted_context_oracle_retrieval_adapter import (
    BudgetedContextOracleRetrievalAdapter,
)
from app.services.context_expanded_oracle_retrieval_adapter import (
    ContextExpandedRetrievalAdapterError,
)
from tests.services.test_context_expanded_oracle_retrieval_adapter import (
    ADJACENT_CHUNK_ID,
    BASE_CHUNK_ID,
    EVIDENCE_ID,
    FACT_ID,
    REPORT_ID,
    FakeHitProvider,
    build_adjacent_chunk,
    build_base_chunk,
    build_bundle,
    build_chunk,
    build_hit,
    build_source,
)


NOISE_CHUNK_ID = (
    f"chunk_{REPORT_ID}_{'f' * 24}"
)


def build_budgeted_adapter(
    *,
    bundle=None,
    provider=None,
    sources=None,
    policy: ContextBudgetPolicy | None = None,
) -> BudgetedContextOracleRetrievalAdapter:
    base_chunk = build_base_chunk()

    actual_provider = (
        provider
        if provider is not None
        else FakeHitProvider(
            hits=(
                build_hit(
                    chunk=base_chunk,
                    rank=1,
                ),
            )
        )
    )

    actual_sources = (
        sources
        if sources is not None
        else {
            REPORT_ID: build_source(),
        }
    )

    actual_policy = (
        policy
        if policy is not None
        else ContextBudgetPolicy()
    )

    return BudgetedContextOracleRetrievalAdapter(
        registry_bundle=(
            bundle
            if bundle is not None
            else build_bundle()
        ),
        hit_provider=actual_provider,
        chunk_sources_by_report_id=(
            actual_sources
        ),
        policy=actual_policy,
    )


def build_query():
    from tests.services.test_context_expanded_oracle_retrieval_adapter import (
        build_query as original_build_query,
    )

    return original_build_query()


def test_expands_when_base_context_is_unresolved(
) -> None:
    base_chunk = build_base_chunk()
    adjacent_chunk = build_adjacent_chunk()

    adapter = build_budgeted_adapter(
        provider=FakeHitProvider(
            hits=(
                build_hit(
                    chunk=base_chunk,
                    rank=1,
                ),
            )
        ),
        sources={
            REPORT_ID: build_source(
                chunks=(
                    base_chunk,
                    adjacent_chunk,
                )
            ),
        },
    )

    trace = adapter.retrieve(
        query=build_query(),
        top_k=5,
    )

    assert trace.status == "completed"

    assert trace.retrieved_fact_ids == (
        FACT_ID,
    )

    assert trace.retrieved_evidence_ids == (
        EVIDENCE_ID,
    )

    assert trace.retrieved_chunk_ids == (
        BASE_CHUNK_ID,
        ADJACENT_CHUNK_ID,
    )

    assert len(
        adapter.full_expansion_audit_records
    ) == 1

    assert len(
        adapter.base_resolution_audit_records
    ) == 1

    assert len(
        adapter.budget_selection_audit_records
    ) == 1

    assert len(
        adapter.final_resolution_audit_records
    ) == 1

    base_resolution = (
        adapter
        .base_resolution_audit_records[0]
    )

    assert base_resolution.fact_ids == ()

    selection = (
        adapter
        .budget_selection_audit_records[0]
    )

    assert (
        selection.gate_decision
        == "expansion_required"
    )

    assert (
        selection.selected_expanded_chunk_ids
        == (
            ADJACENT_CHUNK_ID,
        )
    )

    assert (
        selection.selected_expanded_item_count
        == 1
    )

    final_resolution = (
        adapter
        .final_resolution_audit_records[0]
    )

    assert final_resolution.fact_ids == (
        FACT_ID,
    )

    assert (
        final_resolution.expanded_fact_ids
        == (
            FACT_ID,
        )
    )


def test_skips_expansion_when_base_resolves_fact(
) -> None:
    base_chunk = build_base_chunk()
    adjacent_chunk = build_adjacent_chunk()

    adapter = build_budgeted_adapter(
        bundle=build_bundle(
            evidence_chunk_id=(
                BASE_CHUNK_ID
            ),
            evidence_pdf_page=121,
        ),
        provider=FakeHitProvider(
            hits=(
                build_hit(
                    chunk=base_chunk,
                    rank=1,
                ),
            )
        ),
        sources={
            REPORT_ID: build_source(
                chunks=(
                    base_chunk,
                    adjacent_chunk,
                )
            ),
        },
    )

    trace = adapter.retrieve(
        query=build_query(),
        top_k=5,
    )

    assert trace.status == "completed"

    assert trace.retrieved_fact_ids == (
        FACT_ID,
    )

    assert trace.retrieved_chunk_ids == (
        BASE_CHUNK_ID,
    )

    full_expansion = (
        adapter
        .full_expansion_audit_records[0]
    )

    assert (
        full_expansion.expanded_chunk_ids
        == (
            ADJACENT_CHUNK_ID,
        )
    )

    selection = (
        adapter
        .budget_selection_audit_records[0]
    )

    assert (
        selection.gate_decision
        == "base_resolved"
    )

    assert selection.base_fact_ids == (
        FACT_ID,
    )

    assert selection.candidates == ()

    assert (
        selection.selected_expanded_chunk_ids
        == ()
    )

    assert (
        selection.selected_context
        .expanded_item_count
        == 0
    )

    final_resolution = (
        adapter
        .final_resolution_audit_records[0]
    )

    assert final_resolution.base_fact_ids == (
        FACT_ID,
    )

    assert (
        final_resolution.expanded_fact_ids
        == ()
    )


def test_adapter_applies_item_budget(
) -> None:
    base_chunk = build_base_chunk()

    relevant_chunk = (
        build_adjacent_chunk()
    )

    noise_chunk = build_chunk(
        chunk_id=NOISE_CHUNK_ID,
        pdf_page=120,
        chunk_index=9,
        text=(
            "corporate governance "
            "board committee"
        ),
    )

    adapter = build_budgeted_adapter(
        provider=FakeHitProvider(
            hits=(
                build_hit(
                    chunk=base_chunk,
                    rank=1,
                ),
            )
        ),
        sources={
            REPORT_ID: build_source(
                chunks=(
                    noise_chunk,
                    base_chunk,
                    relevant_chunk,
                )
            ),
        },
        policy=ContextBudgetPolicy(
            max_expanded_items=1,
            max_expanded_chars=1600,
        ),
    )

    trace = adapter.retrieve(
        query=build_query(),
        top_k=5,
    )

    assert trace.status == "completed"

    assert trace.retrieved_fact_ids == (
        FACT_ID,
    )

    selection = (
        adapter
        .budget_selection_audit_records[0]
    )

    assert (
        selection.candidate_item_count
        == 2
    )

    assert (
        selection.selected_expanded_chunk_ids
        == (
            ADJACENT_CHUNK_ID,
        )
    )

    assert (
        selection.dropped_expanded_chunk_ids
        == (
            NOISE_CHUNK_ID,
        )
    )

    assert (
        selection.selected_expanded_item_count
        == 1
    )

    assert (
        selection.selected_expanded_char_count
        <= 1600
    )


def test_empty_hits_return_completed_without_audit(
) -> None:
    adapter = build_budgeted_adapter(
        provider=FakeHitProvider(
            hits=(),
        )
    )

    trace = adapter.retrieve(
        query=build_query(),
        top_k=5,
    )

    assert trace.status == "completed"
    assert trace.retrieved_fact_ids == ()
    assert trace.retrieved_evidence_ids == ()
    assert trace.retrieved_chunk_ids == ()

    assert (
        adapter.full_expansion_audit_records
        == ()
    )

    assert (
        adapter.base_resolution_audit_records
        == ()
    )

    assert (
        adapter.budget_selection_audit_records
        == ()
    )

    assert (
        adapter.final_resolution_audit_records
        == ()
    )


def test_provider_error_returns_failed_trace(
) -> None:
    adapter = build_budgeted_adapter(
        provider=FakeHitProvider(
            error=RuntimeError(
                "provider unavailable"
            ),
        )
    )

    trace = adapter.retrieve(
        query=build_query(),
        top_k=5,
    )

    assert trace.status == "failed"

    assert (
        trace.error_message
        == "provider unavailable"
    )

    assert trace.retrieved_fact_ids == ()
    assert trace.retrieved_chunk_ids == ()

    assert (
        adapter.budget_selection_audit_records
        == ()
    )


def test_clear_audit_records_clears_all_stages(
) -> None:
    adapter = build_budgeted_adapter()

    trace = adapter.retrieve(
        query=build_query(),
        top_k=5,
    )

    assert trace.status == "completed"

    assert (
        adapter.full_expansion_audit_records
    )

    assert (
        adapter.base_resolution_audit_records
    )

    assert (
        adapter.budget_selection_audit_records
    )

    assert (
        adapter.final_resolution_audit_records
    )

    adapter.clear_audit_records()

    assert (
        adapter.full_expansion_audit_records
        == ()
    )

    assert (
        adapter.base_resolution_audit_records
        == ()
    )

    assert (
        adapter.budget_selection_audit_records
        == ()
    )

    assert (
        adapter.final_resolution_audit_records
        == ()
    )


def test_retriever_id_contains_frozen_policy(
) -> None:
    adapter = build_budgeted_adapter()

    assert (
        adapter.retriever_id
        == (
            "fake_hybrid_reranker_v1_"
            "adjacent_page_context_v1_"
            "gated_lexical_"
            "adjacent_budget_v1_"
            "registry_context_"
            "fact_resolver_v1"
        )
    )

    assert (
        adapter.policy.policy_id
        == (
            "gated_lexical_"
            "adjacent_budget_v1"
        )
    )

    assert (
        adapter.policy.max_expanded_items
        == 2
    )

    assert (
        adapter.policy.max_expanded_chars
        == 1600
    )


def test_rejects_non_positive_top_k(
) -> None:
    adapter = build_budgeted_adapter()

    with pytest.raises(
        ContextExpandedRetrievalAdapterError,
        match="top_k must be greater than 0",
    ):
        adapter.retrieve(
            query=build_query(),
            top_k=0,
        )