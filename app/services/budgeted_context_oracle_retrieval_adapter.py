from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from time import perf_counter

from app.schemas.complex_plan_eval_result import (
    ComplexRetrievalQueryOutput,
    ComplexRetrievalTrace,
)
from app.schemas.context_budget import (
    BudgetedContextSelection,
    ContextBudgetPolicy,
)
from app.schemas.context_expansion import (
    AdjacentPageContextExpansion,
)
from app.schemas.reranker import (
    RerankedRetrievalHit,
)
from app.services.context_budgeting import (
    build_base_only_context,
    select_budgeted_adjacent_context,
)
from app.services.context_expanded_oracle_retrieval_adapter import (
    ContextExpandedOracleRetrievalAdapter,
    ContextExpandedRetrievalAdapterError,
)
from app.services.context_expansion import (
    expand_adjacent_page_context,
)
from app.services.context_fact_resolver import (
    ContextFactResolution,
    resolve_facts_from_context,
)


@dataclass(slots=True)
class BudgetedContextOracleRetrievalAdapter(
    ContextExpandedOracleRetrievalAdapter
):
    """
    Resolve FinancialFacts using gated and budgeted
    adjacent-page context.

    Runtime flow:

    1. Run the original Top-k retrieval.
    2. Resolve Facts using only the original Top-k.
    3. If the base context resolves a Fact, skip expansion.
    4. Otherwise rank adjacent-page candidates.
    5. Select candidates under the frozen item and
       character budgets.
    6. Resolve Facts from the final selected context.

    This adapter does not use Gold Fact IDs, Gold Evidence
    IDs, Gold PDF pages or case-specific rules.
    """

    policy: ContextBudgetPolicy = field(
        default_factory=ContextBudgetPolicy,
    )

    _base_resolution_audit_records: list[
        ContextFactResolution
    ] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    _budget_selection_audit_records: list[
        BudgetedContextSelection
    ] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        ContextExpandedOracleRetrievalAdapter.__post_init__(
            self
        )

        self.policy = (
            ContextBudgetPolicy.model_validate(
                self.policy
            )
        )

    @property
    def retriever_id(self) -> str:
        provider_id = (
            self.hit_provider.provider_id.strip()
        )

        if not provider_id:
            return ""

        return "_".join(
            (
                provider_id,
                self.context_strategy_id,
                self.policy.policy_id,
                self.resolver_version,
            )
        )

    @property
    def full_expansion_audit_records(
        self,
    ) -> tuple[
        AdjacentPageContextExpansion,
        ...,
    ]:
        return self.expansion_audit_records

    @property
    def base_resolution_audit_records(
        self,
    ) -> tuple[
        ContextFactResolution,
        ...,
    ]:
        return tuple(
            self._base_resolution_audit_records
        )

    @property
    def budget_selection_audit_records(
        self,
    ) -> tuple[
        BudgetedContextSelection,
        ...,
    ]:
        return tuple(
            self._budget_selection_audit_records
        )

    @property
    def final_resolution_audit_records(
        self,
    ) -> tuple[
        ContextFactResolution,
        ...,
    ]:
        return self.resolution_audit_records

    def clear_audit_records(
        self,
    ) -> None:
        ContextExpandedOracleRetrievalAdapter.clear_audit_records(
            self
        )

        self._base_resolution_audit_records.clear()
        self._budget_selection_audit_records.clear()

    def retrieve(
        self,
        *,
        query: ComplexRetrievalQueryOutput,
        top_k: int,
    ) -> ComplexRetrievalTrace:
        """
        Retrieve Top-k chunks and conditionally add
        budgeted adjacent-page context.
        """

        if top_k <= 0:
            raise ContextExpandedRetrievalAdapterError(
                "top_k must be greater than 0"
            )

        timer_start = perf_counter()

        try:
            self._validate_provider_id()

            self._validate_query_report(
                query
            )

            source = self._get_chunk_source(
                query
            )

            raw_hits = self.hit_provider.search(
                query=query,
                top_k=top_k,
            )

            hits = tuple(
                RerankedRetrievalHit.model_validate(
                    hit
                )
                for hit in raw_hits
            )

            self._validate_hits(
                query=query,
                hits=hits,
                top_k=top_k,
            )

            if not hits:
                latency_ms = (
                    perf_counter()
                    - timer_start
                ) * 1000

                return ComplexRetrievalTrace(
                    query_id=query.query_id,
                    status="completed",
                    retrieved_fact_ids=(),
                    retrieved_evidence_ids=(),
                    retrieved_chunk_ids=(),
                    top_k=top_k,
                    latency_ms=latency_ms,
                    error_message=None,
                )

            full_expansion = (
                expand_adjacent_page_context(
                    original_query=(
                        query.semantic_query
                    ),
                    query=query,
                    hits=hits,
                    chunks=source.chunks,
                    base_top_k=top_k,
                    include_same_page_siblings=(
                        self.context_strategy_id
                        == (
                            "same_and_adjacent_"
                            "page_context_v2"
                        )
                    ),
                )
            )

            base_context = (
                build_base_only_context(
                    full_expansion
                )
            )

            base_resolution = (
                resolve_facts_from_context(
                    registry_bundle=(
                        self.registry_bundle
                    ),
                    query=query,
                    expansion=base_context,
                )
            )

            budget_selection = (
                select_budgeted_adjacent_context(
                    query=query,
                    full_expansion=(
                        full_expansion
                    ),
                    base_resolution=(
                        base_resolution
                    ),
                    policy=self.policy,
                    metric_hints=(
                        (
                            self.registry_bundle
                            .metrics
                            .require(
                                query.metric_id
                            )
                            .display_name_cn,
                        )
                        if (
                            self.policy
                            .lexical_score_version
                            == (
                                "metric_name_query_"
                                "bigram_v1"
                            )
                        )
                        else ()
                    ),
                )
            )

            if (
                budget_selection.gate_decision
                == "base_resolved"
            ):
                final_resolution = (
                    base_resolution
                )
            else:
                final_resolution = (
                    resolve_facts_from_context(
                        registry_bundle=(
                            self.registry_bundle
                        ),
                        query=query,
                        expansion=(
                            budget_selection
                            .selected_context
                        ),
                    )
                )

            self._expansion_audit_records.append(
                full_expansion
            )

            self._base_resolution_audit_records.append(
                base_resolution
            )

            self._budget_selection_audit_records.append(
                budget_selection
            )

            self._resolution_audit_records.append(
                final_resolution
            )

            latency_ms = (
                perf_counter()
                - timer_start
            ) * 1000

            return ComplexRetrievalTrace(
                query_id=query.query_id,
                status="completed",
                retrieved_fact_ids=(
                    final_resolution.fact_ids
                ),
                retrieved_evidence_ids=(
                    final_resolution.evidence_ids
                ),
                retrieved_chunk_ids=(
                    budget_selection
                    .selected_context
                    .used_chunk_ids
                ),
                top_k=top_k,
                latency_ms=latency_ms,
                error_message=None,
            )

        except Exception as exc:
            latency_ms = (
                perf_counter()
                - timer_start
            ) * 1000

            return ComplexRetrievalTrace(
                query_id=query.query_id,
                status="failed",
                retrieved_fact_ids=(),
                retrieved_evidence_ids=(),
                retrieved_chunk_ids=(),
                top_k=top_k,
                latency_ms=latency_ms,
                error_message=str(exc),
            )
