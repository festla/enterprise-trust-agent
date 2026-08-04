from __future__ import annotations

from collections.abc import (
    Mapping,
)
from dataclasses import (
    dataclass,
    field,
)
from time import perf_counter

from app.schemas.complex_plan_eval_result import (
    ComplexRetrievalQueryOutput,
    ComplexRetrievalTrace,
)
from app.schemas.context_expansion import (
    AdjacentPageContextExpansion,
)
from app.schemas.reranker import (
    RerankedRetrievalHit,
)
from app.services.chunk_dataset_source import (
    LoadedChunkDataset,
)
from app.services.complex_oracle_retrieval_adapter import (
    ComplexRerankedHitProvider,
)
from app.services.context_expansion import (
    expand_adjacent_page_context,
)
from app.services.context_fact_resolver import (
    ContextFactResolution,
    resolve_facts_from_context,
)
from app.services.registry import (
    RegistryBundle,
)


class ContextExpandedRetrievalAdapterError(
    ValueError
):
    """Context-expanded retrieval failed."""


@dataclass(slots=True)
class ContextExpandedOracleRetrievalAdapter:
    """
    Resolve FinancialFacts from Top-k retrieval results
    plus auditable adjacent-page context.

    The original Top-k remains unchanged. Expanded chunks
    are context items and are not assigned fake retrieval
    ranks.
    """

    registry_bundle: RegistryBundle

    hit_provider: (
        ComplexRerankedHitProvider
    )

    chunk_sources_by_report_id: Mapping[
        str,
        LoadedChunkDataset,
    ]

    resolver_version: str = (
        "registry_context_fact_resolver_v1"
    )

    context_strategy_id: str = (
        "adjacent_page_context_v1"
    )

    _expansion_audit_records: list[
        AdjacentPageContextExpansion
    ] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    _resolution_audit_records: list[
        ContextFactResolution
    ] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        normalized_resolver_version = (
            self.resolver_version.strip()
        )

        normalized_strategy_id = (
            self.context_strategy_id.strip()
        )

        if not normalized_resolver_version:
            raise (
                ContextExpandedRetrievalAdapterError(
                    "resolver_version cannot be empty"
                )
            )

        if not normalized_strategy_id:
            raise (
                ContextExpandedRetrievalAdapterError(
                    "context_strategy_id cannot "
                    "be empty"
                )
            )

        if not (
            self.chunk_sources_by_report_id
        ):
            raise (
                ContextExpandedRetrievalAdapterError(
                    "at least one ChunkDataset "
                    "source is required"
                )
            )

        normalized_sources: dict[
            str,
            LoadedChunkDataset,
        ] = {}

        for (
            report_id,
            source,
        ) in (
            self
            .chunk_sources_by_report_id
            .items()
        ):
            normalized_report_id = (
                report_id.strip()
            )

            if not normalized_report_id:
                raise (
                    ContextExpandedRetrievalAdapterError(
                        "chunk source report_id "
                        "cannot be empty"
                    )
                )

            if (
                normalized_report_id
                != report_id
            ):
                raise (
                    ContextExpandedRetrievalAdapterError(
                        "chunk source report_id "
                        "cannot contain leading or "
                        "trailing whitespace"
                    )
                )

            if (
                source.manifest.report_id
                != normalized_report_id
            ):
                raise (
                    ContextExpandedRetrievalAdapterError(
                        "ChunkDataset source "
                        "report_id mismatch: "
                        f"route={normalized_report_id}, "
                        "manifest="
                        f"{source.manifest.report_id}"
                    )
                )

            normalized_sources[
                normalized_report_id
            ] = source

        self.resolver_version = (
            normalized_resolver_version
        )

        self.context_strategy_id = (
            normalized_strategy_id
        )

        self.chunk_sources_by_report_id = (
            normalized_sources
        )

    @property
    def retriever_id(self) -> str:
        provider_id = (
            self.hit_provider
            .provider_id
            .strip()
        )

        if not provider_id:
            return ""

        return (
            f"{provider_id}_"
            f"{self.context_strategy_id}_"
            f"{self.resolver_version}"
        )

    @property
    def expansion_audit_records(
        self,
    ) -> tuple[
        AdjacentPageContextExpansion,
        ...,
    ]:
        return tuple(
            self._expansion_audit_records
        )

    @property
    def resolution_audit_records(
        self,
    ) -> tuple[
        ContextFactResolution,
        ...,
    ]:
        return tuple(
            self._resolution_audit_records
        )

    def clear_audit_records(
        self,
    ) -> None:
        self._expansion_audit_records.clear()
        self._resolution_audit_records.clear()

    def retrieve(
        self,
        *,
        query: ComplexRetrievalQueryOutput,
        top_k: int,
    ) -> ComplexRetrievalTrace:
        """
        Retrieve Top-k chunks, expand adjacent pages and
        resolve verified FinancialFacts.
        """

        if top_k <= 0:
            raise (
                ContextExpandedRetrievalAdapterError(
                    "top_k must be greater than 0"
                )
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

            raw_hits = (
                self.hit_provider.search(
                    query=query,
                    top_k=top_k,
                )
            )

            hits = tuple(
                RerankedRetrievalHit
                .model_validate(hit)
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

            expansion = (
                expand_adjacent_page_context(
                    # GoldOracleRetriever only exposes
                    # the atomic retrieval query here.
                    original_query=(
                        query.semantic_query
                    ),
                    query=query,
                    hits=hits,
                    chunks=source.chunks,
                    base_top_k=top_k,
                )
            )

            resolution = (
                resolve_facts_from_context(
                    registry_bundle=(
                        self.registry_bundle
                    ),
                    query=query,
                    expansion=expansion,
                )
            )

            self._expansion_audit_records.append(
                expansion
            )

            self._resolution_audit_records.append(
                resolution
            )

            latency_ms = (
                perf_counter()
                - timer_start
            ) * 1000

            return ComplexRetrievalTrace(
                query_id=query.query_id,
                status="completed",
                retrieved_fact_ids=(
                    resolution.fact_ids
                ),
                retrieved_evidence_ids=(
                    resolution.evidence_ids
                ),
                # These are all context chunks used by
                # resolution. top_k still describes only
                # the base reranked retrieval count.
                retrieved_chunk_ids=(
                    expansion.used_chunk_ids
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

    def _validate_provider_id(
        self,
    ) -> None:
        if not (
            self.hit_provider
            .provider_id
            .strip()
        ):
            raise (
                ContextExpandedRetrievalAdapterError(
                    "hit provider provider_id "
                    "cannot be empty"
                )
            )

    def _validate_query_report(
        self,
        query: ComplexRetrievalQueryOutput,
    ) -> None:
        report = (
            self.registry_bundle
            .reports
            .get(query.report_id)
        )

        if report is None:
            raise (
                ContextExpandedRetrievalAdapterError(
                    "query references an unknown "
                    f"report: {query.report_id}"
                )
            )

        if (
            report.company_id
            != query.company_id
        ):
            raise (
                ContextExpandedRetrievalAdapterError(
                    "query company_id does not "
                    "match Report"
                )
            )

        if (
            report.fiscal_year
            != query.fiscal_year
        ):
            raise (
                ContextExpandedRetrievalAdapterError(
                    "query fiscal_year does not "
                    "match Report"
                )
            )

        if (
            report.report_type
            is not query.report_type
        ):
            raise (
                ContextExpandedRetrievalAdapterError(
                    "query report_type does not "
                    "match Report"
                )
            )

    def _get_chunk_source(
        self,
        query: ComplexRetrievalQueryOutput,
    ) -> LoadedChunkDataset:
        source = (
            self.chunk_sources_by_report_id
            .get(query.report_id)
        )

        if source is None:
            raise (
                ContextExpandedRetrievalAdapterError(
                    "no ChunkDataset source for "
                    f"report: {query.report_id}"
                )
            )

        manifest = source.manifest

        if (
            manifest.company_id
            != query.company_id
        ):
            raise (
                ContextExpandedRetrievalAdapterError(
                    "ChunkDataset company_id does "
                    "not match query"
                )
            )

        if (
            manifest.fiscal_year
            != query.fiscal_year
        ):
            raise (
                ContextExpandedRetrievalAdapterError(
                    "ChunkDataset fiscal_year does "
                    "not match query"
                )
            )

        if (
            manifest.report_type
            is not query.report_type
        ):
            raise (
                ContextExpandedRetrievalAdapterError(
                    "ChunkDataset report_type does "
                    "not match query"
                )
            )

        return source

    def _validate_hits(
        self,
        *,
        query: ComplexRetrievalQueryOutput,
        hits: tuple[
            RerankedRetrievalHit,
            ...,
        ],
        top_k: int,
    ) -> None:
        if len(hits) > top_k:
            raise (
                ContextExpandedRetrievalAdapterError(
                    "hit count exceeds top_k: "
                    f"hit_count={len(hits)}, "
                    f"top_k={top_k}"
                )
            )

        expected_ranks = tuple(
            range(
                1,
                len(hits) + 1,
            )
        )

        actual_ranks = tuple(
            hit.rank
            for hit in hits
        )

        if actual_ranks != expected_ranks:
            raise (
                ContextExpandedRetrievalAdapterError(
                    "reranked hit ranks must start "
                    "at 1 and increase continuously"
                )
            )

        chunk_ids = tuple(
            hit.chunk_id
            for hit in hits
        )

        if len(chunk_ids) != len(
            set(chunk_ids)
        ):
            raise (
                ContextExpandedRetrievalAdapterError(
                    "reranked hits contain "
                    "duplicate chunk_id values"
                )
            )

        for hit in hits:
            if (
                hit.company_id
                != query.company_id
            ):
                raise (
                    ContextExpandedRetrievalAdapterError(
                        "hit company_id does not "
                        "match query"
                    )
                )

            if (
                hit.report_id
                != query.report_id
            ):
                raise (
                    ContextExpandedRetrievalAdapterError(
                        "hit report_id does not "
                        "match query"
                    )
                )

            if (
                hit.fiscal_year
                != query.fiscal_year
            ):
                raise (
                    ContextExpandedRetrievalAdapterError(
                        "hit fiscal_year does not "
                        "match query"
                    )
                )

            if (
                hit.report_type
                is not query.report_type
            ):
                raise (
                    ContextExpandedRetrievalAdapterError(
                        "hit report_type does not "
                        "match query"
                    )
                )