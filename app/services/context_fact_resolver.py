from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.schemas.complex_plan_eval_result import (
    ComplexRetrievalQueryOutput,
)
from app.schemas.context_expansion import (
    AdjacentPageContextExpansion,
    ContextItemOrigin,
)
from app.schemas.enums import (
    ValidationStatus,
)
from app.schemas.evidence import (
    SourceEvidence,
)
from app.services.registry import (
    RegistryBundle,
)


EvidenceMatchMode = Literal[
    "chunk_id",
    "pdf_page",
]


class ContextFactResolverError(
    ValueError
):
    """Context-based fact resolution failed."""


@dataclass(
    frozen=True,
    slots=True,
)
class ResolvedContextFactSupport:
    """One Fact resolved from one context item."""

    fact_id: str
    evidence_id: str

    supporting_chunk_id: str
    supporting_context_order: int

    supporting_origin: (
        ContextItemOrigin
    )

    evidence_match_mode: (
        EvidenceMatchMode
    )


@dataclass(
    frozen=True,
    slots=True,
)
class ContextFactResolution:
    """Auditable result of resolving Facts from context."""

    query_id: str

    supports: tuple[
        ResolvedContextFactSupport,
        ...,
    ]

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ContextFactResolverError(
                "query_id cannot be empty"
            )

        fact_ids = tuple(
            support.fact_id
            for support in self.supports
        )

        if len(fact_ids) != len(
            set(fact_ids)
        ):
            raise ContextFactResolverError(
                "resolved supports contain "
                "duplicate fact_id values"
            )

    @property
    def fact_ids(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            support.fact_id
            for support in self.supports
        )

    @property
    def evidence_ids(
        self,
    ) -> tuple[str, ...]:
        return _unique_in_order(
            support.evidence_id
            for support in self.supports
        )

    @property
    def supporting_chunk_ids(
        self,
    ) -> tuple[str, ...]:
        return _unique_in_order(
            support.supporting_chunk_id
            for support in self.supports
        )

    @property
    def base_fact_ids(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            support.fact_id
            for support in self.supports
            if (
                support.supporting_origin
                == "retrieved"
            )
        )

    @property
    def expanded_fact_ids(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            support.fact_id
            for support in self.supports
            if (
                support.supporting_origin
                != "retrieved"
            )
        )


def resolve_facts_from_context(
    *,
    registry_bundle: RegistryBundle,
    query: ComplexRetrievalQueryOutput,
    expansion: (
        AdjacentPageContextExpansion
    ),
) -> ContextFactResolution:
    """
    Resolve verified FinancialFacts from expanded context.

    This mirrors the existing Registry Fact Resolver:
    - use structured Query identity to select candidates;
    - require verified Fact and Evidence;
    - prefer exact evidence.chunk_id matching;
    - fall back to report_id + pdf_page when chunk_id
      is not recorded.
    """

    _validate_query_expansion_identity(
        query=query,
        expansion=expansion,
    )

    candidate_facts = (
        registry_bundle
        .financial_facts
        .find(
            company_id=query.company_id,
            report_id=query.report_id,
            metric_id=query.metric_id,
            fiscal_year=query.fiscal_year,
            statement_scope=(
                query.statement_scope.value
            ),
        )
    )

    verified_facts_with_evidence: list[
        tuple[
            object,
            SourceEvidence,
        ]
    ] = []

    for fact in candidate_facts:
        if (
            fact.statement_type
            is not query.statement_type
        ):
            continue

        if (
            fact.validation_status
            is not ValidationStatus.VERIFIED
        ):
            continue

        evidence = (
            registry_bundle
            .evidences
            .get(
                fact.primary_evidence_id
            )
        )

        if evidence is None:
            continue

        if (
            evidence.validation_status
            is not ValidationStatus.VERIFIED
        ):
            continue

        verified_facts_with_evidence.append(
            (
                fact,
                evidence,
            )
        )

    supports: list[
        ResolvedContextFactSupport
    ] = []

    seen_fact_ids: set[str] = set()

    for item in expansion.items:
        for (
            fact,
            evidence,
        ) in verified_facts_with_evidence:
            if fact.fact_id in seen_fact_ids:
                continue

            match_mode = (
                _get_evidence_match_mode(
                    item_report_id=(
                        item.report_id
                    ),
                    item_chunk_id=(
                        item.chunk_id
                    ),
                    item_pdf_page=(
                        item.pdf_page
                    ),
                    evidence=evidence,
                )
            )

            if match_mode is None:
                continue

            seen_fact_ids.add(
                fact.fact_id
            )

            supports.append(
                ResolvedContextFactSupport(
                    fact_id=fact.fact_id,
                    evidence_id=(
                        evidence.evidence_id
                    ),
                    supporting_chunk_id=(
                        item.chunk_id
                    ),
                    supporting_context_order=(
                        item.context_order
                    ),
                    supporting_origin=(
                        item.origin
                    ),
                    evidence_match_mode=(
                        match_mode
                    ),
                )
            )

    return ContextFactResolution(
        query_id=query.query_id,
        supports=tuple(supports),
    )


def _validate_query_expansion_identity(
    *,
    query: ComplexRetrievalQueryOutput,
    expansion: (
        AdjacentPageContextExpansion
    ),
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
            expansion,
            field_name,
        )

        if query_value != expansion_value:
            raise ContextFactResolverError(
                "query and context expansion "
                "identity mismatch: "
                f"{field_name}"
            )


def _get_evidence_match_mode(
    *,
    item_report_id: str,
    item_chunk_id: str,
    item_pdf_page: int,
    evidence: SourceEvidence,
) -> EvidenceMatchMode | None:
    if (
        item_report_id
        != evidence.report_id
    ):
        return None

    if evidence.chunk_id is not None:
        if (
            item_chunk_id
            == evidence.chunk_id
        ):
            return "chunk_id"

        return None

    if (
        item_pdf_page
        == evidence.pdf_page
    ):
        return "pdf_page"

    return None


def _unique_in_order(
    values,
) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        result.append(value)

    return tuple(result)
