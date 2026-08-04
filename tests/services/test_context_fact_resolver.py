from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.schemas.complex_plan_eval_result import (
    ComplexRetrievalQueryOutput,
)
from app.schemas.context_expansion import (
    AdjacentPageContextExpansion,
    ContextExpansionItem,
)
from app.schemas.enums import (
    ReportType,
    StatementScope,
    StatementType,
    ValidationStatus,
)
from app.services.context_fact_resolver import (
    ContextFactResolverError,
    resolve_facts_from_context,
)


COMPANY_ID = "haier_smart_home"
REPORT_ID = "haier_smart_home_2024"

DOCUMENT_ID = (
    f"doc_{REPORT_ID}_{'a' * 24}"
)

BASE_CHUNK_ID = (
    f"chunk_{REPORT_ID}_{'b' * 24}"
)

ADJACENT_CHUNK_ID = (
    f"chunk_{REPORT_ID}_{'c' * 24}"
)

UNKNOWN_CHUNK_ID = (
    f"chunk_{REPORT_ID}_{'d' * 24}"
)

BASE_TEXT = "consolidated statement header"

ADJACENT_TEXT = (
    "net profit 19575612501.68"
)


@dataclass(frozen=True)
class FakeFact:
    fact_id: str
    statement_type: StatementType
    validation_status: ValidationStatus
    primary_evidence_id: str


@dataclass(frozen=True)
class FakeEvidence:
    evidence_id: str
    report_id: str
    chunk_id: str | None
    pdf_page: int
    validation_status: ValidationStatus


class FakeFinancialFactRegistry:
    def __init__(
        self,
        facts: tuple[
            FakeFact,
            ...,
        ],
    ) -> None:
        self.facts = facts
        self.last_filters = None

    def find(
        self,
        **filters,
    ):
        self.last_filters = filters
        return self.facts


class FakeEvidenceRegistry:
    def __init__(
        self,
        evidences: tuple[
            FakeEvidence,
            ...,
        ],
    ) -> None:
        self._evidences = {
            evidence.evidence_id: evidence
            for evidence in evidences
        }

    def get(
        self,
        evidence_id: str,
    ):
        return self._evidences.get(
            evidence_id
        )


@dataclass
class FakeRegistryBundle:
    financial_facts: (
        FakeFinancialFactRegistry
    )

    evidences: FakeEvidenceRegistry


def build_query(
) -> ComplexRetrievalQueryOutput:
    return ComplexRetrievalQueryOutput(
        query_id="q3",
        semantic_query=(
            "net profit fiscal year 2024"
        ),
        company_id=COMPANY_ID,
        report_id=REPORT_ID,
        metric_id="net_profit",
        fiscal_year=2024,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        statement_type=(
            StatementType.INCOME_STATEMENT
        ),
        statement_scope=(
            StatementScope.CONSOLIDATED
        ),
    )


def build_base_item(
) -> ContextExpansionItem:
    return ContextExpansionItem(
        context_order=1,
        origin="retrieved",
        company_id=COMPANY_ID,
        report_id=REPORT_ID,
        fiscal_year=2024,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        document_id=DOCUMENT_ID,
        page_id=(
            f"{DOCUMENT_ID}_page_0121"
        ),
        pdf_page=121,
        printed_page=119,
        chunk_id=BASE_CHUNK_ID,
        chunk_index=10,
        text=BASE_TEXT,
        text_char_count=len(
            BASE_TEXT
        ),
        retrieval_rank=1,
        retrieval_score=1.5,
        anchor_chunk_id=(
            BASE_CHUNK_ID
        ),
        anchor_retrieval_rank=1,
        page_distance=0,
    )


def build_adjacent_item(
) -> ContextExpansionItem:
    return ContextExpansionItem(
        context_order=2,
        origin="adjacent_page",
        company_id=COMPANY_ID,
        report_id=REPORT_ID,
        fiscal_year=2024,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        document_id=DOCUMENT_ID,
        page_id=(
            f"{DOCUMENT_ID}_page_0122"
        ),
        pdf_page=122,
        printed_page=120,
        chunk_id=(
            ADJACENT_CHUNK_ID
        ),
        chunk_index=11,
        text=ADJACENT_TEXT,
        text_char_count=len(
            ADJACENT_TEXT
        ),
        retrieval_rank=None,
        retrieval_score=None,
        anchor_chunk_id=(
            BASE_CHUNK_ID
        ),
        anchor_retrieval_rank=1,
        page_distance=1,
    )


def build_expansion(
) -> AdjacentPageContextExpansion:
    base_item = build_base_item()
    adjacent_item = (
        build_adjacent_item()
    )

    return AdjacentPageContextExpansion(
        schema_version=1,
        strategy_id=(
            "adjacent_page_context_v1"
        ),
        query_id="q3",
        original_query=(
            "What is the net profit?"
        ),
        semantic_query=(
            "net profit fiscal year 2024"
        ),
        company_id=COMPANY_ID,
        report_id=REPORT_ID,
        fiscal_year=2024,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        document_id=DOCUMENT_ID,
        base_top_k=5,
        page_window=1,
        items=(
            base_item,
            adjacent_item,
        ),
        base_chunk_ids=(
            BASE_CHUNK_ID,
        ),
        expanded_chunk_ids=(
            ADJACENT_CHUNK_ID,
        ),
        used_chunk_ids=(
            BASE_CHUNK_ID,
            ADJACENT_CHUNK_ID,
        ),
        base_item_count=1,
        expanded_item_count=1,
        total_item_count=2,
        duplicate_chunk_count=0,
        base_char_count=len(
            BASE_TEXT
        ),
        expanded_char_count=len(
            ADJACENT_TEXT
        ),
        total_char_count=(
            len(BASE_TEXT)
            + len(ADJACENT_TEXT)
        ),
    )


def build_fact(
    *,
    fact_id: str = (
        "fact_haier_smart_home_"
        "2024_net_profit"
    ),
    evidence_id: str = (
        "evidence_haier_smart_home_"
        "2024_net_profit"
    ),
    statement_type: StatementType = (
        StatementType.INCOME_STATEMENT
    ),
    status: ValidationStatus = (
        ValidationStatus.VERIFIED
    ),
) -> FakeFact:
    return FakeFact(
        fact_id=fact_id,
        statement_type=statement_type,
        validation_status=status,
        primary_evidence_id=(
            evidence_id
        ),
    )


def build_evidence(
    *,
    evidence_id: str = (
        "evidence_haier_smart_home_"
        "2024_net_profit"
    ),
    chunk_id: str | None = None,
    pdf_page: int = 122,
    status: ValidationStatus = (
        ValidationStatus.VERIFIED
    ),
) -> FakeEvidence:
    return FakeEvidence(
        evidence_id=evidence_id,
        report_id=REPORT_ID,
        chunk_id=chunk_id,
        pdf_page=pdf_page,
        validation_status=status,
    )


def build_bundle(
    *,
    facts: tuple[
        FakeFact,
        ...,
    ] | None = None,
    evidences: tuple[
        FakeEvidence,
        ...,
    ] | None = None,
) -> FakeRegistryBundle:
    actual_facts = (
        facts
        if facts is not None
        else (
            build_fact(),
        )
    )

    actual_evidences = (
        evidences
        if evidences is not None
        else (
            build_evidence(),
        )
    )

    return FakeRegistryBundle(
        financial_facts=(
            FakeFinancialFactRegistry(
                actual_facts
            )
        ),
        evidences=(
            FakeEvidenceRegistry(
                actual_evidences
            )
        ),
    )


def test_resolve_fact_from_adjacent_page(
) -> None:
    bundle = build_bundle()

    resolution = (
        resolve_facts_from_context(
            registry_bundle=bundle,
            query=build_query(),
            expansion=build_expansion(),
        )
    )

    assert resolution.fact_ids == (
        "fact_haier_smart_home_"
        "2024_net_profit",
    )

    assert resolution.evidence_ids == (
        "evidence_haier_smart_home_"
        "2024_net_profit",
    )

    assert resolution.base_fact_ids == ()

    assert resolution.expanded_fact_ids == (
        "fact_haier_smart_home_"
        "2024_net_profit",
    )

    assert (
        resolution.supporting_chunk_ids
        == (
            ADJACENT_CHUNK_ID,
        )
    )

    support = resolution.supports[0]

    assert (
        support.supporting_origin
        == "adjacent_page"
    )

    assert (
        support.evidence_match_mode
        == "pdf_page"
    )

    assert (
        support.supporting_context_order
        == 2
    )

    assert (
        bundle
        .financial_facts
        .last_filters
        == {
            "company_id": COMPANY_ID,
            "report_id": REPORT_ID,
            "metric_id": "net_profit",
            "fiscal_year": 2024,
            "statement_scope": (
                "consolidated"
            ),
        }
    )


def test_resolve_fact_from_base_chunk_id(
) -> None:
    bundle = build_bundle(
        evidences=(
            build_evidence(
                chunk_id=BASE_CHUNK_ID,
                pdf_page=121,
            ),
        )
    )

    resolution = (
        resolve_facts_from_context(
            registry_bundle=bundle,
            query=build_query(),
            expansion=build_expansion(),
        )
    )

    assert resolution.base_fact_ids == (
        "fact_haier_smart_home_"
        "2024_net_profit",
    )

    assert (
        resolution.expanded_fact_ids
        == ()
    )

    support = resolution.supports[0]

    assert (
        support.supporting_origin
        == "retrieved"
    )

    assert (
        support.evidence_match_mode
        == "chunk_id"
    )


def test_chunk_id_prevents_page_fallback(
) -> None:
    bundle = build_bundle(
        evidences=(
            build_evidence(
                chunk_id=UNKNOWN_CHUNK_ID,
                pdf_page=122,
            ),
        )
    )

    resolution = (
        resolve_facts_from_context(
            registry_bundle=bundle,
            query=build_query(),
            expansion=build_expansion(),
        )
    )

    assert resolution.fact_ids == ()
    assert resolution.evidence_ids == ()
    assert resolution.supports == ()


def test_skip_unverified_fact() -> None:
    bundle = build_bundle(
        facts=(
            build_fact(
                status=(
                    ValidationStatus.PENDING
                ),
            ),
        )
    )

    resolution = (
        resolve_facts_from_context(
            registry_bundle=bundle,
            query=build_query(),
            expansion=build_expansion(),
        )
    )

    assert resolution.fact_ids == ()


def test_skip_unverified_evidence(
) -> None:
    bundle = build_bundle(
        evidences=(
            build_evidence(
                status=(
                    ValidationStatus.PENDING
                ),
            ),
        )
    )

    resolution = (
        resolve_facts_from_context(
            registry_bundle=bundle,
            query=build_query(),
            expansion=build_expansion(),
        )
    )

    assert resolution.fact_ids == ()


def test_skip_statement_type_mismatch(
) -> None:
    bundle = build_bundle(
        facts=(
            build_fact(
                statement_type=(
                    StatementType.BALANCE_SHEET
                ),
            ),
        )
    )

    resolution = (
        resolve_facts_from_context(
            registry_bundle=bundle,
            query=build_query(),
            expansion=build_expansion(),
        )
    )

    assert resolution.fact_ids == ()


def test_reject_query_expansion_mismatch(
) -> None:
    mismatched_expansion = (
        build_expansion().model_copy(
            update={
                "semantic_query": (
                    "different query"
                ),
            }
        )
    )

    with pytest.raises(
        ContextFactResolverError,
        match="semantic_query",
    ):
        resolve_facts_from_context(
            registry_bundle=(
                build_bundle()
            ),
            query=build_query(),
            expansion=(
                mismatched_expansion
            ),
        )


def test_return_empty_without_candidates(
) -> None:
    bundle = build_bundle(
        facts=(),
        evidences=(),
    )

    resolution = (
        resolve_facts_from_context(
            registry_bundle=bundle,
            query=build_query(),
            expansion=build_expansion(),
        )
    )

    assert resolution.query_id == "q3"
    assert resolution.fact_ids == ()
    assert resolution.evidence_ids == ()
    assert resolution.supporting_chunk_ids == ()


def test_deduplicate_shared_evidence_ids(
) -> None:
    evidence_id = (
        "evidence_haier_smart_home_"
        "2024_shared_net_profit"
    )

    first_fact = build_fact(
        fact_id=(
            "fact_haier_smart_home_"
            "2024_net_profit"
        ),
        evidence_id=evidence_id,
    )

    second_fact = build_fact(
        fact_id=(
            "fact_haier_smart_home_"
            "2024_adjusted_net_profit"
        ),
        evidence_id=evidence_id,
    )

    shared_evidence = build_evidence(
        evidence_id=evidence_id,
        pdf_page=122,
    )

    resolution = (
        resolve_facts_from_context(
            registry_bundle=(
                build_bundle(
                    facts=(
                        first_fact,
                        second_fact,
                    ),
                    evidences=(
                        shared_evidence,
                    ),
                )
            ),
            query=build_query(),
            expansion=build_expansion(),
        )
    )

    assert resolution.fact_ids == (
        first_fact.fact_id,
        second_fact.fact_id,
    )

    assert resolution.evidence_ids == (
        evidence_id,
    )

    assert (
        resolution.supporting_chunk_ids
        == (
            ADJACENT_CHUNK_ID,
        )
    )

    assert resolution.expanded_fact_ids == (
        first_fact.fact_id,
        second_fact.fact_id,
    )