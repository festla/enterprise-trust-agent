from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest

from app.schemas.chunk import Chunk
from app.schemas.complex_plan_eval_result import (
    ComplexRetrievalQueryOutput,
)
from app.schemas.enums import (
    ChunkStrategy,
    PageContentType,
    PageMappingStatus,
    PageParseStatus,
    ReportType,
    StatementScope,
    StatementType,
    ValidationStatus,
)
from app.schemas.reranker import (
    RerankedRetrievalHit,
)
from app.services.context_expanded_oracle_retrieval_adapter import (
    ContextExpandedOracleRetrievalAdapter,
    ContextExpandedRetrievalAdapterError,
)


COMPANY_ID = "haier_smart_home"
REPORT_ID = "haier_smart_home_2024"

DOCUMENT_ID = (
    f"doc_{REPORT_ID}_{'a' * 24}"
)

PAGE_DATASET_ID = (
    f"page_dataset_{REPORT_ID}_"
    f"{'b' * 24}"
)

CHUNK_DATASET_ID = (
    f"chunk_dataset_{REPORT_ID}_"
    f"{'c' * 24}"
)

BASE_CHUNK_ID = (
    f"chunk_{REPORT_ID}_{'d' * 24}"
)

ADJACENT_CHUNK_ID = (
    f"chunk_{REPORT_ID}_{'e' * 24}"
)

FACT_ID = (
    "fact_haier_smart_home_"
    "2024_net_profit"
)

EVIDENCE_ID = (
    "evidence_haier_smart_home_"
    "2024_net_profit"
)


@dataclass(frozen=True)
class FakeReport:
    company_id: str
    fiscal_year: int
    report_type: ReportType


class FakeReportRegistry:
    def __init__(
        self,
        reports: dict[
            str,
            FakeReport,
        ],
    ) -> None:
        self._reports = reports

    def get(
        self,
        report_id: str,
    ):
        return self._reports.get(
            report_id
        )


@dataclass(frozen=True)
class FakeFact:
    fact_id: str
    statement_type: StatementType
    validation_status: ValidationStatus
    primary_evidence_id: str


class FakeFinancialFactRegistry:
    def __init__(
        self,
        facts: tuple[
            FakeFact,
            ...,
        ],
    ) -> None:
        self._facts = facts

    def find(
        self,
        **filters,
    ):
        return self._facts


@dataclass(frozen=True)
class FakeEvidence:
    evidence_id: str
    report_id: str
    chunk_id: str | None
    pdf_page: int
    validation_status: ValidationStatus


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
    reports: FakeReportRegistry

    financial_facts: (
        FakeFinancialFactRegistry
    )

    evidences: FakeEvidenceRegistry


@dataclass(frozen=True)
class FakeChunkManifest:
    report_id: str
    company_id: str
    fiscal_year: int
    report_type: ReportType


@dataclass(frozen=True)
class FakeChunkSource:
    manifest: FakeChunkManifest
    chunks: tuple[
        Chunk,
        ...,
    ]


class FakeHitProvider:
    def __init__(
        self,
        *,
        hits: tuple[
            RerankedRetrievalHit,
            ...,
        ] = (),
        provider_id: str = (
            "fake_hybrid_reranker_v1"
        ),
        error: Exception | None = None,
    ) -> None:
        self._hits = hits
        self._provider_id = provider_id
        self._error = error
        self.calls = []

    @property
    def provider_id(self) -> str:
        return self._provider_id

    def search(
        self,
        *,
        query,
        top_k,
    ):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
            }
        )

        if self._error is not None:
            raise self._error

        return self._hits


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


def build_chunk(
    *,
    chunk_id: str,
    pdf_page: int,
    chunk_index: int,
    text: str,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        chunk_dataset_id=(
            CHUNK_DATASET_ID
        ),
        page_dataset_id=(
            PAGE_DATASET_ID
        ),
        company_id=COMPANY_ID,
        report_id=REPORT_ID,
        fiscal_year=2024,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        document_id=DOCUMENT_ID,
        page_id=(
            f"{DOCUMENT_ID}_page_"
            f"{pdf_page:04d}"
        ),
        pdf_page=pdf_page,
        printed_page=pdf_page - 2,
        mapping_status=(
            PageMappingStatus.MAPPED
        ),
        content_type=(
            PageContentType.TEXT
        ),
        parse_status=(
            PageParseStatus.SUCCESS
        ),
        chunk_index=chunk_index,
        strategy=(
            ChunkStrategy.FIXED_LENGTH
        ),
        chunker_version=(
            "fixed_length_chunker_v1"
        ),
        source_text_field=(
            "normalized_text"
        ),
        source_start_char=0,
        source_end_char=len(text),
        text=text,
        char_count=len(text),
        text_sha256=(
            hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()
        ),
        paragraph_start_index=None,
        paragraph_end_index=None,
        section_path=(),
        section_source_page_id=None,
        section_inherited=False,
    )


def build_base_chunk() -> Chunk:
    return build_chunk(
        chunk_id=BASE_CHUNK_ID,
        pdf_page=121,
        chunk_index=10,
        text=(
            "consolidated income "
            "statement header"
        ),
    )


def build_adjacent_chunk() -> Chunk:
    return build_chunk(
        chunk_id=(
            ADJACENT_CHUNK_ID
        ),
        pdf_page=122,
        chunk_index=11,
        text=(
            "net profit "
            "19575612501.68"
        ),
    )


def build_hit(
    *,
    chunk: Chunk,
    rank: int,
) -> RerankedRetrievalHit:
    reranker_score = (
        2.0 - rank * 0.1
    )

    rrf_score = (
        2.0 / (60 + rank)
    )

    return RerankedRetrievalHit(
        rank=rank,
        retriever_type=(
            "hybrid_reranker"
        ),
        score_type=(
            "cross_encoder_logit"
        ),
        score=reranker_score,
        chunk_id=chunk.chunk_id,
        chunk_dataset_id=(
            chunk.chunk_dataset_id
        ),
        company_id=chunk.company_id,
        report_id=chunk.report_id,
        fiscal_year=chunk.fiscal_year,
        report_type=chunk.report_type,
        document_id=chunk.document_id,
        page_id=chunk.page_id,
        pdf_page=chunk.pdf_page,
        printed_page=(
            chunk.printed_page
        ),
        mapping_status=(
            chunk.mapping_status
        ),
        chunk_index=chunk.chunk_index,
        strategy=chunk.strategy,
        source_start_char=(
            chunk.source_start_char
        ),
        source_end_char=(
            chunk.source_end_char
        ),
        section_path=chunk.section_path,
        text=chunk.text,
        dense_rank=rank,
        bm25_rank=rank,
        rrf_rank=rank,
        rrf_score=rrf_score,
        reranker_score=(
            reranker_score
        ),
        source_retrievers=(
            "dense",
            "bm25",
        ),
    )


def build_bundle(
    *,
    evidence_chunk_id: (
        str | None
    ) = None,
    evidence_pdf_page: int = 122,
) -> FakeRegistryBundle:
    report = FakeReport(
        company_id=COMPANY_ID,
        fiscal_year=2024,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
    )

    fact = FakeFact(
        fact_id=FACT_ID,
        statement_type=(
            StatementType.INCOME_STATEMENT
        ),
        validation_status=(
            ValidationStatus.VERIFIED
        ),
        primary_evidence_id=(
            EVIDENCE_ID
        ),
    )

    evidence = FakeEvidence(
        evidence_id=EVIDENCE_ID,
        report_id=REPORT_ID,
        chunk_id=evidence_chunk_id,
        pdf_page=evidence_pdf_page,
        validation_status=(
            ValidationStatus.VERIFIED
        ),
    )

    return FakeRegistryBundle(
        reports=FakeReportRegistry(
            {
                REPORT_ID: report,
            }
        ),
        financial_facts=(
            FakeFinancialFactRegistry(
                (fact,)
            )
        ),
        evidences=(
            FakeEvidenceRegistry(
                (evidence,)
            )
        ),
    )


def build_source(
    *,
    report_id: str = REPORT_ID,
    company_id: str = COMPANY_ID,
    chunks: tuple[
        Chunk,
        ...,
    ] | None = None,
) -> FakeChunkSource:
    actual_chunks = (
        chunks
        if chunks is not None
        else (
            build_base_chunk(),
            build_adjacent_chunk(),
        )
    )

    return FakeChunkSource(
        manifest=FakeChunkManifest(
            report_id=report_id,
            company_id=company_id,
            fiscal_year=2024,
            report_type=(
                ReportType.ANNUAL_REPORT
            ),
        ),
        chunks=actual_chunks,
    )


def build_adapter(
    *,
    bundle: (
        FakeRegistryBundle | None
    ) = None,
    provider: (
        FakeHitProvider | None
    ) = None,
    sources=None,
):
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

    return (
        ContextExpandedOracleRetrievalAdapter(
            registry_bundle=(
                bundle
                if bundle is not None
                else build_bundle()
            ),
            hit_provider=(
                actual_provider
            ),
            chunk_sources_by_report_id=(
                actual_sources
            ),
        )
    )


def test_resolve_fact_from_adjacent_context(
) -> None:
    base_chunk = build_base_chunk()
    adjacent_chunk = (
        build_adjacent_chunk()
    )

    provider = FakeHitProvider(
        hits=(
            build_hit(
                chunk=base_chunk,
                rank=1,
            ),
        )
    )

    adapter = build_adapter(
        provider=provider,
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
    assert trace.top_k == 5

    assert trace.retrieved_fact_ids == (
        FACT_ID,
    )

    assert (
        trace.retrieved_evidence_ids
        == (
            EVIDENCE_ID,
        )
    )

    assert trace.retrieved_chunk_ids == (
        BASE_CHUNK_ID,
        ADJACENT_CHUNK_ID,
    )

    assert provider.calls[0]["top_k"] == 5

    assert (
        len(
            adapter
            .expansion_audit_records
        )
        == 1
    )

    expansion = (
        adapter
        .expansion_audit_records[0]
    )

    assert expansion.base_top_k == 5

    assert expansion.base_chunk_ids == (
        BASE_CHUNK_ID,
    )

    assert expansion.expanded_chunk_ids == (
        ADJACENT_CHUNK_ID,
    )

    resolution = (
        adapter
        .resolution_audit_records[0]
    )

    assert resolution.base_fact_ids == ()

    assert resolution.expanded_fact_ids == (
        FACT_ID,
    )

    assert (
        resolution.supports[0]
        .supporting_origin
        == "adjacent_page"
    )

    assert (
        adapter.retriever_id
        == (
            "fake_hybrid_reranker_v1_"
            "adjacent_page_context_v1_"
            "registry_context_"
            "fact_resolver_v1"
        )
    )


def test_resolve_base_fact_before_expansion(
) -> None:
    base_chunk = build_base_chunk()

    adapter = build_adapter(
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
    )

    trace = adapter.retrieve(
        query=build_query(),
        top_k=5,
    )

    assert trace.status == "completed"

    resolution = (
        adapter
        .resolution_audit_records[0]
    )

    assert resolution.base_fact_ids == (
        FACT_ID,
    )

    assert (
        resolution.expanded_fact_ids
        == ()
    )

    assert (
        resolution.supports[0]
        .supporting_origin
        == "retrieved"
    )


def test_clear_audit_records() -> None:
    adapter = build_adapter()

    trace = adapter.retrieve(
        query=build_query(),
        top_k=5,
    )

    assert trace.status == "completed"

    assert (
        adapter.expansion_audit_records
    )

    assert (
        adapter.resolution_audit_records
    )

    adapter.clear_audit_records()

    assert (
        adapter.expansion_audit_records
        == ()
    )

    assert (
        adapter.resolution_audit_records
        == ()
    )


def test_empty_hits_preserve_completed_trace(
) -> None:
    adapter = build_adapter(
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
        adapter.expansion_audit_records
        == ()
    )

    assert (
        adapter.resolution_audit_records
        == ()
    )


def test_provider_error_returns_failed_trace(
) -> None:
    adapter = build_adapter(
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

    assert trace.retrieved_chunk_ids == ()


def test_missing_chunk_source_returns_failure(
) -> None:
    other_report_id = (
        "other_company_2024"
    )

    other_source = build_source(
        report_id=other_report_id,
        company_id="other_company",
        chunks=(),
    )

    adapter = build_adapter(
        sources={
            other_report_id: (
                other_source
            ),
        }
    )

    trace = adapter.retrieve(
        query=build_query(),
        top_k=5,
    )

    assert trace.status == "failed"

    assert "no ChunkDataset source" in (
        trace.error_message
    )


def test_reject_noncontinuous_hit_ranks(
) -> None:
    base_chunk = build_base_chunk()

    adapter = build_adapter(
        provider=FakeHitProvider(
            hits=(
                build_hit(
                    chunk=base_chunk,
                    rank=2,
                ),
            )
        )
    )

    trace = adapter.retrieve(
        query=build_query(),
        top_k=5,
    )

    assert trace.status == "failed"

    assert "ranks" in trace.error_message


def test_reject_hit_count_above_top_k(
) -> None:
    first_chunk = build_base_chunk()

    second_chunk = build_chunk(
        chunk_id=(
            f"chunk_{REPORT_ID}_"
            f"{'f' * 24}"
        ),
        pdf_page=130,
        chunk_index=20,
        text="another retrieved chunk",
    )

    adapter = build_adapter(
        provider=FakeHitProvider(
            hits=(
                build_hit(
                    chunk=first_chunk,
                    rank=1,
                ),
                build_hit(
                    chunk=second_chunk,
                    rank=2,
                ),
            )
        ),
        sources={
            REPORT_ID: build_source(
                chunks=(
                    first_chunk,
                    second_chunk,
                )
            ),
        },
    )

    trace = adapter.retrieve(
        query=build_query(),
        top_k=1,
    )

    assert trace.status == "failed"

    assert "exceeds top_k" in (
        trace.error_message
    )


def test_reject_source_route_mismatch(
) -> None:
    mismatched_source = build_source(
        report_id=(
            "different_report_2024"
        ),
    )

    with pytest.raises(
        ContextExpandedRetrievalAdapterError,
        match="report_id mismatch",
    ):
        build_adapter(
            sources={
                REPORT_ID: (
                    mismatched_source
                ),
            }
        )


def test_blank_provider_id_returns_failure(
) -> None:
    base_chunk = build_base_chunk()

    adapter = build_adapter(
        provider=FakeHitProvider(
            hits=(
                build_hit(
                    chunk=base_chunk,
                    rank=1,
                ),
            ),
            provider_id="   ",
        )
    )

    trace = adapter.retrieve(
        query=build_query(),
        top_k=5,
    )

    assert trace.status == "failed"

    assert "provider_id" in (
        trace.error_message
    )