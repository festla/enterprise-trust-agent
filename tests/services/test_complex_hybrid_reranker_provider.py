from pathlib import Path

import pytest

from app.schemas.hybrid_retrieval import (
    HybridRetrievalHit,
)
from app.schemas.reranker import (
    RerankerRuntimeConfig,
    RerankerSpec,
)
from app.services.complex_hybrid_reranker_provider import (
    ComplexHybridRerankerProviderError,
    RoutedHybridRerankerHitProvider,
)
from app.services.complex_oracle_retrieval_adapter import (
    ComplexOracleRetrievalAdapter,
)
from app.services.complex_plan_eval_dataset import (
    load_complex_financial_eval_cases,
)
from app.services.complex_plan_oracle import (
    build_gold_oracle_rewrite,
)
from app.services.registry_loader import (
    load_registry_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REGISTRY_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "registries"
)

CASES_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "complex_plan"
    / "complex_plan_pilot_v1.jsonl"
)


@pytest.fixture(scope="module")
def bundle():
    registry_bundle, _, _, _ = (
        load_registry_bundle(
            companies_path=(
                REGISTRY_ROOT
                / "companies.yaml"
            ),
            reports_path=(
                REGISTRY_ROOT
                / "reports.yaml"
            ),
            metrics_path=(
                REGISTRY_ROOT
                / "metrics.yaml"
            ),
            evidences_path=(
                REGISTRY_ROOT
                / "evidences.yaml"
            ),
            financial_facts_path=(
                REGISTRY_ROOT
                / "financial_facts.yaml"
            ),
        )
    )

    return registry_bundle


@pytest.fixture(scope="module")
def query():
    cases = load_complex_financial_eval_cases(
        CASES_PATH
    )

    case = next(
        item
        for item in cases
        if item.case_id == "complex_001"
    )

    rewrite = build_gold_oracle_rewrite(
        case
    )

    return rewrite.retrieval_queries[0]


def make_hybrid_hit(
    *,
    query,
    pdf_page,
    rank,
    text,
):
    document_id = (
        f"doc_{query.report_id}_test"
    )

    page_id = (
        f"{document_id}_page_"
        f"{pdf_page:04d}"
    )

    chunk_id = (
        f"chunk_{query.report_id}_"
        f"{pdf_page}_{rank}"
    )

    rrf_score = (
        1.0 / (60 + rank)
        + 1.0 / (60 + rank)
    )

    return HybridRetrievalHit(
        rank=rank,
        retriever_type="hybrid_rrf",
        score_type="rrf",
        score=rrf_score,
        chunk_id=chunk_id,
        chunk_dataset_id=(
            f"chunk_dataset_"
            f"{query.report_id}_test"
        ),
        company_id=query.company_id,
        report_id=query.report_id,
        fiscal_year=query.fiscal_year,
        report_type=query.report_type,
        document_id=document_id,
        page_id=page_id,
        pdf_page=pdf_page,
        printed_page=max(
            1,
            pdf_page - 1,
        ),
        mapping_status="mapped",
        chunk_index=rank - 1,
        strategy="fixed_length",
        source_start_char=0,
        source_end_char=20,
        section_path=(),
        text=text,
        dense_rank=rank,
        bm25_rank=rank,
        rrf_score=rrf_score,
        source_retrievers=(
            "dense",
            "bm25",
        ),
    )


class FakeHybridRetriever:
    def __init__(self, hits):
        self.hits = tuple(hits)
        self.calls = []

    def search(
        self,
        *,
        query,
        top_k,
        filters,
    ):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "filters": filters,
            }
        )

        return self.hits[:top_k]


class FakeRerankerProvider:
    def __init__(self):
        self.calls = []

        self._spec = RerankerSpec(
            model_revision="test_revision",
        )

    @property
    def spec(self):
        return self._spec

    def score_pairs(self, pairs):
        pairs = tuple(pairs)
        self.calls.append(pairs)

        return tuple(
            (
                10.0
                if "正确证据" in text
                else 1.0
            )
            for _, text in pairs
        )


def build_provider(
    *,
    report_id,
    hybrid_retriever,
    reranker_provider=None,
    return_count=5,
):
    return RoutedHybridRerankerHitProvider(
        hybrid_retrievers_by_report_id={
            report_id: hybrid_retriever,
        },
        reranker_provider=(
            reranker_provider
            or FakeRerankerProvider()
        ),
        runtime_config=(
            RerankerRuntimeConfig(
                batch_size=2,
                device="cpu",
                local_files_only=True,
                rerank_candidate_count=5,
                return_count=return_count,
            )
        ),
        provider_id=(
            "hybrid_reranker_test_v1"
        ),
    )


def test_routes_query_and_reranks_hits(
    query,
) -> None:
    wrong_hit = make_hybrid_hit(
        query=query,
        pdf_page=100,
        rank=1,
        text="普通候选片段",
    )

    correct_hit = make_hybrid_hit(
        query=query,
        pdf_page=158,
        rank=2,
        text="正确证据：营业收入",
    )

    hybrid_retriever = (
        FakeHybridRetriever(
            (
                wrong_hit,
                correct_hit,
            )
        )
    )

    reranker_provider = (
        FakeRerankerProvider()
    )

    provider = build_provider(
        report_id=query.report_id,
        hybrid_retriever=(
            hybrid_retriever
        ),
        reranker_provider=(
            reranker_provider
        ),
    )

    hits = provider.search(
        query=query,
        top_k=2,
    )

    assert [
        hit.pdf_page
        for hit in hits
    ] == [
        158,
        100,
    ]

    assert [
        hit.rank
        for hit in hits
    ] == [
        1,
        2,
    ]

    assert len(
        hybrid_retriever.calls
    ) == 1

    call = hybrid_retriever.calls[0]

    assert call["query"] == (
        query.semantic_query
    )

    assert call["top_k"] == 5

    filters = call["filters"]

    assert filters.company_ids == (
        query.company_id,
    )

    assert filters.report_ids == (
        query.report_id,
    )

    assert filters.fiscal_years == (
        query.fiscal_year,
    )

    # 确认没有使用 Gold 页码过滤。
    assert filters.pdf_pages == ()


def test_limits_final_results_to_top_k(
    query,
) -> None:
    hits = tuple(
        make_hybrid_hit(
            query=query,
            pdf_page=100 + rank,
            rank=rank,
            text=f"候选 {rank}",
        )
        for rank in range(1, 5)
    )

    provider = build_provider(
        report_id=query.report_id,
        hybrid_retriever=(
            FakeHybridRetriever(hits)
        ),
    )

    results = provider.search(
        query=query,
        top_k=2,
    )

    assert len(results) == 2

    assert [
        hit.rank
        for hit in results
    ] == [1, 2]


def test_missing_report_route_is_rejected(
    query,
) -> None:
    provider = build_provider(
        report_id="other_report_2024",
        hybrid_retriever=(
            FakeHybridRetriever(())
        ),
    )

    with pytest.raises(
        ComplexHybridRerankerProviderError,
        match="没有为 Query 配置",
    ):
        provider.search(
            query=query,
            top_k=5,
        )


def test_top_k_cannot_exceed_return_count(
    query,
) -> None:
    provider = build_provider(
        report_id=query.report_id,
        hybrid_retriever=(
            FakeHybridRetriever(())
        ),
        return_count=3,
    )

    with pytest.raises(
        ComplexHybridRerankerProviderError,
        match="return_count",
    ):
        provider.search(
            query=query,
            top_k=5,
        )


def test_empty_hybrid_result_is_allowed(
    query,
) -> None:
    reranker_provider = (
        FakeRerankerProvider()
    )

    provider = build_provider(
        report_id=query.report_id,
        hybrid_retriever=(
            FakeHybridRetriever(())
        ),
        reranker_provider=(
            reranker_provider
        ),
    )

    results = provider.search(
        query=query,
        top_k=5,
    )

    assert results == ()

    # 没有候选时不调用 Cross-Encoder。
    assert reranker_provider.calls == []


def test_blank_provider_id_is_rejected(
    query,
) -> None:
    with pytest.raises(
        ComplexHybridRerankerProviderError,
        match="provider_id 不能为空",
    ):
        RoutedHybridRerankerHitProvider(
            hybrid_retrievers_by_report_id={
                query.report_id: (
                    FakeHybridRetriever(())
                ),
            },
            reranker_provider=(
                FakeRerankerProvider()
            ),
            runtime_config=(
                RerankerRuntimeConfig(
                    rerank_candidate_count=5,
                    return_count=5,
                )
            ),
            provider_id="   ",
        )


def test_provider_and_fact_resolver_work_together(
    query,
    bundle,
) -> None:
    wrong_hit = make_hybrid_hit(
        query=query,
        pdf_page=100,
        rank=1,
        text="普通候选片段",
    )

    correct_hit = make_hybrid_hit(
        query=query,
        pdf_page=158,
        rank=2,
        text="正确证据：营业收入",
    )

    hit_provider = build_provider(
        report_id=query.report_id,
        hybrid_retriever=(
            FakeHybridRetriever(
                (
                    wrong_hit,
                    correct_hit,
                )
            )
        ),
    )

    adapter = (
        ComplexOracleRetrievalAdapter(
            registry_bundle=bundle,
            hit_provider=hit_provider,
        )
    )

    trace = adapter.retrieve(
        query=query,
        top_k=5,
    )

    assert trace.status == "completed"

    assert trace.retrieved_fact_ids == (
        "fact_midea_group_2024_revenue",
    )

    assert trace.retrieved_evidence_ids == (
        "evidence_midea_group_2024_revenue",
    )

    assert trace.retrieved_chunk_ids[0] == (
        correct_hit.chunk_id
    )