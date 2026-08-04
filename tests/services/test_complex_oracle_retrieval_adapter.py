from pathlib import Path

import pytest

from app.schemas.reranker import (
    RerankedRetrievalHit,
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
def cases_by_id():
    cases = load_complex_financial_eval_cases(
        CASES_PATH
    )

    return {
        case.case_id: case
        for case in cases
    }


def make_hit(
    *,
    query,
    pdf_page,
    rank=1,
    chunk_id=None,
    company_id=None,
    report_id=None,
):
    actual_company_id = (
        company_id or query.company_id
    )

    actual_report_id = (
        report_id or query.report_id
    )

    document_id = (
        f"doc_{actual_report_id}_test"
    )

    page_id = (
        f"{document_id}_page_"
        f"{pdf_page:04d}"
    )

    actual_chunk_id = (
        chunk_id
        or (
            f"chunk_{actual_report_id}_"
            f"{pdf_page}_{rank}"
        )
    )

    score = 10.0 - rank

    return RerankedRetrievalHit(
        rank=rank,
        retriever_type=(
            "hybrid_reranker"
        ),
        score_type=(
            "cross_encoder_logit"
        ),
        score=score,
        chunk_id=actual_chunk_id,
        chunk_dataset_id=(
            f"chunk_dataset_"
            f"{actual_report_id}_test"
        ),
        company_id=actual_company_id,
        report_id=actual_report_id,
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
        text="用于测试的财务报表片段。",
        dense_rank=rank,
        bm25_rank=rank,
        rrf_rank=rank,
        rrf_score=(
            1.0 / (60 + rank)
        ),
        reranker_score=score,
        source_retrievers=(
            "dense",
            "bm25",
        ),
    )


class FakeHitProvider:
    def __init__(
        self,
        hits_by_query_id,
    ):
        self.hits_by_query_id = (
            hits_by_query_id
        )

    @property
    def provider_id(self) -> str:
        return "fake_hybrid_reranker_v1"

    def search(
        self,
        *,
        query,
        top_k,
    ):
        return tuple(
            self.hits_by_query_id.get(
                query.query_id,
                (),
            )
        )[:top_k]


class ExplodingHitProvider:
    @property
    def provider_id(self) -> str:
        return "exploding_provider_v1"

    def search(
        self,
        *,
        query,
        top_k,
    ):
        raise RuntimeError(
            "模拟检索器异常"
        )


def get_query(
    cases_by_id,
    case_id,
    query_id,
):
    case = cases_by_id[case_id]

    rewrite = build_gold_oracle_rewrite(
        case
    )

    return next(
        query
        for query
        in rewrite.retrieval_queries
        if query.query_id == query_id
    )


def test_resolves_page_hit_to_fact_and_evidence(
    bundle,
    cases_by_id,
) -> None:
    query = get_query(
        cases_by_id,
        "complex_001",
        "q1",
    )

    hit = make_hit(
        query=query,
        pdf_page=158,
    )

    adapter = (
        ComplexOracleRetrievalAdapter(
            registry_bundle=bundle,
            hit_provider=FakeHitProvider(
                {"q1": (hit,)}
            ),
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

    assert trace.retrieved_chunk_ids == (
        hit.chunk_id,
    )


def test_wrong_page_does_not_resolve_fact(
    bundle,
    cases_by_id,
) -> None:
    query = get_query(
        cases_by_id,
        "complex_001",
        "q1",
    )

    hit = make_hit(
        query=query,
        pdf_page=100,
    )

    adapter = (
        ComplexOracleRetrievalAdapter(
            registry_bundle=bundle,
            hit_provider=FakeHitProvider(
                {"q1": (hit,)}
            ),
        )
    )

    trace = adapter.retrieve(
        query=query,
        top_k=5,
    )

    assert trace.status == "completed"
    assert trace.retrieved_fact_ids == ()
    assert trace.retrieved_evidence_ids == ()
    assert trace.retrieved_chunk_ids == (
        hit.chunk_id,
    )


def test_metric_distinguishes_facts_on_same_page(
    bundle,
    cases_by_id,
) -> None:
    revenue_query = get_query(
        cases_by_id,
        "complex_002",
        "q1",
    )

    cost_query = get_query(
        cases_by_id,
        "complex_002",
        "q2",
    )

    revenue_hit = make_hit(
        query=revenue_query,
        pdf_page=116,
    )

    cost_hit = make_hit(
        query=cost_query,
        pdf_page=116,
    )

    adapter = (
        ComplexOracleRetrievalAdapter(
            registry_bundle=bundle,
            hit_provider=FakeHitProvider(
                {
                    "q1": (revenue_hit,),
                    "q2": (cost_hit,),
                }
            ),
        )
    )

    revenue_trace = adapter.retrieve(
        query=revenue_query,
        top_k=5,
    )

    cost_trace = adapter.retrieve(
        query=cost_query,
        top_k=5,
    )

    assert revenue_trace.retrieved_fact_ids == (
        "fact_hisense_home_2024_revenue",
    )

    assert cost_trace.retrieved_fact_ids == (
        "fact_hisense_home_2024_operating_cost",
    )


def test_rejects_hit_from_wrong_report(
    bundle,
    cases_by_id,
) -> None:
    query = get_query(
        cases_by_id,
        "complex_001",
        "q1",
    )

    hit = make_hit(
        query=query,
        pdf_page=158,
        company_id="gree_electric",
        report_id="gree_electric_2024",
    )

    adapter = (
        ComplexOracleRetrievalAdapter(
            registry_bundle=bundle,
            hit_provider=FakeHitProvider(
                {"q1": (hit,)}
            ),
        )
    )

    trace = adapter.retrieve(
        query=query,
        top_k=5,
    )

    assert trace.status == "failed"
    assert "报告身份不一致" in (
        trace.error_message or ""
    )


def test_rejects_non_continuous_ranks(
    bundle,
    cases_by_id,
) -> None:
    query = get_query(
        cases_by_id,
        "complex_001",
        "q1",
    )

    hit_1 = make_hit(
        query=query,
        pdf_page=100,
        rank=1,
    )

    hit_3 = make_hit(
        query=query,
        pdf_page=158,
        rank=3,
    )

    adapter = (
        ComplexOracleRetrievalAdapter(
            registry_bundle=bundle,
            hit_provider=FakeHitProvider(
                {
                    "q1": (
                        hit_1,
                        hit_3,
                    )
                }
            ),
        )
    )

    trace = adapter.retrieve(
        query=query,
        top_k=5,
    )

    assert trace.status == "failed"
    assert "连续递增" in (
        trace.error_message or ""
    )


def test_provider_exception_becomes_failed_trace(
    bundle,
    cases_by_id,
) -> None:
    query = get_query(
        cases_by_id,
        "complex_001",
        "q1",
    )

    adapter = (
        ComplexOracleRetrievalAdapter(
            registry_bundle=bundle,
            hit_provider=(
                ExplodingHitProvider()
            ),
        )
    )

    trace = adapter.retrieve(
        query=query,
        top_k=5,
    )

    assert trace.status == "failed"
    assert "模拟检索器异常" in (
        trace.error_message or ""
    )


def test_unknown_metric_returns_no_fact(
    bundle,
    cases_by_id,
) -> None:
    query = get_query(
        cases_by_id,
        "complex_001",
        "q1",
    ).model_copy(
        update={
            "metric_id": (
                "metric_not_registered"
            ),
        }
    )

    hit = make_hit(
        query=query,
        pdf_page=158,
    )

    adapter = (
        ComplexOracleRetrievalAdapter(
            registry_bundle=bundle,
            hit_provider=FakeHitProvider(
                {"q1": (hit,)}
            ),
        )
    )

    trace = adapter.retrieve(
        query=query,
        top_k=5,
    )

    assert trace.status == "completed"
    assert trace.retrieved_fact_ids == ()


def test_retriever_id_contains_provider_and_resolver(
    bundle,
) -> None:
    adapter = (
        ComplexOracleRetrievalAdapter(
            registry_bundle=bundle,
            hit_provider=FakeHitProvider(
                {}
            ),
        )
    )

    assert adapter.retriever_id == (
        "fake_hybrid_reranker_v1_"
        "registry_fact_resolver_v1"
    )