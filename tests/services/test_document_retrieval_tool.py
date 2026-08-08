from __future__ import annotations

from typing import Any

import pytest

from app.schemas.hybrid_retrieval import (
    HybridRetrievalHit,
)
from app.schemas.reranker import (
    RerankerRuntimeConfig,
    RerankerSpec,
)
from app.schemas.tool_registry import (
    DocumentEvidenceQuery,
    RetrieveDocumentsInput,
    RetrieveDocumentsOutput,
)
from app.services.document_retrieval_tool import (
    DocumentRetrievalToolError,
    RetrieveDocumentsTool,
    RoutedDocumentRerankerProvider,
    register_retrieve_documents_tool,
)
from app.services.tool_registry import (
    ToolExecutor,
    ToolRegistry,
)


# ============================================================
# 【复制即可】
#
# 这里只构造测试 Hit。
# Week 5 已经单独测试过 HybridRetrievalHit。
# ============================================================


def _build_query(
) -> DocumentEvidenceQuery:
    return DocumentEvidenceQuery(
        query_id="q1",
        semantic_query=(
            "美的集团营业收入增长原因"
        ),
        company_id="midea_group",
        report_id="midea_group_2024",
        fiscal_year=2024,
        report_type="annual_report",
    )


def _make_hybrid_hit(
    *,
    query: DocumentEvidenceQuery,
    pdf_page: int,
    rank: int,
    text: str,
    company_id: str | None = None,
) -> HybridRetrievalHit:
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
        company_id=(
            company_id
            or query.company_id
        ),
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
        section_path=(
            "管理层讨论与分析",
        ),
        text=text,
        dense_rank=rank,
        bm25_rank=rank,
        rrf_score=rrf_score,
        source_retrievers=(
            "dense",
            "bm25",
        ),
    )


# ============================================================
# 【复制即可】
#
# FakeHybridRetriever 用于验证：
#
# Provider 有没有正确把 query/top_k/filter
# 传给 Week 5 检索层。
# ============================================================


class FakeHybridRetriever:
    def __init__(
        self,
        hits,
    ) -> None:
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
    def __init__(
        self,
    ) -> None:
        self.calls = []

        self._spec = RerankerSpec(
            model_revision=(
                "test_revision"
            ),
        )

    @property
    def spec(self):
        return self._spec

    def score_pairs(
        self,
        pairs,
    ):
        pairs = tuple(pairs)

        self.calls.append(
            pairs
        )

        return tuple(
            (
                10.0
                if "正确证据" in text
                else 1.0
            )
            for _, text
            in pairs
        )


def _build_provider(
    *,
    query: DocumentEvidenceQuery,
    hits=(),
    return_count: int = 5,
):
    hybrid_retriever = (
        FakeHybridRetriever(
            hits
        )
    )

    reranker_provider = (
        FakeRerankerProvider()
    )

    provider = (
        RoutedDocumentRerankerProvider(
            hybrid_retrievers_by_report_id={
                query.report_id: (
                    hybrid_retriever
                ),
            },
            reranker_provider=(
                reranker_provider
            ),
            runtime_config=(
                RerankerRuntimeConfig(
                    batch_size=2,
                    device="cpu",
                    local_files_only=True,
                    rerank_candidate_count=5,
                    return_count=(
                        return_count
                    ),
                )
            ),
            provider_id=(
                "document_test_v1"
            ),
        )
    )

    return (
        provider,
        hybrid_retriever,
        reranker_provider,
    )


# ============================================================
# 【手敲 + 重点理解】
#
# 建议自己敲这个测试。
#
# 它验证：
#
# Document Query
# → metadata filters
# → Hybrid
# → Reranker
#
# 同时明确验证没有 Gold PDF Page。
# ============================================================


def test_document_provider_routes_and_reranks(
) -> None:
    query = _build_query()

    wrong_hit = _make_hybrid_hit(
        query=query,
        pdf_page=100,
        rank=1,
        text="普通候选内容",
    )

    correct_hit = _make_hybrid_hit(
        query=query,
        pdf_page=158,
        rank=2,
        text="正确证据：营业收入增长",
    )

    (
        provider,
        hybrid_retriever,
        _,
    ) = _build_provider(
        query=query,
        hits=(
            wrong_hit,
            correct_hit,
        ),
    )

    results = provider.search(
        query=query,
        top_k=2,
    )

    assert tuple(
        hit.pdf_page
        for hit in results
    ) == (
        158,
        100,
    )

    assert tuple(
        hit.rank
        for hit in results
    ) == (
        1,
        2,
    )

    assert len(
        hybrid_retriever.calls
    ) == 1

    call = hybrid_retriever.calls[0]

    assert (
        call["query"]
        == query.semantic_query
    )

    assert call["top_k"] == 5

    filters = call["filters"]

    assert filters.company_ids == (
        "midea_group",
    )

    assert filters.report_ids == (
        "midea_group_2024",
    )

    assert filters.fiscal_years == (
        2024,
    )

    assert filters.pdf_pages == ()


# 【复制即可】
def test_document_provider_rejects_missing_route(
) -> None:
    query = _build_query()

    provider = (
        RoutedDocumentRerankerProvider(
            hybrid_retrievers_by_report_id={
                "other_report_2024": (
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
        )
    )

    with pytest.raises(
        DocumentRetrievalToolError,
        match="没有为文档查询配置",
    ):
        provider.search(
            query=query,
            top_k=5,
        )


# 【复制即可】
def test_document_provider_rejects_large_top_k(
) -> None:
    query = _build_query()

    (
        provider,
        _,
        _,
    ) = _build_provider(
        query=query,
        return_count=3,
    )

    with pytest.raises(
        DocumentRetrievalToolError,
        match="return_count",
    ):
        provider.search(
            query=query,
            top_k=4,
        )


# ============================================================
# 【手敲 + 重点理解】
#
# 这个测试建议自己敲。
#
# 重点是理解：
#
# RerankedRetrievalHit
#        ↓
# Runtime RetrievedDocument
#
# Runtime 只保留回答和审计真正需要的字段。
# ============================================================


def test_retrieve_documents_maps_runtime_documents(
) -> None:
    query = _build_query()

    hit = _make_hybrid_hit(
        query=query,
        pdf_page=158,
        rank=1,
        text="正确证据：营业收入增长",
    )

    (
        provider,
        _,
        _,
    ) = _build_provider(
        query=query,
        hits=(hit,),
    )

    tool = RetrieveDocumentsTool(
        hit_provider=provider
    )

    output = tool.handle(
        RetrieveDocumentsInput(
            query=query,
            top_k=1,
        )
    )

    assert isinstance(
        output,
        RetrieveDocumentsOutput,
    )

    assert output.query_id == "q1"

    assert len(
        output.documents
    ) == 1

    document = (
        output.documents[0]
    )

    assert document.rank == 1

    assert (
        document.pdf_page
        == 158
    )

    assert (
        document.printed_page
        == 157
    )

    assert (
        document.report_id
        == "midea_group_2024"
    )

    assert (
        document.text
        == "正确证据：营业收入增长"
    )

    assert document.score == 10.0


# 【复制即可】
def test_retrieve_documents_allows_empty_result(
) -> None:
    query = _build_query()

    (
        provider,
        _,
        reranker_provider,
    ) = _build_provider(
        query=query,
        hits=(),
    )

    tool = RetrieveDocumentsTool(
        hit_provider=provider
    )

    output = tool.handle(
        RetrieveDocumentsInput(
            query=query,
            top_k=5,
        )
    )

    assert output.documents == ()

    # 没有 Hybrid Candidate，
    # 不应该调用 Cross-Encoder。
    assert reranker_provider.calls == []


# 【复制即可】
def test_document_provider_rejects_wrong_identity(
) -> None:
    query = _build_query()

    wrong_hit = _make_hybrid_hit(
        query=query,
        pdf_page=158,
        rank=1,
        text="错误公司命中",
        company_id="gree_electric",
    )

    (
        provider,
        _,
        _,
    ) = _build_provider(
        query=query,
        hits=(wrong_hit,),
    )

    with pytest.raises(
        DocumentRetrievalToolError,
        match="company_id",
    ):
        provider.search(
            query=query,
            top_k=1,
        )


# ============================================================
# 【重点理解，代码可复制】
#
# 这是 6B 的完整 Runtime 集成测试：
#
# ToolExecutor
# → permission
# → Pydantic input
# → idempotency
# → retrieve_documents
# → Hybrid + Rerank
# → Runtime Output
# ============================================================


def test_retrieve_documents_runs_through_executor(
) -> None:
    query = _build_query()

    hit = _make_hybrid_hit(
        query=query,
        pdf_page=158,
        rank=1,
        text="正确证据：营业收入增长",
    )

    (
        provider,
        _,
        _,
    ) = _build_provider(
        query=query,
        hits=(hit,),
    )

    tool_registry = ToolRegistry()

    register_retrieve_documents_tool(
        tool_registry=tool_registry,
        hit_provider=provider,
    )

    executor = ToolExecutor(
        tool_registry,
        retry_backoff_seconds=0,
    )

    arguments: dict[
        str,
        Any,
    ] = (
        RetrieveDocumentsInput(
            query=query,
            top_k=1,
        ).model_dump(
            mode="json"
        )
    )

    first_result = (
        executor.execute(
            tool_name=(
                "retrieve_documents"
            ),
            arguments=arguments,
            request_id="request_1",
            run_id="run_1",
            step_id="s1",
            granted_permissions={
                "read_documents",
            },
        )
    )

    assert (
        first_result.reused
        is False
    )

    assert (
        first_result.traces[0].status
        == "succeeded"
    )

    assert (
        first_result.output[
            "query_id"
        ]
        == "q1"
    )

    documents = (
        first_result.output[
            "documents"
        ]
    )

    assert len(documents) == 1

    assert (
        documents[0]["pdf_page"]
        == 158
    )

    second_result = (
        executor.execute(
            tool_name=(
                "retrieve_documents"
            ),
            arguments=arguments,
            request_id="request_1",
            run_id="run_1",
            step_id="s1",
            granted_permissions={
                "read_documents",
            },
        )
    )

    assert (
        second_result.reused
        is True
    )

    assert (
        second_result.traces[0].status
        == "reused"
    )