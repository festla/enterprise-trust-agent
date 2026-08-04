from __future__ import annotations

from math import log2

import pytest
from pydantic import ValidationError

from app.rag.query_planning import (
    build_financial_fact_query_plan,
)
from app.rag.retrieval_evaluation import (
    RetrievalEvaluationError,
    evaluate_financial_fact_retrieval,
    summarize_retrieval_results,
)
from app.schemas.enums import (
    ChunkStrategy,
    PageMappingStatus,
    ReportType,
    StatementScope,
    StatementType,
)
from app.schemas.retrieval import RetrievalHit
from app.schemas.retrieval_eval import (
    FinancialFactRetrievalEvalCase,
    RetrievalEvalResult,
)
from app.schemas.hybrid_retrieval import (
    HybridRetrievalHit,
)
from app.schemas.reranker import (
    RerankedRetrievalHit,
)

REPORT_ID = "midea_group_2024"

DOCUMENT_ID = (
    f"doc_{REPORT_ID}_{'a' * 24}"
)

CHUNK_DATASET_ID = (
    f"chunk_dataset_{REPORT_ID}_"
    f"{'b' * 24}"
)

VECTOR_INDEX_ID = (
    f"vector_index_{REPORT_ID}_"
    f"{'c' * 24}"
)


def build_case(
    *,
    gold_pdf_pages: tuple[int, ...] = (
        158,
    ),
) -> FinancialFactRetrievalEvalCase:
    return FinancialFactRetrievalEvalCase(
        case_id="fact_001",
        question=(
            "美的集团2024年营业收入是多少？"
        ),
        metric_name="营业收入",
        company_id="midea_group",
        report_id=REPORT_ID,
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
        gold_pdf_pages=gold_pdf_pages,
    )


def build_plan(
    *,
    pdf_pages: tuple[int, ...] = (),
):
    return build_financial_fact_query_plan(
        original_query=(
            "美的集团2024年营业收入是多少？"
        ),
        metric_name="营业收入",
        fiscal_year=2024,
        company_id="midea_group",
        report_id=REPORT_ID,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        statement_type=(
            StatementType.INCOME_STATEMENT
        ),
        statement_scope=(
            StatementScope.CONSOLIDATED
        ),
        pdf_pages=pdf_pages,
    )


def build_hit(
    *,
    rank: int,
    pdf_page: int,
) -> RetrievalHit:
    text = (
        "其中：营业收入 "
        "407,149,600"
        if pdf_page == 158
        else f"非目标证据页面 {pdf_page}"
    )

    return RetrievalHit(
        rank=rank,
        score=max(
            0.1,
            0.9 - rank * 0.05,
        ),
        chunk_id=(
            f"chunk_{REPORT_ID}_"
            f"{rank:024x}"
        ),
        chunk_dataset_id=(
            CHUNK_DATASET_ID
        ),
        company_id="midea_group",
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
        printed_page=pdf_page,
        mapping_status=(
            PageMappingStatus.MAPPED
        ),
        chunk_index=rank - 1,
        strategy=(
            ChunkStrategy.FIXED_LENGTH
        ),
        source_start_char=0,
        source_end_char=len(text),
        section_path=(),
        text=text,
    )

def build_reranked_hit(
) -> RerankedRetrievalHit:
    source_hit = build_hit(
        rank=1,
        pdf_page=158,
    )

    hybrid_hit = (
        HybridRetrievalHit
        .from_source_hit(
            rank=4,
            rrf_score=(
                1 / 62
                + 1 / 61
            ),
            source_hit=source_hit,
            dense_rank=2,
            bm25_rank=1,
        )
    )

    return (
        RerankedRetrievalHit
        .from_hybrid_hit(
            rank=1,
            reranker_score=3.25,
            source_hit=hybrid_hit,
        )
    )


def evaluate(
    hits: tuple[RetrievalHit, ...],
):
    return evaluate_financial_fact_retrieval(
        case=build_case(),
        plan=build_plan(),
        strategy=(
            ChunkStrategy.FIXED_LENGTH
        ),
        chunk_dataset_id=(
            CHUNK_DATASET_ID
        ),
        vector_index_id=(
            VECTOR_INDEX_ID
        ),
        hits=hits,
    )


def test_normalize_gold_pdf_pages() -> None:
    case = build_case(
        gold_pdf_pages=(
            159,
            158,
            159,
        )
    )

    assert case.gold_pdf_pages == (
        158,
        159,
    )


def test_reject_empty_gold_pdf_pages(
) -> None:
    with pytest.raises(
        ValidationError,
        match="gold_pdf_pages",
    ):
        build_case(
            gold_pdf_pages=()
        )


def test_rank_one_recall() -> None:
    result = evaluate(
        (
            build_hit(
                rank=1,
                pdf_page=158,
            ),
            build_hit(
                rank=2,
                pdf_page=247,
            ),
        )
    )

    assert result.first_relevant_rank == 1
    assert result.recall_at_1 is True
    assert result.recall_at_3 is True
    assert result.recall_at_5 is True
    assert result.top_hits[0].pdf_page == 158


def test_rank_four_recall() -> None:
    result = evaluate(
        (
            build_hit(rank=1, pdf_page=1),
            build_hit(rank=2, pdf_page=165),
            build_hit(rank=3, pdf_page=247),
            build_hit(rank=4, pdf_page=158),
        )
    )

    assert result.first_relevant_rank == 4
    assert result.recall_at_1 is False
    assert result.recall_at_3 is False
    assert result.recall_at_5 is True


def test_no_relevant_page() -> None:
    result = evaluate(
        (
            build_hit(rank=1, pdf_page=1),
            build_hit(rank=2, pdf_page=247),
        )
    )

    assert result.first_relevant_rank is None
    assert result.recall_at_1 is False
    assert result.recall_at_3 is False
    assert result.recall_at_5 is False


def test_reject_non_continuous_ranks(
) -> None:
    with pytest.raises(
        RetrievalEvaluationError,
        match="连续递增",
    ):
        evaluate(
            (
                build_hit(
                    rank=1,
                    pdf_page=1,
                ),
                build_hit(
                    rank=3,
                    pdf_page=158,
                ),
            )
        )


def test_reject_gold_page_filter_leakage(
) -> None:
    with pytest.raises(
        RetrievalEvaluationError,
        match="Gold Evidence 泄漏",
    ):
        evaluate_financial_fact_retrieval(
            case=build_case(),
            plan=build_plan(
                pdf_pages=(158,)
            ),
            strategy=(
                ChunkStrategy.FIXED_LENGTH
            ),
            chunk_dataset_id=(
                CHUNK_DATASET_ID
            ),
            vector_index_id=(
                VECTOR_INDEX_ID
            ),
            hits=(
                build_hit(
                    rank=1,
                    pdf_page=158,
                ),
            ),
        )


def test_summarize_retrieval_results(
) -> None:
    rank_one = evaluate(
        (
            build_hit(rank=1, pdf_page=158),
        )
    )

    rank_four = evaluate(
        (
            build_hit(rank=1, pdf_page=1),
            build_hit(rank=2, pdf_page=2),
            build_hit(rank=3, pdf_page=3),
            build_hit(rank=4, pdf_page=158),
        )
    ).model_copy(
        update={
            "case_id": "fact_002",
        }
    )

    summary = summarize_retrieval_results(
        evaluation_set_id=(
            "midea_fact_dev_v1"
        ),
        results=(
            rank_one,
            rank_four,
        ),
    )

    assert summary.case_count == 2
    assert summary.hit_at_1_count == 1
    assert summary.hit_at_3_count == 1
    assert summary.hit_at_5_count == 2
    assert summary.recall_at_1 == 0.5
    assert summary.recall_at_3 == 0.5
    assert summary.recall_at_5 == 1.0


def test_reject_mixed_vector_indexes(
) -> None:
    first = evaluate(
        (
            build_hit(rank=1, pdf_page=158),
        )
    )

    second = first.model_copy(
        update={
            "case_id": "fact_002",
            "index_id": (
                f"vector_index_{REPORT_ID}_"
                f"{'d' * 24}"
            ),
        }
    )
    with pytest.raises(
        RetrievalEvaluationError,
        match="不同索引",
    ):
        summarize_retrieval_results(
            evaluation_set_id=(
                "midea_fact_dev_v1"
            ),
            results=(first, second),
        )

def test_calculate_mrr_and_ndcg_for_rank_two(
) -> None:
    result = evaluate(
        (
            build_hit(
                rank=1,
                pdf_page=1,
            ),
            build_hit(
                rank=2,
                pdf_page=158,
            ),
        )
    )

    assert result.first_relevant_rank == 2

    assert result.reciprocal_rank == (
        pytest.approx(0.5)
    )

    assert result.ndcg_at_1 == 0

    assert result.ndcg_at_3 == (
        pytest.approx(
            1 / log2(3)
        )
    )

    assert result.ndcg_at_5 == (
        pytest.approx(
            1 / log2(3)
        )
    )

def test_ndcg_deduplicates_same_gold_page(
) -> None:
    result = evaluate(
        (
            build_hit(
                rank=1,
                pdf_page=158,
            ),
            build_hit(
                rank=2,
                pdf_page=158,
            ),
            build_hit(
                rank=3,
                pdf_page=1,
            ),
        )
    )

    assert result.first_relevant_rank == 1
    assert result.ndcg_at_3 == (
        pytest.approx(1.0)
    )

def test_summarize_mrr_and_ndcg(
) -> None:
    rank_one = evaluate(
        (
            build_hit(
                rank=1,
                pdf_page=158,
            ),
        )
    )

    rank_two = evaluate(
        (
            build_hit(
                rank=1,
                pdf_page=1,
            ),
            build_hit(
                rank=2,
                pdf_page=158,
            ),
        )
    ).model_copy(
        update={
            "case_id": "fact_002",
        }
    )

    summary = summarize_retrieval_results(
        evaluation_set_id=(
            "midea_fact_dev_v1"
        ),
        results=(
            rank_one,
            rank_two,
        ),
    )

    assert summary.mrr == pytest.approx(
        (1.0 + 0.5) / 2
    )

    assert summary.ndcg_at_1 == (
        pytest.approx(0.5)
    )

    assert summary.ndcg_at_3 == (
        pytest.approx(
            (
                1.0
                + 1 / log2(3)
            )
            / 2
        )
    )

    assert summary.ndcg_at_5 == (
        pytest.approx(
            (
                1.0
                + 1 / log2(3)
            )
            / 2
        )
    )


def test_evaluate_hybrid_retrieval(
) -> None:
    source_hit = build_hit(
        rank=1,
        pdf_page=158,
    )

    hybrid_hit = (
        HybridRetrievalHit.from_source_hit(
            rank=1,
            rrf_score=(
                1 / 61
                + 1 / 62
            ),
            source_hit=source_hit,
            dense_rank=1,
            bm25_rank=2,
        )
    )

    result = (
        evaluate_financial_fact_retrieval(
            case=build_case(),
            plan=build_plan(),
            strategy=(
                ChunkStrategy.FIXED_LENGTH
            ),
            chunk_dataset_id=(
                CHUNK_DATASET_ID
            ),
            index_id=(
                f"hybrid_run_{REPORT_ID}_"
                f"{'e' * 24}"
            ),
            retriever_type=(
                "hybrid_rrf"
            ),
            hits=(hybrid_hit,),
        )
    )

    assert result.recall_at_1
    assert result.reciprocal_rank == 1
    assert result.retriever_type == (
        "hybrid_rrf"
    )

    saved_hit = result.top_hits[0]

    assert isinstance(
        saved_hit,
        HybridRetrievalHit,
    )

    assert saved_hit.dense_rank == 1
    assert saved_hit.bm25_rank == 2


def test_preserve_hybrid_hit_audit_fields(
) -> None:
    hybrid_hit = (
        HybridRetrievalHit.from_source_hit(
            rank=1,
            rrf_score=(
                1 / 61
                + 1 / 62
            ),
            source_hit=build_hit(
                rank=1,
                pdf_page=158,
            ),
            dense_rank=1,
            bm25_rank=2,
        )
    )

    result = RetrievalEvalResult(
        case_id="fact_001",
        question="营业收入是多少？",
        semantic_query="营业收入 合并利润表",
        company_id="midea_group",
        report_id=REPORT_ID,
        fiscal_year=2024,
        strategy=(
            ChunkStrategy.FIXED_LENGTH
        ),
        retriever_type="hybrid_rrf",
        chunk_dataset_id=(
            CHUNK_DATASET_ID
        ),
        index_id=(
            f"hybrid_run_{REPORT_ID}_"
            f"{'e' * 24}"
        ),
        gold_pdf_pages=(158,),
        evaluated_hit_count=1,
        first_relevant_rank=1,
        recall_at_1=True,
        recall_at_3=True,
        recall_at_5=True,
        top_hits=(hybrid_hit,),
    )

    dumped = result.model_dump(
        mode="json"
    )

    dumped_hit = dumped[
        "top_hits"
    ][0]

    assert dumped_hit[
        "dense_rank"
    ] == 1

    assert dumped_hit[
        "bm25_rank"
    ] == 2

    assert (
        dumped_hit[
            "source_retrievers"
        ]
        == ["dense", "bm25"]
    )

def test_evaluate_reranker_retrieval(
) -> None:
    reranked_hit = (
        build_reranked_hit()
    )

    result = (
        evaluate_financial_fact_retrieval(
            case=build_case(),
            plan=build_plan(),
            strategy=(
                ChunkStrategy.FIXED_LENGTH
            ),
            chunk_dataset_id=(
                CHUNK_DATASET_ID
            ),
            index_id=(
                f"reranker_run_"
                f"{REPORT_ID}_"
                f"{'f' * 24}"
            ),
            retriever_type=(
                "hybrid_reranker"
            ),
            hits=(reranked_hit,),
        )
    )

    assert result.recall_at_1
    assert result.first_relevant_rank == 1
    assert result.reciprocal_rank == 1

    assert result.retriever_type == (
        "hybrid_reranker"
    )

    assert result.index_id.startswith(
        "reranker_run_"
    )

    saved_hit = result.top_hits[0]

    assert isinstance(
        saved_hit,
        RerankedRetrievalHit,
    )

    assert saved_hit.rank == 1
    assert saved_hit.rrf_rank == 4

    assert saved_hit.dense_rank == 2
    assert saved_hit.bm25_rank == 1

    assert saved_hit.reranker_score == (
        pytest.approx(3.25)
    )

    summary = (
        summarize_retrieval_results(
            evaluation_set_id=(
                "midea_reranker_dev_v1"
            ),
            results=(result,),
        )
    )

    assert summary.retriever_type == (
        "hybrid_reranker"
    )

    assert summary.index_id.startswith(
        "reranker_run_"
    )

    assert summary.case_count == 1
    assert summary.recall_at_1 == 1


def test_preserve_reranker_audit_fields(
) -> None:
    result = (
        evaluate_financial_fact_retrieval(
            case=build_case(),
            plan=build_plan(),
            strategy=(
                ChunkStrategy.FIXED_LENGTH
            ),
            chunk_dataset_id=(
                CHUNK_DATASET_ID
            ),
            index_id=(
                f"reranker_run_"
                f"{REPORT_ID}_"
                f"{'f' * 24}"
            ),
            retriever_type=(
                "hybrid_reranker"
            ),
            hits=(
                build_reranked_hit(),
            ),
        )
    )

    dumped = result.model_dump(
        mode="json"
    )

    dumped_hit = dumped[
        "top_hits"
    ][0]

    assert dumped_hit[
        "reranker_score"
    ] == pytest.approx(3.25)

    assert dumped_hit[
        "rrf_rank"
    ] == 4

    assert dumped_hit[
        "rrf_score"
    ] == pytest.approx(
        1 / 62
        + 1 / 61
    )

    assert dumped_hit[
        "dense_rank"
    ] == 2

    assert dumped_hit[
        "bm25_rank"
    ] == 1

    assert (
        dumped_hit[
            "source_retrievers"
        ]
        == ["dense", "bm25"]
    )

    assert dumped_hit[
        "retriever_type"
    ] == "hybrid_reranker"

    assert dumped_hit[
        "score_type"
    ] == "cross_encoder_logit"