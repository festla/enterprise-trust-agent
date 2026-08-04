from __future__ import annotations

from collections.abc import Sequence

from app.schemas.enums import ChunkStrategy
from app.schemas.retrieval import (
    RetrievalHit,
    RetrievalQueryPlan,
)
from app.schemas.retrieval_eval import (
    FinancialFactRetrievalEvalCase,
    RetrievalEvalResult,
    RetrievalEvalSummary,
    RetrievalMethod,
)

class RetrievalEvaluationError(
    ValueError
):
    """检索评测基础异常。"""


def _resolve_index_id(
    *,
    index_id: str | None,
    vector_index_id: str | None,
) -> str:
    """兼容旧 Dense 参数名。"""

    if (
        index_id is None
        and vector_index_id is None
    ):
        raise RetrievalEvaluationError(
            "必须提供 index_id"
        )

    if (
        index_id is not None
        and vector_index_id is not None
        and index_id != vector_index_id
    ):
        raise RetrievalEvaluationError(
            "index_id 与 vector_index_id "
            "不能冲突"
        )

    resolved = (
        index_id
        if index_id is not None
        else vector_index_id
    )

    # 上面的校验已经保证二者至少有一个不为空。
    assert resolved is not None

    return resolved


def _validate_case_and_plan(
    *,
    case: FinancialFactRetrievalEvalCase,
    plan: RetrievalQueryPlan,
) -> None:
    """检查评测题与检索计划是否对应。"""

    if plan.original_query != case.question:
        raise RetrievalEvaluationError(
            "Query Plan 的 original_query "
            "与评测题不一致"
        )

    if plan.metric_name != case.metric_name:
        raise RetrievalEvaluationError(
            "Query Plan 的 metric_name "
            "与评测题不一致"
        )

    if plan.fiscal_year != case.fiscal_year:
        raise RetrievalEvaluationError(
            "Query Plan 的 fiscal_year "
            "与评测题不一致"
        )

    if (
        plan.statement_type
        != case.statement_type
    ):
        raise RetrievalEvaluationError(
            "Query Plan 的 statement_type "
            "与评测题不一致"
        )

    if (
        plan.statement_scope
        != case.statement_scope
    ):
        raise RetrievalEvaluationError(
            "Query Plan 的 statement_scope "
            "与评测题不一致"
        )

    if (
        case.company_id
        not in plan.filters.company_ids
    ):
        raise RetrievalEvaluationError(
            "Query Plan 缺少目标 company_id 过滤"
        )

    if (
        case.report_id
        not in plan.filters.report_ids
    ):
        raise RetrievalEvaluationError(
            "Query Plan 缺少目标 report_id 过滤"
        )

    if (
        case.fiscal_year
        not in plan.filters.fiscal_years
    ):
        raise RetrievalEvaluationError(
            "Query Plan 缺少目标 fiscal_year 过滤"
        )

    if (
        case.report_type
        not in plan.filters.report_types
    ):
        raise RetrievalEvaluationError(
            "Query Plan 缺少目标 report_type 过滤"
        )

    # 评测时禁止提前告诉检索器正确页面。
    if (
        plan.filters.pdf_pages
        or plan.filters.page_ids
    ):
        raise RetrievalEvaluationError(
            "检索评测不能使用页面级过滤，"
            "否则会造成 Gold Evidence 泄漏"
        )


def evaluate_financial_fact_retrieval(
    *,
    case: FinancialFactRetrievalEvalCase,
    plan: RetrievalQueryPlan,
    strategy: ChunkStrategy,
    chunk_dataset_id: str,
    hits: Sequence[RetrievalHit],
    retriever_type: RetrievalMethod = "dense",
    index_id: str | None = None,
    vector_index_id: str | None = None,
) -> RetrievalEvalResult:
    """根据人工 Gold PDF 页计算单题指标。"""

    _validate_case_and_plan(
        case=case,
        plan=plan,
    )

    resolved_index_id = _resolve_index_id(
        index_id=index_id,
        vector_index_id=vector_index_id,
    )

    actual_ranks = tuple(
        hit.rank
        for hit in hits
    )

    expected_ranks = tuple(
        range(
            1,
            len(hits) + 1,
        )
    )

    if actual_ranks != expected_ranks:
        raise RetrievalEvaluationError(
            "RetrievalHit 的 rank 必须从 1 "
            "开始连续递增"
        )

    for hit in hits:
        if (
            hit.retriever_type
            != retriever_type
        ):
            raise RetrievalEvaluationError(
                "检索结果的 retriever_type "
                "不一致"
            )

        if (
            hit.chunk_dataset_id
            != chunk_dataset_id
        ):
            raise RetrievalEvaluationError(
                "检索结果引用了不同的 "
                "chunk_dataset_id"
            )

        if hit.strategy != strategy:
            raise RetrievalEvaluationError(
                "检索结果的 Chunk 策略不一致"
            )

        if hit.company_id != case.company_id:
            raise RetrievalEvaluationError(
                "检索结果的 company_id 不一致"
            )

        if hit.report_id != case.report_id:
            raise RetrievalEvaluationError(
                "检索结果的 report_id 不一致"
            )

        if hit.fiscal_year != case.fiscal_year:
            raise RetrievalEvaluationError(
                "检索结果的 fiscal_year 不一致"
            )

        if hit.report_type != case.report_type:
            raise RetrievalEvaluationError(
                "检索结果的 report_type 不一致"
            )

    gold_page_set = set(
        case.gold_pdf_pages
    )

    relevant_ranks = tuple(
        hit.rank
        for hit in hits
        if hit.pdf_page in gold_page_set
    )

    first_relevant_rank = (
        min(relevant_ranks)
        if relevant_ranks
        else None
    )

    return RetrievalEvalResult(
        case_id=case.case_id,
        question=case.question,
        semantic_query=plan.semantic_query,
        company_id=case.company_id,
        report_id=case.report_id,
        fiscal_year=case.fiscal_year,
        strategy=strategy,
        retriever_type=retriever_type,
        chunk_dataset_id=chunk_dataset_id,
        index_id=resolved_index_id,
        gold_pdf_pages=case.gold_pdf_pages,
        evaluated_hit_count=len(hits),
        first_relevant_rank=first_relevant_rank,
        recall_at_1=(
            first_relevant_rank is not None
            and first_relevant_rank <= 1
        ),
        recall_at_3=(
            first_relevant_rank is not None
            and first_relevant_rank <= 3
        ),
        recall_at_5=(
            first_relevant_rank is not None
            and first_relevant_rank <= 5
        ),
        top_hits=tuple(hits[:5]),
    )



def summarize_retrieval_results(
    *,
    evaluation_set_id: str,
    results: Sequence[RetrievalEvalResult],
) -> RetrievalEvalSummary:
    """汇总同一索引上的逐题检索结果。"""

    if not results:
        raise RetrievalEvaluationError(
            "至少需要一条评测结果"
        )

    first = results[0]

    case_ids = tuple(
        result.case_id
        for result in results
    )

    if len(case_ids) != len(set(case_ids)):
        raise RetrievalEvaluationError(
            "评测结果包含重复 case_id"
        )

    for result in results:
        if result.strategy != first.strategy:
            raise RetrievalEvaluationError(
                "不能汇总不同 Chunk 策略的结果"
            )

        if (
            result.chunk_dataset_id
            != first.chunk_dataset_id
        ):
            raise RetrievalEvaluationError(
                "不能汇总不同 ChunkDataset 的结果"
            )

        if (
            result.retriever_type
            != first.retriever_type
        ):
            raise RetrievalEvaluationError(
                "不能汇总不同检索器的结果"
            )

        if result.index_id != first.index_id:
            raise RetrievalEvaluationError(
                "不能汇总不同索引的结果"
            )

    case_count = len(results)

    hit_at_1_count = sum(
        result.recall_at_1
        for result in results
    )

    hit_at_3_count = sum(
        result.recall_at_3
        for result in results
    )

    hit_at_5_count = sum(
        result.recall_at_5
        for result in results
    )

    mrr = sum(
        result.reciprocal_rank
        for result in results
    ) / case_count

    ndcg_at_1 = sum(
        result.ndcg_at_1
        for result in results
    ) / case_count

    ndcg_at_3 = sum(
        result.ndcg_at_3
        for result in results
    ) / case_count

    ndcg_at_5 = sum(
        result.ndcg_at_5
        for result in results
    ) / case_count

    return RetrievalEvalSummary(
        evaluation_set_id=evaluation_set_id,
        strategy=first.strategy,
        retriever_type=(
            first.retriever_type
        ),
        chunk_dataset_id=(
            first.chunk_dataset_id
        ),
        index_id=first.index_id,
        case_ids=case_ids,
        case_count=case_count,
        hit_at_1_count=hit_at_1_count,
        hit_at_3_count=hit_at_3_count,
        hit_at_5_count=hit_at_5_count,
        recall_at_1=(
            hit_at_1_count / case_count
        ),
        recall_at_3=(
            hit_at_3_count / case_count
        ),
        recall_at_5=(
            hit_at_5_count / case_count
        ),
        mrr=mrr,
        ndcg_at_1=ndcg_at_1,
        ndcg_at_3=ndcg_at_3,
        ndcg_at_5=ndcg_at_5,
    )