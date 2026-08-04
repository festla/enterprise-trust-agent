from __future__ import annotations

import re
from collections.abc import Sequence

from app.schemas.enums import (
    StatementType,
)
from app.schemas.evidence_context import (
    EvidenceCitation,
    EvidenceContext,
    EvidenceContextItem,
    EvidenceReadinessResult,
)
from app.schemas.report import Report
from app.schemas.retrieval import (
    RetrievalHit,
    RetrievalQueryPlan,
)


class EvidenceContextError(ValueError):
    """Evidence Context 基础异常。"""


class EvidenceSourceMismatchError(
    EvidenceContextError
):
    """检索证据与目标报告身份不一致。"""


class InvalidEvidenceBudgetError(
    EvidenceContextError
):
    """Evidence Context 字符或条数预算无效。"""


class InvalidEvidenceRankError(
    EvidenceContextError
):
    """检索结果排名不是连续的稳定序列。"""


_STATEMENT_PHRASES = {
    StatementType.BALANCE_SHEET: (
        "合并资产负债表",
        "公司资产负债表",
        "合并及公司资产负债表",
    ),
    StatementType.INCOME_STATEMENT: (
        "合并利润表",
        "公司利润表",
        "合并及公司利润表",
    ),
    StatementType.CASH_FLOW_STATEMENT: (
        "合并现金流量表",
        "公司现金流量表",
        "合并及公司现金流量表",
    ),
    (
        StatementType
        .STATEMENT_OF_CHANGES_IN_EQUITY
    ): (
        "所有者权益变动表",
        "股东权益变动表",
    ),
    StatementType.FINANCIAL_SUMMARY: (
        "主要财务数据",
    ),
    StatementType.NOTE: (
        "财务报表附注",
    ),
}

_FINANCIAL_VALUE_PATTERN = re.compile(
    r"(?<!\d)"
    r"[+-]?"
    r"(?:"
    r"\(\s*"
    r")?"
    r"(?:"
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|"
    r"\d+\.\d+"
    r"|"
    r"\d{4,}"
    r")"
    r"(?:\s*\))?"
    r"%?"
    r"(?!\d)"
)


def _normalize_for_match(
    value: str,
) -> str:
    """移除空白，降低 PDF 文本断行影响。"""

    return re.sub(
        r"\s+",
        "",
        value,
    ).casefold()


def _validate_report_and_plan(
    *,
    report: Report,
    plan: RetrievalQueryPlan,
) -> None:
    """检查 Query Plan 是否属于目标报告。"""

    if (
        plan.filters.report_ids
        and report.report_id
        not in plan.filters.report_ids
    ):
        raise EvidenceSourceMismatchError(
            "Query Plan 与目标 report_id 不一致"
        )

    if (
        plan.filters.company_ids
        and report.company_id
        not in plan.filters.company_ids
    ):
        raise EvidenceSourceMismatchError(
            "Query Plan 与目标 company_id 不一致"
        )

    if (
        plan.filters.fiscal_years
        and report.fiscal_year
        not in plan.filters.fiscal_years
    ):
        raise EvidenceSourceMismatchError(
            "Query Plan 与目标 fiscal_year 不一致"
        )

    if (
        plan.filters.report_types
        and report.report_type
        not in plan.filters.report_types
    ):
        raise EvidenceSourceMismatchError(
            "Query Plan 与目标 report_type 不一致"
        )


def _validate_hit_source(
    *,
    report: Report,
    hit: RetrievalHit,
) -> None:
    """检查一个 Hit 是否来自当前有效报告。"""

    if hit.report_id != report.report_id:
        raise EvidenceSourceMismatchError(
            "RetrievalHit 的 report_id "
            "与目标报告不一致"
        )

    if hit.company_id != report.company_id:
        raise EvidenceSourceMismatchError(
            "RetrievalHit 的 company_id "
            "与目标报告不一致"
        )

    if hit.fiscal_year != report.fiscal_year:
        raise EvidenceSourceMismatchError(
            "RetrievalHit 的 fiscal_year "
            "与目标报告不一致"
        )

    if hit.report_type != report.report_type:
        raise EvidenceSourceMismatchError(
            "RetrievalHit 的 report_type "
            "与目标报告不一致"
        )

    if (
        report.active_document_id is not None
        and hit.document_id
        != report.active_document_id
    ):
        raise EvidenceSourceMismatchError(
            "RetrievalHit 不是当前有效的 "
            "Document 版本"
        )


def _build_citation(
    *,
    citation_id: str,
    report: Report,
    hit: RetrievalHit,
) -> EvidenceCitation:
    return EvidenceCitation(
        citation_id=citation_id,
        company_id=report.company_id,
        report_id=report.report_id,
        report_title=report.title,
        source_name=report.source_name,
        fiscal_year=report.fiscal_year,
        document_id=hit.document_id,
        page_id=hit.page_id,
        pdf_page=hit.pdf_page,
        printed_page=hit.printed_page,
        chunk_id=hit.chunk_id,
        retrieval_rank=hit.rank,
        retrieval_score=hit.score,
    )


def _build_header(
    *,
    citation: EvidenceCitation,
) -> str:
    printed_page_text = (
        str(citation.printed_page)
        if citation.printed_page is not None
        else "未映射"
    )

    return (
        f"[{citation.citation_id}] "
        f"{citation.report_title} | "
        f"{citation.fiscal_year}年度 | "
        f"PDF第{citation.pdf_page}页 | "
        f"印刷第{printed_page_text}页 | "
        f"Rank {citation.retrieval_rank} | "
        f"Score {citation.retrieval_score:.6f}"
    )


def build_evidence_context(
    *,
    report: Report,
    plan: RetrievalQueryPlan,
    hits: Sequence[RetrievalHit],
    max_hits: int = 5,
    max_chars: int = 4000,
) -> EvidenceContext:
    """将 Top-k 检索结果组装为可追溯上下文。"""

    if max_hits < 1:
        raise InvalidEvidenceBudgetError(
            "max_hits 必须大于等于 1"
        )

    if max_chars < 200:
        raise InvalidEvidenceBudgetError(
            "max_chars 必须大于等于 200"
        )

    _validate_report_and_plan(
        report=report,
        plan=plan,
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
        raise InvalidEvidenceRankError(
            "RetrievalHit 的 rank "
            "必须从 1 连续递增"
        )

    unique_hits: list[RetrievalHit] = []
    seen_chunk_ids: set[str] = set()
    duplicate_hit_count = 0

    for hit in hits:
        _validate_hit_source(
            report=report,
            hit=hit,
        )

        if hit.chunk_id in seen_chunk_ids:
            duplicate_hit_count += 1
            continue

        seen_chunk_ids.add(hit.chunk_id)
        unique_hits.append(hit)

    candidate_hits = unique_hits[:max_hits]

    items: list[EvidenceContextItem] = []
    blocks: list[str] = []

    for hit in candidate_hits:
        citation_id = (
            f"E{len(items) + 1}"
        )

        citation = _build_citation(
            citation_id=citation_id,
            report=report,
            hit=hit,
        )

        header = _build_header(
            citation=citation,
        )

        original_text = hit.text.strip()

        full_block = (
            f"{header}\n"
            f"{original_text}"
        )

        candidate_context = (
            "\n\n".join(
                (*blocks, full_block)
            )
        )

        if len(candidate_context) <= max_chars:
            included_text = original_text

        else:
            # 为了保持 Rank 顺序，不跳过高排名证据
            # 再尝试加入后面的低排名证据。
            if items:
                break

            prefix = f"{header}\n"

            available_text_chars = (
                max_chars - len(prefix)
            )

            if available_text_chars < 2:
                raise InvalidEvidenceBudgetError(
                    "字符预算不足以保存第一条引用"
                )

            included_text = original_text[
                :available_text_chars
            ]

            if (
                len(included_text)
                < len(original_text)
            ):
                included_text = (
                    included_text[:-1]
                    + "…"
                )

            full_block = (
                f"{header}\n"
                f"{included_text}"
            )

        item = EvidenceContextItem(
            citation=citation,
            text=included_text,
            original_char_count=len(
                original_text
            ),
            included_char_count=len(
                included_text
            ),
            text_truncated=(
                len(included_text)
                < len(original_text)
            ),
        )

        items.append(item)
        blocks.append(full_block)

        if item.text_truncated:
            break

    context_text = "\n\n".join(
        blocks
    )

    omitted_hit_count = (
        len(unique_hits) - len(items)
    )

    return EvidenceContext(
        original_query=plan.original_query,
        semantic_query=plan.semantic_query,
        company_id=report.company_id,
        report_id=report.report_id,
        report_title=report.title,
        source_name=report.source_name,
        fiscal_year=report.fiscal_year,
        items=tuple(items),
        context_text=context_text,
        max_hits=max_hits,
        max_chars=max_chars,
        used_chars=len(context_text),
        duplicate_hit_count=(
            duplicate_hit_count
        ),
        omitted_hit_count=(
            omitted_hit_count
        ),
        truncated=(
            omitted_hit_count > 0
            or any(
                item.text_truncated
                for item in items
            )
        ),
        used_chunk_ids=tuple(
            item.citation.chunk_id
            for item in items
        ),
    )


def _contains_statement_heading(
    *,
    statement_type: StatementType,
    text: str,
) -> bool:
    """检查文本是否包含目标报表的正式标题短语。"""

    phrases = _STATEMENT_PHRASES.get(
        statement_type
    )

    if phrases is None:
        return False

    normalized_text = _normalize_for_match(
        text
    )

    return any(
        _normalize_for_match(phrase)
        in normalized_text
        for phrase in phrases
    )

def _has_metric_nearby_value(
    *,
    item: EvidenceContextItem,
    metric_name: str,
    fiscal_year: int,
    window_chars: int = 120,
) -> bool:
    """检查目标指标后方是否紧邻财务型数值。

    只查看指标出现位置之后的局部窗口，避免使用
    Chunk 中其他项目、页码、年份或章节编号作为支持。
    """

    normalized_text = _normalize_for_match(
        item.text
    )

    normalized_metric = _normalize_for_match(
        metric_name
    )

    search_start = 0

    excluded_values = {
        str(fiscal_year),
    }

    while True:
        metric_position = normalized_text.find(
            normalized_metric,
            search_start,
        )

        if metric_position < 0:
            return False

        value_window_start = (
            metric_position
            + len(normalized_metric)
        )

        value_window = normalized_text[
            value_window_start:
            value_window_start + window_chars
        ]

        for match in (
            _FINANCIAL_VALUE_PATTERN
            .finditer(value_window)
        ):
            raw_value = match.group()

            normalized_value = (
                raw_value
                .replace("(", "")
                .replace(")", "")
                .replace(",", "")
                .replace("%", "")
                .lstrip("+-")
            )

            integer_part = normalized_value.split(
                ".",
                maxsplit=1,
            )[0]

            if integer_part in excluded_values:
                continue

            return True

        search_start = (
            metric_position
            + len(normalized_metric)
        )

def assess_financial_fact_evidence(
    *,
    plan: RetrievalQueryPlan,
    context: EvidenceContext,
) -> EvidenceReadinessResult:
    """用保守规则判断 Context 能否进入生成。

    至少需要一个 Evidence Item 同时满足：
    1. 包含目标报表的正式标题；
    2. 包含目标指标；
    3. 目标指标后方的局部窗口内存在财务型数值。

    不允许使用同一 Chunk 中距离指标较远的无关数字，
    也不允许将多个 Chunk 中的零散信息拼接为完整证据。
    """

    if not context.items:
        return EvidenceReadinessResult(
            status="no_retrieval_hits",
            metric_name=plan.metric_name,
            supporting_citation_ids=(),
            reason=(
                "检索没有返回任何可用证据"
            ),
            context=context,
        )

    if (
        plan.statement_type
        not in _STATEMENT_PHRASES
    ):
        return EvidenceReadinessResult(
            status="insufficient_evidence",
            metric_name=plan.metric_name,
            supporting_citation_ids=(),
            reason=(
                "当前确定性证据 Gate "
                "不支持该 statement_type"
            ),
            context=context,
        )

    supporting_ids: list[str] = []

    for item in context.items:
        has_statement_heading = (
            _contains_statement_heading(
                statement_type=(
                    plan.statement_type
                ),
                text=item.text,
            )
        )

        has_metric_value_pair = (
            _has_metric_nearby_value(
                item=item,
                metric_name=plan.metric_name,
                fiscal_year=plan.fiscal_year,
            )
        )

        # 必须在同一个 Evidence Item 中同时满足：
        # 1. 出现正式报表标题；
        # 2. 目标指标后方存在邻近的财务数值。
        #
        # 不能将多个页面中的零散内容拼成完整证据，
        # 也不能用 Chunk 中远离目标指标的无关数字。
        if (
            has_statement_heading
            and has_metric_value_pair
        ):
            supporting_ids.append(
                item.citation.citation_id
            )

    if supporting_ids:
        return EvidenceReadinessResult(
            status="ready_for_generation",
            metric_name=plan.metric_name,
            supporting_citation_ids=tuple(
                supporting_ids
            ),
            reason=(
                "至少一个 Evidence Item "
                "同时包含目标报表标题，且目标指标后方"
                "存在邻近的财务数值"
            ),
            context=context,
        )

    return EvidenceReadinessResult(
        status="insufficient_evidence",
        metric_name=plan.metric_name,
        supporting_citation_ids=(),
        reason=(
            "Top-k 证据未在同一个 Chunk 中"
            "同时提供目标报表标题，以及与目标指标"
            "邻近且可归属的财务数值"
        ),
        context=context,
    )