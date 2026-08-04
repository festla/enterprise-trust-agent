from __future__ import annotations

from app.schemas.chunk import Chunk
from app.schemas.retrieval import (
    RetrievalFilter,
)


def matches_retrieval_filter(
    *,
    chunk: Chunk,
    filters: RetrievalFilter,
) -> bool:
    """判断 Chunk 是否满足全部元数据过滤条件。"""

    if (
        filters.company_ids
        and chunk.company_id
        not in filters.company_ids
    ):
        return False

    if (
        filters.report_ids
        and chunk.report_id
        not in filters.report_ids
    ):
        return False

    if (
        filters.fiscal_years
        and chunk.fiscal_year
        not in filters.fiscal_years
    ):
        return False

    if (
        filters.report_types
        and chunk.report_type
        not in filters.report_types
    ):
        return False

    if (
        filters.document_ids
        and chunk.document_id
        not in filters.document_ids
    ):
        return False

    if (
        filters.page_ids
        and chunk.page_id
        not in filters.page_ids
    ):
        return False

    if (
        filters.pdf_pages
        and chunk.pdf_page
        not in filters.pdf_pages
    ):
        return False

    return True