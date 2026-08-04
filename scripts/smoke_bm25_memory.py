from __future__ import annotations

import hashlib

from app.rag.bm25 import (
    ExactBM25Index,
)
from app.rag.tokenization import (
    DeterministicChineseBigramTokenizer,
)
from app.schemas.chunk import Chunk
from app.schemas.enums import (
    ChunkStrategy,
    PageContentType,
    PageMappingStatus,
    PageParseStatus,
    ReportType,
)


REPORT_ID = "midea_group_2024"

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


def build_chunk(
    *,
    suffix: str,
    text: str,
    pdf_page: int,
) -> Chunk:
    return Chunk(
        chunk_id=(
            f"chunk_{REPORT_ID}_{suffix * 24}"
        ),
        chunk_dataset_id=CHUNK_DATASET_ID,
        page_dataset_id=PAGE_DATASET_ID,
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
        content_type=PageContentType.TEXT,
        parse_status=PageParseStatus.SUCCESS,
        chunk_index=pdf_page - 1,
        strategy=ChunkStrategy.FIXED_LENGTH,
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
        text_sha256=hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest(),
    )


def main() -> None:
    chunks = (
        build_chunk(
            suffix="1",
            text=(
                "存货会计政策与存货计量方法，"
                "公司采用成本与可变现净值"
                "孰低计量。"
            ),
            pdf_page=1,
        ),
        build_chunk(
            suffix="2",
            text=(
                "公司计提存货跌价准备，"
                "并说明存货跌价准备"
                "转回方法。"
            ),
            pdf_page=2,
        ),
        build_chunk(
            suffix="3",
            text=(
                "2024年末合并口径的"
                "资产负债表中，"
                "存货金额为12345元。"
            ),
            pdf_page=3,
        ),
        build_chunk(
            suffix="4",
            text=(
                "2024年度合并利润表中，"
                "营业收入金额为98765元。"
            ),
            pdf_page=4,
        ),
    )

    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    index = ExactBM25Index.build(
        chunks=chunks,
        tokenizer=tokenizer,
    )

    query = (
        "2024年末合并口径的"
        "存货是多少？"
    )

    hits = index.search(
        query=query,
        tokenizer=tokenizer,
        top_k=4,
    )

    print(f"query={query}")
    print(f"candidate_count={len(hits)}")

    for hit in hits:
        print(
            "rank="
            f"{hit.rank} "
            "pdf_page="
            f"{hit.pdf_page} "
            "retriever_type="
            f"{hit.retriever_type} "
            "score="
            f"{hit.score:.6f}"
        )

        print(f"text={hit.text}")

    if hits[0].pdf_page != 3:
        raise RuntimeError(
            "BM25 极小样本验收失败："
            "正式报表事实 Chunk 未排在第一"
        )

    print("smoke_test_passed=true")


if __name__ == "__main__":
    main()