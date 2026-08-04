from __future__ import annotations

from app.rag.rrf import (
    reciprocal_rank_fusion,
)
from app.schemas.enums import (
    ChunkStrategy,
    PageMappingStatus,
    ReportType,
)
from app.schemas.hybrid_retrieval import (
    RRFConfig,
)
from app.schemas.retrieval import (
    RetrievalHit,
)


REPORT_ID = "midea_group_2024"

DOCUMENT_ID = (
    f"doc_{REPORT_ID}_{'a' * 24}"
)

CHUNK_DATASET_ID = (
    f"chunk_dataset_{REPORT_ID}_"
    f"{'b' * 24}"
)


CHUNK_VALUES = {
    "a": {
        "pdf_page": 10,
        "text": "合并利润表营业收入",
    },
    "b": {
        "pdf_page": 20,
        "text": "合并资产负债表资产总计",
    },
    "c": {
        "pdf_page": 30,
        "text": (
            "经营活动产生的现金流量净额"
        ),
    },
    "d": {
        "pdf_page": 40,
        "text": "应收账款期末余额",
    },
}


def build_hit(
    *,
    suffix: str,
    rank: int,
    retriever_type: str,
) -> RetrievalHit:
    values = CHUNK_VALUES[suffix]

    pdf_page = int(
        values["pdf_page"]
    )

    text = str(
        values["text"]
    )

    if retriever_type == "dense":
        score_type = "cosine_similarity"
        score = 0.95 - rank * 0.05
    else:
        score_type = "bm25"
        score = 30.0 - rank

    return RetrievalHit(
        rank=rank,
        retriever_type=retriever_type,
        score_type=score_type,
        score=score,
        chunk_id=(
            f"chunk_{REPORT_ID}_"
            f"{suffix * 24}"
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
        chunk_index=pdf_page,
        strategy=(
            ChunkStrategy.FIXED_LENGTH
        ),
        source_start_char=0,
        source_end_char=len(text),
        section_path=(),
        text=text,
    )


def main() -> None:
    dense_hits = (
        build_hit(
            suffix="a",
            rank=1,
            retriever_type="dense",
        ),
        build_hit(
            suffix="b",
            rank=2,
            retriever_type="dense",
        ),
        build_hit(
            suffix="c",
            rank=3,
            retriever_type="dense",
        ),
    )

    bm25_hits = (
        build_hit(
            suffix="b",
            rank=1,
            retriever_type="bm25",
        ),
        build_hit(
            suffix="d",
            rank=2,
            retriever_type="bm25",
        ),
        build_hit(
            suffix="a",
            rank=3,
            retriever_type="bm25",
        ),
    )

    config = RRFConfig(
        rank_constant=60,
        dense_candidate_count=3,
        bm25_candidate_count=3,
    )

    hits = reciprocal_rank_fusion(
        dense_hits=dense_hits,
        bm25_hits=bm25_hits,
        config=config,
        top_k=4,
    )

    print(
        "dense_order="
        + ",".join(
            hit.chunk_id[-24]
            for hit in dense_hits
        )
    )

    print(
        "bm25_order="
        + ",".join(
            hit.chunk_id[-24]
            for hit in bm25_hits
        )
    )

    for hit in hits:
        print(
            f"rank={hit.rank} "
            f"chunk={hit.chunk_id[-24]} "
            f"dense_rank={hit.dense_rank} "
            f"bm25_rank={hit.bm25_rank} "
            f"rrf_score={hit.rrf_score:.8f} "
            f"sources="
            f"{','.join(hit.source_retrievers)}"
        )

    expected_order = (
        "b" * 24,
        "a" * 24,
        "d" * 24,
        "c" * 24,
    )

    actual_order = tuple(
        hit.chunk_id[-24:]
        for hit in hits
    )

    if actual_order != expected_order:
        raise RuntimeError(
            "RRF Smoke Test 排名不符合预期："
            f"{actual_order}"
        )

    first = hits[0]

    expected_first_score = (
        1 / (60 + 2)
        + 1 / (60 + 1)
    )

    if (
        abs(
            first.rrf_score
            - expected_first_score
        )
        > 1e-12
    ):
        raise RuntimeError(
            "RRF Smoke Test 分数计算错误"
        )

    print("smoke_test_passed=true")


if __name__ == "__main__":
    main()