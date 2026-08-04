from __future__ import annotations

from collections.abc import (
    Sequence,
)

import pytest

from app.rag.reranking import (
    DuplicateRerankerChunkError,
    InvalidHybridHitsError,
    InvalidRerankerQueryError,
    InvalidRerankerScoreError,
    RerankerOutputCountMismatchError,
    rerank_hybrid_hits,
)
from app.schemas.enums import (
    ChunkStrategy,
    PageMappingStatus,
    ReportType,
)
from app.schemas.hybrid_retrieval import (
    HybridRetrievalHit,
)
from app.schemas.reranker import (
    RerankerRuntimeConfig,
    RerankerSpec,
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
        "text": "营业收入相关附注",
    },
    "b": {
        "pdf_page": 20,
        "text": (
            "合并资产负债表 "
            "资产总计 100 元"
        ),
    },
    "c": {
        "pdf_page": 30,
        "text": "资产结构与资产质量分析",
    },
    "d": {
        "pdf_page": 40,
        "text": "应收账款期末余额",
    },
}


def build_source_hit(
    *,
    suffix: str,
) -> RetrievalHit:
    values = CHUNK_VALUES[suffix]

    pdf_page = int(
        values["pdf_page"]
    )

    text = str(
        values["text"]
    )

    return RetrievalHit(
        rank=1,
        retriever_type="dense",
        score_type=(
            "cosine_similarity"
        ),
        score=0.8,
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


def build_hybrid_hit(
    *,
    suffix: str,
    rank: int,
    dense_rank: int | None,
    bm25_rank: int | None,
) -> HybridRetrievalHit:
    score = 0.0

    if dense_rank is not None:
        score += 1 / (
            60 + dense_rank
        )

    if bm25_rank is not None:
        score += 1 / (
            60 + bm25_rank
        )

    return (
        HybridRetrievalHit
        .from_source_hit(
            rank=rank,
            rrf_score=score,
            source_hit=(
                build_source_hit(
                    suffix=suffix
                )
            ),
            dense_rank=dense_rank,
            bm25_rank=bm25_rank,
        )
    )


def build_hybrid_hits(
) -> tuple[
    HybridRetrievalHit,
    ...,
]:
    return (
        build_hybrid_hit(
            suffix="a",
            rank=1,
            dense_rank=1,
            bm25_rank=3,
        ),
        build_hybrid_hit(
            suffix="b",
            rank=2,
            dense_rank=2,
            bm25_rank=1,
        ),
        build_hybrid_hit(
            suffix="c",
            rank=3,
            dense_rank=3,
            bm25_rank=4,
        ),
        build_hybrid_hit(
            suffix="d",
            rank=4,
            dense_rank=None,
            bm25_rank=2,
        ),
    )


class FakeRerankerProvider:
    """按候选顺序返回预设分数。"""

    def __init__(
        self,
        *,
        scores: Sequence[object],
        respect_pair_count: bool = True,
    ) -> None:
        self._spec = RerankerSpec(
            model_name="test/fake-reranker",
            model_revision="fake_v1",
            max_length=128,
        )

        self._scores = tuple(
            scores
        )

        self._respect_pair_count = (
            respect_pair_count
        )

        self.pairs_received: tuple[
            tuple[str, str],
            ...,
        ] = ()

    @property
    def spec(self) -> RerankerSpec:
        return self._spec

    def score_pairs(
        self,
        pairs: Sequence[
            tuple[str, str]
        ],
    ) -> Sequence[float]:
        self.pairs_received = tuple(
            pairs
        )

        if self._respect_pair_count:
            return self._scores[
                :len(self.pairs_received)
            ]  # type: ignore[return-value]

        return self._scores  # type: ignore[return-value]


def test_rerank_candidates_by_provider_score(
) -> None:
    provider = FakeRerankerProvider(
        scores=(
            0.1,
            3.0,
            2.0,
            -1.0,
        )
    )

    hits = rerank_hybrid_hits(
        query="资产总计是多少？",
        hits=build_hybrid_hits(),
        provider=provider,
        config=RerankerRuntimeConfig(
            rerank_candidate_count=4,
            return_count=4,
        ),
    )

    assert [
        hit.pdf_page
        for hit in hits
    ] == [
        20,
        30,
        10,
        40,
    ]

    assert [
        hit.rank
        for hit in hits
    ] == [
        1,
        2,
        3,
        4,
    ]


def test_preserve_hybrid_audit_fields(
) -> None:
    provider = FakeRerankerProvider(
        scores=(
            0.1,
            3.0,
            2.0,
            -1.0,
        )
    )

    hits = rerank_hybrid_hits(
        query="资产总计是多少？",
        hits=build_hybrid_hits(),
        provider=provider,
        config=RerankerRuntimeConfig(
            rerank_candidate_count=4,
            return_count=4,
        ),
    )

    first = hits[0]

    assert first.pdf_page == 20
    assert first.rank == 1
    assert first.rrf_rank == 2

    assert first.dense_rank == 2
    assert first.bm25_rank == 1

    assert first.reranker_score == (
        pytest.approx(3.0)
    )

    assert first.score == pytest.approx(
        3.0
    )

    assert first.source_retrievers == (
        "dense",
        "bm25",
    )


def test_equal_scores_preserve_rrf_order(
) -> None:
    provider = FakeRerankerProvider(
        scores=(
            2.0,
            2.0,
            2.0,
            2.0,
        )
    )

    hits = rerank_hybrid_hits(
        query="资产总计是多少？",
        hits=build_hybrid_hits(),
        provider=provider,
        config=RerankerRuntimeConfig(
            rerank_candidate_count=4,
            return_count=4,
        ),
    )

    assert [
        hit.rrf_rank
        for hit in hits
    ] == [
        1,
        2,
        3,
        4,
    ]


def test_apply_rerank_candidate_limit(
) -> None:
    provider = FakeRerankerProvider(
        scores=(
            0.1,
            3.0,
            100.0,
            100.0,
        )
    )

    hits = rerank_hybrid_hits(
        query="资产总计是多少？",
        hits=build_hybrid_hits(),
        provider=provider,
        config=RerankerRuntimeConfig(
            rerank_candidate_count=2,
            return_count=2,
        ),
    )

    assert {
        hit.rrf_rank
        for hit in hits
    } == {
        1,
        2,
    }

    assert len(
        provider.pairs_received
    ) == 2


def test_apply_return_count_after_reranking(
) -> None:
    provider = FakeRerankerProvider(
        scores=(
            0.1,
            3.0,
            2.0,
            -1.0,
        )
    )

    hits = rerank_hybrid_hits(
        query="资产总计是多少？",
        hits=build_hybrid_hits(),
        provider=provider,
        config=RerankerRuntimeConfig(
            rerank_candidate_count=4,
            return_count=2,
        ),
    )

    assert len(hits) == 2

    assert [
        hit.pdf_page
        for hit in hits
    ] == [
        20,
        30,
    ]


def test_provider_receives_query_and_text(
) -> None:
    provider = FakeRerankerProvider(
        scores=(
            1.0,
            0.0,
        )
    )

    rerank_hybrid_hits(
        query="资产总计是多少？",
        hits=build_hybrid_hits(),
        provider=provider,
        config=RerankerRuntimeConfig(
            rerank_candidate_count=2,
            return_count=2,
        ),
    )

    assert provider.pairs_received == (
        (
            "资产总计是多少？",
            "营业收入相关附注",
        ),
        (
            "资产总计是多少？",
            (
                "合并资产负债表 "
                "资产总计 100 元"
            ),
        ),
    )


def test_empty_hits_return_empty(
) -> None:
    provider = FakeRerankerProvider(
        scores=()
    )

    hits = rerank_hybrid_hits(
        query="资产总计是多少？",
        hits=(),
        provider=provider,
    )

    assert hits == ()
    assert provider.pairs_received == ()


def test_reject_blank_query() -> None:
    provider = FakeRerankerProvider(
        scores=(1.0,)
    )

    with pytest.raises(
        InvalidRerankerQueryError,
    ):
        rerank_hybrid_hits(
            query="   ",
            hits=build_hybrid_hits(),
            provider=provider,
        )


def test_reject_non_continuous_rrf_ranks(
) -> None:
    hits = (
        build_hybrid_hit(
            suffix="a",
            rank=1,
            dense_rank=1,
            bm25_rank=3,
        ),
        build_hybrid_hit(
            suffix="b",
            rank=3,
            dense_rank=2,
            bm25_rank=1,
        ),
    )

    provider = FakeRerankerProvider(
        scores=(1.0, 2.0)
    )

    with pytest.raises(
        InvalidHybridHitsError,
        match="连续递增",
    ):
        rerank_hybrid_hits(
            query="资产总计是多少？",
            hits=hits,
            provider=provider,
        )


def test_reject_wrong_retriever_type(
) -> None:
    invalid_hit = (
        build_hybrid_hits()[0]
        .model_copy(
            update={
                "retriever_type": "dense",
            }
        )
    )

    provider = FakeRerankerProvider(
        scores=(1.0,)
    )

    with pytest.raises(
        InvalidHybridHitsError,
        match="hybrid_rrf",
    ):
        rerank_hybrid_hits(
            query="资产总计是多少？",
            hits=(invalid_hit,),
            provider=provider,
        )


def test_reject_duplicate_chunk_id(
) -> None:
    hits = (
        build_hybrid_hit(
            suffix="a",
            rank=1,
            dense_rank=1,
            bm25_rank=3,
        ),
        build_hybrid_hit(
            suffix="a",
            rank=2,
            dense_rank=2,
            bm25_rank=1,
        ),
    )

    provider = FakeRerankerProvider(
        scores=(1.0, 2.0)
    )

    with pytest.raises(
        DuplicateRerankerChunkError,
        match="重复 chunk_id",
    ):
        rerank_hybrid_hits(
            query="资产总计是多少？",
            hits=hits,
            provider=provider,
        )


def test_reject_provider_output_count_mismatch(
) -> None:
    provider = FakeRerankerProvider(
        scores=(1.0,),
        respect_pair_count=False,
    )

    with pytest.raises(
        RerankerOutputCountMismatchError,
        match="数量",
    ):
        rerank_hybrid_hits(
            query="资产总计是多少？",
            hits=build_hybrid_hits(),
            provider=provider,
            config=RerankerRuntimeConfig(
                rerank_candidate_count=4,
                return_count=4,
            ),
        )


@pytest.mark.parametrize(
    "invalid_score",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
        "1.0",
        True,
        None,
    ),
)
def test_reject_invalid_provider_score(
    invalid_score: object,
) -> None:
    provider = FakeRerankerProvider(
        scores=(
            invalid_score,
        )
    )

    with pytest.raises(
        InvalidRerankerScoreError,
    ):
        rerank_hybrid_hits(
            query="资产总计是多少？",
            hits=(
                build_hybrid_hits()[0],
            ),
            provider=provider,
            config=RerankerRuntimeConfig(
                rerank_candidate_count=1,
                return_count=1,
            ),
        )