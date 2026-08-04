from __future__ import annotations

from math import log2
from typing import (
    Any,
    Literal,
    Self,
)
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .enums import (
    ChunkStrategy,
    ReportType,
    StatementScope,
    StatementType,
)
from .retrieval import RetrievalHit
from app.schemas.hybrid_retrieval import (
    HybridRetrievalHit,
)
from app.schemas.reranker import (
    RerankedRetrievalHit,
)
RetrievalMethod = Literal[
    "dense",
    "bm25",
    "hybrid_rrf",
    "hybrid_reranker",
]


_INDEX_ID_PATTERN = (
    r"^(?:"
    r"vector_index"
    r"|bm25_index"
    r"|hybrid_run"
    r"|reranker_run"
    r")_"
    r"[a-z0-9_]+_[0-9a-f]{24}$"
)


def _read_hit_value(
    hit: object,
    field_name: str,
) -> Any:
    """同时支持模型对象和反序列化字典。"""

    if isinstance(hit, dict):
        return hit[field_name]

    return getattr(
        hit,
        field_name,
    )


def _calculate_page_level_ndcg(
    *,
    ranked_pdf_pages: tuple[int, ...],
    gold_pdf_pages: tuple[int, ...],
    top_k: int,
) -> float:
    """计算去重 Gold 页后的二元 NDCG。"""

    gold_page_set = set(
        gold_pdf_pages
    )

    seen_relevant_pages: set[int] = set()

    dcg = 0.0

    for rank, pdf_page in enumerate(
        ranked_pdf_pages[:top_k],
        start=1,
    ):
        if (
            pdf_page in gold_page_set
            and pdf_page
            not in seen_relevant_pages
        ):
            dcg += 1.0 / log2(rank + 1)

            seen_relevant_pages.add(
                pdf_page
            )

    ideal_relevant_count = min(
        top_k,
        len(gold_page_set),
    )

    idcg = sum(
        1.0 / log2(rank + 1)
        for rank in range(
            1,
            ideal_relevant_count + 1,
        )
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


class FinancialFactRetrievalEvalCase(
    BaseModel
):
    """一条经过人工核验的财务事实检索题。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: int = Field(
        default=1,
        ge=1,
        le=1,
    )

    case_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    question: str = Field(
        min_length=1,
    )

    metric_name: str = Field(
        min_length=1,
    )

    company_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    report_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    fiscal_year: int = Field(
        ge=2000,
        le=2100,
    )

    report_type: ReportType

    statement_type: StatementType

    statement_scope: StatementScope

    gold_pdf_pages: tuple[int, ...]

    @field_validator("gold_pdf_pages")
    @classmethod
    def normalize_gold_pdf_pages(
        cls,
        value: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Gold 页面必须非空、合法且顺序稳定。"""

        if not value:
            raise ValueError(
                "gold_pdf_pages 不能为空"
            )

        if any(page < 1 for page in value):
            raise ValueError(
                "Gold PDF 页码必须大于等于 1"
            )

        return tuple(
            sorted(set(value))
        )

    @model_validator(mode="after")
    def validate_report_identity(
        self,
    ) -> Self:
        """当前数据约定下，报告 ID 绑定公司和年份。"""

        expected_report_id = (
            f"{self.company_id}_"
            f"{self.fiscal_year}"
        )

        if self.report_id != expected_report_id:
            raise ValueError(
                "report_id 必须由 company_id "
                "和 fiscal_year 组成"
            )

        return self


class RetrievalEvalResult(BaseModel):
    """一条题目在一个检索索引上的评测结果。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
        allow_inf_nan=False,
        populate_by_name=True,
    )

    schema_version: int = Field(
        default=1,
        ge=1,
        le=1,
    )

    case_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    question: str = Field(
        min_length=1,
    )

    semantic_query: str = Field(
        min_length=1,
    )

    company_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    report_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    fiscal_year: int = Field(
        ge=2000,
        le=2100,
    )

    strategy: ChunkStrategy

    retriever_type: RetrievalMethod = (
        "dense"
    )

    chunk_dataset_id: str = Field(
        pattern=(
            r"^chunk_dataset_[a-z0-9_]+_"
            r"[0-9a-f]{24}$"
        ),
    )

    index_id: str = Field(
        validation_alias=AliasChoices(
            "index_id",
            "vector_index_id",
        ),
        pattern=_INDEX_ID_PATTERN,
    )

    gold_pdf_pages: tuple[int, ...]

    evaluated_hit_count: int = Field(
        ge=0,
    )

    first_relevant_rank: int | None = Field(
        default=None,
        ge=1,
    )

    recall_at_1: bool
    recall_at_3: bool
    recall_at_5: bool

    reciprocal_rank: float = Field(
        ge=0,
        le=1,
    )

    ndcg_at_1: float = Field(
        ge=0,
        le=1,
    )

    ndcg_at_3: float = Field(
        ge=0,
        le=1,
    )

    ndcg_at_5: float = Field(
        ge=0,
        le=1,
    )

    top_hits: tuple[
        RerankedRetrievalHit
        | HybridRetrievalHit
        | RetrievalHit,
        ...,
    ] = ()

    top_pdf_pages: tuple[int, ...] = ()

    top_scores: tuple[float, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def populate_derived_metrics(
        cls,
        value: object,
    ) -> object:
        """兼容旧构造方式并生成派生指标。"""

        if not isinstance(value, dict):
            return value

        values = dict(value)

        raw_hits = tuple(
            values.get(
                "top_hits",
                (),
            )
        )

        pdf_pages = tuple(
            int(
                _read_hit_value(
                    hit,
                    "pdf_page",
                )
            )
            for hit in raw_hits
        )

        scores = tuple(
            float(
                _read_hit_value(
                    hit,
                    "score",
                )
            )
            for hit in raw_hits
        )

        values.setdefault(
            "top_pdf_pages",
            pdf_pages,
        )

        values.setdefault(
            "top_scores",
            scores,
        )

        first_rank = values.get(
            "first_relevant_rank"
        )

        reciprocal_rank = (
            0.0
            if first_rank is None
            else 1.0 / int(first_rank)
        )

        values.setdefault(
            "reciprocal_rank",
            reciprocal_rank,
        )

        gold_pdf_pages = tuple(
            values.get(
                "gold_pdf_pages",
                (),
            )
        )

        values.setdefault(
            "ndcg_at_1",
            _calculate_page_level_ndcg(
                ranked_pdf_pages=pdf_pages,
                gold_pdf_pages=(
                    gold_pdf_pages
                ),
                top_k=1,
            ),
        )

        values.setdefault(
            "ndcg_at_3",
            _calculate_page_level_ndcg(
                ranked_pdf_pages=pdf_pages,
                gold_pdf_pages=(
                    gold_pdf_pages
                ),
                top_k=3,
            ),
        )

        values.setdefault(
            "ndcg_at_5",
            _calculate_page_level_ndcg(
                ranked_pdf_pages=pdf_pages,
                gold_pdf_pages=(
                    gold_pdf_pages
                ),
                top_k=5,
            ),
        )

        return values

    @property
    def vector_index_id(self) -> str:
        """兼容旧 Dense 评测代码。"""

        return self.index_id

    @model_validator(mode="after")
    def validate_metrics(
        self,
    ) -> Self:
        """检查来源、排名和派生指标。"""

        expected_prefix_by_retriever = {
            "dense": "vector_index_",
            "bm25": "bm25_index_",
            "hybrid_rrf": "hybrid_run_",
            "hybrid_reranker": (
                "reranker_run_"
            ),
        }
        expected_prefix = (
            expected_prefix_by_retriever[
                self.retriever_type
            ]
        )

        if not self.index_id.startswith(
            expected_prefix
        ):
            raise ValueError(
                "index_id 与 retriever_type "
                "不一致"
            )

        if len(self.top_hits) > 5:
            raise ValueError(
                "top_hits 最多保存 5 条"
            )

        expected_top_ranks = tuple(
            range(
                1,
                len(self.top_hits) + 1,
            )
        )

        actual_top_ranks = tuple(
            hit.rank
            for hit in self.top_hits
        )

        if actual_top_ranks != expected_top_ranks:
            raise ValueError(
                "top_hits 的 rank 必须从 1 "
                "开始连续递增"
            )

        if any(
            hit.retriever_type
            != self.retriever_type
            for hit in self.top_hits
        ):
            raise ValueError(
                "top_hits 的检索器类型不一致"
            )

        expected_pdf_pages = tuple(
            hit.pdf_page
            for hit in self.top_hits
        )

        if (
            self.top_pdf_pages
            != expected_pdf_pages
        ):
            raise ValueError(
                "top_pdf_pages 与 top_hits 不一致"
            )

        expected_scores = tuple(
            hit.score
            for hit in self.top_hits
        )

        if (
            len(self.top_scores)
            != len(expected_scores)
            or any(
                abs(actual - expected) > 1e-12
                for actual, expected
                in zip(
                    self.top_scores,
                    expected_scores,
                    strict=True,
                )
            )
        ):
            raise ValueError(
                "top_scores 与 top_hits 不一致"
            )

        if (
            self.evaluated_hit_count
            < len(self.top_hits)
        ):
            raise ValueError(
                "评测结果数量不能少于 "
                "保存的 Top Hits 数量"
            )

        rank = self.first_relevant_rank

        if rank is None:
            if (
                self.recall_at_1
                or self.recall_at_3
                or self.recall_at_5
            ):
                raise ValueError(
                    "未命中时 Recall 必须均为 False"
                )

        else:
            if rank > self.evaluated_hit_count:
                raise ValueError(
                    "首次命中排名不能超过 "
                    "评测结果数量"
                )

            if self.recall_at_1 != (rank <= 1):
                raise ValueError(
                    "recall_at_1 与排名不一致"
                )

            if self.recall_at_3 != (rank <= 3):
                raise ValueError(
                    "recall_at_3 与排名不一致"
                )

            if self.recall_at_5 != (rank <= 5):
                raise ValueError(
                    "recall_at_5 与排名不一致"
                )

        expected_reciprocal_rank = (
            0.0
            if rank is None
            else 1.0 / rank
        )

        if (
            abs(
                self.reciprocal_rank
                - expected_reciprocal_rank
            )
            > 1e-12
        ):
            raise ValueError(
                "reciprocal_rank "
                "与首次命中排名不一致"
            )

        expected_ndcg_values = (
            _calculate_page_level_ndcg(
                ranked_pdf_pages=(
                    self.top_pdf_pages
                ),
                gold_pdf_pages=(
                    self.gold_pdf_pages
                ),
                top_k=1,
            ),
            _calculate_page_level_ndcg(
                ranked_pdf_pages=(
                    self.top_pdf_pages
                ),
                gold_pdf_pages=(
                    self.gold_pdf_pages
                ),
                top_k=3,
            ),
            _calculate_page_level_ndcg(
                ranked_pdf_pages=(
                    self.top_pdf_pages
                ),
                gold_pdf_pages=(
                    self.gold_pdf_pages
                ),
                top_k=5,
            ),
        )

        actual_ndcg_values = (
            self.ndcg_at_1,
            self.ndcg_at_3,
            self.ndcg_at_5,
        )

        if any(
            abs(actual - expected) > 1e-12
            for actual, expected
            in zip(
                actual_ndcg_values,
                expected_ndcg_values,
                strict=True,
            )
        ):
            raise ValueError(
                "NDCG 与 Top Hits 不一致"
            )

        return self


class RetrievalEvalSummary(BaseModel):
    """一个评测集在同一索引上的汇总结果。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
        populate_by_name=True,
    )

    schema_version: int = Field(
        default=1,
        ge=1,
        le=1,
    )

    evaluation_set_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    strategy: ChunkStrategy

    retriever_type: RetrievalMethod = (
        "dense"
    )

    chunk_dataset_id: str = Field(
        pattern=(
            r"^chunk_dataset_[a-z0-9_]+_"
            r"[0-9a-f]{24}$"
        ),
    )

    index_id: str = Field(
        validation_alias=AliasChoices(
            "index_id",
            "vector_index_id",
        ),
        pattern=_INDEX_ID_PATTERN,
    )

    case_ids: tuple[str, ...]

    case_count: int = Field(
        ge=1,
    )

    hit_at_1_count: int = Field(
        ge=0,
    )

    hit_at_3_count: int = Field(
        ge=0,
    )

    hit_at_5_count: int = Field(
        ge=0,
    )

    recall_at_1: float = Field(
        ge=0,
        le=1,
    )

    recall_at_3: float = Field(
        ge=0,
        le=1,
    )

    recall_at_5: float = Field(
        ge=0,
        le=1,
    )

    mrr: float = Field(
        default=0,
        ge=0,
        le=1,
    )

    ndcg_at_1: float = Field(
        default=0,
        ge=0,
        le=1,
    )

    ndcg_at_3: float = Field(
        default=0,
        ge=0,
        le=1,
    )

    ndcg_at_5: float = Field(
        default=0,
        ge=0,
        le=1,
    )

    @property
    def vector_index_id(self) -> str:
        """兼容旧 Dense 评测代码。"""

        return self.index_id

    @field_validator("case_ids")
    @classmethod
    def validate_case_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not value:
            raise ValueError(
                "case_ids 不能为空"
            )

        if len(value) != len(set(value)):
            raise ValueError(
                "case_ids 不能重复"
            )

        return value

    @model_validator(mode="after")
    def validate_summary(
        self,
    ) -> Self:
        expected_prefix_by_retriever = {
            "dense": "vector_index_",
            "bm25": "bm25_index_",
            "hybrid_rrf": "hybrid_run_",
            "hybrid_reranker": (
                "reranker_run_"
            ),
        }

        expected_prefix = (
            expected_prefix_by_retriever[
                self.retriever_type
            ]
        )

        if not self.index_id.startswith(
            expected_prefix
        ):
            raise ValueError(
                "index_id 与 retriever_type "
                "不一致"
            )

        if len(self.case_ids) != self.case_count:
            raise ValueError(
                "case_ids 数量必须与 "
                "case_count 一致"
            )

        counts = (
            self.hit_at_1_count,
            self.hit_at_3_count,
            self.hit_at_5_count,
        )

        if any(
            count > self.case_count
            for count in counts
        ):
            raise ValueError(
                "命中数量不能超过题目数量"
            )

        if not (
            self.hit_at_1_count
            <= self.hit_at_3_count
            <= self.hit_at_5_count
        ):
            raise ValueError(
                "Top-k 命中数量必须单调递增"
            )

        expected_recalls = tuple(
            count / self.case_count
            for count in counts
        )

        actual_recalls = (
            self.recall_at_1,
            self.recall_at_3,
            self.recall_at_5,
        )

        if any(
            abs(actual - expected) > 1e-12
            for actual, expected
            in zip(
                actual_recalls,
                expected_recalls,
                strict=True,
            )
        ):
            raise ValueError(
                "Recall 与命中数量不一致"
            )

        return self