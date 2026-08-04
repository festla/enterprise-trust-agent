from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.rag.hybrid_retriever import (
    HybridRetriever,
)
from app.rag.reranking import (
    RerankerProvider,
    rerank_hybrid_hits,
)
from app.schemas.complex_plan_eval_result import (
    ComplexRetrievalQueryOutput,
)
from app.schemas.reranker import (
    RerankedRetrievalHit,
    RerankerRuntimeConfig,
)
from app.schemas.retrieval import (
    RetrievalFilter,
)


class ComplexHybridRerankerProviderError(
    ValueError
):
    """复杂问题 Hybrid+Reranker 路由失败。"""


@dataclass(slots=True)
class RoutedHybridRerankerHitProvider:
    """按照 report_id 路由到对应 Hybrid 索引并重排。"""

    hybrid_retrievers_by_report_id: Mapping[
        str,
        HybridRetriever,
    ]

    reranker_provider: RerankerProvider

    runtime_config: RerankerRuntimeConfig

    provider_id: str

    def __post_init__(self) -> None:
        normalized_provider_id = (
            self.provider_id.strip()
        )

        if not normalized_provider_id:
            raise (
                ComplexHybridRerankerProviderError(
                    "provider_id 不能为空"
                )
            )

        if not (
            self.hybrid_retrievers_by_report_id
        ):
            raise (
                ComplexHybridRerankerProviderError(
                    "至少需要配置一个 "
                    "HybridRetriever"
                )
            )

        normalized_routes: dict[
            str,
            HybridRetriever,
        ] = {}

        for (
            report_id,
            retriever,
        ) in (
            self
            .hybrid_retrievers_by_report_id
            .items()
        ):
            normalized_report_id = (
                report_id.strip()
            )

            if not normalized_report_id:
                raise (
                    ComplexHybridRerankerProviderError(
                        "路由中的 report_id "
                        "不能为空"
                    )
                )

            if (
                normalized_report_id
                in normalized_routes
            ):
                raise (
                    ComplexHybridRerankerProviderError(
                        "路由中出现重复 "
                        f"report_id："
                        f"{normalized_report_id}"
                    )
                )

            normalized_routes[
                normalized_report_id
            ] = retriever

        self.provider_id = (
            normalized_provider_id
        )

        self.hybrid_retrievers_by_report_id = (
            normalized_routes
        )

    def search(
        self,
        *,
        query: ComplexRetrievalQueryOutput,
        top_k: int,
    ) -> Sequence[
        RerankedRetrievalHit
    ]:
        """执行报告路由、Hybrid 检索和 Reranker。"""

        if top_k <= 0:
            raise (
                ComplexHybridRerankerProviderError(
                    "top_k 必须大于 0"
                )
            )

        if (
            top_k
            > self.runtime_config.return_count
        ):
            raise (
                ComplexHybridRerankerProviderError(
                    "top_k 不能大于 "
                    "RerankerRuntimeConfig."
                    "return_count："
                    f"top_k={top_k}, "
                    "return_count="
                    f"{self.runtime_config.return_count}"
                )
            )

        hybrid_retriever = (
            self
            .hybrid_retrievers_by_report_id
            .get(query.report_id)
        )

        if hybrid_retriever is None:
            raise (
                ComplexHybridRerankerProviderError(
                    "没有为 Query 配置 "
                    "HybridRetriever："
                    f"report_id={query.report_id}"
                )
            )

        # 这里只使用 Query 自己的结构化身份。
        # 不能使用 Gold PDF 页码。
        filters = RetrievalFilter(
            company_ids=(
                query.company_id,
            ),
            report_ids=(
                query.report_id,
            ),
            fiscal_years=(
                query.fiscal_year,
            ),
            report_types=(
                query.report_type,
            ),
            document_ids=(),
            page_ids=(),
            pdf_pages=(),
        )

        hybrid_hits = (
            hybrid_retriever.search(
                query=query.semantic_query,
                top_k=(
                    self.runtime_config
                    .rerank_candidate_count
                ),
                filters=filters,
            )
        )

        reranked_hits = rerank_hybrid_hits(
            query=query.semantic_query,
            hits=hybrid_hits,
            provider=self.reranker_provider,
            config=self.runtime_config,
        )

        # runtime_config 可能返回多于本次 Oracle
        # 需要的数量，这里保留最终 Top-k。
        return tuple(
            reranked_hits[:top_k]
        )