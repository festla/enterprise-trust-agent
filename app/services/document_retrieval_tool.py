from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from app.rag.hybrid_retriever import (
    HybridRetriever,
)
from app.rag.reranking import (
    RerankerProvider,
    rerank_hybrid_hits,
)
from app.schemas.reranker import (
    RerankedRetrievalHit,
    RerankerRuntimeConfig,
)
from app.schemas.retrieval import (
    RetrievalFilter,
)
from app.schemas.tool_registry import (
    DocumentEvidenceQuery,
    RetrievedDocument,
    RetrieveDocumentsInput,
    RetrieveDocumentsOutput,
    ToolDefinition,
)
from app.services.tool_registry import (
    ToolRegistry,
)


class DocumentRetrievalToolError(
    ValueError
):
    """Runtime 文档检索工具基础异常。"""


# ============================================================
# Tool 不直接依赖某一个具体 Retriever 类，
# 而是依赖一个最小接口：
#
#       search(query, top_k)
#
# 这就是依赖倒置：
#
# RetrieveDocumentsTool
#         ↓
# DocumentHitProvider Protocol
#         ↓
# 具体生产 Provider / Fake Provider 都可以实现它
#
# 后面测试时就不需要真的加载 BGE 和 Cross-Encoder。
# ============================================================

class DocumentHitProvider(Protocol):
    def search(
        self,
        *,
        query: DocumentEvidenceQuery,
        top_k: int,
    ) -> Sequence[
        RerankedRetrievalHit
    ]:
        """返回已经完成 Hybrid + Rerank 的命中。"""


# ============================================================
# 这是 6B 最重要的一层。
#
# 我们没有重新写：
#
#   Dense
#   BM25
#   RRF
#   Cross-Encoder
#
# 只是：
#
#   1. 根据 report_id 找到正确索引
#   2. 构造 RetrievalFilter
#   3. 调用已有 HybridRetriever
#   4. 调用已有 rerank_hybrid_hits
#
# 这仍然属于 thin adapter / routing adapter。
# ============================================================

@dataclass(slots=True)
class RoutedDocumentRerankerProvider:
    """面向 DocumentEvidenceQuery 的报告级检索路由。"""
    """负责“去哪个report的索引搜，以及怎么搜”。"""

    hybrid_retrievers_by_report_id: Mapping[
        str,
        HybridRetriever,
    ]

    reranker_provider: RerankerProvider

    runtime_config: RerankerRuntimeConfig

    provider_id: str = (
        "document_hybrid_reranker_v1"
    )

    def __post_init__(self) -> None:
        normalized_provider_id = (
            self.provider_id.strip()
        )

        if not normalized_provider_id:
            raise DocumentRetrievalToolError(
                "provider_id 不能为空"
            )

        if not (
            self.hybrid_retrievers_by_report_id
        ):
            raise DocumentRetrievalToolError(
                "至少需要配置一个 "
                "HybridRetriever"
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
                raise DocumentRetrievalToolError(
                    "report_id 路由不能为空"
                )

            if (
                normalized_report_id
                != report_id
            ):
                raise DocumentRetrievalToolError(
                    "report_id 路由不能包含"
                    "首尾空白字符"
                )

            if (
                normalized_report_id
                in normalized_routes
            ):
                raise DocumentRetrievalToolError(
                    "出现重复 report_id 路由："
                    f"{normalized_report_id}"
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
        query: DocumentEvidenceQuery,
        top_k: int,
    ) -> tuple[
        RerankedRetrievalHit,
        ...,
    ]:
        # ====================================================
        # top_k 是“调用方希望最终拿到多少条”。
        #
        # runtime_config.return_count 是
        # Reranker 当前最多允许返回多少条。
        #
        # 两层预算必须一致。
        # ====================================================

        if top_k <= 0:
            raise DocumentRetrievalToolError(
                "top_k 必须大于 0"
            )

        if (
            top_k
            > self.runtime_config.return_count
        ):
            raise DocumentRetrievalToolError(
                "top_k 不能大于 "
                "RerankerRuntimeConfig."
                "return_count："
                f"top_k={top_k}, "
                "return_count="
                f"{self.runtime_config.return_count}"
            )

        hybrid_retriever = (
            self
            .hybrid_retrievers_by_report_id
            .get(query.report_id)
        )

        if hybrid_retriever is None:
            raise DocumentRetrievalToolError(
                "没有为文档查询配置 "
                "HybridRetriever："
                f"report_id={query.report_id}"
            )

        # ====================================================
        # 这里是生产 Runtime 非常重要的边界：
        #
        # 只允许使用 Query 本身携带的结构化条件：
        #
        # company_id
        # report_id
        # fiscal_year
        # report_type
        #
        # 绝对不能：
        #
        # filters.pdf_pages = gold_pages
        #
        # 否则就是评测数据泄漏。
        # ====================================================

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

        # ====================================================
        # 第一阶段：
        #
        # Dense + BM25 + RRF
        #
        # 这里传入的不是最终 top_k，
        # 而是 Reranker 需要看到的候选数量。
        #
        # 例如：
        #
        # Hybrid → 50 candidates
        # Reranker → final 5
        # ====================================================

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

        # ====================================================
        # 第二阶段：
        #
        # Cross-Encoder 对 query + chunk text
        # 做更精细的相关性判断。
        #
        # 不在 Tool 里面重新实现排序算法。
        # ====================================================

        reranked_hits = rerank_hybrid_hits(
            query=query.semantic_query,
            hits=hybrid_hits,
            provider=self.reranker_provider,
            config=self.runtime_config,
        )

        selected_hits = tuple(
            reranked_hits[:top_k]
        )

        self._validate_hits(
            query=query,
            hits=selected_hits,
        )

        return selected_hits

    @staticmethod
    def _validate_hits(
        *,
        query: DocumentEvidenceQuery,
        hits: tuple[
            RerankedRetrievalHit,
            ...,
        ],
    ) -> None:
        """防止错误路由结果进入 Agent Runtime。"""

        expected_ranks = tuple(
            range(
                1,
                len(hits) + 1,
            )
        )

        actual_ranks = tuple(
            hit.rank
            for hit in hits
        )

        if actual_ranks != expected_ranks:
            raise DocumentRetrievalToolError(
                "文档检索 rank 必须从 1 "
                "连续递增"
            )

        chunk_ids = tuple(
            hit.chunk_id
            for hit in hits
        )

        if len(chunk_ids) != len(
            set(chunk_ids)
        ):
            raise DocumentRetrievalToolError(
                "文档检索结果不能包含"
                "重复 chunk_id"
            )

        # ====================================================
        # RetrievalFilter 是输入约束。
        #
        # 这里再次验证输出身份，
        # 属于 defense in depth：
        #
        # 不因为“前面已经过滤了”
        # 就盲目信任 Retriever 返回值。
        # ====================================================

        for hit in hits:
            if (
                hit.company_id
                != query.company_id
            ):
                raise DocumentRetrievalToolError(
                    "命中的 company_id "
                    "与 Query 不一致"
                )

            if (
                hit.report_id
                != query.report_id
            ):
                raise DocumentRetrievalToolError(
                    "命中的 report_id "
                    "与 Query 不一致"
                )

            if (
                hit.fiscal_year
                != query.fiscal_year
            ):
                raise DocumentRetrievalToolError(
                    "命中的 fiscal_year "
                    "与 Query 不一致"
                )

            if (
                hit.report_type
                is not query.report_type
            ):
                raise DocumentRetrievalToolError(
                    "命中的 report_type "
                    "与 Query 不一致"
                )


# ============================================================
# Provider 解决：
#
#   “去哪里搜、怎么搜”
#
# Tool 解决：
#
#   “Runtime 暴露什么输入输出”
#
# 两者不要混成一个大类。
# ============================================================

@dataclass(
    frozen=True,
    slots=True,
)
class RetrieveDocumentsTool:
    """把 Hybrid+Reranker 包装成 Runtime 文档工具。"""

    hit_provider: DocumentHitProvider

    tool_name: str = "retrieve_documents"

    tool_version: str = "1.0.0"

    def build_definition(
        self,
    ) -> ToolDefinition:
        return ToolDefinition(
            tool_name=self.tool_name,
            description=(
                "在指定公司年报中执行 "
                "Dense+BM25+RRF+Reranker "
                "文档证据检索。"
            ),
            version=self.tool_version,
            input_schema=(
                RetrieveDocumentsInput
                .model_json_schema()
            ),
            output_schema=(
                RetrieveDocumentsOutput
                .model_json_schema()
            ),
            permission="read_documents",

            # Hybrid + embedding + reranker
            # 明显比 Registry 查询更重。
            timeout_seconds=30.0,

            # 当前底层主要为本地确定性检索。
            # 暂不在这一层自动重试。
            max_retries=0,

            # 文档查询没有副作用。
            idempotent=True,

            # top_k=50 且单段最多 8000 字符，
            # 这里预留足够空间。
            max_result_bytes=2_000_000,
        )

    def handle(
        self,
        input_value: BaseModel,
    ) -> RetrieveDocumentsOutput:
        if not isinstance(
            input_value,
            RetrieveDocumentsInput,
        ):
            raise TypeError(
                "retrieve_documents "
                "必须接受 "
                "RetrieveDocumentsInput"
            )

        query = input_value.query

        hits = self.hit_provider.search(
            query=query,
            top_k=input_value.top_k,
        )

        # ====================================================
        # RerankedRetrievalHit
        #        ↓
        # RetrievedDocument
        #
        # Adapter 的核心，就是明确做这种
        # Contract A → Contract B 的映射。
        #
        # 不应该直接把底层 RerankedRetrievalHit
        # 暴露给 Agent Runtime。
        # ====================================================

        documents = tuple(
            RetrievedDocument(
                query_id=query.query_id,
                rank=hit.rank,
                chunk_id=hit.chunk_id,
                document_id=(
                    hit.document_id
                ),
                page_id=hit.page_id,
                company_id=(
                    hit.company_id
                ),
                report_id=hit.report_id,
                fiscal_year=(
                    hit.fiscal_year
                ),
                pdf_page=hit.pdf_page,
                printed_page=(
                    hit.printed_page
                ),
                score=hit.score,

                section_path=(
                    hit.section_path
                ),
                text=hit.text,
            )
            for hit in hits
        )

        return RetrieveDocumentsOutput(
            query_id=query.query_id,
            documents=documents,
        )

# ============================================================
# Runtime 不直接 new Tool 然后到处传。
#
# 通过统一注册函数完成依赖注入。
# ============================================================

def register_retrieve_documents_tool(
    *,
    tool_registry: ToolRegistry,
    hit_provider: DocumentHitProvider,
) -> RetrieveDocumentsTool:
    tool = RetrieveDocumentsTool(
        hit_provider=hit_provider
    )

    tool_registry.register(
        definition=(
            tool.build_definition()
        ),
        input_model=(
            RetrieveDocumentsInput
        ),
        output_model=(
            RetrieveDocumentsOutput
        ),
        handler=tool.handle,
    )

    return tool