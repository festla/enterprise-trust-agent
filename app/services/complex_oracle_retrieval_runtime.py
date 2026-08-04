from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.rag.embedding import (
    EmbeddingProvider,
)
from app.rag.embedders import (
    BGE_SMALL_ZH_V15_REVISION,
    SentenceTransformerEmbeddingProvider,
    build_bge_small_zh_v15_spec,
)
from app.rag.hybrid_retriever import (
    HybridRetriever,
)
from app.rag.reranking import (
    RerankerProvider,
)
from app.rag.rerankers import (
    SentenceTransformerCrossEncoderProvider,
)
from app.rag.tokenization import (
    DeterministicChineseBigramTokenizer,
)
from app.schemas.hybrid_retrieval import (
    RRFConfig,
)
from app.schemas.reranker import (
    RerankerRuntimeConfig,
    RerankerSpec,
)
from app.services.bm25_index import (
    load_bm25_index,
)
from app.services.complex_hybrid_reranker_provider import (
    RoutedHybridRerankerHitProvider,
)
from app.services.vector_index import (
    build_vector_index,
)


RERANKER_MODEL_NAME = (
    "BAAI/bge-reranker-base"
)

RERANKER_MODEL_REVISION = (
    "2cfc18c9415c912f9d8155881c133215df768a70"
)


class ComplexOracleRetrievalRuntimeError(
    ValueError
):
    """复杂问题真实检索运行时装配失败。"""


@dataclass(frozen=True, slots=True)
class ComplexReportRetrievalRoute:
    """一份报告对应的 Chunk 和 BM25 索引路径。"""

    report_id: str

    chunk_dataset_directory: Path

    bm25_index_directory: Path

    def __post_init__(self) -> None:
        if not self.report_id.strip():
            raise (
                ComplexOracleRetrievalRuntimeError(
                    "route.report_id 不能为空"
                )
            )

        if self.report_id != self.report_id.strip():
            raise (
                ComplexOracleRetrievalRuntimeError(
                    "route.report_id "
                    "不能包含首尾空白"
                )
            )


def build_complex_oracle_hit_provider(
    *,
    routes: Sequence[
        ComplexReportRetrievalRoute
    ],
    vector_index_output_root: Path,
    embedding_provider: EmbeddingProvider,
    reranker_provider: RerankerProvider,
    rrf_config: RRFConfig,
    reranker_runtime_config: (
        RerankerRuntimeConfig
    ),
    provider_id: str,
) -> RoutedHybridRerankerHitProvider:
    """加载各报告索引并构造路由检索器。"""

    if not routes:
        raise (
            ComplexOracleRetrievalRuntimeError(
                "routes 不能为空"
            )
        )

    if not provider_id.strip():
        raise (
            ComplexOracleRetrievalRuntimeError(
                "provider_id 不能为空"
            )
        )

    if not vector_index_output_root.is_dir():
        raise (
            ComplexOracleRetrievalRuntimeError(
                "Vector Index 根目录不存在："
                f"{vector_index_output_root}"
            )
        )

    report_ids = [
        route.report_id
        for route in routes
    ]

    if len(report_ids) != len(
        set(report_ids)
    ):
        raise (
            ComplexOracleRetrievalRuntimeError(
                "routes 包含重复 report_id"
            )
        )

    hybrid_retrievers: dict[
        str,
        HybridRetriever,
    ] = {}

    for route in routes:
        if not (
            route
            .chunk_dataset_directory
            .is_dir()
        ):
            raise (
                ComplexOracleRetrievalRuntimeError(
                    "ChunkDataset 目录不存在："
                    f"{route.chunk_dataset_directory}"
                )
            )

        if not (
            route
            .bm25_index_directory
            .is_dir()
        ):
            raise (
                ComplexOracleRetrievalRuntimeError(
                    "BM25 Index 目录不存在："
                    f"{route.bm25_index_directory}"
                )
            )

        dense_result = build_vector_index(
            chunk_dataset_directory=(
                route
                .chunk_dataset_directory
            ),
            output_root=(
                vector_index_output_root
            ),
            provider=embedding_provider,
        )

        bm25_result = load_bm25_index(
            route.bm25_index_directory
        )

        dense_manifest = (
            dense_result.manifest
        )

        bm25_manifest = (
            bm25_result.manifest
        )

        if (
            dense_manifest.report_id
            != route.report_id
        ):
            raise (
                ComplexOracleRetrievalRuntimeError(
                    "Dense Index 的 report_id "
                    "与 Route 不一致："
                    f"route={route.report_id}, "
                    "dense="
                    f"{dense_manifest.report_id}"
                )
            )

        if (
            bm25_manifest.report_id
            != route.report_id
        ):
            raise (
                ComplexOracleRetrievalRuntimeError(
                    "BM25 Index 的 report_id "
                    "与 Route 不一致："
                    f"route={route.report_id}, "
                    "bm25="
                    f"{bm25_manifest.report_id}"
                )
            )

        _validate_index_identity(
            dense_manifest=(
                dense_manifest
            ),
            bm25_manifest=(
                bm25_manifest
            ),
        )

        tokenizer = (
            DeterministicChineseBigramTokenizer(
                spec=(
                    bm25_manifest
                    .tokenizer_spec
                )
            )
        )

        hybrid_retriever = (
            HybridRetriever(
                dense_index=(
                    dense_result.index
                ),
                bm25_index=(
                    bm25_result.index
                ),
                provider=(
                    embedding_provider
                ),
                tokenizer=tokenizer,
                config=rrf_config,
            )
        )

        hybrid_retrievers[
            route.report_id
        ] = hybrid_retriever

    return RoutedHybridRerankerHitProvider(
        hybrid_retrievers_by_report_id=(
            hybrid_retrievers
        ),
        reranker_provider=(
            reranker_provider
        ),
        runtime_config=(
            reranker_runtime_config
        ),
        provider_id=(
            provider_id.strip()
        ),
    )


def build_default_complex_oracle_hit_provider(
    *,
    routes: Sequence[
        ComplexReportRetrievalRoute
    ],
    vector_index_output_root: Path,
    device: str = "cpu",
    embedding_batch_size: int = 16,
    reranker_batch_size: int = 8,
    local_files_only: bool = False,
    dense_candidate_count: int = 50,
    bm25_candidate_count: int = 50,
    rerank_candidate_count: int = 50,
    return_count: int = 5,
    rank_constant: int = 60,
    embedding_model_revision: str = (
        BGE_SMALL_ZH_V15_REVISION
    ),
    reranker_model_revision: str = (
        RERANKER_MODEL_REVISION
    ),
    show_progress_bar: bool = True,
) -> RoutedHybridRerankerHitProvider:
    """使用项目固定 BGE 模型构造正式运行时。"""

    embedding_spec = (
        build_bge_small_zh_v15_spec(
            model_revision=(
                embedding_model_revision
            )
        )
    )

    embedding_provider = (
        SentenceTransformerEmbeddingProvider(
            spec=embedding_spec,
            batch_size=(
                embedding_batch_size
            ),
            device=device,
            local_files_only=(
                local_files_only
            ),
        )
    )

    reranker_spec = RerankerSpec(
        model_name=(
            RERANKER_MODEL_NAME
        ),
        model_revision=(
            reranker_model_revision
        ),
        max_length=512,
    )

    reranker_runtime_config = (
        RerankerRuntimeConfig(
            batch_size=(
                reranker_batch_size
            ),
            device=device,
            local_files_only=(
                local_files_only
            ),
            rerank_candidate_count=(
                rerank_candidate_count
            ),
            return_count=return_count,
        )
    )

    reranker_provider = (
        SentenceTransformerCrossEncoderProvider(
            spec=reranker_spec,
            runtime_config=(
                reranker_runtime_config
            ),
            show_progress_bar=(
                show_progress_bar
            ),
        )
    )

    rrf_config = RRFConfig(
        rank_constant=rank_constant,
        dense_candidate_count=(
            dense_candidate_count
        ),
        bm25_candidate_count=(
            bm25_candidate_count
        ),
    )

    return build_complex_oracle_hit_provider(
        routes=routes,
        vector_index_output_root=(
            vector_index_output_root
        ),
        embedding_provider=(
            embedding_provider
        ),
        reranker_provider=(
            reranker_provider
        ),
        rrf_config=rrf_config,
        reranker_runtime_config=(
            reranker_runtime_config
        ),
        provider_id=(
            "complex_hybrid_reranker_"
            "bge_fixed_v1"
        ),
    )


def _validate_index_identity(
    *,
    dense_manifest: object,
    bm25_manifest: object,
) -> None:
    """保证 Dense 和 BM25 使用同一 ChunkDataset。"""

    compared_fields = (
        "company_id",
        "report_id",
        "fiscal_year",
        "report_type",
        "document_id",
        "chunk_dataset_id",
        "chunk_strategy",
    )

    for field_name in compared_fields:
        dense_value = getattr(
            dense_manifest,
            field_name,
        )

        bm25_value = getattr(
            bm25_manifest,
            field_name,
        )

        if dense_value != bm25_value:
            raise (
                ComplexOracleRetrievalRuntimeError(
                    "Dense 与 BM25 Index "
                    "来源身份不一致："
                    f"field={field_name}, "
                    f"dense={dense_value!r}, "
                    f"bm25={bm25_value!r}"
                )
            )