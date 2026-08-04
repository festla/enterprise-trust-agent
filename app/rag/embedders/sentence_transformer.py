from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from app.rag.embedding import FloatArray
from app.schemas.embedding import EmbeddingSpec

BGE_SMALL_ZH_V15_MODEL_NAME = (
    "BAAI/bge-small-zh-v1.5"
)

BGE_SMALL_ZH_V15_REVISION = (
    "534b6bfaaf500e70bcb9f3771cebc940c23b219d"
)

BGE_SMALL_ZH_V15_DIMENSION = 512

BGE_SMALL_ZH_V15_QUERY_PREFIX = (
    "为这个句子生成表示以用于检索相关文章："
)

BGE_SMALL_ZH_V15_MAX_SEQUENCE_LENGTH = 512

class SentenceTransformerProviderError(
    ValueError
):
    """Sentence Transformer Provider 基础异常。"""


class EmbeddingBackendUnavailableError(
    SentenceTransformerProviderError
):
    """Sentence Transformers 依赖或模型无法加载。"""


class EmbeddingModelDimensionMismatchError(
    SentenceTransformerProviderError
):
    """模型实际输出维度与 EmbeddingSpec 不一致。"""


class InvalidEmbeddingBackendOutputError(
    SentenceTransformerProviderError
):
    """模型后端返回了形状或数值无效的向量。"""


class SentenceTransformerBackend(Protocol):
    """供真实模型和测试 Fake Backend 共同满足的接口。"""

    max_seq_length: int

    def get_embedding_dimension(
        self,
    ) -> int | None:
        """返回模型向量维度。"""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embedding: bool,
    ) -> object:
        """生成一批文本向量。"""


def build_bge_small_zh_v15_spec(
    *,
    model_revision: str = (
        BGE_SMALL_ZH_V15_REVISION
    ),
) -> EmbeddingSpec:
    """构造固定版本的中文 BGE 基线配置。"""

    return EmbeddingSpec(
        provider="sentence_transformers",
        model_name=(
            BGE_SMALL_ZH_V15_MODEL_NAME
        ),
        model_version=model_revision,
        dimension=(
            BGE_SMALL_ZH_V15_DIMENSION
        ),
        dtype="float32",
        normalize_embeddings=True,
        query_prefix=(
            BGE_SMALL_ZH_V15_QUERY_PREFIX
        ),
        document_prefix="",
        max_sequence_length=(
            BGE_SMALL_ZH_V15_MAX_SEQUENCE_LENGTH
        ),
    )


class SentenceTransformerEmbeddingProvider:
    """基于 Sentence Transformers 的真实 Provider。
    
    Query 和 Document 前缀由 ExactVectorIndex
    根据 EmbeddingSpec 统一添加，本类不重复添加。
    """

    def __init__(
        self,
        *,
        spec: EmbeddingSpec,
        batch_size: int = 16,
        device: str = "cpu",
        cache_folder: Path | None = None,
        local_files_only: bool = False,
        show_progress_bar: bool = False,
        backend: (
            SentenceTransformerBackend | None
        ) = None,
    ) -> None:
        if (
            spec.provider != "sentence_transformers"
        ):
            raise SentenceTransformerProviderError(
                "EmbeddingSpec.provider 必须为 "
                "sentence_transformers"
            )

        if batch_size < 1:
            raise SentenceTransformerProviderError(
                "batch_size 必须大于等于 1"
            )

        self._spec = spec
        self._batch_size = batch_size
        self._show_progress_bar = (
            show_progress_bar
        )

        if backend is None:
            try:
                from sentence_transformers import (
                    SentenceTransformer,
                )
            except ImportError as exc:
                raise (
                    EmbeddingBackendUnavailableError(
                        "缺少 sentence-transformers "
                        "依赖"
                    )
                ) from exc

            try:
                backend = SentenceTransformer(
                    model_name_or_path=(
                        spec.model_name
                    ),
                    revision=spec.model_version,
                    device=device,
                    cache_folder=(
                        str(cache_folder)
                        if cache_folder is not None
                        else None
                    ),
                    trust_remote_code=False,
                    local_files_only=(
                        local_files_only
                    ),
                )
            except Exception as exc:
                raise (
                    EmbeddingBackendUnavailableError(
                        "无法加载 Embedding 模型："
                        f"{spec.model_name}@"
                        f"{spec.model_version}"
                    )
                ) from exc

        actual_dimension = (
            backend
            .get_embedding_dimension()
        )

        if (
            actual_dimension is None
            or int(actual_dimension) != spec.dimension
        ):
            raise (
                EmbeddingModelDimensionMismatchError(
                    "模型实际向量维度与 "
                    "EmbeddingSpec 不一致："
                    f"actual={actual_dimension}, "
                    f"expected={spec.dimension}"
                )
            )

        if spec.max_sequence_length is not None:
            backend.max_seq_length = (
                spec.max_sequence_length
            )

        self._backend = backend

    @property
    def spec(self) -> EmbeddingSpec:
        """返回不可变模型配置。"""

        return self._spec

    def _encode(
        self,
        texts: tuple[str, ...],
    ) -> FloatArray:
        """调用后端并检查输出矩阵。"""

        if not texts:
            return np.empty(
                (
                    0,
                    self._spec.dimension,
                ),
                dtype=np.float32,
            )

        try:
            raw_vectors = self._backend.encode(
                list(texts),
                batch_size=self._batch_size,
                show_progress_bar=(
                    self._show_progress_bar
                ),
                convert_to_numpy=True,
                normalize_embeddings=(
                    self._spec
                    .normalize_embeddings
                ),
            )
        except Exception as exc:
            raise SentenceTransformerProviderError(
                "Embedding 模型推理失败"
            ) from exc

        matrix = np.asarray(
            raw_vectors,
            dtype=np.float32,
        )

        if (
            matrix.ndim == 1
            and len(texts) == 1
        ):
            matrix = matrix.reshape(1, -1)

        expected_shape = (
            len(texts),
            self._spec.dimension,
        )

        if matrix.shape != expected_shape:
            raise (
                InvalidEmbeddingBackendOutputError(
                    "Embedding 后端输出形状错误："
                    f"actual={matrix.shape}, "
                    f"expected={expected_shape}"
                )
            )

        if not np.isfinite(matrix).all():
            raise (
                InvalidEmbeddingBackendOutputError(
                    "Embedding 后端输出包含 "
                    "NaN 或 Infinity"
                )
            )

        return np.ascontiguousarray(
            matrix,
            dtype=np.float32,
        )

    
    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> FloatArray:
        """批量生成文档向量。"""

        return self._encode(
            tuple(texts)
        )

    def embed_query(
        self,
        text: str,
    ) -> FloatArray:
        """生成一条查询向量。"""

        if not text.strip():
            raise SentenceTransformerProviderError(
                "查询文本不能为空"
            )

        matrix = self._encode((text,))

        return np.ascontiguousarray(
            matrix[0],
            dtype=np.float32,
        )