from __future__ import annotations

from collections.abc import Sequence
from typing import (
    Protocol,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray

from app.schemas.embedding import (
    EmbeddingSpec,
)


FloatArray = NDArray[np.float32]


class EmbeddingError(ValueError):
    """Embedding 数据基础异常。"""


class InvalidEmbeddingShapeError(
    EmbeddingError
):
    """Embedding 维度或形状不正确。"""


class InvalidEmbeddingValueError(
    EmbeddingError
):
    """Embedding 包含无效数值或零向量。"""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """模型实现必须满足的最小接口。"""

    @property
    def spec(self) -> EmbeddingSpec:
        """返回模型与执行配置。"""

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> FloatArray:
        """批量生成文档向量。"""

    def embed_query(
        self,
        text: str,
    ) -> FloatArray:
        """生成单条查询向量。"""


def normalize_embedding_matrix(
    value: object,
    *,
    expected_rows: int,
    expected_dimension: int,
) -> FloatArray:
    """验证并按行执行 L2 归一化。"""

    matrix = np.asarray(
        value,
        dtype=np.float32,
    )

    if matrix.ndim != 2:
        raise InvalidEmbeddingShapeError(
            "文档 Embedding 必须是二维矩阵"
        )

    if matrix.shape != (
        expected_rows,
        expected_dimension,
    ):
        raise InvalidEmbeddingShapeError(
            "文档 Embedding 形状不一致："
            f"actual={matrix.shape}, "
            f"expected="
            f"({expected_rows}, "
            f"{expected_dimension})"
        )

    if not np.isfinite(matrix).all():
        raise InvalidEmbeddingValueError(
            "文档 Embedding 包含 NaN "
            "或 Infinity"
        )

    norms = np.linalg.norm(
        matrix,
        axis=1,
    )

    if np.any(norms <= 0):
        raise InvalidEmbeddingValueError(
            "文档 Embedding 不能包含零向量"
        )

    normalized = (
        matrix / norms[:, np.newaxis]
    ).astype(
        np.float32,
        copy=False,
    )

    return np.ascontiguousarray(
        normalized
    )


def normalize_query_vector(
    value: object,
    *,
    expected_dimension: int,
) -> FloatArray:
    """验证并归一化查询向量。"""

    vector = np.asarray(
        value,
        dtype=np.float32,
    )

    if vector.ndim == 2:
        if vector.shape[0] != 1:
            raise InvalidEmbeddingShapeError(
                "二维查询向量只能包含一行"
            )

        vector = vector[0]

    if vector.ndim != 1:
        raise InvalidEmbeddingShapeError(
            "查询 Embedding 必须是一维向量"
        )

    if vector.shape[0] != expected_dimension:
        raise InvalidEmbeddingShapeError(
            "查询 Embedding 维度不一致："
            f"actual={vector.shape[0]}, "
            f"expected={expected_dimension}"
        )

    if not np.isfinite(vector).all():
        raise InvalidEmbeddingValueError(
            "查询 Embedding 包含 NaN "
            "或 Infinity"
        )

    norm = float(
        np.linalg.norm(vector)
    )

    if norm <= 0:
        raise InvalidEmbeddingValueError(
            "查询 Embedding 不能是零向量"
        )

    normalized = (
        vector / norm
    ).astype(
        np.float32,
        copy=False,
    )

    return np.ascontiguousarray(
        normalized
    )