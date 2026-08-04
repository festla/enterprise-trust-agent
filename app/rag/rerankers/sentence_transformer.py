from __future__ import annotations

from collections.abc import (
    Callable,
    Sequence,
)
from pathlib import Path
from typing import Protocol

import numpy as np

from app.schemas.reranker import (
    RerankerRuntimeConfig,
    RerankerSpec,
)


class CrossEncoderProviderError(
    ValueError
):
    """Cross-Encoder Provider 基础异常。"""


class CrossEncoderBackendUnavailableError(
    CrossEncoderProviderError
):
    """依赖、模型或本地缓存不可用。"""


class InvalidCrossEncoderInputError(
    CrossEncoderProviderError
):
    """Cross-Encoder 输入对不合法。"""


class InvalidCrossEncoderOutputError(
    CrossEncoderProviderError
):
    """Cross-Encoder 返回的分数形状或数值非法。"""


class CrossEncoderInferenceError(
    CrossEncoderProviderError
):
    """Cross-Encoder 推理过程失败。"""


class CrossEncoderBackend(Protocol):
    """真实 CrossEncoder 与测试 Fake Backend 的统一接口。"""

    def predict(
        self,
        inputs: list[
            tuple[str, str]
        ],
        *,
        batch_size: int,
        show_progress_bar: bool,
        activation_fn: Callable[
            [object],
            object,
        ],
        apply_softmax: bool,
        convert_to_numpy: bool,
    ) -> object:
        """为文本对批量生成相关性分数。"""


def _identity_activation(
    value: object,
) -> object:
    """保留模型原始 Logit，不应用 Sigmoid。"""

    return value


class SentenceTransformerCrossEncoderProvider:
    """基于 sentence-transformers 的 Cross-Encoder Provider。"""

    def __init__(
        self,
        *,
        spec: RerankerSpec,
        runtime_config: RerankerRuntimeConfig,
        cache_folder: Path | None = None,
        show_progress_bar: bool = False,
        backend: CrossEncoderBackend | None = None,
    ) -> None:
        if (
            spec.provider
            != "sentence_transformers_cross_encoder"
        ):
            raise CrossEncoderProviderError(
                "RerankerSpec.provider 必须为 "
                "sentence_transformers_cross_encoder"
            )

        self._spec = spec

        self._runtime_config = (
            runtime_config
        )

        self._show_progress_bar = (
            show_progress_bar
        )

        if backend is not None:
            self._backend = backend
            return

        try:
            from sentence_transformers import (
                CrossEncoder,
            )

            self._backend = CrossEncoder(
                spec.model_name,
                device=(
                    runtime_config.device
                ),
                cache_folder=(
                    None
                    if cache_folder is None
                    else str(cache_folder)
                ),
                revision=(
                    spec.model_revision
                ),
                local_files_only=(
                    runtime_config
                    .local_files_only
                ),
                max_length=(
                    spec.max_length
                ),
            )

        except (
            ImportError,
            OSError,
            ValueError,
        ) as exc:
            raise (
                CrossEncoderBackendUnavailableError(
                    "无法加载 Cross-Encoder "
                    "依赖或模型："
                    f"{spec.model_name}@"
                    f"{spec.model_revision}"
                )
            ) from exc

    @property
    def spec(self) -> RerankerSpec:
        return self._spec

    @property
    def runtime_config(
        self,
    ) -> RerankerRuntimeConfig:
        return self._runtime_config

    def score_pairs(
        self,
        pairs: Sequence[
            tuple[str, str]
        ],
    ) -> tuple[float, ...]:
        """批量计算 query-passage 原始相关性 Logit。"""

        normalized_pairs: list[
            tuple[str, str]
        ] = []

        for position, pair in enumerate(
            pairs
        ):
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
            ):
                raise InvalidCrossEncoderInputError(
                    "Cross-Encoder 输入必须是 "
                    "(query, passage) 二元组："
                    f"position={position}"
                )

            query, passage = pair

            if (
                not isinstance(query, str)
                or not query.strip()
            ):
                raise InvalidCrossEncoderInputError(
                    "Cross-Encoder query "
                    "不能为空："
                    f"position={position}"
                )

            if (
                not isinstance(passage, str)
                or not passage.strip()
            ):
                raise InvalidCrossEncoderInputError(
                    "Cross-Encoder passage "
                    "不能为空："
                    f"position={position}"
                )

            normalized_pairs.append(
                (
                    query,
                    passage,
                )
            )

        if not normalized_pairs:
            return ()

        try:
            raw_output = (
                self._backend.predict(
                    normalized_pairs,
                    batch_size=(
                        self._runtime_config
                        .batch_size
                    ),
                    show_progress_bar=(
                        self._show_progress_bar
                    ),
                    # CrossEncoder 默认可能使用
                    # Sigmoid；这里显式保留 Logit。
                    activation_fn=(
                        _identity_activation
                    ),
                    apply_softmax=False,
                    convert_to_numpy=True,
                )
            )

        except Exception as exc:
            raise CrossEncoderInferenceError(
                "Cross-Encoder 推理失败"
            ) from exc

        scores = np.asarray(
            raw_output,
            dtype=np.float64,
        )

        # 单个输入可能由某些 Backend
        # 返回零维标量。
        if scores.ndim == 0:
            scores = scores.reshape(1)

        # 接受常见的 (N, 1) 单标签输出。
        if (
            scores.ndim == 2
            and scores.shape[1] == 1
        ):
            scores = scores[:, 0]

        if scores.ndim != 1:
            raise InvalidCrossEncoderOutputError(
                "Cross-Encoder 必须为每个输入"
                "返回一个标量分数："
                f"shape={scores.shape}"
            )

        if (
            scores.shape[0]
            != len(normalized_pairs)
        ):
            raise InvalidCrossEncoderOutputError(
                "Cross-Encoder 输出数量"
                "与输入数量不一致："
                f"expected={len(normalized_pairs)}, "
                f"actual={scores.shape[0]}"
            )

        if not np.isfinite(scores).all():
            raise InvalidCrossEncoderOutputError(
                "Cross-Encoder 输出必须全部"
                "为有限数值"
            )

        return tuple(
            float(score)
            for score in scores
        )