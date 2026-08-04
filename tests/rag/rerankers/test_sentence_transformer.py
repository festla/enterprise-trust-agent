from __future__ import annotations

from collections.abc import (
    Callable,
)

import numpy as np
import pytest

from app.rag.rerankers import (
    CrossEncoderInferenceError,
    CrossEncoderProviderError,
    InvalidCrossEncoderInputError,
    InvalidCrossEncoderOutputError,
    SentenceTransformerCrossEncoderProvider,
)
from app.schemas.reranker import (
    RerankerRuntimeConfig,
    RerankerSpec,
)


class FakeCrossEncoderBackend:
    """返回预设输出并记录调用参数。"""

    def __init__(
        self,
        *,
        output: object,
        error: Exception | None = None,
    ) -> None:
        self.output = output
        self.error = error

        self.calls: list[
            dict[str, object]
        ] = []

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
        self.calls.append(
            {
                "inputs": tuple(inputs),
                "batch_size": batch_size,
                "show_progress_bar": (
                    show_progress_bar
                ),
                "activation_fn": (
                    activation_fn
                ),
                "apply_softmax": (
                    apply_softmax
                ),
                "convert_to_numpy": (
                    convert_to_numpy
                ),
            }
        )

        if self.error is not None:
            raise self.error

        return self.output


def build_spec() -> RerankerSpec:
    return RerankerSpec(
        model_name=(
            "test/fake-cross-encoder"
        ),
        model_revision="fake_revision_v1",
        max_length=128,
    )


def build_runtime_config(
) -> RerankerRuntimeConfig:
    return RerankerRuntimeConfig(
        batch_size=4,
        device="cpu",
        local_files_only=True,
        rerank_candidate_count=10,
        return_count=5,
    )


def build_provider(
    *,
    output: object,
    error: Exception | None = None,
) -> tuple[
    SentenceTransformerCrossEncoderProvider,
    FakeCrossEncoderBackend,
]:
    backend = FakeCrossEncoderBackend(
        output=output,
        error=error,
    )

    provider = (
        SentenceTransformerCrossEncoderProvider(
            spec=build_spec(),
            runtime_config=(
                build_runtime_config()
            ),
            show_progress_bar=False,
            backend=backend,
        )
    )

    return provider, backend


def test_score_pairs_returns_raw_scores(
) -> None:
    provider, backend = build_provider(
        output=np.asarray(
            (
                3.5,
                -1.25,
            ),
            dtype=np.float32,
        )
    )

    scores = provider.score_pairs(
        (
            (
                "资产总计是多少？",
                "合并资产负债表资产总计",
            ),
            (
                "资产总计是多少？",
                "营业收入和营业成本",
            ),
        )
    )

    assert scores == pytest.approx(
        (
            3.5,
            -1.25,
        )
    )

    assert len(backend.calls) == 1

    call = backend.calls[0]

    assert call["batch_size"] == 4
    assert not call["show_progress_bar"]
    assert not call["apply_softmax"]
    assert call["convert_to_numpy"]

    activation_fn = call[
        "activation_fn"
    ]

    marker = object()

    assert callable(activation_fn)
    assert activation_fn(marker) is marker


def test_provider_exposes_spec_and_runtime(
) -> None:
    provider, _ = build_provider(
        output=np.asarray(
            (1.0,)
        )
    )

    assert provider.spec == build_spec()

    assert (
        provider.runtime_config
        == build_runtime_config()
    )


def test_empty_pairs_return_empty_without_backend_call(
) -> None:
    provider, backend = build_provider(
        output=np.asarray(())
    )

    scores = provider.score_pairs(())

    assert scores == ()
    assert backend.calls == []


@pytest.mark.parametrize(
    "pairs",
    (
        (
            (
                "",
                "有效文档",
            ),
        ),
        (
            (
                "有效问题",
                "   ",
            ),
        ),
        (
            (
                123,
                "有效文档",
            ),
        ),
    ),
)
def test_reject_invalid_pairs(
    pairs: object,
) -> None:
    provider, _ = build_provider(
        output=np.asarray((1.0,))
    )

    with pytest.raises(
        InvalidCrossEncoderInputError,
    ):
        provider.score_pairs(
            pairs  # type: ignore[arg-type]
        )


def test_accept_scalar_for_single_pair(
) -> None:
    provider, _ = build_provider(
        output=np.asarray(2.5)
    )

    scores = provider.score_pairs(
        (
            (
                "问题",
                "文档",
            ),
        )
    )

    assert scores == pytest.approx(
        (2.5,)
    )


def test_accept_single_label_column_output(
) -> None:
    provider, _ = build_provider(
        output=np.asarray(
            (
                (2.0,),
                (-1.0,),
            )
        )
    )

    scores = provider.score_pairs(
        (
            ("问题一", "文档一"),
            ("问题二", "文档二"),
        )
    )

    assert scores == pytest.approx(
        (
            2.0,
            -1.0,
        )
    )


def test_reject_output_count_mismatch(
) -> None:
    provider, _ = build_provider(
        output=np.asarray(
            (1.0,)
        )
    )

    with pytest.raises(
        InvalidCrossEncoderOutputError,
        match="数量",
    ):
        provider.score_pairs(
            (
                ("问题一", "文档一"),
                ("问题二", "文档二"),
            )
        )


def test_reject_multi_label_output(
) -> None:
    provider, _ = build_provider(
        output=np.asarray(
            (
                (1.0, 2.0),
                (3.0, 4.0),
            )
        )
    )

    with pytest.raises(
        InvalidCrossEncoderOutputError,
        match="一个标量",
    ):
        provider.score_pairs(
            (
                ("问题一", "文档一"),
                ("问题二", "文档二"),
            )
        )


@pytest.mark.parametrize(
    "invalid_score",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_reject_non_finite_output(
    invalid_score: float,
) -> None:
    provider, _ = build_provider(
        output=np.asarray(
            (invalid_score,)
        )
    )

    with pytest.raises(
        InvalidCrossEncoderOutputError,
        match="有限数值",
    ):
        provider.score_pairs(
            (
                ("问题", "文档"),
            )
        )


def test_wrap_backend_inference_error(
) -> None:
    provider, _ = build_provider(
        output=None,
        error=RuntimeError(
            "backend failed"
        ),
    )

    with pytest.raises(
        CrossEncoderInferenceError,
        match="推理失败",
    ):
        provider.score_pairs(
            (
                ("问题", "文档"),
            )
        )


def test_reject_wrong_provider_spec(
) -> None:
    invalid_spec = build_spec().model_copy(
        update={
            "provider": "other_provider",
        }
    )

    backend = FakeCrossEncoderBackend(
        output=np.asarray((1.0,))
    )

    with pytest.raises(
        CrossEncoderProviderError,
        match="provider",
    ):
        SentenceTransformerCrossEncoderProvider(
            spec=invalid_spec,
            runtime_config=(
                build_runtime_config()
            ),
            backend=backend,
        )