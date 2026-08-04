from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.bm25 import (
    BM25Config,
    BM25TokenizerSpec,
)


def test_build_default_bm25_config() -> None:
    config = BM25Config()

    assert config.algorithm == "bm25"
    assert config.algorithm_version == (
        "bm25_v1"
    )
    assert config.k1 == pytest.approx(1.2)
    assert config.b == pytest.approx(0.75)
    assert config.idf_variant == (
        "robertson_log1p"
    )
    assert config.unique_query_terms is True


def test_build_default_tokenizer_spec(
) -> None:
    spec = BM25TokenizerSpec()

    assert spec.tokenizer_name == (
        "deterministic_chinese_bigram"
    )
    assert spec.cjk_ngram_size == 2
    assert (
        spec.unicode_normalization
        == "NFKC"
    )
    assert spec.lowercase_ascii is True


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("k1", 0),
        ("k1", -1),
        ("b", -0.1),
        ("b", 1.1),
        ("k1", float("nan")),
        ("b", float("inf")),
    ),
)
def test_reject_invalid_bm25_parameters(
    field_name: str,
    invalid_value: float,
) -> None:
    with pytest.raises(ValidationError):
        BM25Config(
            **{
                field_name: invalid_value,
            }
        )


def test_reject_unknown_bm25_field() -> None:
    with pytest.raises(ValidationError):
        BM25Config(
            unknown_parameter=True,
        )


def test_reject_changed_tokenizer_rule(
) -> None:
    with pytest.raises(ValidationError):
        BM25TokenizerSpec(
            cjk_ngram_size=3,
        )


def test_bm25_config_is_frozen() -> None:
    config = BM25Config()

    with pytest.raises(ValidationError):
        config.k1 = 2.0  # type: ignore[misc]


def test_tokenizer_spec_is_frozen() -> None:
    spec = BM25TokenizerSpec()

    with pytest.raises(ValidationError):
        spec.cjk_ngram_size = 3  # type: ignore[misc]