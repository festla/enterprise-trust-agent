from __future__ import annotations

from app.rag.tokenization import (
    DeterministicChineseBigramTokenizer,
)


def test_tokenize_financial_metric() -> None:
    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    tokens = tokenizer.tokenize(
        "应收账款"
    )

    assert tokens == (
        "应收",
        "收账",
        "账款",
    )


def test_preserve_token_order_and_frequency(
) -> None:
    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    tokens = tokenizer.tokenize(
        "存货存货"
    )

    assert tokens == (
        "存货",
        "货存",
        "存货",
    )


def test_normalize_full_width_ascii() -> None:
    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    tokens = tokenizer.tokenize(
        "ＡＢＣ２０２４年度"
    )

    assert tokens == (
        "abc2024",
        "年度",
    )


def test_punctuation_creates_boundaries(
) -> None:
    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    tokens = tokenizer.tokenize(
        "存货，会计政策"
    )

    assert tokens == (
        "存货",
        "会计",
        "计政",
        "政策",
    )


def test_keep_single_cjk_character() -> None:
    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    tokens = tokenizer.tokenize("元")

    assert tokens == ("元",)


def test_return_empty_for_only_boundaries(
) -> None:
    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    tokens = tokenizer.tokenize(
        "，。！？  "
    )

    assert tokens == ()


def test_tokenization_is_deterministic(
) -> None:
    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    text = (
        "海信家电2024年末合并口径的"
        "存货是多少？"
    )

    first_tokens = tokenizer.tokenize(text)
    second_tokens = tokenizer.tokenize(text)

    assert first_tokens == second_tokens


def test_expose_tokenizer_spec() -> None:
    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    assert tokenizer.spec.tokenizer_version == (
        "deterministic_chinese_bigram_v1"
    )