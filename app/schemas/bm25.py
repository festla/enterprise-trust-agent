from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)
import hashlib
import json

class BM25TokenizerSpec(BaseModel):
    """确定性中文 BM25 分词器配置。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: int = Field(
        default=1,
        ge=1,
        le=1,
    )

    tokenizer_name: Literal[
        "deterministic_chinese_bigram"
    ] = "deterministic_chinese_bigram"

    tokenizer_version: Literal[
        "deterministic_chinese_bigram_v1"
    ] = "deterministic_chinese_bigram_v1"

    unicode_normalization: Literal[
        "NFKC"
    ] = "NFKC"

    lowercase_ascii: Literal[
        True
    ] = True

    cjk_token_rule: Literal[
        "character_ngram"
    ] = "character_ngram"

    cjk_ngram_size: Literal[
        2
    ] = 2

    single_cjk_fallback: Literal[
        "keep_single_character"
    ] = "keep_single_character"

    ascii_token_rule: Literal[
        "contiguous_alnum"
    ] = "contiguous_alnum"

    boundary_rule: Literal[
        "non_cjk_non_ascii_alnum"
    ] = "non_cjk_non_ascii_alnum"


class BM25Config(BaseModel):
    """BM25 核心计算的确定性配置。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    schema_version: int = Field(
        default=1,
        ge=1,
        le=1,
    )

    algorithm: Literal[
        "bm25"
    ] = "bm25"

    algorithm_version: Literal[
        "bm25_v1"
    ] = "bm25_v1"

    k1: float = Field(
        default=1.2,
        gt=0,
        description=(
            "控制词频饱和速度；"
            "值越大，重复词频的影响越明显"
        ),
    )

    b: float = Field(
        default=0.75,
        ge=0,
        le=1,
        description=(
            "控制文档长度归一化程度"
        ),
    )

    idf_variant: Literal[
        "robertson_log1p"
    ] = "robertson_log1p"

    unique_query_terms: Literal[
        True
    ] = True


def _canonical_json_bytes(
    value: object,
) -> bytes:
    """生成稳定、可哈希的 JSON 字节。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def calculate_bm25_tokenizer_spec_sha256(
    spec: BM25TokenizerSpec,
) -> str:
    """计算 TokenizerSpec 的稳定哈希。"""

    return hashlib.sha256(
        _canonical_json_bytes(
            spec.model_dump(mode="json")
        )
    ).hexdigest()


def calculate_bm25_config_sha256(
    config: BM25Config,
) -> str:
    """计算 BM25Config 的稳定哈希。"""

    return hashlib.sha256(
        _canonical_json_bytes(
            config.model_dump(mode="json")
        )
    ).hexdigest()