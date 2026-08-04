from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


def _canonical_json_bytes(
    value: object,
) -> bytes:
    """生成稳定、可哈希的JSON字节。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class EmbeddingSpec(BaseModel):
    """生成检索向量时使用的模型契约。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )

    schema_version: int = Field(
        default=1,
        ge=1,
        le=1,
    )

    provider: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
        description="Embedding 实现提供方",
    )

    model_name: str = Field(
        min_length=1,
        description="模型名称或模型仓库 ID",
    )

    model_version: str = Field(
        min_length=1,
        description=(
            "模型权重版本、提交哈希或明确版本号"
        ),
    )

    dimension: int = Field(
        ge=1,
        description="Embedding 向量维度",
    )

    dtype: Literal["float32"] = "float32"

    normalize_embeddings: bool = True

    query_prefix: str = ""

    document_prefix: str = ""

    max_sequence_length: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Provider 实际使用的最大输入长度；"
            "为空表示由模型默认配置决定"
        ),
    )

    @field_validator(
        "provider",
        "model_name",
        "model_version",
        mode="before",
    )
    @classmethod
    def normalize_identity_fields(
        cls,
        value: object,
    ) -> object:
        """规范身份字段，但不修改 Embedding 前缀。"""

        if not isinstance(value, str):
            return value

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "Embedding 身份字段不能为空"
            )

        return normalized


def calculate_embedding_spec_sha256(
    spec: EmbeddingSpec,
) -> str:
    """计算 Embedding 配置的稳定哈希。"""

    return hashlib.sha256(
        _canonical_json_bytes(
            spec.model_dump(mode="json")
        )
    ).hexdigest()