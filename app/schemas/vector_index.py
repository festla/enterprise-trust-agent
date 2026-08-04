from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .embedding import (
    EmbeddingSpec,
    calculate_embedding_spec_sha256,
)
from .enums import (
    ChunkStrategy,
    ReportType,
)


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


class VectorIndexManifest(BaseModel):
    """一个持久化精确向量索引的审计记录。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    schema_version: int = Field(
        default=1,
        ge=1,
        le=1,
    )

    index_id: str = Field(
        pattern=(
            r"^vector_index_[a-z0-9_]+_"
            r"[0-9a-f]{24}$"
        ),
    )

    chunk_dataset_id: str = Field(
        pattern=(
            r"^chunk_dataset_[a-z0-9_]+_"
            r"[0-9a-f]{24}$"
        ),
    )

    report_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    company_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    fiscal_year: int = Field(
        ge=2000,
        le=2100,
    )

    report_type: ReportType

    document_id: str = Field(
        pattern=(
            r"^doc_[a-z0-9_]+_"
            r"[0-9a-f]{24}$"
        ),
    )

    chunk_strategy: ChunkStrategy

    chunk_dataset_manifest_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    source_chunks_jsonl_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    embedding_spec: EmbeddingSpec

    embedding_spec_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    embedding_input_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "按 Chunk 顺序对实际文档模型输入计算的哈希"
        ),
    )

    index_type: Literal[
        "exact_cosine"
    ] = "exact_cosine"

    index_version: Literal[
        "exact_cosine_v1"
    ] = "exact_cosine_v1"

    numpy_version: str = Field(
        min_length=1,
    )

    stored_vectors_normalized: Literal[
        True
    ] = True

    vector_count: int = Field(
        ge=1,
    )

    vector_dimension: int = Field(
        ge=1,
    )

    vector_dtype: Literal[
        "float32"
    ] = "float32"

    metadata_record_count: int = Field(
        ge=1,
    )

    vectors_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    metadata_jsonl_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    quality_gate_passed: bool

    quality_gate_errors: tuple[str, ...] = ()

    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def validate_created_at_timezone(
        cls,
        value: datetime,
    ) -> datetime:
        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "created_at 必须包含时区"
            )

        return value

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        """检查索引来源、模型和文件统计。"""

        if not self.index_id.startswith(
            f"vector_index_{self.report_id}_"
        ):
            raise ValueError(
                "index_id 必须属于 report_id"
            )

        if not self.chunk_dataset_id.startswith(
            f"chunk_dataset_{self.report_id}_"
        ):
            raise ValueError(
                "chunk_dataset_id 必须属于 report_id"
            )

        if not self.document_id.startswith(
            f"doc_{self.report_id}_"
        ):
            raise ValueError(
                "document_id 必须属于 report_id"
            )

        expected_spec_sha256 = (
            calculate_embedding_spec_sha256(
                self.embedding_spec
            )
        )

        if (
            self.embedding_spec_sha256
            != expected_spec_sha256
        ):
            raise ValueError(
                "embedding_spec_sha256 "
                "与 EmbeddingSpec 不一致"
            )

        if (
            self.vector_dimension
            != self.embedding_spec.dimension
        ):
            raise ValueError(
                "vector_dimension 必须与 "
                "EmbeddingSpec.dimension 一致"
            )

        if (
            self.vector_dtype
            != self.embedding_spec.dtype
        ):
            raise ValueError(
                "vector_dtype 必须与 "
                "EmbeddingSpec.dtype 一致"
            )

        if (
            self.vector_count
            != self.metadata_record_count
        ):
            raise ValueError(
                "向量数量必须与元数据数量一致"
            )

        expected_quality_passed = (
            len(self.quality_gate_errors) == 0
        )

        if (
            self.quality_gate_passed
            != expected_quality_passed
        ):
            raise ValueError(
                "quality_gate_passed 必须与 "
                "quality_gate_errors 一致"
            )

        return self