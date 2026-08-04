from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .chunk import ChunkingConfig
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


def calculate_chunking_config_sha256(
    config: ChunkingConfig,
) -> str:
    """计算 Chunk 配置的稳定哈希。"""

    return hashlib.sha256(
        _canonical_json_bytes(
            config.model_dump(mode="json")
        )
    ).hexdigest()


class ChunkDatasetManifest(BaseModel):
    """一份 Chunk JSONL 数据集的审计记录。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    schema_version: int = Field(
        default=1,
        ge=1,
        le=1,
    )

    dataset_id: str = Field(
        pattern=(
            r"^chunk_dataset_[a-z0-9_]+_"
            r"[0-9a-f]{24}$"
        ),
    )

    page_dataset_id: str = Field(
        pattern=(
            r"^page_dataset_[a-z0-9_]+_"
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

    page_dataset_manifest_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    source_pages_jsonl_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    report_snapshot_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    chunk_schema_version: int = Field(
        default=1,
        ge=1,
        le=1,
    )

    strategy: ChunkStrategy

    chunker_name: str = Field(
        min_length=1,
    )

    chunker_version: str = Field(
        min_length=1,
    )

    chunking_config: ChunkingConfig

    chunking_config_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    chunks_jsonl_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    input_page_count: int = Field(
        ge=1,
    )

    eligible_page_count: int = Field(
        ge=0,
    )

    chunked_page_count: int = Field(
        ge=0,
    )

    skipped_page_count: int = Field(
        ge=0,
    )

    skipped_page_ids: tuple[str, ...] = ()

    chunk_record_count: int = Field(
        ge=1,
    )

    chunk_char_count_total: int = Field(
        ge=1,
        description=(
            "全部 Chunk 字符数量之和；"
            "包含 overlap 重复字符"
        ),
    )

    quality_gate_passed: bool

    quality_gate_errors: tuple[str, ...] = ()

    quality_warnings: tuple[str, ...] = ()

    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def validate_created_at_timezone(
        cls,
        value: datetime,
    ) -> datetime:
        """创建时间必须包含时区。"""

        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "datetime 必须包含时区信息"
            )

        return value

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        """检查身份、配置和统计是否一致。"""

        if not self.dataset_id.startswith(
            f"chunk_dataset_{self.report_id}_"
        ):
            raise ValueError(
                "dataset_id 必须属于 report_id"
            )

        if not self.page_dataset_id.startswith(
            f"page_dataset_{self.report_id}_"
        ):
            raise ValueError(
                "page_dataset_id 必须属于 report_id"
            )

        if not self.document_id.startswith(
            f"doc_{self.report_id}_"
        ):
            raise ValueError(
                "document_id 必须属于 report_id"
            )

        if (
            self.chunking_config.strategy
            != self.strategy
        ):
            raise ValueError(
                "strategy 必须与 "
                "chunking_config.strategy 一致"
            )

        if (
            self.chunking_config.chunker_name
            != self.chunker_name
        ):
            raise ValueError(
                "chunker_name 必须与配置一致"
            )

        if (
            self.chunking_config.chunker_version
            != self.chunker_version
        ):
            raise ValueError(
                "chunker_version 必须与配置一致"
            )

        expected_config_sha256 = (
            calculate_chunking_config_sha256(
                self.chunking_config
            )
        )

        if (
            self.chunking_config_sha256
            != expected_config_sha256
        ):
            raise ValueError(
                "chunking_config_sha256 "
                "与配置内容不一致"
            )

        if (
            self.eligible_page_count
            + self.skipped_page_count
            != self.input_page_count
        ):
            raise ValueError(
                "eligible 与 skipped 页数之和 "
                "必须等于输入页数"
            )

        if (
            self.chunked_page_count
            != self.eligible_page_count
        ):
            raise ValueError(
                "当前版本要求全部 eligible 页面 "
                "均成功生成 Chunk"
            )

        normalized_skipped_ids = tuple(
            sorted(set(self.skipped_page_ids))
        )

        if (
            self.skipped_page_ids
            != normalized_skipped_ids
        ):
            raise ValueError(
                "skipped_page_ids 必须升序且不能重复"
            )

        if (
            len(self.skipped_page_ids)
            != self.skipped_page_count
        ):
            raise ValueError(
                "skipped_page_ids 数量必须与 "
                "skipped_page_count 一致"
            )

        if (
            self.chunk_record_count
            < self.chunked_page_count
        ):
            raise ValueError(
                "每个已切分页面至少应有一个 Chunk"
            )

        if (
            self.chunk_char_count_total
            < self.chunk_record_count
        ):
            raise ValueError(
                "Chunk 字符总数不能小于 Chunk 数量"
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