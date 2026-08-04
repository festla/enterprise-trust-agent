from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .bm25 import (
    BM25Config,
    BM25TokenizerSpec,
    calculate_bm25_config_sha256,
    calculate_bm25_tokenizer_spec_sha256,
)
from .enums import (
    ChunkStrategy,
    ReportType,
)


class BM25IndexDocumentRecord(BaseModel):
    """一个 Chunk 的持久化 BM25 统计。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    chunk_id: str = Field(
        pattern=(
            r"^chunk_[a-z0-9_]+_"
            r"[0-9a-f]{24}$"
        ),
    )

    document_length: int = Field(
        ge=1,
    )

    term_frequencies: dict[
        str,
        int,
    ]

    @field_validator(
        "term_frequencies"
    )
    @classmethod
    def validate_term_frequencies(
        cls,
        value: dict[str, int],
    ) -> dict[str, int]:
        if not value:
            raise ValueError(
                "term_frequencies 不能为空"
            )

        if any(
            not term
            for term in value
        ):
            raise ValueError(
                "BM25 Token 不能为空"
            )

        if any(
            count < 1
            for count in value.values()
        ):
            raise ValueError(
                "词频必须大于等于 1"
            )

        return dict(
            sorted(value.items())
        )

    @model_validator(mode="after")
    def validate_document_record(
        self,
    ) -> Self:
        if (
            sum(self.term_frequencies.values())
            != self.document_length
        ):
            raise ValueError(
                "document_length 必须等于 "
                "term_frequencies 之和"
            )

        return self


class BM25IndexData(BaseModel):
    """index.json 的可验证内容。"""

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

    document_records: tuple[
        BM25IndexDocumentRecord,
        ...,
    ]

    document_frequencies: dict[
        str,
        int,
    ]

    document_count: int = Field(
        ge=1,
    )

    vocabulary_size: int = Field(
        ge=1,
    )

    total_token_count: int = Field(
        ge=1,
    )

    average_document_length: float = Field(
        gt=0,
    )

    @field_validator(
        "document_records"
    )
    @classmethod
    def validate_document_records(
        cls,
        value: tuple[
            BM25IndexDocumentRecord,
            ...,
        ],
    ) -> tuple[
        BM25IndexDocumentRecord,
        ...,
    ]:
        if not value:
            raise ValueError(
                "document_records 不能为空"
            )

        chunk_ids = tuple(
            record.chunk_id
            for record in value
        )

        if (
            len(chunk_ids)
            != len(set(chunk_ids))
        ):
            raise ValueError(
                "document_records "
                "不能包含重复 chunk_id"
            )

        return value

    @field_validator(
        "document_frequencies"
    )
    @classmethod
    def validate_document_frequencies(
        cls,
        value: dict[str, int],
    ) -> dict[str, int]:
        if not value:
            raise ValueError(
                "document_frequencies 不能为空"
            )

        if any(
            not term
            for term in value
        ):
            raise ValueError(
                "BM25 Token 不能为空"
            )

        if any(
            count < 1
            for count in value.values()
        ):
            raise ValueError(
                "文档频率必须大于等于 1"
            )

        return dict(
            sorted(value.items())
        )

    @model_validator(mode="after")
    def validate_index_data(
        self,
    ) -> Self:
        if (
            len(self.document_records)
            != self.document_count
        ):
            raise ValueError(
                "document_count 与 "
                "document_records 数量不一致"
            )

        if (
            len(self.document_frequencies)
            != self.vocabulary_size
        ):
            raise ValueError(
                "vocabulary_size 与词项数量不一致"
            )

        if any(
            frequency > self.document_count
            for frequency
            in self.document_frequencies.values()
        ):
            raise ValueError(
                "文档频率不能超过文档数量"
            )

        expected_total_token_count = sum(
            record.document_length
            for record in self.document_records
        )

        if (
            expected_total_token_count
            != self.total_token_count
        ):
            raise ValueError(
                "total_token_count "
                "与文档长度之和不一致"
            )

        expected_average = (
            self.total_token_count
            / self.document_count
        )

        if (
            abs(
                self.average_document_length
                - expected_average
            )
            > 1e-12
        ):
            raise ValueError(
                "average_document_length "
                "与文档统计不一致"
            )

        calculated_frequencies: dict[
            str,
            int,
        ] = {}

        for record in self.document_records:
            for term in record.term_frequencies:
                calculated_frequencies[term] = (
                    calculated_frequencies.get(
                        term,
                        0,
                    )
                    + 1
                )

        calculated_frequencies = dict(
            sorted(
                calculated_frequencies.items()
            )
        )

        if (
            calculated_frequencies
            != self.document_frequencies
        ):
            raise ValueError(
                "document_frequencies "
                "与文档词频记录不一致"
            )

        return self


class BM25IndexManifest(BaseModel):
    """持久化 BM25 索引的审计记录。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )

    schema_version: int = Field(
        default=1,
        ge=1,
        le=1,
    )

    index_id: str = Field(
        pattern=(
            r"^bm25_index_[a-z0-9_]+_"
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

    tokenizer_spec: BM25TokenizerSpec

    tokenizer_spec_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    bm25_config: BM25Config

    bm25_config_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    tokenized_corpus_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "按 Chunk 顺序对实际 Token 序列"
            "计算的语义哈希"
        ),
    )

    index_type: Literal[
        "exact_bm25"
    ] = "exact_bm25"

    index_version: Literal[
        "exact_bm25_v1"
    ] = "exact_bm25_v1"

    document_count: int = Field(
        ge=1,
    )

    metadata_record_count: int = Field(
        ge=1,
    )

    vocabulary_size: int = Field(
        ge=1,
    )

    total_token_count: int = Field(
        ge=1,
    )

    average_document_length: float = Field(
        gt=0,
    )

    index_json_sha256: str = Field(
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
    def validate_manifest(
        self,
    ) -> Self:
        if not self.index_id.startswith(
            f"bm25_index_{self.report_id}_"
        ):
            raise ValueError(
                "index_id 必须属于 report_id"
            )

        if not self.chunk_dataset_id.startswith(
            f"chunk_dataset_{self.report_id}_"
        ):
            raise ValueError(
                "chunk_dataset_id "
                "必须属于 report_id"
            )

        if not self.document_id.startswith(
            f"doc_{self.report_id}_"
        ):
            raise ValueError(
                "document_id 必须属于 report_id"
            )

        expected_tokenizer_sha256 = (
            calculate_bm25_tokenizer_spec_sha256(
                self.tokenizer_spec
            )
        )

        if (
            self.tokenizer_spec_sha256
            != expected_tokenizer_sha256
        ):
            raise ValueError(
                "tokenizer_spec_sha256 "
                "与 TokenizerSpec 不一致"
            )

        expected_config_sha256 = (
            calculate_bm25_config_sha256(
                self.bm25_config
            )
        )

        if (
            self.bm25_config_sha256
            != expected_config_sha256
        ):
            raise ValueError(
                "bm25_config_sha256 "
                "与 BM25Config 不一致"
            )

        if (
            self.document_count
            != self.metadata_record_count
        ):
            raise ValueError(
                "文档数量必须与元数据数量一致"
            )

        expected_average = (
            self.total_token_count
            / self.document_count
        )

        if (
            abs(
                self.average_document_length
                - expected_average
            )
            > 1e-12
        ):
            raise ValueError(
                "平均文档长度与 Token 统计不一致"
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