from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import hashlib
import json
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.schemas.bm25 import (
    BM25Config,
    BM25TokenizerSpec,
    calculate_bm25_config_sha256,
    calculate_bm25_tokenizer_spec_sha256,
)


INDEX_IDENTITY_EXCLUDED_FIELDS = frozenset(
    {
        "index_id",
        "index_identity_sha256",
        "created_at",
    }
)


def _normalize_identity_value(
    value: object,
) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(
            mode="json"
        )

    if isinstance(value, Mapping):
        return {
            str(key): (
                _normalize_identity_value(
                    item
                )
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            _normalize_identity_value(
                item
            )
            for item in value
        ]

    if isinstance(value, datetime):
        return value.isoformat()

    return value


def calculate_competition_bm25_identity(
    payload: Mapping[str, object],
) -> str:
    identity_payload = {
        key: _normalize_identity_value(
            value
        )
        for key, value in payload.items()
        if (
            key
            not in INDEX_IDENTITY_EXCLUDED_FIELDS
        )
    }

    canonical_bytes = json.dumps(
        identity_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(
        canonical_bytes
    ).hexdigest()


def build_competition_bm25_index_id(
    identity_sha256: str,
) -> str:
    if (
        len(identity_sha256) != 64
        or any(
            char
            not in "0123456789abcdef"
            for char in identity_sha256
        )
    ):
        raise ValueError(
            "identity_sha256 必须是"
            "64位小写十六进制字符串"
        )

    return (
        "competition_bm25_index_"
        f"{identity_sha256[:16]}"
    )


class CompetitionBM25DocumentRecord(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: int = Field(
        default=1,
        ge=1,
        le=1,
    )

    chunk_id: str = Field(
        min_length=1,
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
    def validate_record(
        self,
    ) -> Self:
        if (
            sum(
                self.term_frequencies.values()
            )
            != self.document_length
        ):
            raise ValueError(
                "document_length 必须等于"
                " term_frequencies 之和"
            )

        return self


class CompetitionBM25IndexData(
    BaseModel
):
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
        CompetitionBM25DocumentRecord,
        ...,
    ] = Field(
        min_length=1,
    )

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
            CompetitionBM25DocumentRecord,
            ...,
        ],
    ) -> tuple[
        CompetitionBM25DocumentRecord,
        ...,
    ]:
        chunk_ids = tuple(
            record.chunk_id
            for record in value
        )

        if (
            len(chunk_ids)
            != len(set(chunk_ids))
        ):
            raise ValueError(
                "document_records 包含重复"
                " chunk_id"
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
                "document_count 与记录数量不一致"
            )

        if (
            len(self.document_frequencies)
            != self.vocabulary_size
        ):
            raise ValueError(
                "vocabulary_size 与词项数量不一致"
            )

        if any(
            frequency
            > self.document_count
            for frequency
            in self.document_frequencies.values()
        ):
            raise ValueError(
                "文档频率不能超过文档数量"
            )

        expected_total = sum(
            record.document_length
            for record
            in self.document_records
        )

        if (
            expected_total
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

        for record in (
            self.document_records
        ):
            for term in (
                record.term_frequencies
            ):
                calculated_frequencies[
                    term
                ] = (
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
                "与词频记录不一致"
            )

        return self


class CompetitionBM25IndexManifest(
    BaseModel
):
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

    index_id: str = Field(
        pattern=(
            r"^competition_bm25_index_"
            r"[0-9a-f]{16}$"
        ),
    )

    index_identity_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    corpus_id: str = Field(
        pattern=(
            r"^competition_corpus_"
            r"[0-9a-f]{16}$"
        ),
    )

    corpus_identity_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    corpus_manifest_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    corpus_chunks_jsonl_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    corpus_source_count: int = Field(
        ge=1,
    )

    corpus_doc_count: int = Field(
        ge=1,
    )

    corpus_chunk_count: int = Field(
        ge=1,
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
    )

    index_type: str = Field(
        default="exact_bm25",
        pattern=r"^exact_bm25$",
    )

    index_version: str = Field(
        default=(
            "competition_exact_bm25_v1"
        ),
        pattern=(
            r"^competition_exact_bm25_v1$"
        ),
    )

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

    quality_gate_errors: tuple[
        str,
        ...,
    ] = ()

    created_at: datetime

    @field_validator(
        "created_at"
    )
    @classmethod
    def validate_created_at(
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
                "与配置不一致"
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
                "与配置不一致"
            )

        if (
            self.document_count
            != self.corpus_chunk_count
            or self.metadata_record_count
            != self.corpus_chunk_count
        ):
            raise ValueError(
                "BM25 文档数量必须等于"
                " Corpus Chunk 数量"
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
                "平均文档长度与 Token "
                "统计不一致"
            )

        expected_quality = (
            len(
                self.quality_gate_errors
            )
            == 0
        )

        if (
            self.quality_gate_passed
            != expected_quality
        ):
            raise ValueError(
                "quality_gate_passed 必须与"
                " quality_gate_errors 一致"
            )

        identity_payload = (
            self.model_dump(
                mode="json",
                exclude=(
                    INDEX_IDENTITY_EXCLUDED_FIELDS
                ),
            )
        )

        expected_identity = (
            calculate_competition_bm25_identity(
                identity_payload
            )
        )

        if (
            self.index_identity_sha256
            != expected_identity
        ):
            raise ValueError(
                "index_identity_sha256 "
                "与 Manifest 内容不一致"
            )

        expected_index_id = (
            build_competition_bm25_index_id(
                expected_identity
            )
        )

        if (
            self.index_id
            != expected_index_id
        ):
            raise ValueError(
                "index_id 与 Manifest "
                "身份不一致"
            )

        return self