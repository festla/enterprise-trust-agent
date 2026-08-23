from __future__ import annotations

from collections import Counter
import hashlib
import json
from collections.abc import Mapping
from typing import (
    Literal,
    Self,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.schemas.competition_chunk import (
    CompetitionChunkType,
)
from app.schemas.competition_text import (
    CompetitionTextSourceType,
)


CORPUS_IDENTITY_FIELDS = frozenset(
    {
        "corpus_id",
        "corpus_identity_sha256",
    }
)


def calculate_competition_corpus_identity(
    payload: Mapping[
        str,
        object,
    ],
) -> str:
    """
    根据不包含 corpus_id 和 identity 字段的
    Manifest 内容生成稳定 SHA256。

    source_records 会先经过 Pydantic 规范化，
    保证原始字典和已构造 Model 得到相同身份。
    """

    identity_payload = {
        key: value
        for key, value in payload.items()
        if key not in CORPUS_IDENTITY_FIELDS
    }

    raw_source_records = (
        identity_payload.get(
            "source_records"
        )
    )

    if isinstance(
        raw_source_records,
        (
            list,
            tuple,
        ),
    ):
        identity_payload[
            "source_records"
        ] = [
            (
                CompetitionCorpusSourceRecord
                .model_validate(
                    record
                )
                .model_dump(
                    mode="json"
                )
            )
            for record in raw_source_records
        ]

    canonical_bytes = json.dumps(
        identity_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        allow_nan=False,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        canonical_bytes
    ).hexdigest()


def build_competition_corpus_id(
    identity_sha256: str,
) -> str:
    if (
        len(identity_sha256) != 64
        or any(
            char not in "0123456789abcdef"
            for char in identity_sha256
        )
    ):
        raise ValueError(
            "identity_sha256 必须是 "
            "64位小写十六进制字符串"
        )

    return (
        "competition_corpus_"
        f"{identity_sha256[:16]}"
    )


class CompetitionCorpusSourceRecord(
    BaseModel
):
    """冻结 Corpus 中一个来源文档的统计记录。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    source_id: str = Field(
        min_length=1,
    )

    doc_id: str = Field(
        min_length=1,
    )

    source_type: (
        CompetitionTextSourceType
    )

    relative_path: str = Field(
        min_length=1,
    )

    source_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    chunk_count: int = Field(
        ge=1,
    )

    chunk_char_count: int = Field(
        ge=1,
    )

    text_chunk_count: int = Field(
        ge=0,
    )

    table_chunk_count: int = Field(
        ge=0,
    )

    @model_validator(
        mode="after"
    )
    def validate_source_record(
        self,
    ) -> Self:
        if (
            self.chunk_count
            != (
                self.text_chunk_count
                + self.table_chunk_count
            )
        ):
            raise ValueError(
                "chunk_count 必须等于 "
                "text_chunk_count 与 "
                "table_chunk_count 之和"
            )

        return self


class CompetitionChunkCorpusManifest(
    BaseModel
):
    """
    Competition 统一文本 Chunk Corpus 的
    可验证 Manifest。

    Manifest 不保存问题、答案或 evidence，
    只保存来源与 Chunk 统计身份。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    corpus_scope: Literal[
        "qa_used_all"
    ] = "qa_used_all"

    chunker_version: Literal[
        "competition_text_chunker_v1"
    ] = "competition_text_chunker_v1"

    max_chars: int = Field(
        ge=1,
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

    chunks_jsonl_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    source_count: int = Field(
        ge=1,
    )

    doc_count: int = Field(
        ge=1,
    )

    chunk_count: int = Field(
        ge=1,
    )

    chunk_char_count: int = Field(
        ge=1,
    )

    text_chunk_count: int = Field(
        ge=0,
    )

    table_chunk_count: int = Field(
        ge=0,
    )

    source_type_counts: dict[
        CompetitionTextSourceType,
        int,
    ]

    chunk_type_counts: dict[
        CompetitionChunkType,
        int,
    ]

    source_records: tuple[
        CompetitionCorpusSourceRecord,
        ...
    ] = Field(
        min_length=1,
    )

    @model_validator(
        mode="after"
    )
    def validate_manifest(
        self,
    ) -> Self:
        source_ids = [
            record.source_id
            for record in self.source_records
        ]

        doc_ids = [
            record.doc_id
            for record in self.source_records
        ]

        if source_ids != sorted(
            source_ids
        ):
            raise ValueError(
                "source_records 必须按照 "
                "source_id 排序"
            )

        if (
            len(source_ids)
            != len(set(source_ids))
        ):
            raise ValueError(
                "source_records 包含重复 "
                "source_id"
            )

        if (
            len(doc_ids)
            != len(set(doc_ids))
        ):
            raise ValueError(
                "source_records 包含重复 "
                "doc_id"
            )

        if (
            self.source_count
            != len(self.source_records)
        ):
            raise ValueError(
                "source_count 与 "
                "source_records 数量不一致"
            )

        if (
            self.doc_count
            != len(set(doc_ids))
        ):
            raise ValueError(
                "doc_count 与唯一 doc_id "
                "数量不一致"
            )

        expected_chunk_count = sum(
            record.chunk_count
            for record in self.source_records
        )

        expected_char_count = sum(
            record.chunk_char_count
            for record in self.source_records
        )

        expected_text_count = sum(
            record.text_chunk_count
            for record in self.source_records
        )

        expected_table_count = sum(
            record.table_chunk_count
            for record in self.source_records
        )

        if (
            self.chunk_count
            != expected_chunk_count
        ):
            raise ValueError(
                "chunk_count 与来源统计不一致"
            )

        if (
            self.chunk_char_count
            != expected_char_count
        ):
            raise ValueError(
                "chunk_char_count "
                "与来源统计不一致"
            )

        if (
            self.text_chunk_count
            != expected_text_count
        ):
            raise ValueError(
                "text_chunk_count "
                "与来源统计不一致"
            )

        if (
            self.table_chunk_count
            != expected_table_count
        ):
            raise ValueError(
                "table_chunk_count "
                "与来源统计不一致"
            )

        if (
            self.chunk_count
            != (
                self.text_chunk_count
                + self.table_chunk_count
            )
        ):
            raise ValueError(
                "chunk_count 必须等于 "
                "text/table Chunk 之和"
            )

        expected_source_types = dict(
            Counter(
                record.source_type
                for record in self.source_records
            )
        )

        if (
            self.source_type_counts
            != expected_source_types
        ):
            raise ValueError(
                "source_type_counts "
                "与来源记录不一致"
            )

        expected_chunk_types = {
            "text": (
                self.text_chunk_count
            ),
            "table": (
                self.table_chunk_count
            ),
        }

        if (
            self.chunk_type_counts
            != expected_chunk_types
        ):
            raise ValueError(
                "chunk_type_counts "
                "与 Chunk 统计不一致"
            )

        if any(
            count < 0
            for count in (
                *self.source_type_counts.values(),
                *self.chunk_type_counts.values(),
            )
        ):
            raise ValueError(
                "类型统计不能小于 0"
            )

        identity_payload = (
            self.model_dump(
                mode="json",
                exclude={
                    "corpus_id",
                    "corpus_identity_sha256",
                },
            )
        )

        expected_identity = (
            calculate_competition_corpus_identity(
                identity_payload
            )
        )

        if (
            self.corpus_identity_sha256
            != expected_identity
        ):
            raise ValueError(
                "corpus_identity_sha256 "
                "与 Manifest 内容不一致"
            )

        expected_corpus_id = (
            build_competition_corpus_id(
                expected_identity
            )
        )

        if (
            self.corpus_id
            != expected_corpus_id
        ):
            raise ValueError(
                "corpus_id 与 Manifest "
                "身份不一致"
            )

        return self