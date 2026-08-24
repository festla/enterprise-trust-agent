from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from app.schemas.competition_chunk import (
    CompetitionChunkDocument,
    CompetitionTextChunk,
)
from app.schemas.competition_corpus import (
    CompetitionChunkCorpusManifest,
    CompetitionCorpusSourceRecord,
    build_competition_corpus_id,
    calculate_competition_corpus_identity,
)


CHUNKS_FILENAME = "chunks.jsonl"
MANIFEST_FILENAME = "corpus_manifest.json"


class CompetitionCorpusError(
    ValueError
):
    """Competition Corpus 基础异常。"""


class InvalidCompetitionCorpusError(
    CompetitionCorpusError
):
    """Corpus 输入或内部关系无效。"""


class CorruptCompetitionCorpusError(
    CompetitionCorpusError
):
    """已落盘 Corpus 内容损坏。"""


class CompetitionCorpusIdentityConflictError(
    CompetitionCorpusError
):
    """已有 Corpus 与当前预期身份冲突。"""


@dataclass(frozen=True)
class CompetitionCorpusBuildResult:
    manifest: CompetitionChunkCorpusManifest
    corpus_directory: Path
    chunks_path: Path
    manifest_path: Path
    chunks: tuple[
        CompetitionTextChunk,
        ...
    ]


def _sha256_bytes(
    content: bytes,
) -> str:
    return hashlib.sha256(
        content
    ).hexdigest()


def _canonical_json_bytes(
    payload: object,
) -> bytes:
    return json.dumps(
        payload,
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


def serialize_competition_chunks(
    chunks: tuple[
        CompetitionTextChunk,
        ...
    ],
) -> bytes:
    if not chunks:
        raise InvalidCompetitionCorpusError(
            "Corpus 不能没有 Chunk"
        )

    lines = [
        _canonical_json_bytes(
            chunk.model_dump(
                mode="json"
            )
        )
        for chunk in chunks
    ]

    return (
        b"\n".join(lines)
        + b"\n"
    )


def _normalize_documents(
    documents: tuple[
        CompetitionChunkDocument,
        ...
    ],
) -> tuple[
    CompetitionChunkDocument,
    ...
]:
    if not documents:
        raise InvalidCompetitionCorpusError(
            "Corpus 不能没有来源文档"
        )

    ordered = tuple(
        sorted(
            documents,
            key=lambda document: (
                document.source.source_id
            ),
        )
    )

    source_ids = [
        document.source.source_id
        for document in ordered
    ]

    doc_ids = [
        document.source.doc_id
        for document in ordered
    ]

    if (
        len(source_ids)
        != len(set(source_ids))
    ):
        raise InvalidCompetitionCorpusError(
            "Corpus 包含重复 source_id"
        )

    if (
        len(doc_ids)
        != len(set(doc_ids))
    ):
        raise InvalidCompetitionCorpusError(
            "Corpus 包含重复 doc_id"
        )

    for document in ordered:
        if document.source.sha256 is None:
            raise InvalidCompetitionCorpusError(
                "Corpus 来源缺少 sha256: "
                f"{document.source.source_id}"
            )

    return ordered


def _flatten_chunks(
    documents: tuple[
        CompetitionChunkDocument,
        ...
    ],
) -> tuple[
    CompetitionTextChunk,
    ...
]:
    chunks = tuple(
        chunk
        for document in documents
        for chunk in document.chunks
    )

    chunk_ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    if (
        len(chunk_ids)
        != len(set(chunk_ids))
    ):
        raise InvalidCompetitionCorpusError(
            "Corpus 包含重复 chunk_id"
        )

    return chunks


def _build_source_records(
    documents: tuple[
        CompetitionChunkDocument,
        ...
    ],
) -> tuple[
    CompetitionCorpusSourceRecord,
    ...
]:
    records = []

    for document in documents:
        source = document.source
        chunks = document.chunks

        records.append(
            CompetitionCorpusSourceRecord(
                source_id=source.source_id,
                doc_id=source.doc_id,
                source_type=(
                    source.source_type
                ),
                relative_path=(
                    source.relative_path
                ),
                source_sha256=(
                    source.sha256
                ),
                chunk_count=len(chunks),
                chunk_char_count=sum(
                    chunk.char_count
                    for chunk in chunks
                ),
                text_chunk_count=sum(
                    chunk.chunk_type
                    == "text"
                    for chunk in chunks
                ),
                table_chunk_count=sum(
                    chunk.chunk_type
                    == "table"
                    for chunk in chunks
                ),
            )
        )

    return tuple(records)


def _build_manifest(
    *,
    documents: tuple[
        CompetitionChunkDocument,
        ...
    ],
    chunks: tuple[
        CompetitionTextChunk,
        ...
    ],
    chunks_bytes: bytes,
    max_chars: int,
) -> CompetitionChunkCorpusManifest:
    source_records = (
        _build_source_records(
            documents
        )
    )

    payload: dict[
        str,
        object,
    ] = {
        "schema_version": 1,
        "corpus_scope": "qa_used_all",
        "chunker_version": (
            "competition_text_chunker_v1"
        ),
        "max_chars": max_chars,
        "chunks_jsonl_sha256": (
            _sha256_bytes(
                chunks_bytes
            )
        ),
        "source_count": len(
            source_records
        ),
        "doc_count": len(
            {
                record.doc_id
                for record in source_records
            }
        ),
        "chunk_count": len(chunks),
        "chunk_char_count": sum(
            chunk.char_count
            for chunk in chunks
        ),
        "text_chunk_count": sum(
            chunk.chunk_type == "text"
            for chunk in chunks
        ),
        "table_chunk_count": sum(
            chunk.chunk_type == "table"
            for chunk in chunks
        ),
        "source_type_counts": dict(
            Counter(
                record.source_type
                for record in source_records
            )
        ),
        "chunk_type_counts": {
            "text": sum(
                chunk.chunk_type == "text"
                for chunk in chunks
            ),
            "table": sum(
                chunk.chunk_type == "table"
                for chunk in chunks
            ),
        },
        "source_records": (
            source_records
        ),
    }

    identity = (
        calculate_competition_corpus_identity(
            payload
        )
    )

    return CompetitionChunkCorpusManifest(
        **payload,
        corpus_identity_sha256=identity,
        corpus_id=(
            build_competition_corpus_id(
                identity
            )
        ),
    )


def _validate_chunks_against_manifest(
    *,
    manifest: CompetitionChunkCorpusManifest,
    chunks: tuple[
        CompetitionTextChunk,
        ...
    ],
) -> None:
    if (
        len(chunks)
        != manifest.chunk_count
    ):
        raise CorruptCompetitionCorpusError(
            "Chunk 数量与 Manifest 不一致"
        )

    expected_order = sorted(
        chunks,
        key=lambda chunk: (
            chunk.source_id,
            chunk.chunk_index,
        ),
    )

    if list(chunks) != expected_order:
        raise CorruptCompetitionCorpusError(
            "Chunk 顺序不是 "
            "source_id/chunk_index 稳定顺序"
        )

    chunk_ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    if (
        len(chunk_ids)
        != len(set(chunk_ids))
    ):
        raise CorruptCompetitionCorpusError(
            "Corpus 包含重复 chunk_id"
        )

    records_by_source = {
        record.source_id: record
        for record in manifest.source_records
    }

    for chunk in chunks:
        record = records_by_source.get(
            chunk.source_id
        )

        if record is None:
            raise CorruptCompetitionCorpusError(
                "Chunk 引用了未知 source_id"
            )

        if (
            chunk.doc_id
            != record.doc_id
            or chunk.source_type
            != record.source_type
        ):
            raise CorruptCompetitionCorpusError(
                "Chunk 来源身份与 "
                "Manifest 不一致"
            )

    for record in (
        manifest.source_records
    ):
        source_chunks = [
            chunk
            for chunk in chunks
            if (
                chunk.source_id
                == record.source_id
            )
        ]

        indexes = [
            chunk.chunk_index
            for chunk in source_chunks
        ]

        if indexes != list(
            range(len(source_chunks))
        ):
            raise CorruptCompetitionCorpusError(
                "来源 Chunk index 不连续: "
                f"{record.source_id}"
            )

        if (
            len(source_chunks)
            != record.chunk_count
        ):
            raise CorruptCompetitionCorpusError(
                "来源 Chunk 数量不一致"
            )

        if (
            sum(
                chunk.char_count
                for chunk in source_chunks
            )
            != record.chunk_char_count
        ):
            raise CorruptCompetitionCorpusError(
                "来源 Chunk 字符数不一致"
            )

        if (
            sum(
                chunk.chunk_type == "text"
                for chunk in source_chunks
            )
            != record.text_chunk_count
        ):
            raise CorruptCompetitionCorpusError(
                "来源 text Chunk 数量不一致"
            )

        if (
            sum(
                chunk.chunk_type == "table"
                for chunk in source_chunks
            )
            != record.table_chunk_count
        ):
            raise CorruptCompetitionCorpusError(
                "来源 table Chunk 数量不一致"
            )


def load_competition_chunk_corpus(
    corpus_directory: Path,
) -> CompetitionCorpusBuildResult:
    chunks_path = (
        corpus_directory
        / CHUNKS_FILENAME
    )

    manifest_path = (
        corpus_directory
        / MANIFEST_FILENAME
    )

    if (
        not chunks_path.is_file()
        or not manifest_path.is_file()
    ):
        raise CorruptCompetitionCorpusError(
            "Corpus 缺少 chunks.jsonl "
            "或 corpus_manifest.json"
        )

    try:
        manifest = (
            CompetitionChunkCorpusManifest
            .model_validate_json(
                manifest_path.read_bytes()
            )
        )
    except Exception as exc:
        raise CorruptCompetitionCorpusError(
            "Corpus Manifest 无效"
        ) from exc

    if (
        corpus_directory.name
        != manifest.corpus_id
    ):
        raise CorruptCompetitionCorpusError(
            "Corpus 目录名与 "
            "corpus_id 不一致"
        )

    chunks_bytes = (
        chunks_path.read_bytes()
    )

    if (
        _sha256_bytes(chunks_bytes)
        != manifest.chunks_jsonl_sha256
    ):
        raise CorruptCompetitionCorpusError(
            "chunks.jsonl SHA256 "
            "与 Manifest 不一致"
        )

    try:
        chunks = tuple(
            CompetitionTextChunk
            .model_validate_json(
                line
            )
            for line in (
                chunks_bytes
                .splitlines()
            )
            if line.strip()
        )
    except Exception as exc:
        raise CorruptCompetitionCorpusError(
            "chunks.jsonl 包含无效记录"
        ) from exc

    _validate_chunks_against_manifest(
        manifest=manifest,
        chunks=chunks,
    )

    return CompetitionCorpusBuildResult(
        manifest=manifest,
        corpus_directory=(
            corpus_directory
        ),
        chunks_path=chunks_path,
        manifest_path=manifest_path,
        chunks=chunks,
    )


def build_competition_chunk_corpus(
    *,
    documents: tuple[
        CompetitionChunkDocument,
        ...
    ],
    output_root: Path,
    max_chars: int = 900,
) -> CompetitionCorpusBuildResult:
    if max_chars <= 0:
        raise ValueError(
            "max_chars 必须大于 0"
        )

    ordered_documents = (
        _normalize_documents(
            documents
        )
    )

    chunks = _flatten_chunks(
        ordered_documents
    )

    chunks_bytes = (
        serialize_competition_chunks(
            chunks
        )
    )

    manifest = _build_manifest(
        documents=ordered_documents,
        chunks=chunks,
        chunks_bytes=chunks_bytes,
        max_chars=max_chars,
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    corpus_directory = (
        output_root
        / manifest.corpus_id
    )

    if corpus_directory.exists():
        loaded = (
            load_competition_chunk_corpus(
                corpus_directory
            )
        )

        if (
            loaded.manifest != manifest
            or loaded.chunks != chunks
        ):
            raise (
                CompetitionCorpusIdentityConflictError(
                    "已有 Corpus 与当前 "
                    "预期内容不一致"
                )
            )

        return loaded

    manifest_bytes = (
        _canonical_json_bytes(
            manifest.model_dump(
                mode="json"
            )
        )
        + b"\n"
    )

    with TemporaryDirectory(
        prefix=".competition-corpus-",
        dir=output_root,
    ) as temporary_root:
        temporary_directory = (
            Path(temporary_root)
            / manifest.corpus_id
        )

        temporary_directory.mkdir()

        (
            temporary_directory
            / CHUNKS_FILENAME
        ).write_bytes(
            chunks_bytes
        )

        (
            temporary_directory
            / MANIFEST_FILENAME
        ).write_bytes(
            manifest_bytes
        )

        temporary_directory.replace(
            corpus_directory
        )

    return load_competition_chunk_corpus(
        corpus_directory
    )