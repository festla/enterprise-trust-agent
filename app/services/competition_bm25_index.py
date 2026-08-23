from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType

from pydantic import ValidationError

from app.rag.competition_bm25 import (
    CompetitionExactBM25Index,
)
from app.rag.tokenization import (
    BM25Tokenizer,
    DeterministicChineseBigramTokenizer,
)
from app.schemas.bm25 import (
    BM25Config,
    calculate_bm25_config_sha256,
    calculate_bm25_tokenizer_spec_sha256,
)
from app.schemas.competition_bm25 import (
    CompetitionBM25DocumentRecord,
    CompetitionBM25IndexData,
    CompetitionBM25IndexManifest,
    build_competition_bm25_index_id,
    calculate_competition_bm25_identity,
)
from app.schemas.competition_chunk import (
    CompetitionTextChunk,
)
from app.services.competition_corpus import (
    CompetitionCorpusBuildResult,
    CompetitionCorpusError,
    load_competition_chunk_corpus,
    serialize_competition_chunks,
)


INDEX_FILENAME = "index.json"
METADATA_FILENAME = "metadata.jsonl"
MANIFEST_FILENAME = "index_manifest.json"


class CompetitionBM25IndexServiceError(
    ValueError
):
    """Competition BM25 持久化服务异常。"""


class InvalidCompetitionBM25SourceError(
    CompetitionBM25IndexServiceError
):
    """来源 Corpus 无效。"""


class CorruptCompetitionBM25IndexError(
    CompetitionBM25IndexServiceError
):
    """索引文件损坏或内部关系不一致。"""


class CompetitionBM25IndexIdentityConflictError(
    CompetitionBM25IndexServiceError
):
    """索引与指定 Corpus 身份不一致。"""


class CompetitionBM25IndexWriteError(
    CompetitionBM25IndexServiceError
):
    """索引无法安全写入磁盘。"""


@dataclass(frozen=True, slots=True)
class CompetitionBM25IndexResult:
    index_directory: Path
    index_path: Path
    metadata_path: Path
    manifest_path: Path
    manifest: CompetitionBM25IndexManifest
    index: CompetitionExactBM25Index


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(
    payload: object,
) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _calculate_tokenized_corpus_sha256(
    *,
    chunks: tuple[CompetitionTextChunk, ...],
    tokenizer: BM25Tokenizer,
) -> str:
    records = tuple(
        {
            "chunk_id": chunk.chunk_id,
            "tokens": tokenizer.tokenize(
                chunk.text
            ),
        }
        for chunk in chunks
    )

    return _sha256_bytes(
        _canonical_json_bytes(records)
    )


def _build_index_data(
    index: CompetitionExactBM25Index,
) -> CompetitionBM25IndexData:
    records = tuple(
        CompetitionBM25DocumentRecord(
            chunk_id=chunk.chunk_id,
            document_length=(
                index.document_lengths[
                    position
                ]
            ),
            term_frequencies=dict(
                index.term_frequencies[
                    position
                ]
            ),
        )
        for position, chunk
        in enumerate(index.chunks)
    )

    total_token_count = sum(
        index.document_lengths
    )

    return CompetitionBM25IndexData(
        document_records=records,
        document_frequencies=dict(
            index.document_frequencies
        ),
        document_count=len(
            index.chunks
        ),
        vocabulary_size=len(
            index.document_frequencies
        ),
        total_token_count=(
            total_token_count
        ),
        average_document_length=(
            index.average_document_length
        ),
    )


def _serialize_index_data(
    data: CompetitionBM25IndexData,
) -> bytes:
    return _canonical_json_bytes(
        data.model_dump(mode="json")
    )


def _serialize_manifest(
    manifest: CompetitionBM25IndexManifest,
) -> bytes:
    return (
        manifest.model_dump_json(indent=2)
        + "\n"
    ).encode("utf-8")


def _load_metadata(
    metadata_bytes: bytes,
) -> tuple[CompetitionTextChunk, ...]:
    try:
        text = metadata_bytes.decode(
            "utf-8"
        )
    except UnicodeDecodeError as exc:
        raise CorruptCompetitionBM25IndexError(
            "metadata.jsonl 不是合法 UTF-8"
        ) from exc

    lines = text.splitlines()

    if (
        not lines
        or any(
            not line.strip()
            for line in lines
        )
    ):
        raise CorruptCompetitionBM25IndexError(
            "metadata.jsonl 为空或包含空记录"
        )

    try:
        return tuple(
            CompetitionTextChunk
            .model_validate_json(line)
            for line in lines
        )
    except ValidationError as exc:
        raise CorruptCompetitionBM25IndexError(
            "metadata.jsonl 包含无效 Chunk"
        ) from exc


def _validate_loaded_data(
    *,
    index_directory: Path,
    manifest: CompetitionBM25IndexManifest,
    data: CompetitionBM25IndexData,
    chunks: tuple[CompetitionTextChunk, ...],
) -> None:
    if (
        index_directory.name
        != manifest.index_id
    ):
        raise CorruptCompetitionBM25IndexError(
            "索引目录名与 index_id 不一致"
        )

    if not manifest.quality_gate_passed:
        raise CorruptCompetitionBM25IndexError(
            "BM25 Index 未通过质量门禁"
        )

    if (
        len(chunks)
        != manifest.metadata_record_count
        or len(chunks)
        != manifest.corpus_chunk_count
    ):
        raise CorruptCompetitionBM25IndexError(
            "Chunk 数量与 Manifest 不一致"
        )

    if (
        data.document_count
        != manifest.document_count
    ):
        raise CorruptCompetitionBM25IndexError(
            "index.json 文档数量与 Manifest 不一致"
        )

    if (
        data.vocabulary_size
        != manifest.vocabulary_size
    ):
        raise CorruptCompetitionBM25IndexError(
            "词表大小与 Manifest 不一致"
        )

    if (
        data.total_token_count
        != manifest.total_token_count
    ):
        raise CorruptCompetitionBM25IndexError(
            "Token 总数与 Manifest 不一致"
        )

    if (
        abs(
            data.average_document_length
            - manifest.average_document_length
        )
        > 1e-12
    ):
        raise CorruptCompetitionBM25IndexError(
            "平均文档长度与 Manifest 不一致"
        )

    chunk_ids = tuple(
        chunk.chunk_id
        for chunk in chunks
    )

    record_chunk_ids = tuple(
        record.chunk_id
        for record
        in data.document_records
    )

    if chunk_ids != record_chunk_ids:
        raise CorruptCompetitionBM25IndexError(
            "index.json 与 metadata.jsonl "
            "的 Chunk 顺序不一致"
        )

    if (
        len(chunk_ids)
        != len(set(chunk_ids))
    ):
        raise CorruptCompetitionBM25IndexError(
            "metadata.jsonl 包含重复 chunk_id"
        )

    expected_order = tuple(
        sorted(
            chunks,
            key=lambda chunk: (
                chunk.source_id,
                chunk.chunk_index,
            ),
        )
    )

    if chunks != expected_order:
        raise CorruptCompetitionBM25IndexError(
            "Chunk 顺序不是稳定 Corpus 顺序"
        )

    if (
        len(
            {
                chunk.source_id
                for chunk in chunks
            }
        )
        != manifest.corpus_source_count
    ):
        raise CorruptCompetitionBM25IndexError(
            "source_id 数量与 Manifest 不一致"
        )

    if (
        len(
            {
                chunk.doc_id
                for chunk in chunks
            }
        )
        != manifest.corpus_doc_count
    ):
        raise CorruptCompetitionBM25IndexError(
            "doc_id 数量与 Manifest 不一致"
        )


def _validate_token_statistics(
    *,
    manifest: CompetitionBM25IndexManifest,
    data: CompetitionBM25IndexData,
    chunks: tuple[CompetitionTextChunk, ...],
) -> None:
    tokenizer = (
        DeterministicChineseBigramTokenizer(
            spec=manifest.tokenizer_spec
        )
    )

    actual_sha256 = (
        _calculate_tokenized_corpus_sha256(
            chunks=chunks,
            tokenizer=tokenizer,
        )
    )

    if (
        actual_sha256
        != manifest.tokenized_corpus_sha256
    ):
        raise CorruptCompetitionBM25IndexError(
            "Tokenized Corpus 哈希校验失败"
        )

    for chunk, record in zip(
        chunks,
        data.document_records,
        strict=True,
    ):
        tokens = tokenizer.tokenize(
            chunk.text
        )

        actual_term_frequencies = dict(
            sorted(
                Counter(tokens).items()
            )
        )

        if (
            actual_term_frequencies
            != record.term_frequencies
        ):
            raise CorruptCompetitionBM25IndexError(
                "持久化词频与 Chunk 文本不一致："
                f"{chunk.chunk_id}"
            )

        if (
            len(tokens)
            != record.document_length
        ):
            raise CorruptCompetitionBM25IndexError(
                "文档长度与分词结果不一致："
                f"{chunk.chunk_id}"
            )


def _restore_memory_index(
    *,
    manifest: CompetitionBM25IndexManifest,
    data: CompetitionBM25IndexData,
    chunks: tuple[CompetitionTextChunk, ...],
) -> CompetitionExactBM25Index:
    return CompetitionExactBM25Index(
        chunks=chunks,
        term_frequencies=tuple(
            MappingProxyType(
                dict(
                    record.term_frequencies
                )
            )
            for record
            in data.document_records
        ),
        document_frequencies=(
            MappingProxyType(
                dict(
                    data.document_frequencies
                )
            )
        ),
        document_lengths=tuple(
            record.document_length
            for record
            in data.document_records
        ),
        average_document_length=(
            data.average_document_length
        ),
        tokenizer_spec=(
            manifest.tokenizer_spec
        ),
        config=manifest.bm25_config,
    )


def _load_source_corpus(
    corpus_directory: Path,
) -> CompetitionCorpusBuildResult:
    try:
        return load_competition_chunk_corpus(
            corpus_directory
        )
    except CompetitionCorpusError as exc:
        raise InvalidCompetitionBM25SourceError(
            str(exc)
        ) from exc


def _validate_source_binding(
    *,
    result: CompetitionBM25IndexResult,
    source: CompetitionCorpusBuildResult,
) -> None:
    manifest = result.manifest

    expected = {
        "corpus_id": (
            source.manifest.corpus_id
        ),
        "corpus_identity_sha256": (
            source.manifest
            .corpus_identity_sha256
        ),
        "corpus_manifest_sha256": (
            _sha256_bytes(
                source.manifest_path
                .read_bytes()
            )
        ),
        "corpus_chunks_jsonl_sha256": (
            source.manifest
            .chunks_jsonl_sha256
        ),
        "corpus_source_count": (
            source.manifest.source_count
        ),
        "corpus_doc_count": (
            source.manifest.doc_count
        ),
        "corpus_chunk_count": (
            source.manifest.chunk_count
        ),
    }

    actual = {
        key: getattr(manifest, key)
        for key in expected
    }

    if (
        actual != expected
        or result.index.chunks
        != source.chunks
    ):
        raise (
            CompetitionBM25IndexIdentityConflictError(
                "BM25 索引与指定 Corpus "
                "身份不一致"
            )
        )


def load_competition_bm25_index(
    index_directory: Path,
    *,
    corpus_directory: Path | None = None,
) -> CompetitionBM25IndexResult:
    index_path = (
        index_directory
        / INDEX_FILENAME
    )
    metadata_path = (
        index_directory
        / METADATA_FILENAME
    )
    manifest_path = (
        index_directory
        / MANIFEST_FILENAME
    )

    if not all(
        path.is_file()
        for path in (
            index_path,
            metadata_path,
            manifest_path,
        )
    ):
        raise CorruptCompetitionBM25IndexError(
            "BM25 索引目录缺少必要文件"
        )

    try:
        index_bytes = (
            index_path.read_bytes()
        )
        metadata_bytes = (
            metadata_path.read_bytes()
        )
        manifest_bytes = (
            manifest_path.read_bytes()
        )
    except OSError as exc:
        raise CorruptCompetitionBM25IndexError(
            "无法读取 BM25 索引文件"
        ) from exc

    try:
        manifest = (
            CompetitionBM25IndexManifest
            .model_validate_json(
                manifest_bytes
            )
        )
        data = (
            CompetitionBM25IndexData
            .model_validate_json(
                index_bytes
            )
        )
    except ValidationError as exc:
        raise CorruptCompetitionBM25IndexError(
            "BM25 索引 Schema 校验失败"
        ) from exc

    if (
        _sha256_bytes(index_bytes)
        != manifest.index_json_sha256
    ):
        raise CorruptCompetitionBM25IndexError(
            "index.json SHA256 校验失败"
        )

    if (
        _sha256_bytes(metadata_bytes)
        != manifest.metadata_jsonl_sha256
    ):
        raise CorruptCompetitionBM25IndexError(
            "metadata.jsonl SHA256 校验失败"
        )

    chunks = _load_metadata(
        metadata_bytes
    )

    _validate_loaded_data(
        index_directory=index_directory,
        manifest=manifest,
        data=data,
        chunks=chunks,
    )

    _validate_token_statistics(
        manifest=manifest,
        data=data,
        chunks=chunks,
    )

    index = _restore_memory_index(
        manifest=manifest,
        data=data,
        chunks=chunks,
    )

    result = CompetitionBM25IndexResult(
        index_directory=index_directory,
        index_path=index_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        manifest=manifest,
        index=index,
    )

    if corpus_directory is not None:
        source = _load_source_corpus(
            corpus_directory
        )

        _validate_source_binding(
            result=result,
            source=source,
        )

    return result


def _build_manifest(
    *,
    source: CompetitionCorpusBuildResult,
    tokenizer: BM25Tokenizer,
    config: BM25Config,
    tokenized_corpus_sha256: str,
    data: CompetitionBM25IndexData,
    index_bytes: bytes,
    metadata_bytes: bytes,
) -> CompetitionBM25IndexManifest:
    payload: dict[str, object] = {
        "schema_version": 1,
        "corpus_id": (
            source.manifest.corpus_id
        ),
        "corpus_identity_sha256": (
            source.manifest
            .corpus_identity_sha256
        ),
        "corpus_manifest_sha256": (
            _sha256_bytes(
                source.manifest_path
                .read_bytes()
            )
        ),
        "corpus_chunks_jsonl_sha256": (
            source.manifest
            .chunks_jsonl_sha256
        ),
        "corpus_source_count": (
            source.manifest.source_count
        ),
        "corpus_doc_count": (
            source.manifest.doc_count
        ),
        "corpus_chunk_count": (
            source.manifest.chunk_count
        ),
        "tokenizer_spec": tokenizer.spec,
        "tokenizer_spec_sha256": (
            calculate_bm25_tokenizer_spec_sha256(
                tokenizer.spec
            )
        ),
        "bm25_config": config,
        "bm25_config_sha256": (
            calculate_bm25_config_sha256(
                config
            )
        ),
        "tokenized_corpus_sha256": (
            tokenized_corpus_sha256
        ),
        "index_type": "exact_bm25",
        "index_version": (
            "competition_exact_bm25_v1"
        ),
        "document_count": (
            data.document_count
        ),
        "metadata_record_count": (
            len(source.chunks)
        ),
        "vocabulary_size": (
            data.vocabulary_size
        ),
        "total_token_count": (
            data.total_token_count
        ),
        "average_document_length": (
            data.average_document_length
        ),
        "index_json_sha256": (
            _sha256_bytes(index_bytes)
        ),
        "metadata_jsonl_sha256": (
            _sha256_bytes(metadata_bytes)
        ),
        "quality_gate_passed": True,
        "quality_gate_errors": (),
        "created_at": datetime.now(
            timezone.utc
        ),
    }

    identity = (
        calculate_competition_bm25_identity(
            payload
        )
    )

    return CompetitionBM25IndexManifest(
        **payload,
        index_identity_sha256=identity,
        index_id=(
            build_competition_bm25_index_id(
                identity
            )
        ),
    )


def build_competition_bm25_index(
    *,
    corpus_directory: Path,
    output_root: Path,
    tokenizer: BM25Tokenizer,
    config: BM25Config | None = None,
) -> CompetitionBM25IndexResult:
    source = _load_source_corpus(
        corpus_directory
    )

    active_config = (
        config
        if config is not None
        else BM25Config()
    )

    index = (
        CompetitionExactBM25Index.build(
            chunks=source.chunks,
            tokenizer=tokenizer,
            config=active_config,
        )
    )

    data = _build_index_data(index)
    index_bytes = (
        _serialize_index_data(data)
    )
    metadata_bytes = (
        serialize_competition_chunks(
            source.chunks
        )
    )

    tokenized_corpus_sha256 = (
        _calculate_tokenized_corpus_sha256(
            chunks=source.chunks,
            tokenizer=tokenizer,
        )
    )

    manifest = _build_manifest(
        source=source,
        tokenizer=tokenizer,
        config=active_config,
        tokenized_corpus_sha256=(
            tokenized_corpus_sha256
        ),
        data=data,
        index_bytes=index_bytes,
        metadata_bytes=metadata_bytes,
    )

    try:
        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )
    except OSError as exc:
        raise CompetitionBM25IndexWriteError(
            "无法创建 BM25 输出目录"
        ) from exc

    index_directory = (
        output_root
        / manifest.index_id
    )

    if index_directory.exists():
        result = (
            load_competition_bm25_index(
                index_directory
            )
        )

        _validate_source_binding(
            result=result,
            source=source,
        )

        if (
            result.manifest
            .index_identity_sha256
            != manifest
            .index_identity_sha256
        ):
            raise (
                CompetitionBM25IndexIdentityConflictError(
                    "已有索引与本次构建输入不一致"
                )
            )

        return result

    manifest_bytes = (
        _serialize_manifest(manifest)
    )

    try:
        temporary_directory = Path(
            tempfile.mkdtemp(
                prefix=(
                    f".{manifest.index_id}_"
                ),
                dir=output_root,
            )
        )
    except OSError as exc:
        raise CompetitionBM25IndexWriteError(
            "无法创建 BM25 临时目录"
        ) from exc

    try:
        (
            temporary_directory
            / INDEX_FILENAME
        ).write_bytes(index_bytes)

        (
            temporary_directory
            / METADATA_FILENAME
        ).write_bytes(
            metadata_bytes
        )

        (
            temporary_directory
            / MANIFEST_FILENAME
        ).write_bytes(
            manifest_bytes
        )

        try:
            temporary_directory.rename(
                index_directory
            )
        except OSError as exc:
            if index_directory.exists():
                shutil.rmtree(
                    temporary_directory,
                    ignore_errors=True,
                )

                result = (
                    load_competition_bm25_index(
                        index_directory
                    )
                )

                _validate_source_binding(
                    result=result,
                    source=source,
                )

                if (
                    result.manifest
                    .index_identity_sha256
                    != manifest
                    .index_identity_sha256
                ):
                    raise (
                        CompetitionBM25IndexIdentityConflictError(
                            "并发生成的索引身份不一致"
                        )
                    )

                return result

            raise CompetitionBM25IndexWriteError(
                "无法原子提交 BM25 索引目录"
            ) from exc

    except Exception:
        if temporary_directory.exists():
            shutil.rmtree(
                temporary_directory,
                ignore_errors=True,
            )
        raise

    return load_competition_bm25_index(
        index_directory,
        corpus_directory=(
            source.corpus_directory
        ),
    )