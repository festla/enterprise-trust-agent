from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from types import MappingProxyType

from pydantic import ValidationError

from app.rag.bm25 import ExactBM25Index
from app.rag.tokenization import (
    BM25Tokenizer,
    DeterministicChineseBigramTokenizer,
)
from app.schemas.bm25 import (
    BM25Config,
    calculate_bm25_config_sha256,
    calculate_bm25_tokenizer_spec_sha256,
)
from app.schemas.bm25_index import (
    BM25IndexData,
    BM25IndexDocumentRecord,
    BM25IndexManifest,
)
from app.schemas.chunk import Chunk
from app.services.chunk_dataset_source import (
    ChunkDatasetSourceError,
    LoadedChunkDataset,
    canonical_json_bytes,
    load_chunk_dataset_source,
    serialize_chunks,
)


INDEX_FILENAME = "index.json"

METADATA_FILENAME = "metadata.jsonl"

MANIFEST_FILENAME = "index_manifest.json"


class BM25IndexServiceError(ValueError):
    """BM25 持久化索引服务基础异常。"""


class InvalidBM25SourceError(
    BM25IndexServiceError
):
    """来源 ChunkDataset 无效。"""


class CorruptBM25IndexError(
    BM25IndexServiceError
):
    """持久化 BM25 索引损坏或不一致。"""


class BM25IndexIdentityConflictError(
    BM25IndexServiceError
):
    """已有目录与本次构建身份不一致。"""


class BM25IndexWriteError(
    BM25IndexServiceError
):
    """BM25 索引无法安全写入磁盘。"""


@dataclass(frozen=True, slots=True)
class BM25IndexResult:
    """构建或加载后的 BM25 索引结果。"""

    index_directory: Path
    manifest: BM25IndexManifest
    index: ExactBM25Index


def _sha256_bytes(
    value: bytes,
) -> str:
    return hashlib.sha256(value).hexdigest()


def _calculate_tokenized_corpus_sha256(
    *,
    chunks: tuple[Chunk, ...],
    tokenizer: BM25Tokenizer,
) -> str:
    """计算实际 Token 序列的稳定语义哈希。"""

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
        canonical_json_bytes(records)
    )


def _build_index_data(
    index: ExactBM25Index,
) -> BM25IndexData:
    """将内存 BM25 结构转换为可验证数据。"""

    document_records = tuple(
        BM25IndexDocumentRecord(
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

    return BM25IndexData(
        document_records=document_records,
        document_frequencies=dict(
            index.document_frequencies
        ),
        document_count=len(index.chunks),
        vocabulary_size=len(
            index.document_frequencies
        ),
        total_token_count=total_token_count,
        average_document_length=(
            index.average_document_length
        ),
    )


def _serialize_index_data(
    data: BM25IndexData,
) -> bytes:
    """稳定序列化 index.json。"""

    return canonical_json_bytes(
        data.model_dump(mode="json")
    )


def _serialize_manifest(
    manifest: BM25IndexManifest,
) -> bytes:
    """使用可读格式保存 Manifest。"""

    return (
        manifest.model_dump_json(indent=2)
        + "\n"
    ).encode("utf-8")


def _build_index_id(
    *,
    source: LoadedChunkDataset,
    tokenizer_spec_sha256: str,
    bm25_config_sha256: str,
    tokenized_corpus_sha256: str,
) -> str:
    """根据全部确定性输入生成索引身份。"""

    identity_payload = {
        "index_version": "exact_bm25_v1",
        "chunk_dataset_id": (
            source.manifest.dataset_id
        ),
        "chunk_dataset_manifest_sha256": (
            source.manifest_sha256
        ),
        "source_chunks_jsonl_sha256": (
            source.manifest.chunks_jsonl_sha256
        ),
        "tokenizer_spec_sha256": (
            tokenizer_spec_sha256
        ),
        "bm25_config_sha256": (
            bm25_config_sha256
        ),
        "tokenized_corpus_sha256": (
            tokenized_corpus_sha256
        ),
    }

    suffix = _sha256_bytes(
        canonical_json_bytes(
            identity_payload
        )
    )[:24]

    return (
        f"bm25_index_"
        f"{source.manifest.report_id}_"
        f"{suffix}"
    )


def _load_metadata(
    metadata_bytes: bytes,
) -> tuple[Chunk, ...]:
    """读取并校验完整 Chunk 元数据。"""

    try:
        text = metadata_bytes.decode(
            "utf-8"
        )
    except UnicodeDecodeError as exc:
        raise CorruptBM25IndexError(
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
        raise CorruptBM25IndexError(
            "metadata.jsonl "
            "为空或包含空记录"
        )

    try:
        return tuple(
            Chunk.model_validate_json(line)
            for line in lines
        )
    except ValidationError as exc:
        raise CorruptBM25IndexError(
            "metadata.jsonl "
            "包含无效 Chunk"
        ) from exc


def _validate_loaded_data(
    *,
    manifest: BM25IndexManifest,
    data: BM25IndexData,
    chunks: tuple[Chunk, ...],
) -> None:
    """交叉验证 Manifest、IndexData 和 Chunk。"""

    if not manifest.quality_gate_passed:
        raise CorruptBM25IndexError(
            "BM25 Index 未通过质量门禁"
        )

    if (
        len(chunks)
        != manifest.metadata_record_count
    ):
        raise CorruptBM25IndexError(
            "元数据记录数量与 Manifest 不一致"
        )

    if (
        data.document_count
        != manifest.document_count
    ):
        raise CorruptBM25IndexError(
            "index.json 文档数量与 "
            "Manifest 不一致"
        )

    if (
        data.vocabulary_size
        != manifest.vocabulary_size
    ):
        raise CorruptBM25IndexError(
            "词表大小与 Manifest 不一致"
        )

    if (
        data.total_token_count
        != manifest.total_token_count
    ):
        raise CorruptBM25IndexError(
            "Token 总数与 Manifest 不一致"
        )

    if (
        abs(
            data.average_document_length
            - manifest.average_document_length
        )
        > 1e-12
    ):
        raise CorruptBM25IndexError(
            "平均文档长度与 Manifest 不一致"
        )

    chunk_ids = tuple(
        chunk.chunk_id
        for chunk in chunks
    )

    record_chunk_ids = tuple(
        record.chunk_id
        for record in data.document_records
    )

    if chunk_ids != record_chunk_ids:
        raise CorruptBM25IndexError(
            "index.json 与 metadata.jsonl "
            "的 Chunk 顺序不一致"
        )

    for chunk in chunks:
        if (
            chunk.chunk_dataset_id
            != manifest.chunk_dataset_id
        ):
            raise CorruptBM25IndexError(
                "ChunkDataset 身份不一致"
            )

        if (
            chunk.report_id
            != manifest.report_id
        ):
            raise CorruptBM25IndexError(
                "Chunk 的 report_id 不一致"
            )

        if (
            chunk.company_id
            != manifest.company_id
        ):
            raise CorruptBM25IndexError(
                "Chunk 的 company_id 不一致"
            )

        if (
            chunk.fiscal_year
            != manifest.fiscal_year
        ):
            raise CorruptBM25IndexError(
                "Chunk 的 fiscal_year 不一致"
            )

        if (
            chunk.report_type
            != manifest.report_type
        ):
            raise CorruptBM25IndexError(
                "Chunk 的 report_type 不一致"
            )

        if (
            chunk.document_id
            != manifest.document_id
        ):
            raise CorruptBM25IndexError(
                "Chunk 的 document_id 不一致"
            )

        if (
            chunk.strategy
            != manifest.chunk_strategy
        ):
            raise CorruptBM25IndexError(
                "Chunk Strategy 与 Manifest 不一致"
            )


def _validate_token_statistics(
    *,
    manifest: BM25IndexManifest,
    data: BM25IndexData,
    chunks: tuple[Chunk, ...],
) -> None:
    """验证持久化 TF 与实际 Chunk 文本一致。"""

    tokenizer = (
        DeterministicChineseBigramTokenizer(
            spec=manifest.tokenizer_spec
        )
    )

    actual_corpus_sha256 = (
        _calculate_tokenized_corpus_sha256(
            chunks=chunks,
            tokenizer=tokenizer,
        )
    )

    if (
        actual_corpus_sha256
        != manifest.tokenized_corpus_sha256
    ):
        raise CorruptBM25IndexError(
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

        actual_term_frequency = dict(
            sorted(
                Counter(tokens).items()
            )
        )

        if (
            actual_term_frequency
            != record.term_frequencies
        ):
            raise CorruptBM25IndexError(
                "持久化词频与 Chunk 文本不一致："
                f"{chunk.chunk_id}"
            )

        if (
            len(tokens)
            != record.document_length
        ):
            raise CorruptBM25IndexError(
                "持久化文档长度与 Chunk "
                "分词结果不一致："
                f"{chunk.chunk_id}"
            )


def _restore_memory_index(
    *,
    manifest: BM25IndexManifest,
    data: BM25IndexData,
    chunks: tuple[Chunk, ...],
) -> ExactBM25Index:
    """由持久化统计恢复内存 BM25 Index。"""

    return ExactBM25Index(
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


def load_bm25_index(
    index_directory: Path,
) -> BM25IndexResult:
    """加载并完整验证持久化 BM25 Index。"""

    index_path = (
        index_directory / INDEX_FILENAME
    )

    metadata_path = (
        index_directory / METADATA_FILENAME
    )

    manifest_path = (
        index_directory / MANIFEST_FILENAME
    )

    if not all(
        path.is_file()
        for path in (
            index_path,
            metadata_path,
            manifest_path,
        )
    ):
        raise CorruptBM25IndexError(
            "BM25 索引目录缺少必要文件"
        )

    try:
        index_bytes = index_path.read_bytes()
        metadata_bytes = (
            metadata_path.read_bytes()
        )
        manifest_bytes = (
            manifest_path.read_bytes()
        )
    except OSError as exc:
        raise CorruptBM25IndexError(
            "无法读取 BM25 索引文件"
        ) from exc

    try:
        manifest = (
            BM25IndexManifest
            .model_validate_json(
                manifest_bytes
            )
        )

        data = (
            BM25IndexData
            .model_validate_json(
                index_bytes
            )
        )
    except ValidationError as exc:
        raise CorruptBM25IndexError(
            "BM25 索引 Schema 校验失败"
        ) from exc

    if (
        _sha256_bytes(index_bytes)
        != manifest.index_json_sha256
    ):
        raise CorruptBM25IndexError(
            "index.json 哈希校验失败"
        )

    if (
        _sha256_bytes(metadata_bytes)
        != manifest.metadata_jsonl_sha256
    ):
        raise CorruptBM25IndexError(
            "metadata.jsonl 哈希校验失败"
        )

    chunks = _load_metadata(
        metadata_bytes
    )

    _validate_loaded_data(
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

    return BM25IndexResult(
        index_directory=index_directory,
        manifest=manifest,
        index=index,
    )


def _validate_existing_identity(
    *,
    result: BM25IndexResult,
    source: LoadedChunkDataset,
    tokenizer: BM25Tokenizer,
    config: BM25Config,
    tokenized_corpus_sha256: str,
) -> None:
    """检查已有同名索引是否属于本次输入。"""

    manifest = result.manifest

    expected_values = {
        "chunk_dataset_id": (
            source.manifest.dataset_id
        ),
        "chunk_dataset_manifest_sha256": (
            source.manifest_sha256
        ),
        "source_chunks_jsonl_sha256": (
            source.manifest.chunks_jsonl_sha256
        ),
        "tokenizer_spec_sha256": (
            calculate_bm25_tokenizer_spec_sha256(
                tokenizer.spec
            )
        ),
        "bm25_config_sha256": (
            calculate_bm25_config_sha256(
                config
            )
        ),
        "tokenized_corpus_sha256": (
            tokenized_corpus_sha256
        ),
    }

    actual_values = {
        key: getattr(manifest, key)
        for key in expected_values
    }

    if actual_values != expected_values:
        raise BM25IndexIdentityConflictError(
            "已有 BM25 索引与本次构建输入不一致"
        )


def build_bm25_index(
    *,
    chunk_dataset_directory: Path,
    output_root: Path,
    tokenizer: BM25Tokenizer,
    config: BM25Config | None = None,
) -> BM25IndexResult:
    """构建并持久化确定性 BM25 Index。"""

    try:
        source = load_chunk_dataset_source(
            chunk_dataset_directory
        )
    except ChunkDatasetSourceError as exc:
        raise InvalidBM25SourceError(
            str(exc)
        ) from exc

    active_config = (
        config
        if config is not None
        else BM25Config()
    )

    index = ExactBM25Index.build(
        chunks=source.chunks,
        tokenizer=tokenizer,
        config=active_config,
    )

    tokenizer_spec_sha256 = (
        calculate_bm25_tokenizer_spec_sha256(
            tokenizer.spec
        )
    )

    bm25_config_sha256 = (
        calculate_bm25_config_sha256(
            active_config
        )
    )

    tokenized_corpus_sha256 = (
        _calculate_tokenized_corpus_sha256(
            chunks=source.chunks,
            tokenizer=tokenizer,
        )
    )

    index_id = _build_index_id(
        source=source,
        tokenizer_spec_sha256=(
            tokenizer_spec_sha256
        ),
        bm25_config_sha256=(
            bm25_config_sha256
        ),
        tokenized_corpus_sha256=(
            tokenized_corpus_sha256
        ),
    )

    index_directory = (
        output_root / index_id
    )

    if index_directory.exists():
        result = load_bm25_index(
            index_directory
        )

        _validate_existing_identity(
            result=result,
            source=source,
            tokenizer=tokenizer,
            config=active_config,
            tokenized_corpus_sha256=(
                tokenized_corpus_sha256
            ),
        )

        return result

    index_data = _build_index_data(
        index
    )

    index_bytes = _serialize_index_data(
        index_data
    )

    metadata_bytes = serialize_chunks(
        source.chunks
    )

    manifest = BM25IndexManifest(
        index_id=index_id,
        chunk_dataset_id=(
            source.manifest.dataset_id
        ),
        report_id=(
            source.manifest.report_id
        ),
        company_id=(
            source.manifest.company_id
        ),
        fiscal_year=(
            source.manifest.fiscal_year
        ),
        report_type=(
            source.manifest.report_type
        ),
        document_id=(
            source.manifest.document_id
        ),
        chunk_strategy=(
            source.manifest.strategy
        ),
        chunk_dataset_manifest_sha256=(
            source.manifest_sha256
        ),
        source_chunks_jsonl_sha256=(
            source.manifest.chunks_jsonl_sha256
        ),
        tokenizer_spec=tokenizer.spec,
        tokenizer_spec_sha256=(
            tokenizer_spec_sha256
        ),
        bm25_config=active_config,
        bm25_config_sha256=(
            bm25_config_sha256
        ),
        tokenized_corpus_sha256=(
            tokenized_corpus_sha256
        ),
        document_count=(
            index_data.document_count
        ),
        metadata_record_count=len(
            source.chunks
        ),
        vocabulary_size=(
            index_data.vocabulary_size
        ),
        total_token_count=(
            index_data.total_token_count
        ),
        average_document_length=(
            index_data.average_document_length
        ),
        index_json_sha256=(
            _sha256_bytes(index_bytes)
        ),
        metadata_jsonl_sha256=(
            _sha256_bytes(metadata_bytes)
        ),
        quality_gate_passed=True,
        quality_gate_errors=(),
        created_at=datetime.now(
            timezone.utc
        ),
    )

    manifest_bytes = _serialize_manifest(
        manifest
    )

    try:
        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_directory = Path(
            tempfile.mkdtemp(
                prefix=(
                    f".{index_id}_"
                ),
                dir=output_root,
            )
        )

    except OSError as exc:
        raise BM25IndexWriteError(
            "无法创建 BM25 索引输出目录"
        ) from exc

    try:
        (
            temporary_directory
            / INDEX_FILENAME
        ).write_bytes(index_bytes)

        (
            temporary_directory
            / METADATA_FILENAME
        ).write_bytes(metadata_bytes)

        (
            temporary_directory
            / MANIFEST_FILENAME
        ).write_bytes(manifest_bytes)

        try:
            temporary_directory.rename(
                index_directory
            )
        except OSError as exc:
            # 可能有另一个并发构建刚刚提交。
            if index_directory.exists():
                shutil.rmtree(
                    temporary_directory,
                    ignore_errors=True,
                )

                result = load_bm25_index(
                    index_directory
                )

                _validate_existing_identity(
                    result=result,
                    source=source,
                    tokenizer=tokenizer,
                    config=active_config,
                    tokenized_corpus_sha256=(
                        tokenized_corpus_sha256
                    ),
                )

                return result

            raise BM25IndexWriteError(
                "无法原子提交 BM25 索引目录"
            ) from exc

    except Exception:
        if temporary_directory.exists():
            shutil.rmtree(
                temporary_directory,
                ignore_errors=True,
            )

        raise

    return BM25IndexResult(
        index_directory=index_directory,
        manifest=manifest,
        index=index,
    )