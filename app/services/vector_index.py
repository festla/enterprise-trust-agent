from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from app.rag.embedding import (
    EmbeddingProvider,
)
from app.rag.vector_index import (
    ExactVectorIndex,
)
from app.schemas.chunk import Chunk
from app.schemas.chunk_dataset import (
    ChunkDatasetManifest,
)
from app.schemas.embedding import (
    EmbeddingSpec,
    calculate_embedding_spec_sha256,
)
from app.schemas.vector_index import (
    VectorIndexManifest,
)
from app.services.chunk_dataset_source import (
    ChunkDatasetSourceError,
    LoadedChunkDataset,
    canonical_json_bytes,
    load_chunk_dataset_source,
    serialize_chunks,
)


VECTORS_FILENAME = "vectors.npy"
METADATA_FILENAME = "metadata.jsonl"
INDEX_MANIFEST_FILENAME = (
    "index_manifest.json"
)

INDEX_TYPE = "exact_cosine"
INDEX_VERSION = "exact_cosine_v1"


class VectorIndexDatasetError(ValueError):
    """持久化向量索引基础异常。"""


class InvalidSourceChunkDatasetError(
    VectorIndexDatasetError
):
    """来源 Chunk 数据集无效。"""


class InvalidVectorIndexDatasetError(
    VectorIndexDatasetError
):
    """新生成的向量索引无效。"""


class ExistingVectorIndexDatasetError(
    VectorIndexDatasetError
):
    """已有向量索引损坏或与输入冲突。"""


class VectorIndexDatasetWriteError(
    VectorIndexDatasetError
):
    """向量索引无法安全提交到磁盘。"""


@dataclass(frozen=True, slots=True)
class VectorIndexBuildResult:
    """一次向量索引构建结果。"""

    manifest: VectorIndexManifest
    index: ExactVectorIndex

    index_directory: Path
    vectors_path: Path
    metadata_path: Path
    manifest_path: Path

    created: bool


@dataclass(frozen=True, slots=True)
class LoadedChunkDataset:
    """经过验证的来源 Chunk 数据集。"""

    manifest: ChunkDatasetManifest
    chunks: tuple[Chunk, ...]
    manifest_sha256: str


def _serialize_vectors(
    vectors: np.ndarray,
) -> bytes:
    """将向量矩阵安全序列化为 NPY 字节。"""

    buffer = io.BytesIO()

    np.save(
        buffer,
        vectors,
        allow_pickle=False,
    )

    return buffer.getvalue()


def _calculate_embedding_input_sha256(
    *,
    chunks: tuple[Chunk, ...],
    spec: EmbeddingSpec,
) -> str:
    """对实际送入文档 Embedding 的输入计算哈希。"""

    payload = [
        {
            "chunk_id": chunk.chunk_id,
            "input": (
                spec.document_prefix
                + chunk.text
            ),
        }
        for chunk in chunks
    ]

    return hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()


def build_vector_index_id(
    *,
    source: LoadedChunkDataset,
    spec: EmbeddingSpec,
) -> str:
    """根据来源、模型和索引算法生成稳定 ID。"""

    embedding_input_sha256 = (
        _calculate_embedding_input_sha256(
            chunks=source.chunks,
            spec=spec,
        )
    )

    payload = {
        "chunk_dataset_id": (
            source.manifest.dataset_id
        ),
        "chunk_dataset_manifest_sha256": (
            source.manifest_sha256
        ),
        "source_chunks_jsonl_sha256": (
            source.manifest
            .chunks_jsonl_sha256
        ),
        "embedding_spec": (
            spec.model_dump(mode="json")
        ),
        "embedding_input_sha256": (
            embedding_input_sha256
        ),
        "index_type": INDEX_TYPE,
        "index_version": INDEX_VERSION,
        "numpy_version": np.__version__,
    }

    identity_sha256 = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()

    return (
        f"vector_index_"
        f"{source.manifest.report_id}_"
        f"{identity_sha256[:24]}"
    )


def _build_index_manifest(
    *,
    index_id: str,
    source: LoadedChunkDataset,
    spec: EmbeddingSpec,
    vectors_bytes: bytes,
    metadata_bytes: bytes,
    created_at: datetime,
) -> VectorIndexManifest:
    """构造持久化索引 Manifest。"""

    return VectorIndexManifest(
        index_id=index_id,
        chunk_dataset_id=(
            source.manifest.dataset_id
        ),
        report_id=source.manifest.report_id,
        company_id=source.manifest.company_id,
        fiscal_year=source.manifest.fiscal_year,
        report_type=source.manifest.report_type,
        document_id=source.manifest.document_id,
        chunk_strategy=source.manifest.strategy,
        chunk_dataset_manifest_sha256=(
            source.manifest_sha256
        ),
        source_chunks_jsonl_sha256=(
            source.manifest
            .chunks_jsonl_sha256
        ),
        embedding_spec=spec,
        embedding_spec_sha256=(
            calculate_embedding_spec_sha256(
                spec
            )
        ),
        embedding_input_sha256=(
            _calculate_embedding_input_sha256(
                chunks=source.chunks,
                spec=spec,
            )
        ),
        numpy_version=np.__version__,
        vector_count=len(source.chunks),
        vector_dimension=spec.dimension,
        metadata_record_count=len(
            source.chunks
        ),
        vectors_sha256=hashlib.sha256(
            vectors_bytes
        ).hexdigest(),
        metadata_jsonl_sha256=(
            hashlib.sha256(
                metadata_bytes
            ).hexdigest()
        ),
        quality_gate_passed=True,
        quality_gate_errors=(),
        created_at=created_at,
    )


def _load_vectors(
    *,
    vectors_bytes: bytes,
    expected_rows: int,
    expected_dimension: int,
) -> np.ndarray:
    """读取并验证持久化向量矩阵。"""

    try:
        with io.BytesIO(
            vectors_bytes
        ) as buffer:
            vectors = np.load(
                buffer,
                allow_pickle=False,
            )
    except (
        OSError,
        ValueError,
    ) as exc:
        raise ExistingVectorIndexDatasetError(
            "vectors.npy 无法安全读取"
        ) from exc

    if vectors.dtype != np.dtype("float32"):
        raise ExistingVectorIndexDatasetError(
            "vectors.npy 必须使用 float32"
        )

    if vectors.shape != (
        expected_rows,
        expected_dimension,
    ):
        raise ExistingVectorIndexDatasetError(
            "向量矩阵形状与 Manifest 不一致"
        )

    if not np.isfinite(vectors).all():
        raise ExistingVectorIndexDatasetError(
            "持久化向量包含 NaN 或 Infinity"
        )

    norms = np.linalg.norm(
        vectors,
        axis=1,
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ExistingVectorIndexDatasetError(
            "持久化向量未完成 L2 归一化"
        )

    result = np.ascontiguousarray(
        vectors,
        dtype=np.float32,
    )

    result.setflags(write=False)

    return result


def _read_existing_index(
    *,
    index_directory: Path,
    source: LoadedChunkDataset,
    provider: EmbeddingProvider,
    expected_index_id: str,
) -> VectorIndexBuildResult:
    """读取并完整校验已有索引。"""

    vectors_path = (
        index_directory / VECTORS_FILENAME
    )

    metadata_path = (
        index_directory / METADATA_FILENAME
    )

    manifest_path = (
        index_directory
        / INDEX_MANIFEST_FILENAME
    )

    if (
        not vectors_path.is_file()
        or not metadata_path.is_file()
        or not manifest_path.is_file()
    ):
        raise ExistingVectorIndexDatasetError(
            "已有索引目录缺少必要文件"
        )

    try:
        manifest = (
            VectorIndexManifest
            .model_validate_json(
                manifest_path.read_bytes()
            )
        )

        vectors_bytes = (
            vectors_path.read_bytes()
        )

        metadata_bytes = (
            metadata_path.read_bytes()
        )
    except (
        OSError,
        ValidationError,
    ) as exc:
        raise ExistingVectorIndexDatasetError(
            "已有索引文件无法读取或校验"
        ) from exc

    if manifest.index_id != expected_index_id:
        raise ExistingVectorIndexDatasetError(
            "已有 index_id 与当前输入不一致"
        )

    if (
        manifest.chunk_dataset_id
        != source.manifest.dataset_id
    ):
        raise ExistingVectorIndexDatasetError(
            "已有索引来源 ChunkDataset 不一致"
        )

    if manifest.embedding_spec != provider.spec:
        raise ExistingVectorIndexDatasetError(
            "已有索引 EmbeddingSpec 不一致"
        )

    if (
        hashlib.sha256(
            vectors_bytes
        ).hexdigest()
        != manifest.vectors_sha256
    ):
        raise ExistingVectorIndexDatasetError(
            "vectors.npy 哈希校验失败"
        )

    if (
        hashlib.sha256(
            metadata_bytes
        ).hexdigest()
        != manifest.metadata_jsonl_sha256
    ):
        raise ExistingVectorIndexDatasetError(
            "metadata.jsonl 哈希校验失败"
        )

    try:
        metadata_text = metadata_bytes.decode(
            "utf-8"
        )

        metadata_chunks = tuple(
            Chunk.model_validate_json(line)
            for line
            in metadata_text.splitlines()
            if line.strip()
        )
    except (
        UnicodeDecodeError,
        ValidationError,
    ) as exc:
        raise ExistingVectorIndexDatasetError(
            "metadata.jsonl 包含无效记录"
        ) from exc

    if metadata_chunks != source.chunks:
        raise ExistingVectorIndexDatasetError(
            "索引元数据的内容或顺序 "
            "与来源 ChunkDataset 不一致"
        )

    vectors = _load_vectors(
        vectors_bytes=vectors_bytes,
        expected_rows=len(metadata_chunks),
        expected_dimension=(
            provider.spec.dimension
        ),
    )

    rebuilt_manifest = _build_index_manifest(
        index_id=expected_index_id,
        source=source,
        spec=provider.spec,
        vectors_bytes=vectors_bytes,
        metadata_bytes=metadata_bytes,
        created_at=manifest.created_at,
    )

    if (
        rebuilt_manifest.model_dump(
            mode="json"
        )
        != manifest.model_dump(
            mode="json"
        )
    ):
        raise ExistingVectorIndexDatasetError(
            "索引 Manifest 与实际文件不一致"
        )

    index = ExactVectorIndex(
        chunks=metadata_chunks,
        vectors=vectors,
        embedding_spec=provider.spec,
    )

    return VectorIndexBuildResult(
        manifest=manifest,
        index=index,
        index_directory=index_directory,
        vectors_path=vectors_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        created=False,
    )


def _write_bytes_synced(
    *,
    path: Path,
    content: bytes,
) -> None:
    with path.open("xb") as file:
        file.write(content)
        file.flush()
        os.fsync(file.fileno())


def build_vector_index(
    *,
    chunk_dataset_directory: Path,
    output_root: Path,
    provider: EmbeddingProvider,
    created_at: datetime | None = None,
) -> VectorIndexBuildResult:
    """构建不可变、可复现的精确向量索引。"""

    try:
        source = load_chunk_dataset_source(
            chunk_dataset_directory
        )

    except ChunkDatasetSourceError as exc:
        raise InvalidSourceChunkDatasetError(
            str(exc)
        ) from exc

    index_id = build_vector_index_id(
        source=source,
        spec=provider.spec,
    )

    index_directory = (
        output_root
        / source.manifest.report_id
        / source.manifest.dataset_id
        / index_id
    )

    if index_directory.exists():
        return _read_existing_index(
            index_directory=index_directory,
            source=source,
            provider=provider,
            expected_index_id=index_id,
        )

    index = ExactVectorIndex.build(
        chunks=source.chunks,
        provider=provider,
    )

    vectors_bytes = _serialize_vectors(
        index.vectors
    )

    metadata_bytes = serialize_chunks(
        source.chunks
    )

    created_at_value = (
        created_at
        if created_at is not None
        else datetime.now(timezone.utc)
    )

    manifest = _build_index_manifest(
        index_id=index_id,
        source=source,
        spec=provider.spec,
        vectors_bytes=vectors_bytes,
        metadata_bytes=metadata_bytes,
        created_at=created_at_value,
    )

    parent_directory = index_directory.parent

    parent_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{index_id}_",
            dir=parent_directory,
        )
    )

    try:
        _write_bytes_synced(
            path=(
                temporary_directory
                / VECTORS_FILENAME
            ),
            content=vectors_bytes,
        )

        _write_bytes_synced(
            path=(
                temporary_directory
                / METADATA_FILENAME
            ),
            content=metadata_bytes,
        )

        manifest_bytes = (
            manifest.model_dump_json(
                indent=2
            )
            + "\n"
        ).encode("utf-8")

        _write_bytes_synced(
            path=(
                temporary_directory
                / INDEX_MANIFEST_FILENAME
            ),
            content=manifest_bytes,
        )

        try:
            temporary_directory.rename(
                index_directory
            )

        except FileExistsError:
            shutil.rmtree(
                temporary_directory,
                ignore_errors=True,
            )

            return _read_existing_index(
                index_directory=index_directory,
                source=source,
                provider=provider,
                expected_index_id=index_id,
            )

        except OSError as exc:
            raise VectorIndexDatasetWriteError(
                "无法提交向量索引目录"
            ) from exc

    except Exception:
        if temporary_directory.exists():
            shutil.rmtree(
                temporary_directory,
                ignore_errors=True,
            )

        raise

    return VectorIndexBuildResult(
        manifest=manifest,
        index=index,
        index_directory=index_directory,
        vectors_path=(
            index_directory
            / VECTORS_FILENAME
        ),
        metadata_path=(
            index_directory
            / METADATA_FILENAME
        ),
        manifest_path=(
            index_directory
            / INDEX_MANIFEST_FILENAME
        ),
        created=True,
    )