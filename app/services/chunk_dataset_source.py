from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from app.schemas.chunk import Chunk
from app.schemas.chunk_dataset import (
    ChunkDatasetManifest,
)


CHUNKS_FILENAME = "chunks.jsonl"

CHUNK_MANIFEST_FILENAME = (
    "dataset_manifest.json"
)


class ChunkDatasetSourceError(ValueError):
    """索引来源 ChunkDataset 无效。"""


@dataclass(frozen=True, slots=True)
class LoadedChunkDataset:
    """经过完整校验的来源 ChunkDataset。"""

    manifest: ChunkDatasetManifest
    chunks: tuple[Chunk, ...]
    manifest_sha256: str


def canonical_json_bytes(
    value: object,
) -> bytes:
    """生成稳定、可哈希的 JSON 字节。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def calculate_chunk_dataset_manifest_sha256(
    manifest: ChunkDatasetManifest,
) -> str:
    """计算 ChunkDataset Manifest 的语义哈希。"""

    return hashlib.sha256(
        canonical_json_bytes(
            manifest.model_dump(mode="json")
        )
    ).hexdigest()


def serialize_chunks(
    chunks: tuple[Chunk, ...],
) -> bytes:
    """按照索引顺序序列化完整 Chunk 元数据。"""

    lines = [
        json.dumps(
            chunk.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for chunk in chunks
    ]

    return (
        "\n".join(lines) + "\n"
    ).encode("utf-8")


def load_chunk_dataset_source(
    directory: Path,
) -> LoadedChunkDataset:
    """读取并完整校验 ChunkDataset。"""

    chunks_path = (
        directory / CHUNKS_FILENAME
    )

    manifest_path = (
        directory
        / CHUNK_MANIFEST_FILENAME
    )

    if (
        not chunks_path.is_file()
        or not manifest_path.is_file()
    ):
        raise ChunkDatasetSourceError(
            "Chunk 数据集目录缺少 "
            "chunks.jsonl 或 dataset_manifest.json"
        )

    try:
        chunks_bytes = chunks_path.read_bytes()

        manifest = (
            ChunkDatasetManifest
            .model_validate_json(
                manifest_path.read_bytes()
            )
        )

    except (
        OSError,
        ValidationError,
    ) as exc:
        raise ChunkDatasetSourceError(
            "无法读取有效的 Chunk 数据集"
        ) from exc

    actual_chunks_sha256 = hashlib.sha256(
        chunks_bytes
    ).hexdigest()

    if (
        actual_chunks_sha256
        != manifest.chunks_jsonl_sha256
    ):
        raise ChunkDatasetSourceError(
            "来源 chunks.jsonl 哈希校验失败"
        )

    try:
        text = chunks_bytes.decode("utf-8")

    except UnicodeDecodeError as exc:
        raise ChunkDatasetSourceError(
            "来源 chunks.jsonl 不是合法 UTF-8"
        ) from exc

    lines = text.splitlines()

    if (
        not lines
        or any(
            not line.strip()
            for line in lines
        )
    ):
        raise ChunkDatasetSourceError(
            "来源 chunks.jsonl "
            "为空或包含空记录"
        )

    try:
        chunks = tuple(
            Chunk.model_validate_json(line)
            for line in lines
        )

    except ValidationError as exc:
        raise ChunkDatasetSourceError(
            "来源 chunks.jsonl "
            "包含无效 Chunk"
        ) from exc

    if (
        len(chunks)
        != manifest.chunk_record_count
    ):
        raise ChunkDatasetSourceError(
            "Chunk 数量与 Manifest 不一致"
        )

    actual_char_count = sum(
        chunk.char_count
        for chunk in chunks
    )

    if (
        actual_char_count
        != manifest.chunk_char_count_total
    ):
        raise ChunkDatasetSourceError(
            "Chunk 字符总数与 Manifest 不一致"
        )

    chunk_ids = tuple(
        chunk.chunk_id
        for chunk in chunks
    )

    if (
        len(chunk_ids)
        != len(set(chunk_ids))
    ):
        raise ChunkDatasetSourceError(
            "来源数据包含重复 chunk_id"
        )

    for chunk in chunks:
        if (
            chunk.chunk_dataset_id
            != manifest.dataset_id
        ):
            raise ChunkDatasetSourceError(
                "Chunk 的 chunk_dataset_id "
                "与 Manifest 不一致"
            )

        if (
            chunk.report_id
            != manifest.report_id
        ):
            raise ChunkDatasetSourceError(
                "Chunk 的 report_id "
                "与 Manifest 不一致"
            )

        if (
            chunk.company_id
            != manifest.company_id
        ):
            raise ChunkDatasetSourceError(
                "Chunk 的 company_id "
                "与 Manifest 不一致"
            )

        if (
            chunk.fiscal_year
            != manifest.fiscal_year
        ):
            raise ChunkDatasetSourceError(
                "Chunk 的 fiscal_year "
                "与 Manifest 不一致"
            )

        if (
            chunk.report_type
            != manifest.report_type
        ):
            raise ChunkDatasetSourceError(
                "Chunk 的 report_type "
                "与 Manifest 不一致"
            )

        if (
            chunk.document_id
            != manifest.document_id
        ):
            raise ChunkDatasetSourceError(
                "Chunk 的 document_id "
                "与 Manifest 不一致"
            )

        if (
            chunk.strategy
            != manifest.strategy
        ):
            raise ChunkDatasetSourceError(
                "Chunk 的 strategy "
                "与 Manifest 不一致"
            )

    return LoadedChunkDataset(
        manifest=manifest,
        chunks=chunks,
        manifest_sha256=(
            calculate_chunk_dataset_manifest_sha256(
                manifest
            )
        ),
    )