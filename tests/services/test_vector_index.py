from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from app.schemas.chunk import (
    Chunk,
    FixedLengthChunkingConfig,
)
from app.schemas.chunk_dataset import (
    ChunkDatasetManifest,
    calculate_chunking_config_sha256,
)
from app.schemas.embedding import EmbeddingSpec
from app.schemas.enums import (
    ChunkStrategy,
    PageContentType,
    PageMappingStatus,
    PageParseStatus,
    ReportType,
)
from app.services.vector_index import (
    ExistingVectorIndexDatasetError,
    InvalidSourceChunkDatasetError,
    build_vector_index,
)


REPORT_ID = "midea_group_2024"

DOCUMENT_ID = (
    f"doc_{REPORT_ID}_{'a' * 24}"
)

PAGE_DATASET_ID = (
    f"page_dataset_{REPORT_ID}_"
    f"{'b' * 24}"
)

CHUNK_DATASET_ID = (
    f"chunk_dataset_{REPORT_ID}_"
    f"{'c' * 24}"
)

NOW = datetime(
    2026,
    7,
    27,
    tzinfo=timezone.utc,
)


def build_chunk(
    *,
    suffix: str,
    text: str,
    pdf_page: int,
) -> Chunk:
    return Chunk(
        chunk_id=(
            f"chunk_{REPORT_ID}_"
            f"{suffix * 24}"
        ),
        chunk_dataset_id=CHUNK_DATASET_ID,
        page_dataset_id=PAGE_DATASET_ID,
        company_id="midea_group",
        report_id=REPORT_ID,
        fiscal_year=2024,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        document_id=DOCUMENT_ID,
        page_id=(
            f"{DOCUMENT_ID}_page_"
            f"{pdf_page:04d}"
        ),
        pdf_page=pdf_page,
        printed_page=pdf_page,
        mapping_status=(
            PageMappingStatus.MAPPED
        ),
        content_type=PageContentType.TEXT,
        parse_status=PageParseStatus.SUCCESS,
        chunk_index=0,
        strategy=ChunkStrategy.FIXED_LENGTH,
        chunker_version=(
            "fixed_length_chunker_v1"
        ),
        source_text_field=(
            "normalized_text"
        ),
        source_start_char=0,
        source_end_char=len(text),
        text=text,
        char_count=len(text),
        text_sha256=hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest(),
    )


class FakeEmbeddingProvider:
    def __init__(
        self,
        *,
        model_version: str = "fake_v1",
    ) -> None:
        self._spec = EmbeddingSpec(
            provider="test",
            model_name="fake_embedding",
            model_version=model_version,
            dimension=3,
        )

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.float32]:
        mapping = {
            "营业收入": (1, 0, 0),
            "净利润": (0, 1, 0),
        }

        return np.asarray(
            [mapping[text] for text in texts],
            dtype=np.float32,
        )

    def embed_query(
        self,
        text: str,
    ) -> NDArray[np.float32]:
        mapping = {
            "收入是多少": (1, 0, 0),
            "利润是多少": (0, 1, 0),
        }

        return np.asarray(
            mapping[text],
            dtype=np.float32,
        )


def write_chunk_dataset(
    root: Path,
) -> Path:
    chunks = (
        build_chunk(
            suffix="1",
            text="营业收入",
            pdf_page=1,
        ),
        build_chunk(
            suffix="2",
            text="净利润",
            pdf_page=2,
        ),
    )

    chunks_bytes = (
        "\n".join(
            json.dumps(
                chunk.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for chunk in chunks
        )
        + "\n"
    ).encode("utf-8")

    config = FixedLengthChunkingConfig(
        max_chars=800,
        overlap_chars=120,
    )

    manifest = ChunkDatasetManifest(
        dataset_id=CHUNK_DATASET_ID,
        page_dataset_id=PAGE_DATASET_ID,
        report_id=REPORT_ID,
        company_id="midea_group",
        fiscal_year=2024,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        document_id=DOCUMENT_ID,
        page_dataset_manifest_sha256=(
            "d" * 64
        ),
        source_pages_jsonl_sha256=(
            "e" * 64
        ),
        report_snapshot_sha256=(
            "f" * 64
        ),
        strategy=ChunkStrategy.FIXED_LENGTH,
        chunker_name="fixed_length",
        chunker_version=(
            "fixed_length_chunker_v1"
        ),
        chunking_config=config,
        chunking_config_sha256=(
            calculate_chunking_config_sha256(
                config
            )
        ),
        chunks_jsonl_sha256=(
            hashlib.sha256(
                chunks_bytes
            ).hexdigest()
        ),
        input_page_count=2,
        eligible_page_count=2,
        chunked_page_count=2,
        skipped_page_count=0,
        skipped_page_ids=(),
        chunk_record_count=2,
        chunk_char_count_total=sum(
            chunk.char_count
            for chunk in chunks
        ),
        quality_gate_passed=True,
        quality_gate_errors=(),
        quality_warnings=(),
        created_at=NOW,
    )

    dataset_directory = (
        root / CHUNK_DATASET_ID
    )

    dataset_directory.mkdir(
        parents=True
    )

    (
        dataset_directory
        / "chunks.jsonl"
    ).write_bytes(chunks_bytes)

    (
        dataset_directory
        / "dataset_manifest.json"
    ).write_text(
        manifest.model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )

    return dataset_directory


def test_build_and_load_searchable_index(
    tmp_path: Path,
) -> None:
    source_directory = (
        write_chunk_dataset(tmp_path)
    )

    provider = FakeEmbeddingProvider()

    result = build_vector_index(
        chunk_dataset_directory=(
            source_directory
        ),
        output_root=tmp_path / "indexes",
        provider=provider,
        created_at=NOW,
    )

    assert result.created is True
    assert result.vectors_path.is_file()
    assert result.metadata_path.is_file()
    assert result.manifest_path.is_file()

    hits = result.index.search(
        query="利润是多少",
        provider=provider,
        top_k=1,
    )

    assert hits[0].text == "净利润"


def test_repeated_build_is_idempotent(
    tmp_path: Path,
) -> None:
    source_directory = (
        write_chunk_dataset(tmp_path)
    )

    provider = FakeEmbeddingProvider()

    first = build_vector_index(
        chunk_dataset_directory=(
            source_directory
        ),
        output_root=tmp_path / "indexes",
        provider=provider,
        created_at=NOW,
    )

    second = build_vector_index(
        chunk_dataset_directory=(
            source_directory
        ),
        output_root=tmp_path / "indexes",
        provider=provider,
    )

    assert first.created is True
    assert second.created is False

    assert (
        first.manifest.index_id
        == second.manifest.index_id
    )


def test_model_version_creates_new_index(
    tmp_path: Path,
) -> None:
    source_directory = (
        write_chunk_dataset(tmp_path)
    )

    first = build_vector_index(
        chunk_dataset_directory=(
            source_directory
        ),
        output_root=tmp_path / "indexes",
        provider=FakeEmbeddingProvider(
            model_version="fake_v1"
        ),
        created_at=NOW,
    )

    second = build_vector_index(
        chunk_dataset_directory=(
            source_directory
        ),
        output_root=tmp_path / "indexes",
        provider=FakeEmbeddingProvider(
            model_version="fake_v2"
        ),
        created_at=NOW,
    )

    assert (
        first.manifest.index_id
        != second.manifest.index_id
    )


def test_detect_tampered_vectors_file(
    tmp_path: Path,
) -> None:
    source_directory = (
        write_chunk_dataset(tmp_path)
    )

    provider = FakeEmbeddingProvider()

    result = build_vector_index(
        chunk_dataset_directory=(
            source_directory
        ),
        output_root=tmp_path / "indexes",
        provider=provider,
        created_at=NOW,
    )

    with result.vectors_path.open("ab") as file:
        file.write(b"tampered")

    with pytest.raises(
        ExistingVectorIndexDatasetError,
        match="vectors.npy 哈希",
    ):
        build_vector_index(
            chunk_dataset_directory=(
                source_directory
            ),
            output_root=tmp_path / "indexes",
            provider=provider,
        )


def test_detect_tampered_metadata_file(
    tmp_path: Path,
) -> None:
    source_directory = (
        write_chunk_dataset(tmp_path)
    )

    provider = FakeEmbeddingProvider()

    result = build_vector_index(
        chunk_dataset_directory=(
            source_directory
        ),
        output_root=tmp_path / "indexes",
        provider=provider,
        created_at=NOW,
    )

    with result.metadata_path.open(
        "ab"
    ) as file:
        file.write(b"tampered")

    with pytest.raises(
        ExistingVectorIndexDatasetError,
        match="metadata.jsonl 哈希",
    ):
        build_vector_index(
            chunk_dataset_directory=(
                source_directory
            ),
            output_root=tmp_path / "indexes",
            provider=provider,
        )