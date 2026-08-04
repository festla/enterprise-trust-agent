from __future__ import annotations

from types import SimpleNamespace

import pytest
import hashlib

import app.services.bm25_index as service
from app.rag.tokenization import (
    DeterministicChineseBigramTokenizer,
)
from app.schemas.enums import (
    ChunkStrategy,
    ReportType,
)
from app.services.bm25_index import (
    CorruptBM25IndexError,
    InvalidBM25SourceError,
    build_bm25_index,
    load_bm25_index,
)
from app.services.chunk_dataset_source import (
    ChunkDatasetSourceError,
)
from app.schemas.chunk import Chunk

from app.schemas.enums import (
    ChunkStrategy,
    PageContentType,
    PageMappingStatus,
    PageParseStatus,
    ReportType,
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


def build_chunk(
    *,
    suffix: str,
    text: str,
    pdf_page: int,
) -> Chunk:
    return Chunk(
        chunk_id=(
            f"chunk_{REPORT_ID}_{suffix * 24}"
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
        chunk_index=pdf_page - 1,
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


def build_chunks() -> tuple[Chunk, ...]:
    return (
        build_chunk(
            suffix="1",
            text="存货会计政策与计量方法",
            pdf_page=1,
        ),
        build_chunk(
            suffix="2",
            text=(
                "合并资产负债表存货金额"
                "为12345元"
            ),
            pdf_page=2,
        ),
        build_chunk(
            suffix="3",
            text=(
                "合并利润表营业收入金额"
                "为98765元"
            ),
            pdf_page=3,
        ),
    )



def build_loaded_source():
    chunks = build_chunks()

    manifest = SimpleNamespace(
        dataset_id=CHUNK_DATASET_ID,
        report_id=REPORT_ID,
        company_id="midea_group",
        fiscal_year=2024,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        document_id=DOCUMENT_ID,
        strategy=(
            ChunkStrategy.FIXED_LENGTH
        ),
        chunks_jsonl_sha256="a" * 64,
    )

    return SimpleNamespace(
        manifest=manifest,
        chunks=chunks,
        manifest_sha256="b" * 64,
    )


def test_build_and_load_bm25_index(
    tmp_path,
    monkeypatch,
) -> None:
    source = build_loaded_source()

    monkeypatch.setattr(
        service,
        "load_chunk_dataset_source",
        lambda _: source,
    )

    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    result = build_bm25_index(
        chunk_dataset_directory=(
            tmp_path / "source"
        ),
        output_root=(
            tmp_path / "indices"
        ),
        tokenizer=tokenizer,
    )

    assert (
        result.index_directory
        / "index.json"
    ).is_file()

    assert (
        result.index_directory
        / "metadata.jsonl"
    ).is_file()

    assert (
        result.index_directory
        / "index_manifest.json"
    ).is_file()

    loaded = load_bm25_index(
        result.index_directory
    )

    hits = loaded.index.search(
        query="合并资产负债表存货金额",
        tokenizer=tokenizer,
        top_k=3,
    )

    assert hits[0].pdf_page == 2
    assert hits[0].retriever_type == "bm25"


def test_repeated_build_is_idempotent(
    tmp_path,
    monkeypatch,
) -> None:
    source = build_loaded_source()

    monkeypatch.setattr(
        service,
        "load_chunk_dataset_source",
        lambda _: source,
    )

    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    first = build_bm25_index(
        chunk_dataset_directory=(
            tmp_path / "source"
        ),
        output_root=(
            tmp_path / "indices"
        ),
        tokenizer=tokenizer,
    )

    first_manifest_bytes = (
        first.index_directory
        / "index_manifest.json"
    ).read_bytes()

    second = build_bm25_index(
        chunk_dataset_directory=(
            tmp_path / "source"
        ),
        output_root=(
            tmp_path / "indices"
        ),
        tokenizer=tokenizer,
    )

    second_manifest_bytes = (
        second.index_directory
        / "index_manifest.json"
    ).read_bytes()

    assert (
        first.index_directory
        == second.index_directory
    )

    assert (
        first.manifest.index_id
        == second.manifest.index_id
    )

    assert (
        first_manifest_bytes
        == second_manifest_bytes
    )


def test_detect_index_json_tampering(
    tmp_path,
    monkeypatch,
) -> None:
    source = build_loaded_source()

    monkeypatch.setattr(
        service,
        "load_chunk_dataset_source",
        lambda _: source,
    )

    result = build_bm25_index(
        chunk_dataset_directory=(
            tmp_path / "source"
        ),
        output_root=(
            tmp_path / "indices"
        ),
        tokenizer=(
            DeterministicChineseBigramTokenizer()
        ),
    )

    index_path = (
        result.index_directory
        / "index.json"
    )

    index_path.write_bytes(
        index_path.read_bytes() + b" "
    )

    with pytest.raises(
        CorruptBM25IndexError,
        match="index.json",
    ):
        load_bm25_index(
            result.index_directory
        )


def test_detect_metadata_tampering(
    tmp_path,
    monkeypatch,
) -> None:
    source = build_loaded_source()

    monkeypatch.setattr(
        service,
        "load_chunk_dataset_source",
        lambda _: source,
    )

    result = build_bm25_index(
        chunk_dataset_directory=(
            tmp_path / "source"
        ),
        output_root=(
            tmp_path / "indices"
        ),
        tokenizer=(
            DeterministicChineseBigramTokenizer()
        ),
    )

    metadata_path = (
        result.index_directory
        / "metadata.jsonl"
    )

    metadata_path.write_bytes(
        metadata_path.read_bytes() + b"\n"
    )

    with pytest.raises(
        CorruptBM25IndexError,
        match="metadata.jsonl",
    ):
        load_bm25_index(
            result.index_directory
        )


def test_reject_missing_index_file(
    tmp_path,
    monkeypatch,
) -> None:
    source = build_loaded_source()

    monkeypatch.setattr(
        service,
        "load_chunk_dataset_source",
        lambda _: source,
    )

    result = build_bm25_index(
        chunk_dataset_directory=(
            tmp_path / "source"
        ),
        output_root=(
            tmp_path / "indices"
        ),
        tokenizer=(
            DeterministicChineseBigramTokenizer()
        ),
    )

    (
        result.index_directory
        / "index.json"
    ).unlink()

    with pytest.raises(
        CorruptBM25IndexError,
        match="缺少必要文件",
    ):
        load_bm25_index(
            result.index_directory
        )


def test_wrap_invalid_chunk_dataset_error(
    tmp_path,
    monkeypatch,
) -> None:
    def raise_source_error(_):
        raise ChunkDatasetSourceError(
            "来源 chunks.jsonl 哈希校验失败"
        )

    monkeypatch.setattr(
        service,
        "load_chunk_dataset_source",
        raise_source_error,
    )

    with pytest.raises(
        InvalidBM25SourceError,
        match="来源 chunks.jsonl",
    ):
        build_bm25_index(
            chunk_dataset_directory=(
                tmp_path / "source"
            ),
            output_root=(
                tmp_path / "indices"
            ),
            tokenizer=(
                DeterministicChineseBigramTokenizer()
            ),
        )

import hashlib
import json


def test_detect_semantic_index_tampering(
    tmp_path,
    monkeypatch,
) -> None:
    source = build_loaded_source()

    monkeypatch.setattr(
        service,
        "load_chunk_dataset_source",
        lambda _: source,
    )

    result = build_bm25_index(
        chunk_dataset_directory=(
            tmp_path / "source"
        ),
        output_root=(
            tmp_path / "indices"
        ),
        tokenizer=(
            DeterministicChineseBigramTokenizer()
        ),
    )

    index_path = (
        result.index_directory
        / "index.json"
    )

    manifest_path = (
        result.index_directory
        / "index_manifest.json"
    )

    index_data = json.loads(
        index_path.read_text(
            encoding="utf-8"
        )
    )

    first_record = (
        index_data["document_records"][0]
    )

    term = next(
        iter(
            first_record[
                "term_frequencies"
            ]
        )
    )

    first_record[
        "term_frequencies"
    ][term] += 1

    first_record["document_length"] += 1

    index_data["total_token_count"] += 1

    index_data[
        "average_document_length"
    ] = (
        index_data["total_token_count"]
        / index_data["document_count"]
    )

    tampered_bytes = json.dumps(
        index_data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    index_path.write_bytes(
        tampered_bytes
    )

    manifest_data = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    manifest_data[
        "index_json_sha256"
    ] = hashlib.sha256(
        tampered_bytes
    ).hexdigest()

    manifest_data[
        "total_token_count"
    ] = index_data[
        "total_token_count"
    ]

    manifest_data[
        "average_document_length"
    ] = index_data[
        "average_document_length"
    ]

    manifest_path.write_text(
        json.dumps(
            manifest_data,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        CorruptBM25IndexError,
        match="持久化词频",
    ):
        load_bm25_index(
            result.index_directory
        )