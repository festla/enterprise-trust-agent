from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from app.rag.chunking import (
    build_chunk_dataset_id,
    build_chunks_for_pages,
    calculate_report_snapshot_sha256,
)
from app.schemas.chunk import (
    Chunk,
    ChunkingConfig,
)
from app.schemas.chunk_dataset import (
    ChunkDatasetManifest,
    calculate_chunking_config_sha256,
)
from app.schemas.enums import (
    PageParseStatus,
)
from app.schemas.page import ParsedPage
from app.schemas.page_dataset import (
    PageDatasetManifest,
)
from app.schemas.report import Report


PAGES_FILENAME = "pages.jsonl"
PAGE_DATASET_MANIFEST_FILENAME = (
    "dataset_manifest.json"
)

CHUNKS_FILENAME = "chunks.jsonl"
CHUNK_DATASET_MANIFEST_FILENAME = (
    "dataset_manifest.json"
)


class ChunkDatasetError(ValueError):
    """Chunk 数据集基础异常。"""


class InvalidSourcePageDatasetError(
    ChunkDatasetError
):
    """来源页面数据集无效或不完整。"""


class InvalidChunkDatasetError(
    ChunkDatasetError
):
    """生成的 Chunk 数据集无效。"""


class ExistingChunkDatasetError(
    ChunkDatasetError
):
    """已有 Chunk 数据集损坏或冲突。"""


class ChunkDatasetWriteError(
    ChunkDatasetError
):
    """Chunk 数据集无法安全落盘。"""


@dataclass(frozen=True, slots=True)
class ChunkDatasetBuildResult:
    """一次 Chunk 数据集构建结果。"""

    manifest: ChunkDatasetManifest
    dataset_directory: Path
    chunks_path: Path
    manifest_path: Path
    created: bool


@dataclass(frozen=True, slots=True)
class _LoadedPageDataset:
    """经过完整校验的来源页面数据集。"""

    manifest: PageDatasetManifest
    pages: tuple[ParsedPage, ...]
    manifest_sha256: str


def _canonical_json_bytes(
    value: object,
) -> bytes:
    """生成稳定 JSON 字节。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _calculate_page_manifest_sha256(
    manifest: PageDatasetManifest,
) -> str:
    """计算页面 Manifest 的语义哈希。"""

    return hashlib.sha256(
        _canonical_json_bytes(
            manifest.model_dump(mode="json")
        )
    ).hexdigest()


def _validate_source_pages(
    *,
    report: Report,
    manifest: PageDatasetManifest,
    pages: tuple[ParsedPage, ...],
) -> None:
    """检查来源页面数据集是否可进入切分管线。"""

    if manifest.report_id != report.report_id:
        raise InvalidSourcePageDatasetError(
            "PageDatasetManifest 与 Report "
            "的 report_id 不一致"
        )

    if (
        report.active_document_id is not None
        and report.active_document_id
        != manifest.document_id
    ):
        raise InvalidSourcePageDatasetError(
            "页面数据集不是 Report 当前有效的 "
            "document_id"
        )

    if (
        report.expected_pdf_page_count
        is not None
        and report.expected_pdf_page_count
        != manifest.total_pdf_pages
    ):
        raise InvalidSourcePageDatasetError(
            "页面数据集页数与 Report "
            "预期 PDF 页数不一致"
        )

    if not manifest.quality_gate_passed:
        raise InvalidSourcePageDatasetError(
            "来源页面数据集未通过质量门禁"
        )

    if len(pages) != manifest.page_record_count:
        raise InvalidSourcePageDatasetError(
            "页面记录数量与 Manifest 不一致"
        )

    expected_pdf_pages = tuple(
        range(
            1,
            manifest.total_pdf_pages + 1,
        )
    )

    actual_pdf_pages = tuple(
        page.pdf_page
        for page in pages
    )

    if actual_pdf_pages != expected_pdf_pages:
        raise InvalidSourcePageDatasetError(
            "页面必须完整覆盖 PDF 页码并升序排列"
        )

    page_ids = [
        page.page_id
        for page in pages
    ]

    if len(page_ids) != len(set(page_ids)):
        raise InvalidSourcePageDatasetError(
            "来源页面数据集包含重复 page_id"
        )

    for page in pages:
        if page.report_id != manifest.report_id:
            raise InvalidSourcePageDatasetError(
                "页面 report_id 与 Manifest 不一致"
            )

        if page.document_id != manifest.document_id:
            raise InvalidSourcePageDatasetError(
                "页面 document_id 与 Manifest 不一致"
            )

        if (
            page.parse_status
            is PageParseStatus.PARSE_ERROR
        ):
            raise InvalidSourcePageDatasetError(
                "存在 parse_error 页面，"
                "不能构建 Chunk 数据集"
            )


def _load_page_dataset(
    *,
    report: Report,
    page_dataset_directory: Path,
) -> _LoadedPageDataset:
    """读取并完整验证来源页面数据集。"""

    manifest_path = (
        page_dataset_directory
        / PAGE_DATASET_MANIFEST_FILENAME
    )

    pages_path = (
        page_dataset_directory
        / PAGES_FILENAME
    )

    if (
        not manifest_path.is_file()
        or not pages_path.is_file()
    ):
        raise InvalidSourcePageDatasetError(
            "页面数据集目录缺少 "
            "dataset_manifest.json 或 pages.jsonl"
        )

    try:
        manifest_bytes = manifest_path.read_bytes()
        pages_bytes = pages_path.read_bytes()
    except OSError as exc:
        raise InvalidSourcePageDatasetError(
            "来源页面数据集文件无法读取"
        ) from exc

    try:
        manifest = (
            PageDatasetManifest
            .model_validate_json(
                manifest_bytes
            )
        )
    except ValidationError as exc:
        raise InvalidSourcePageDatasetError(
            "来源页面 Manifest 无效"
        ) from exc

    actual_pages_sha256 = hashlib.sha256(
        pages_bytes
    ).hexdigest()

    if (
        actual_pages_sha256
        != manifest.pages_jsonl_sha256
    ):
        raise InvalidSourcePageDatasetError(
            "来源 pages.jsonl 哈希校验失败"
        )

    try:
        pages_text = pages_bytes.decode(
            "utf-8"
        )
    except UnicodeDecodeError as exc:
        raise InvalidSourcePageDatasetError(
            "来源 pages.jsonl 不是合法 UTF-8"
        ) from exc

    lines = pages_text.splitlines()

    if (
        not lines
        or any(not line.strip() for line in lines)
    ):
        raise InvalidSourcePageDatasetError(
            "来源 pages.jsonl 不能为空或包含空记录"
        )

    try:
        pages = tuple(
            ParsedPage.model_validate_json(
                line
            )
            for line in lines
        )
    except ValidationError as exc:
        raise InvalidSourcePageDatasetError(
            "来源 pages.jsonl 包含无效页面记录"
        ) from exc

    _validate_source_pages(
        report=report,
        manifest=manifest,
        pages=pages,
    )

    return _LoadedPageDataset(
        manifest=manifest,
        pages=pages,
        manifest_sha256=(
            _calculate_page_manifest_sha256(
                manifest
            )
        ),
    )


def _eligible_page_ids(
    *,
    pages: tuple[ParsedPage, ...],
    config: ChunkingConfig,
) -> tuple[str, ...]:
    """返回当前配置允许切分的页面 ID。"""

    return tuple(
        page.page_id
        for page in pages
        if (
            page.parse_status
            is PageParseStatus.SUCCESS
            and page.content_type
            in config.include_content_types
            and bool(page.normalized_text)
        )
    )


def _validate_chunks(
    *,
    chunks: tuple[Chunk, ...],
    pages: tuple[ParsedPage, ...],
    expected_dataset_id: str,
    config: ChunkingConfig,
) -> None:
    """检查 Chunk 是否完整对应来源页面。"""

    if not chunks:
        raise InvalidChunkDatasetError(
            "Chunk 数据集不能为空"
        )

    expected_order = tuple(
        sorted(
            chunks,
            key=lambda chunk: (
                chunk.pdf_page,
                chunk.chunk_index,
            ),
        )
    )

    if chunks != expected_order:
        raise InvalidChunkDatasetError(
            "Chunk 必须按 PDF 页码和 "
            "chunk_index 升序排列"
        )

    chunk_ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise InvalidChunkDatasetError(
            "Chunk 数据集中出现重复 chunk_id"
        )

    pages_by_id = {
        page.page_id: page
        for page in pages
    }

    chunk_indices_by_page: dict[
        str,
        list[int],
    ] = defaultdict(list)

    for chunk in chunks:
        if (
            chunk.chunk_dataset_id
            != expected_dataset_id
        ):
            raise InvalidChunkDatasetError(
                "Chunk 的 chunk_dataset_id "
                "与目标数据集不一致"
            )

        page = pages_by_id.get(
            chunk.page_id
        )

        if page is None:
            raise InvalidChunkDatasetError(
                "Chunk 引用了不存在的 page_id"
            )

        if chunk.document_id != page.document_id:
            raise InvalidChunkDatasetError(
                "Chunk 与来源页面 document_id 不一致"
            )

        if chunk.report_id != page.report_id:
            raise InvalidChunkDatasetError(
                "Chunk 与来源页面 report_id 不一致"
            )

        if chunk.pdf_page != page.pdf_page:
            raise InvalidChunkDatasetError(
                "Chunk 与来源页面 pdf_page 不一致"
            )

        if (
            chunk.printed_page
            != page.printed_page
        ):
            raise InvalidChunkDatasetError(
                "Chunk 与来源页面 printed_page 不一致"
            )

        if (
            chunk.source_end_char
            > len(page.normalized_text)
        ):
            raise InvalidChunkDatasetError(
                "Chunk 字符边界超出来源页面文本"
            )

        expected_text = (
            page.normalized_text[
                chunk.source_start_char:
                chunk.source_end_char
            ]
        )

        if chunk.text != expected_text:
            raise InvalidChunkDatasetError(
                "Chunk 文本不能由来源页面字符区间还原"
            )

        chunk_indices_by_page[
            chunk.page_id
        ].append(chunk.chunk_index)

    expected_page_ids = set(
        _eligible_page_ids(
            pages=pages,
            config=config,
        )
    )

    actual_page_ids = set(
        chunk_indices_by_page
    )

    if actual_page_ids != expected_page_ids:
        raise InvalidChunkDatasetError(
            "生成 Chunk 的页面集合与 "
            "eligible 页面集合不一致"
        )

    for (
        page_id,
        chunk_indices,
    ) in chunk_indices_by_page.items():
        expected_indices = list(
            range(len(chunk_indices))
        )

        if chunk_indices != expected_indices:
            raise InvalidChunkDatasetError(
                f"页面 '{page_id}' 的 "
                "chunk_index 必须从 0 连续递增"
            )


def _serialize_chunks(
    chunks: tuple[Chunk, ...],
) -> bytes:
    """确定性序列化 Chunk JSONL。"""

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


def _build_dataset_manifest(
    *,
    dataset_id: str,
    report: Report,
    loaded_source: _LoadedPageDataset,
    config: ChunkingConfig,
    chunks: tuple[Chunk, ...],
    chunks_bytes: bytes,
    created_at: datetime,
) -> ChunkDatasetManifest:
    """根据来源页面与 Chunk 构造 Manifest。"""

    eligible_ids = _eligible_page_ids(
        pages=loaded_source.pages,
        config=config,
    )

    eligible_id_set = set(
        eligible_ids
    )

    skipped_pages = tuple(
        page
        for page in loaded_source.pages
        if page.page_id not in eligible_id_set
    )

    skipped_page_ids = tuple(
        sorted(
            page.page_id
            for page in skipped_pages
        )
    )

    skipped_type_counts = Counter(
        page.content_type.value
        for page in skipped_pages
    )

    quality_warnings = tuple(
        f"跳过 {count} 个 {content_type} 页面"
        for content_type, count
        in sorted(
            skipped_type_counts.items()
        )
    )

    chunked_page_ids = {
        chunk.page_id
        for chunk in chunks
    }

    return ChunkDatasetManifest(
        dataset_id=dataset_id,
        page_dataset_id=(
            loaded_source.manifest.dataset_id
        ),
        report_id=report.report_id,
        company_id=report.company_id,
        fiscal_year=report.fiscal_year,
        report_type=report.report_type,
        document_id=(
            loaded_source.manifest.document_id
        ),
        page_dataset_manifest_sha256=(
            loaded_source.manifest_sha256
        ),
        source_pages_jsonl_sha256=(
            loaded_source.manifest
            .pages_jsonl_sha256
        ),
        report_snapshot_sha256=(
            calculate_report_snapshot_sha256(
                report
            )
        ),
        strategy=config.strategy,
        chunker_name=config.chunker_name,
        chunker_version=(
            config.chunker_version
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
        input_page_count=len(
            loaded_source.pages
        ),
        eligible_page_count=len(
            eligible_ids
        ),
        chunked_page_count=len(
            chunked_page_ids
        ),
        skipped_page_count=len(
            skipped_page_ids
        ),
        skipped_page_ids=(
            skipped_page_ids
        ),
        chunk_record_count=len(chunks),
        chunk_char_count_total=sum(
            chunk.char_count
            for chunk in chunks
        ),
        quality_gate_passed=True,
        quality_gate_errors=(),
        quality_warnings=quality_warnings,
        created_at=created_at,
    )


def _write_bytes_synced(
    *,
    path: Path,
    content: bytes,
) -> None:
    """创建文件、写入并刷新到磁盘。"""

    with path.open("xb") as file:
        file.write(content)
        file.flush()
        os.fsync(file.fileno())


def _read_existing_dataset(
    *,
    dataset_directory: Path,
    report: Report,
    loaded_source: _LoadedPageDataset,
    config: ChunkingConfig,
    expected_dataset_id: str,
) -> ChunkDatasetBuildResult:
    """读取并验证已经存在的 Chunk 数据集。"""

    chunks_path = (
        dataset_directory
        / CHUNKS_FILENAME
    )

    manifest_path = (
        dataset_directory
        / CHUNK_DATASET_MANIFEST_FILENAME
    )

    if (
        not chunks_path.is_file()
        or not manifest_path.is_file()
    ):
        raise ExistingChunkDatasetError(
            "已有 Chunk 数据集目录缺少必要文件"
        )

    try:
        stored_manifest = (
            ChunkDatasetManifest
            .model_validate_json(
                manifest_path.read_bytes()
            )
        )
    except (
        OSError,
        ValidationError,
    ) as exc:
        raise ExistingChunkDatasetError(
            "已有 Chunk Dataset Manifest 无效"
        ) from exc

    if (
        stored_manifest.dataset_id
        != expected_dataset_id
    ):
        raise ExistingChunkDatasetError(
            "已有 Chunk 数据集 ID "
            "与当前输入不一致"
        )

    if (
        stored_manifest.page_dataset_id
        != loaded_source.manifest.dataset_id
    ):
        raise ExistingChunkDatasetError(
            "已有 Chunk 数据集来源 "
            "page_dataset_id 不一致"
        )

    if (
        stored_manifest.report_id
        != report.report_id
    ):
        raise ExistingChunkDatasetError(
            "已有 Chunk 数据集 report_id 不一致"
        )

    try:
        chunks_bytes = chunks_path.read_bytes()
    except OSError as exc:
        raise ExistingChunkDatasetError(
            "已有 chunks.jsonl 无法读取"
        ) from exc

    actual_chunks_sha256 = hashlib.sha256(
        chunks_bytes
    ).hexdigest()

    if (
        actual_chunks_sha256
        != stored_manifest.chunks_jsonl_sha256
    ):
        raise ExistingChunkDatasetError(
            "chunks.jsonl 哈希校验失败，"
            "文件可能被修改或损坏"
        )

    try:
        chunks_text = chunks_bytes.decode(
            "utf-8"
        )
    except UnicodeDecodeError as exc:
        raise ExistingChunkDatasetError(
            "chunks.jsonl 不是合法 UTF-8"
        ) from exc

    lines = chunks_text.splitlines()

    if (
        not lines
        or any(not line.strip() for line in lines)
    ):
        raise ExistingChunkDatasetError(
            "chunks.jsonl 不能为空或包含空记录"
        )

    try:
        chunks = tuple(
            Chunk.model_validate_json(
                line
            )
            for line in lines
        )
    except ValidationError as exc:
        raise ExistingChunkDatasetError(
            "chunks.jsonl 包含无效 Chunk"
        ) from exc

    _validate_chunks(
        chunks=chunks,
        pages=loaded_source.pages,
        expected_dataset_id=(
            expected_dataset_id
        ),
        config=config,
    )

    rebuilt_manifest = (
        _build_dataset_manifest(
            dataset_id=expected_dataset_id,
            report=report,
            loaded_source=loaded_source,
            config=config,
            chunks=chunks,
            chunks_bytes=chunks_bytes,
            created_at=(
                stored_manifest.created_at
            ),
        )
    )

    if (
        rebuilt_manifest.model_dump(
            mode="json"
        )
        != stored_manifest.model_dump(
            mode="json"
        )
    ):
        raise ExistingChunkDatasetError(
            "已有 Chunk Manifest 统计 "
            "与 chunks.jsonl 内容不一致"
        )

    return ChunkDatasetBuildResult(
        manifest=stored_manifest,
        dataset_directory=(
            dataset_directory
        ),
        chunks_path=chunks_path,
        manifest_path=manifest_path,
        created=False,
    )


def build_chunk_dataset(
    *,
    report: Report,
    page_dataset_directory: Path,
    output_root: Path,
    config: ChunkingConfig,
    created_at: datetime | None = None,
) -> ChunkDatasetBuildResult:
    """生成不可变、可重复的 Chunk 数据集。"""

    loaded_source = _load_page_dataset(
        report=report,
        page_dataset_directory=(
            page_dataset_directory
        ),
    )

    dataset_id = build_chunk_dataset_id(
        report=report,
        page_dataset_manifest=(
            loaded_source.manifest
        ),
        config=config,
    )

    report_output_directory = (
        output_root / report.report_id
    )

    dataset_directory = (
        report_output_directory
        / dataset_id
    )

    if dataset_directory.exists():
        return _read_existing_dataset(
            dataset_directory=(
                dataset_directory
            ),
            report=report,
            loaded_source=loaded_source,
            config=config,
            expected_dataset_id=dataset_id,
        )

    chunks = build_chunks_for_pages(
        pages=loaded_source.pages,
        report=report,
        page_dataset_manifest=(
            loaded_source.manifest
        ),
        config=config,
    )

    _validate_chunks(
        chunks=chunks,
        pages=loaded_source.pages,
        expected_dataset_id=dataset_id,
        config=config,
    )

    chunks_bytes = _serialize_chunks(
        chunks
    )

    created_at_value = (
        created_at
        if created_at is not None
        else datetime.now(timezone.utc)
    )

    dataset_manifest = (
        _build_dataset_manifest(
            dataset_id=dataset_id,
            report=report,
            loaded_source=loaded_source,
            config=config,
            chunks=chunks,
            chunks_bytes=chunks_bytes,
            created_at=created_at_value,
        )
    )

    report_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{dataset_id}_",
            dir=report_output_directory,
        )
    )

    try:
        temporary_chunks_path = (
            temporary_directory
            / CHUNKS_FILENAME
        )

        temporary_manifest_path = (
            temporary_directory
            / CHUNK_DATASET_MANIFEST_FILENAME
        )

        _write_bytes_synced(
            path=temporary_chunks_path,
            content=chunks_bytes,
        )

        manifest_bytes = (
            dataset_manifest.model_dump_json(
                indent=2
            )
            + "\n"
        ).encode("utf-8")

        _write_bytes_synced(
            path=temporary_manifest_path,
            content=manifest_bytes,
        )

        try:
            temporary_directory.rename(
                dataset_directory
            )

        except FileExistsError:
            shutil.rmtree(
                temporary_directory,
                ignore_errors=True,
            )

            return _read_existing_dataset(
                dataset_directory=(
                    dataset_directory
                ),
                report=report,
                loaded_source=loaded_source,
                config=config,
                expected_dataset_id=dataset_id,
            )

        except OSError as exc:
            raise ChunkDatasetWriteError(
                "无法将临时 Chunk 数据集 "
                "提交到最终目录"
            ) from exc

    except Exception:
        if temporary_directory.exists():
            shutil.rmtree(
                temporary_directory,
                ignore_errors=True,
            )

        raise

    return ChunkDatasetBuildResult(
        manifest=dataset_manifest,
        dataset_directory=(
            dataset_directory
        ),
        chunks_path=(
            dataset_directory
            / CHUNKS_FILENAME
        ),
        manifest_path=(
            dataset_directory
            / CHUNK_DATASET_MANIFEST_FILENAME
        ),
        created=True,
    )