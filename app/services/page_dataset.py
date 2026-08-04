from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from app.schemas.document import DocumentManifest
from app.schemas.enums import (
    PageContentType,
    PageMappingStatus,
    PageParseStatus,
)
from app.schemas.page import ParsedPage
from app.schemas.page_dataset import (
    PageDatasetManifest,
)
from app.schemas.report import PageMappingSegment
from app.services.page_parser import (
    PAGE_CONTENT_CLASSIFIER_VERSION,
    PAGE_TEXT_NORMALIZER_VERSION,
    get_page_parser_version,
    parse_pdf_pages,
)


PAGES_FILENAME = "pages.jsonl"
DATASET_MANIFEST_FILENAME = (
    "dataset_manifest.json"
)


class PageDatasetError(ValueError):
    """页面数据集处理基础异常。"""


class MissingDatasetMappingError(
    PageDatasetError
):
    """当前报告缺少页码映射配置。"""


class InvalidPageDatasetError(
    PageDatasetError
):
    """页面记录结构或顺序不完整。"""


class ExistingPageDatasetError(
    PageDatasetError
):
    """已经存在的页面数据集损坏或冲突。"""


class PageDatasetWriteError(
    PageDatasetError
):
    """页面数据集无法安全落盘。"""


@dataclass(frozen=True, slots=True)
class PageDatasetBuildResult:
    """一次页面数据集构建结果。"""

    manifest: PageDatasetManifest
    dataset_directory: Path
    pages_path: Path
    manifest_path: Path
    created: bool


def _canonical_json_bytes(
    value: object,
) -> bytes:
    """生成稳定、可哈希的 JSON 字节。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def calculate_mapping_sha256(
    *,
    report_id: str,
    page_mappings: Iterable[
        PageMappingSegment
    ],
) -> str:
    """计算当前报告页码映射配置的稳定哈希。"""

    selected_mappings = sorted(
        (
            mapping
            for mapping in page_mappings
            if mapping.report_id == report_id
        ),
        key=lambda mapping: (
            mapping.pdf_page_start,
            mapping.pdf_page_end,
            mapping.mapping_id,
        ),
    )

    if not selected_mappings:
        raise MissingDatasetMappingError(
            f"Report '{report_id}' "
            "没有页码映射配置"
        )

    payload = [
        mapping.model_dump(mode="json")
        for mapping in selected_mappings
    ]

    return hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()


def _build_dataset_id(
    *,
    manifest: DocumentManifest,
    mapping_sha256: str,
    parser_version: str,
) -> str:
    """根据全部解析输入生成稳定数据集 ID。"""

    identity_payload = {
        "document_id": manifest.document_id,
        "source_sha256": manifest.sha256,
        "page_schema_version": 1,
        "mapping_sha256": mapping_sha256,
        "parser_name": "pymupdf",
        "parser_version": parser_version,
        "normalizer_version": (
            PAGE_TEXT_NORMALIZER_VERSION
        ),
        "classifier_version": (
            PAGE_CONTENT_CLASSIFIER_VERSION
        ),
    }

    identity_sha256 = hashlib.sha256(
        _canonical_json_bytes(
            identity_payload
        )
    ).hexdigest()

    return (
        f"page_dataset_{manifest.report_id}_"
        f"{identity_sha256[:24]}"
    )


def _serialize_pages(
    pages: tuple[ParsedPage, ...],
) -> bytes:
    """将页面记录序列化为确定性的 JSONL。"""

    lines = [
        json.dumps(
            page.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for page in pages
    ]

    return (
        "\n".join(lines) + "\n"
    ).encode("utf-8")


def _validate_pages(
    *,
    source_manifest: DocumentManifest,
    pages: tuple[ParsedPage, ...],
) -> None:
    """检查页面记录是否覆盖完整文档。"""

    if (
        len(pages)
        != source_manifest.pdf_page_count
    ):
        raise InvalidPageDatasetError(
            "页面记录数与 PDF 总页数不一致"
        )

    expected_pdf_pages = tuple(
        range(
            1,
            source_manifest.pdf_page_count + 1,
        )
    )

    actual_pdf_pages = tuple(
        page.pdf_page
        for page in pages
    )

    if actual_pdf_pages != expected_pdf_pages:
        raise InvalidPageDatasetError(
            "页面记录必须按照完整 PDF 页码升序排列"
        )

    page_ids = [
        page.page_id
        for page in pages
    ]

    if len(page_ids) != len(set(page_ids)):
        raise InvalidPageDatasetError(
            "页面数据集中出现重复 page_id"
        )

    for page in pages:
        if (
            page.document_id
            != source_manifest.document_id
        ):
            raise InvalidPageDatasetError(
                "页面 document_id 与来源 Manifest 不一致"
            )

        if (
            page.report_id
            != source_manifest.report_id
        ):
            raise InvalidPageDatasetError(
                "页面 report_id 与来源 Manifest 不一致"
            )

    parser_names = {
        page.parser_name
        for page in pages
    }

    parser_versions = {
        page.parser_version
        for page in pages
    }

    if len(parser_names) != 1:
        raise InvalidPageDatasetError(
            "同一数据集不能混用多个解析器"
        )

    if len(parser_versions) != 1:
        raise InvalidPageDatasetError(
            "同一数据集不能混用多个解析器版本"
        )


def _build_dataset_manifest(
    *,
    dataset_id: str,
    source_manifest: DocumentManifest,
    mapping_sha256: str,
    pages: tuple[ParsedPage, ...],
    pages_bytes: bytes,
    created_at: datetime,
) -> PageDatasetManifest:
    """根据页面记录生成数据集 Manifest。"""

    content_counter = Counter(
        page.content_type
        for page in pages
    )

    parse_counter = Counter(
        page.parse_status
        for page in pages
    )

    content_type_counts = {
        content_type: content_counter.get(
            content_type,
            0,
        )
        for content_type in PageContentType
    }

    parse_status_counts = {
        parse_status: parse_counter.get(
            parse_status,
            0,
        )
        for parse_status in PageParseStatus
    }

    mapped_page_count = sum(
        page.mapping_status
        is PageMappingStatus.MAPPED
        for page in pages
    )

    unmapped_page_count = (
        len(pages) - mapped_page_count
    )

    quality_gate_errors: list[str] = []
    quality_warnings: list[str] = []

    parse_error_count = (
        parse_status_counts[
            PageParseStatus.PARSE_ERROR
        ]
    )

    if parse_error_count > 0:
        quality_gate_errors.append(
            f"存在 {parse_error_count} 个页面解析失败"
        )

    scanned_count = (
        content_type_counts[
            PageContentType.SCANNED
        ]
    )

    if scanned_count > 0:
        quality_warnings.append(
            f"存在 {scanned_count} 个扫描型页面，"
            "当前未执行 OCR"
        )

    empty_count = (
        content_type_counts[
            PageContentType.EMPTY
        ]
    )

    if empty_count > 0:
        quality_warnings.append(
            f"存在 {empty_count} 个空白页面"
        )

    return PageDatasetManifest(
        dataset_id=dataset_id,
        document_id=source_manifest.document_id,
        report_id=source_manifest.report_id,
        source_sha256=source_manifest.sha256,
        mapping_sha256=mapping_sha256,
        pages_jsonl_sha256=hashlib.sha256(
            pages_bytes
        ).hexdigest(),
        parser_name=pages[0].parser_name,
        parser_version=pages[0].parser_version,
        normalizer_version=(
            PAGE_TEXT_NORMALIZER_VERSION
        ),
        classifier_version=(
            PAGE_CONTENT_CLASSIFIER_VERSION
        ),
        total_pdf_pages=(
            source_manifest.pdf_page_count
        ),
        page_record_count=len(pages),
        mapped_page_count=mapped_page_count,
        unmapped_page_count=(
            unmapped_page_count
        ),
        raw_char_count_total=sum(
            page.raw_char_count
            for page in pages
        ),
        normalized_char_count_total=sum(
            page.normalized_char_count
            for page in pages
        ),
        content_type_counts=(
            content_type_counts
        ),
        parse_status_counts=(
            parse_status_counts
        ),
        quality_gate_passed=(
            len(quality_gate_errors) == 0
        ),
        quality_gate_errors=tuple(
            quality_gate_errors
        ),
        quality_warnings=tuple(
            quality_warnings
        ),
        created_at=created_at,
    )


def _write_bytes_synced(
    *,
    path: Path,
    content: bytes,
) -> None:
    """写入并刷新一个新文件。"""

    with path.open("xb") as file:
        file.write(content)
        file.flush()
        os.fsync(file.fileno())


def _read_existing_dataset(
    *,
    dataset_directory: Path,
    source_manifest: DocumentManifest,
    expected_dataset_id: str,
    expected_mapping_sha256: str,
    expected_parser_version: str,
) -> PageDatasetBuildResult:
    """读取并完整校验已有页面数据集。"""

    pages_path = (
        dataset_directory / PAGES_FILENAME
    )

    manifest_path = (
        dataset_directory
        / DATASET_MANIFEST_FILENAME
    )

    if (
        not pages_path.is_file()
        or not manifest_path.is_file()
    ):
        raise ExistingPageDatasetError(
            "已有数据集目录缺少必要文件："
            f"{dataset_directory}"
        )

    try:
        stored_manifest = (
            PageDatasetManifest
            .model_validate_json(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )
        )
    except (
        OSError,
        ValidationError,
    ) as exc:
        raise ExistingPageDatasetError(
            "已有数据集 Manifest 无效"
        ) from exc

    if (
        stored_manifest.dataset_id
        != expected_dataset_id
    ):
        raise ExistingPageDatasetError(
            "已有数据集 ID 与当前输入不一致"
        )

    if (
        stored_manifest.document_id
        != source_manifest.document_id
        or stored_manifest.source_sha256
        != source_manifest.sha256
    ):
        raise ExistingPageDatasetError(
            "已有数据集来源文档身份不一致"
        )

    if (
        stored_manifest.mapping_sha256
        != expected_mapping_sha256
    ):
        raise ExistingPageDatasetError(
            "已有数据集页码映射版本不一致"
        )

    if (
        stored_manifest.parser_version
        != expected_parser_version
    ):
        raise ExistingPageDatasetError(
            "已有数据集解析器版本不一致"
        )

    try:
        pages_bytes = pages_path.read_bytes()
    except OSError as exc:
        raise ExistingPageDatasetError(
            "已有 pages.jsonl 无法读取"
        ) from exc

    actual_pages_sha256 = hashlib.sha256(
        pages_bytes
    ).hexdigest()

    if (
        actual_pages_sha256
        != stored_manifest.pages_jsonl_sha256
    ):
        raise ExistingPageDatasetError(
            "pages.jsonl 哈希校验失败，"
            "文件可能被修改或损坏"
        )

    try:
        text = pages_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExistingPageDatasetError(
            "pages.jsonl 不是合法 UTF-8"
        ) from exc

    lines = text.splitlines()

    if (
        not lines
        or any(not line.strip() for line in lines)
    ):
        raise ExistingPageDatasetError(
            "pages.jsonl 不能为空或包含空记录"
        )

    try:
        pages = tuple(
            ParsedPage.model_validate_json(
                line
            )
            for line in lines
        )
    except ValidationError as exc:
        raise ExistingPageDatasetError(
            "pages.jsonl 中存在无效页面记录"
        ) from exc

    _validate_pages(
        source_manifest=source_manifest,
        pages=pages,
    )

    rebuilt_manifest = (
        _build_dataset_manifest(
            dataset_id=expected_dataset_id,
            source_manifest=source_manifest,
            mapping_sha256=(
                expected_mapping_sha256
            ),
            pages=pages,
            pages_bytes=pages_bytes,
            created_at=(
                stored_manifest.created_at
            ),
        )
    )

    if (
        rebuilt_manifest.model_dump(mode="json")
        != stored_manifest.model_dump(
            mode="json"
        )
    ):
        raise ExistingPageDatasetError(
            "已有数据集统计与页面内容不一致"
        )

    return PageDatasetBuildResult(
        manifest=stored_manifest,
        dataset_directory=(
            dataset_directory
        ),
        pages_path=pages_path,
        manifest_path=manifest_path,
        created=False,
    )


def build_page_dataset(
    *,
    source_manifest: DocumentManifest,
    page_mappings: Iterable[
        PageMappingSegment
    ],
    project_root: Path,
    output_root: Path,
    created_at: datetime | None = None,
) -> PageDatasetBuildResult:
    """解析整份 PDF 并安全生成页面数据集。"""

    page_mappings_tuple = tuple(
        page_mappings
    )

    parser_version = (
        get_page_parser_version()
    )

    mapping_sha256 = (
        calculate_mapping_sha256(
            report_id=source_manifest.report_id,
            page_mappings=(
                page_mappings_tuple
            ),
        )
    )

    dataset_id = _build_dataset_id(
        manifest=source_manifest,
        mapping_sha256=mapping_sha256,
        parser_version=parser_version,
    )

    report_output_directory = (
        output_root
        / source_manifest.report_id
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
            source_manifest=source_manifest,
            expected_dataset_id=dataset_id,
            expected_mapping_sha256=(
                mapping_sha256
            ),
            expected_parser_version=(
                parser_version
            ),
        )

    created_at_value = (
        created_at
        if created_at is not None
        else datetime.now(timezone.utc)
    )

    pages = parse_pdf_pages(
        manifest=source_manifest,
        page_mappings=(
            page_mappings_tuple
        ),
        project_root=project_root,
        pdf_pages=None,
        parsed_at=created_at_value,
    )

    _validate_pages(
        source_manifest=source_manifest,
        pages=pages,
    )

    if pages[0].parser_version != parser_version:
        raise InvalidPageDatasetError(
            "页面解析结果的 parser_version "
            "与当前环境不一致"
        )

    pages_bytes = _serialize_pages(
        pages
    )

    dataset_manifest = (
        _build_dataset_manifest(
            dataset_id=dataset_id,
            source_manifest=source_manifest,
            mapping_sha256=mapping_sha256,
            pages=pages,
            pages_bytes=pages_bytes,
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
        temporary_pages_path = (
            temporary_directory
            / PAGES_FILENAME
        )

        temporary_manifest_path = (
            temporary_directory
            / DATASET_MANIFEST_FILENAME
        )

        _write_bytes_synced(
            path=temporary_pages_path,
            content=pages_bytes,
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
                source_manifest=(
                    source_manifest
                ),
                expected_dataset_id=(
                    dataset_id
                ),
                expected_mapping_sha256=(
                    mapping_sha256
                ),
                expected_parser_version=(
                    parser_version
                ),
            )
        except OSError as exc:
            raise PageDatasetWriteError(
                "无法将临时页面数据集提交到最终目录"
            ) from exc

    except Exception:
        if temporary_directory.exists():
            shutil.rmtree(
                temporary_directory,
                ignore_errors=True,
            )

        raise

    return PageDatasetBuildResult(
        manifest=dataset_manifest,
        dataset_directory=(
            dataset_directory
        ),
        pages_path=(
            dataset_directory
            / PAGES_FILENAME
        ),
        manifest_path=(
            dataset_directory
            / DATASET_MANIFEST_FILENAME
        ),
        created=True,
    )