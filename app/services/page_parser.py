from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import (
    PackageNotFoundError,
    version,
)
from pathlib import Path

import pymupdf

from app.schemas.document import DocumentManifest
from app.schemas.enums import (
    DocumentValidationStatus,
    PageContentType,
    PageParseStatus,
)
from app.schemas.page import (
    PageMappingResult,
    ParsedPage,
)
from app.schemas.report import PageMappingSegment
from app.services.document_ingestion import (
    calculate_file_sha256,
)
from app.services.page_mapping import (
    PageMappingResolver,
)


MIXED_IMAGE_AREA_RATIO_THRESHOLD = 0.5
MAX_PARSE_ERROR_LENGTH = 1000

_EXCESSIVE_BLANK_LINES_PATTERN = re.compile(
    r"\n{3,}"
)


PAGE_TEXT_NORMALIZER_VERSION = "page_text_normalizer_v1"
PAGE_CONTENT_CLASSIFIER_VERSION = "page_content_classifier_v1"


class PageParserError(ValueError):
    """页面解析服务基础异常。"""


class PageSourceFileNotFoundError(
    PageParserError
):
    """Manifest 对应的 PDF 文件不存在。"""


class PageSourceOutsideProjectError(
    PageParserError
):
    """Manifest 指向了项目目录外的文件。"""


class PageSourceHashMismatchError(
    PageParserError
):
    """当前 PDF 内容与 Manifest 文件哈希不一致。"""


class PageSourceOpenError(PageParserError):
    """PDF 文件无法正常打开。"""


class PageSourceEncryptedError(PageParserError):
    """PDF 文件需要密码。"""


class PageSourcePageCountMismatchError(
    PageParserError
):
    """PDF 实际页数与 Manifest 不一致。"""


class EmptyPageSelectionError(PageParserError):
    """调用者没有提供任何需要解析的页面。"""


@dataclass(frozen=True, slots=True)
class PageExtractionSignals:
    """从单个 PyMuPDF 页面取得的解析信号。"""

    raw_text: str
    normalized_text: str
    text_block_count: int
    image_block_count: int
    embedded_image_count: int
    max_image_area_ratio: float


def normalize_page_text(raw_text: str) -> str:
    """执行不会改变文本语义的低风险规范化。"""

    text = raw_text.replace("\x00", "")

    text = text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    normalized_lines = [
        line.rstrip(" \t")
        for line in text.split("\n")
    ]

    text = "\n".join(normalized_lines)

    text = _EXCESSIVE_BLANK_LINES_PATTERN.sub(
        "\n\n",
        text,
    )

    return text.strip()


def classify_page_content(
    *,
    normalized_text: str,
    image_block_count: int,
    embedded_image_count: int,
    max_image_area_ratio: float,
) -> PageContentType:
    """按照透明规则判断页面主要内容类型。"""

    has_text = bool(normalized_text)

    has_image = (
        image_block_count > 0
        or embedded_image_count > 0
        or max_image_area_ratio > 0
    )

    if not has_text:
        if has_image:
            return PageContentType.SCANNED

        return PageContentType.EMPTY

    if (
        max_image_area_ratio
        >= MIXED_IMAGE_AREA_RATIO_THRESHOLD
    ):
        return PageContentType.MIXED

    return PageContentType.TEXT


def get_page_parser_version() -> str:
    """读取当前 PyMuPDF 版本。"""

    try:
        return version("PyMuPDF")
    except PackageNotFoundError as exc:
        raise PageParserError(
            "无法读取 PyMuPDF 包版本"
        ) from exc


def _resolve_manifest_pdf_path(
    *,
    manifest: DocumentManifest,
    project_root: Path,
) -> Path:
    """将 Manifest 相对路径解析为安全的实际路径。"""

    resolved_project_root = project_root.resolve()

    candidate_path = (
        resolved_project_root
        / Path(manifest.source_path)
    )

    try:
        resolved_pdf_path = candidate_path.resolve(
            strict=True
        )
    except FileNotFoundError as exc:
        raise PageSourceFileNotFoundError(
            "Manifest 对应的 PDF 文件不存在："
            f"{candidate_path}"
        ) from exc

    if not resolved_pdf_path.is_file():
        raise PageSourceFileNotFoundError(
            "Manifest 对应路径不是普通文件："
            f"{resolved_pdf_path}"
        )

    try:
        resolved_pdf_path.relative_to(
            resolved_project_root
        )
    except ValueError as exc:
        raise PageSourceOutsideProjectError(
            "Manifest 对应 PDF 位于项目目录外："
            f"{resolved_pdf_path}"
        ) from exc

    return resolved_pdf_path


def _normalize_page_selection(
    *,
    pdf_pages: Iterable[int] | None,
    total_pdf_pages: int,
) -> tuple[int, ...]:
    """将页面选择转换为升序、去重的页码元组。"""

    if pdf_pages is None:
        return tuple(
            range(1, total_pdf_pages + 1)
        )

    normalized_pages = tuple(
        sorted(set(pdf_pages))
    )

    if not normalized_pages:
        raise EmptyPageSelectionError(
            "至少需要选择一个 PDF 页面"
        )

    invalid_pages = [
        pdf_page
        for pdf_page in normalized_pages
        if (
            isinstance(pdf_page, bool)
            or not isinstance(pdf_page, int)
            or pdf_page < 1
            or pdf_page > total_pdf_pages
        )
    ]

    if invalid_pages:
        raise PageParserError(
            "存在超出文档范围的 PDF 页码："
            f"{invalid_pages}"
        )

    return normalized_pages


def _calculate_max_image_area_ratio(
    *,
    page: pymupdf.Page,
    embedded_images: list | tuple,
) -> float:
    """计算单张嵌入图片占页面面积的最大比例。"""

    page_area = float(
        page.rect.get_area()
    )

    if page_area <= 0:
        return 0.0

    maximum_ratio = 0.0

    for image in embedded_images:
        xref = image[0]

        try:
            image_rectangles = (
                page.get_image_rects(xref)
            )
        except Exception:
            continue

        for rectangle in image_rectangles:
            image_area = float(
                rectangle.get_area()
            )

            ratio = image_area / page_area

            ratio = max(
                0.0,
                min(ratio, 1.0),
            )

            maximum_ratio = max(
                maximum_ratio,
                ratio,
            )

    return maximum_ratio


def _extract_page_signals(
    page: pymupdf.Page,
) -> PageExtractionSignals:
    """从一个页面提取文本、块数量和图片信号。"""

    raw_text = page.get_text("text")

    if not isinstance(raw_text, str):
        raise TypeError(
            "PyMuPDF 页面文本结果必须为字符串"
        )

    page_dictionary = page.get_text("dict")

    if not isinstance(page_dictionary, dict):
        raise TypeError(
            "PyMuPDF 页面字典结果必须为映射对象"
        )

    blocks = page_dictionary.get(
        "blocks",
        [],
    )

    if not isinstance(blocks, list):
        raise TypeError(
            "PyMuPDF 页面 blocks 必须为列表"
        )

    text_block_count = sum(
        isinstance(block, dict)
        and block.get("type") == 0
        for block in blocks
    )

    image_block_count = sum(
        isinstance(block, dict)
        and block.get("type") == 1
        for block in blocks
    )

    embedded_images = page.get_images(
        full=True
    )

    max_image_area_ratio = (
        _calculate_max_image_area_ratio(
            page=page,
            embedded_images=embedded_images,
        )
    )

    return PageExtractionSignals(
        raw_text=raw_text,
        normalized_text=normalize_page_text(
            raw_text
        ),
        text_block_count=text_block_count,
        image_block_count=image_block_count,
        embedded_image_count=len(
            embedded_images
        ),
        max_image_area_ratio=(
            max_image_area_ratio
        ),
    )


def _format_parse_error(
    error: Exception,
) -> str:
    """生成有长度限制的页面错误摘要。"""

    error_message = (
        f"{type(error).__name__}: {error}"
    )

    return error_message[
        :MAX_PARSE_ERROR_LENGTH
    ]


def _build_success_page(
    *,
    mapping: PageMappingResult,
    signals: PageExtractionSignals,
    parser_version: str,
    parsed_at: datetime,
) -> ParsedPage:
    """构造成功解析的页面记录。"""

    content_type = classify_page_content(
        normalized_text=(
            signals.normalized_text
        ),
        image_block_count=(
            signals.image_block_count
        ),
        embedded_image_count=(
            signals.embedded_image_count
        ),
        max_image_area_ratio=(
            signals.max_image_area_ratio
        ),
    )

    return ParsedPage(
        **mapping.model_dump(),
        raw_text=signals.raw_text,
        normalized_text=(
            signals.normalized_text
        ),
        raw_char_count=len(
            signals.raw_text
        ),
        normalized_char_count=len(
            signals.normalized_text
        ),
        text_block_count=(
            signals.text_block_count
        ),
        image_block_count=(
            signals.image_block_count
        ),
        embedded_image_count=(
            signals.embedded_image_count
        ),
        max_image_area_ratio=(
            signals.max_image_area_ratio
        ),
        content_type=content_type,
        parse_status=PageParseStatus.SUCCESS,
        parse_error=None,
        parser_name="pymupdf",
        parser_version=parser_version,
        parsed_at=parsed_at,
    )


def _build_error_page(
    *,
    mapping: PageMappingResult,
    error: Exception,
    parser_version: str,
    parsed_at: datetime,
) -> ParsedPage:
    """构造单页解析失败记录。"""

    return ParsedPage(
        **mapping.model_dump(),
        raw_text="",
        normalized_text="",
        raw_char_count=0,
        normalized_char_count=0,
        text_block_count=0,
        image_block_count=0,
        embedded_image_count=0,
        max_image_area_ratio=0,
        content_type=PageContentType.UNKNOWN,
        parse_status=(
            PageParseStatus.PARSE_ERROR
        ),
        parse_error=_format_parse_error(
            error
        ),
        parser_name="pymupdf",
        parser_version=parser_version,
        parsed_at=parsed_at,
    )


def parse_pdf_pages(
    *,
    manifest: DocumentManifest,
    page_mappings: Iterable[
        PageMappingSegment
    ],
    project_root: Path,
    pdf_pages: Iterable[int] | None = None,
    parsed_at: datetime | None = None,
) -> tuple[ParsedPage, ...]:
    """按页解析 Manifest 对应的实际 PDF。"""

    if (
        manifest.validation_status
        is not DocumentValidationStatus.VALID
    ):
        raise PageParserError(
            f"文档 '{manifest.document_id}' "
            "未通过文档接入检查"
        )

    resolved_pdf_path = (
        _resolve_manifest_pdf_path(
            manifest=manifest,
            project_root=project_root,
        )
    )

    current_sha256 = calculate_file_sha256(
        resolved_pdf_path
    )

    if current_sha256 != manifest.sha256:
        raise PageSourceHashMismatchError(
            "当前 PDF 的 SHA-256 与 Manifest 不一致："
            f"{manifest.document_id}"
        )

    selected_pdf_pages = (
        _normalize_page_selection(
            pdf_pages=pdf_pages,
            total_pdf_pages=(
                manifest.pdf_page_count
            ),
        )
    )

    mapping_resolver = PageMappingResolver(
        manifest=manifest,
        page_mappings=page_mappings,
    )

    parser_version = get_page_parser_version()

    parsed_at_value = (
        parsed_at
        if parsed_at is not None
        else datetime.now(timezone.utc)
    )

    try:
        document = pymupdf.open(
            resolved_pdf_path
        )
    except Exception as exc:
        raise PageSourceOpenError(
            "PDF 无法打开或已经损坏："
            f"{resolved_pdf_path}"
        ) from exc

    try:
        if not document.is_pdf:
            raise PageSourceOpenError(
                f"文件不是有效 PDF：{resolved_pdf_path}"
            )

        if document.needs_pass:
            raise PageSourceEncryptedError(
                f"PDF 需要密码：{resolved_pdf_path}"
            )

        if (
            document.page_count
            != manifest.pdf_page_count
        ):
            raise (
                PageSourcePageCountMismatchError(
                    "PDF 实际页数与 Manifest 不一致："
                    f"actual={document.page_count}, "
                    f"manifest={manifest.pdf_page_count}"
                )
            )

        parsed_pages: list[ParsedPage] = []

        for pdf_page in selected_pdf_pages:
            mapping = mapping_resolver.resolve(
                pdf_page
            )

            try:
                page = document.load_page(
                    pdf_page - 1
                )

                signals = _extract_page_signals(
                    page
                )

                parsed_page = _build_success_page(
                    mapping=mapping,
                    signals=signals,
                    parser_version=parser_version,
                    parsed_at=parsed_at_value,
                )

            except Exception as exc:
                parsed_page = _build_error_page(
                    mapping=mapping,
                    error=exc,
                    parser_version=parser_version,
                    parsed_at=parsed_at_value,
                )

            parsed_pages.append(
                parsed_page
            )

        return tuple(parsed_pages)

    finally:
        document.close()