from __future__ import annotations

import re
from dataclasses import dataclass

import hashlib
import json
from collections.abc import Iterable

from app.schemas.chunk import (
    Chunk,
    ChunkingConfig,
    FixedLengthChunkingConfig,
    ParagraphChunkingConfig,
    SectionParagraphChunkingConfig,
)
from app.schemas.enums import PageParseStatus
from app.schemas.page import ParsedPage
from app.schemas.page_dataset import (
    PageDatasetManifest,
)
from app.schemas.report import Report


class ChunkingError(ValueError):
    """Chunk 切分基础异常。"""


class ChunkSourceMismatchError(
    ChunkingError
):
    """页面、报告和页面数据集身份不一致。"""


class UnchunkablePageError(
    ChunkingError
):
    """页面解析失败，不能进入 Chunk 数据集。"""


class ChunkDatasetIdentityMismatchError(
    ChunkingError
):
    """传入的 Chunk 数据集 ID 与输入不一致。"""


_PARAGRAPH_SEPARATOR_PATTERN = re.compile(
    r"\n{2,}"
)


@dataclass(frozen=True, slots=True)
class ParagraphSpan:
    """一个页面内可解释的段落字符区间。"""

    paragraph_index: int
    start_char: int
    end_char: int

    @property
    def char_count(self) -> int:
        return self.end_char - self.start_char


_SECTION_LEVEL_1_PATTERN = re.compile(
    r"^第[一二三四五六七八九十百零〇0-9]+节"
)

_SECTION_LEVEL_2_PATTERN = re.compile(
    r"^[一二三四五六七八九十百零〇]+、"
)

_SECTION_LEVEL_3_PATTERN = re.compile(
    r"^[（(][一二三四五六七八九十百零〇0-9]+[）)]"
)


@dataclass(frozen=True, slots=True)
class SectionHeading:
    """页面中通过显式规则识别出的章节标题。"""

    title: str
    level: int
    start_char: int
    end_char: int

def _canonical_json_bytes(
    value: object,
) -> bytes:
    """生成可重复哈希的 JSON 字节。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def calculate_report_snapshot_sha256(
    report: Report,
) -> str:
    """计算写入 Chunk 的报告元数据快照哈希。"""

    payload = {
        "report_id": report.report_id,
        "company_id": report.company_id,
        "fiscal_year": report.fiscal_year,
        "report_type": (
            report.report_type.value
        ),
        "title": report.title,
    }

    return hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()


def build_chunk_dataset_id(
    *,
    report: Report,
    page_dataset_manifest: (
        PageDatasetManifest
    ),
    config: ChunkingConfig,
) -> str:
    """根据上游数据和切分规则生成稳定 ID。"""

    if (
        page_dataset_manifest.report_id
        != report.report_id
    ):
        raise ChunkSourceMismatchError(
            "PageDatasetManifest 与 Report "
            "的 report_id 不一致"
        )

    payload = {
        "page_dataset_id": (
            page_dataset_manifest.dataset_id
        ),
        "pages_jsonl_sha256": (
            page_dataset_manifest
            .pages_jsonl_sha256
        ),
        "report_snapshot_sha256": (
            calculate_report_snapshot_sha256(
                report
            )
        ),
        "chunk_schema_version": 1,
        "chunking_config": (
            config.model_dump(mode="json")
        ),
    }

    identity_sha256 = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()

    return (
        f"chunk_dataset_{report.report_id}_"
        f"{identity_sha256[:24]}"
    )


def _validate_sources(
    *,
    page: ParsedPage,
    report: Report,
    page_dataset_manifest: (
        PageDatasetManifest
    ),
) -> None:
    """检查 Chunk 的三个上游来源是否一致。"""

    if page.report_id != report.report_id:
        raise ChunkSourceMismatchError(
            "ParsedPage 与 Report 的 "
            "report_id 不一致"
        )

    if (
        page.report_id
        != page_dataset_manifest.report_id
    ):
        raise ChunkSourceMismatchError(
            "ParsedPage 与 PageDatasetManifest "
            "的 report_id 不一致"
        )

    if (
        page.document_id
        != page_dataset_manifest.document_id
    ):
        raise ChunkSourceMismatchError(
            "ParsedPage 与 PageDatasetManifest "
            "的 document_id 不一致"
        )


def _build_chunk_id(
    *,
    report_id: str,
    chunk_dataset_id: str,
    page_id: str,
    chunk_index: int,
    source_start_char: int,
    source_end_char: int,
    text_sha256: str,
) -> str:
    """生成与数据集、页面和字符边界绑定的 ID。"""

    payload = {
        "chunk_dataset_id": chunk_dataset_id,
        "page_id": page_id,
        "chunk_index": chunk_index,
        "source_start_char": (
            source_start_char
        ),
        "source_end_char": source_end_char,
        "text_sha256": text_sha256,
    }

    identity_sha256 = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()

    return (
        f"chunk_{report_id}_"
        f"{identity_sha256[:24]}"
    )


def _build_chunk_record(
    *,
    page: ParsedPage,
    report: Report,
    page_dataset_manifest: (
        PageDatasetManifest
    ),
    chunk_dataset_id: str,
    config: ChunkingConfig,
    chunk_index: int,
    source_start_char: int,
    source_end_char: int,
    paragraph_start_index: (
        int | None
    ) = None,
    paragraph_end_index: (
        int | None
    ) = None,
    section_path: tuple[str, ...] = (),
    section_source_page_id: (
        str | None
    ) = None,
    section_inherited: bool = False,
) -> Chunk:
    """根据页面中的准确字符区间构造 Chunk。"""

    chunk_text = page.normalized_text[
        source_start_char:
        source_end_char
    ]

    text_sha256 = hashlib.sha256(
        chunk_text.encode("utf-8")
    ).hexdigest()

    chunk_id = _build_chunk_id(
        report_id=report.report_id,
        chunk_dataset_id=chunk_dataset_id,
        page_id=page.page_id,
        chunk_index=chunk_index,
        source_start_char=source_start_char,
        source_end_char=source_end_char,
        text_sha256=text_sha256,
    )

    return Chunk(
        chunk_id=chunk_id,
        chunk_dataset_id=chunk_dataset_id,
        page_dataset_id=(
            page_dataset_manifest.dataset_id
        ),
        company_id=report.company_id,
        report_id=report.report_id,
        fiscal_year=report.fiscal_year,
        report_type=report.report_type,
        document_id=page.document_id,
        page_id=page.page_id,
        pdf_page=page.pdf_page,
        printed_page=page.printed_page,
        mapping_status=page.mapping_status,
        content_type=page.content_type,
        parse_status=page.parse_status,
        chunk_index=chunk_index,
        strategy=config.strategy,
        chunker_version=(
            config.chunker_version
        ),
        source_text_field=(
            config.source_text_field
        ),
        source_start_char=(
            source_start_char
        ),
        source_end_char=source_end_char,
        text=chunk_text,
        char_count=len(chunk_text),
        text_sha256=text_sha256,
        paragraph_start_index=(
            paragraph_start_index
        ),
        paragraph_end_index=(
            paragraph_end_index
        ),
        section_path=section_path,
        section_source_page_id=(
            section_source_page_id
        ),
        section_inherited=(
            section_inherited
        ),
    )


def build_fixed_length_chunks(
    *,
    page: ParsedPage,
    report: Report,
    page_dataset_manifest: (
        PageDatasetManifest
    ),
    chunk_dataset_id: str,
    config: FixedLengthChunkingConfig,
) -> tuple[Chunk, ...]:
    """在一个页面内部执行固定字符长度切分。"""

    _validate_sources(
        page=page,
        report=report,
        page_dataset_manifest=(
            page_dataset_manifest
        ),
    )

    expected_dataset_id = (
        build_chunk_dataset_id(
            report=report,
            page_dataset_manifest=(
                page_dataset_manifest
            ),
            config=config,
        )
    )

    if (
        chunk_dataset_id
        != expected_dataset_id
    ):
        raise (
            ChunkDatasetIdentityMismatchError(
                "chunk_dataset_id 与页面数据 "
                "和切分配置不一致"
            )
        )

    if (
        page.parse_status
        is not PageParseStatus.SUCCESS
    ):
        raise UnchunkablePageError(
            f"页面 '{page.page_id}' "
            "解析失败，不能切分"
        )

    if (
        page.content_type
        not in config.include_content_types
    ):
        return ()

    source_text = page.normalized_text

    if not source_text:
        return ()

    chunks: list[Chunk] = []

    source_start_char = 0
    chunk_index = 0

    while (
        source_start_char
        < len(source_text)
    ):
        source_end_char = min(
            source_start_char
            + config.max_chars,
            len(source_text),
        )

        chunks.append(
            _build_chunk_record(
                page=page,
                report=report,
                page_dataset_manifest=(
                    page_dataset_manifest
                ),
                chunk_dataset_id=(
                    chunk_dataset_id
                ),
                config=config,
                chunk_index=chunk_index,
                source_start_char=(
                    source_start_char
                ),
                source_end_char=(
                    source_end_char
                ),
            )
        )

        if (
            source_end_char
            == len(source_text)
        ):
            break

        source_start_char += (
            config.step_chars
        )

        chunk_index += 1

    return tuple(chunks)


def build_fixed_length_chunks_for_pages(
    *,
    pages: Iterable[ParsedPage],
    report: Report,
    page_dataset_manifest: (
        PageDatasetManifest
    ),
    config: FixedLengthChunkingConfig,
) -> tuple[Chunk, ...]:
    """按页切分多页，但绝不跨页面拼接。"""

    pages_tuple = tuple(pages)

    actual_pdf_pages = tuple(
        page.pdf_page
        for page in pages_tuple
    )

    expected_pdf_pages = tuple(
        sorted(set(actual_pdf_pages))
    )

    if (
        actual_pdf_pages
        != expected_pdf_pages
    ):
        raise ChunkSourceMismatchError(
            "页面必须按 PDF 页码升序排列，"
            "并且不能重复"
        )

    chunk_dataset_id = (
        build_chunk_dataset_id(
            report=report,
            page_dataset_manifest=(
                page_dataset_manifest
            ),
            config=config,
        )
    )

    chunks: list[Chunk] = []

    for page in pages_tuple:
        page_chunks = (
            build_fixed_length_chunks(
                page=page,
                report=report,
                page_dataset_manifest=(
                    page_dataset_manifest
                ),
                chunk_dataset_id=(
                    chunk_dataset_id
                ),
                config=config,
            )
        )

        chunks.extend(page_chunks)

    return tuple(chunks)


def detect_paragraph_spans(
    source_text: str,
) -> tuple[ParagraphSpan, ...]:
    """使用连续空白行识别页面内段落边界。

    分隔空白行不属于任何段落；单个换行仍保留在段落中。
    """

    if not source_text:
        return ()

    spans: list[ParagraphSpan] = []

    paragraph_start = 0
    paragraph_index = 0

    for match in (
        _PARAGRAPH_SEPARATOR_PATTERN
        .finditer(source_text)
    ):
        paragraph_end = match.start()

        if paragraph_end > paragraph_start:
            spans.append(
                ParagraphSpan(
                    paragraph_index=(
                        paragraph_index
                    ),
                    start_char=paragraph_start,
                    end_char=paragraph_end,
                )
            )

            paragraph_index += 1

        paragraph_start = match.end()

    if paragraph_start < len(source_text):
        spans.append(
            ParagraphSpan(
                paragraph_index=(
                    paragraph_index
                ),
                start_char=paragraph_start,
                end_char=len(source_text),
            )
        )

    return tuple(spans)


def build_paragraph_chunks(
    *,
    page: ParsedPage,
    report: Report,
    page_dataset_manifest: (
        PageDatasetManifest
    ),
    chunk_dataset_id: str,
    config: ParagraphChunkingConfig,
) -> tuple[Chunk, ...]:
    """在单页内按照完整段落组合生成 Chunk。"""

    _validate_sources(
        page=page,
        report=report,
        page_dataset_manifest=(
            page_dataset_manifest
        ),
    )

    expected_dataset_id = (
        build_chunk_dataset_id(
            report=report,
            page_dataset_manifest=(
                page_dataset_manifest
            ),
            config=config,
        )
    )

    if chunk_dataset_id != expected_dataset_id:
        raise (
            ChunkDatasetIdentityMismatchError(
                "chunk_dataset_id 与页面数据 "
                "和段落切分配置不一致"
            )
        )

    if (
        page.parse_status
        is not PageParseStatus.SUCCESS
    ):
        raise UnchunkablePageError(
            f"页面 '{page.page_id}' "
            "解析失败，不能切分"
        )

    if (
        page.content_type
        not in config.include_content_types
    ):
        return ()

    source_text = page.normalized_text

    if not source_text:
        return ()

    paragraph_spans = (
        detect_paragraph_spans(source_text)
    )

    if not paragraph_spans:
        return ()

    chunks: list[Chunk] = []

    paragraph_position = 0
    chunk_index = 0

    while paragraph_position < len(
        paragraph_spans
    ):
        first_span = paragraph_spans[
            paragraph_position
        ]

        # 单个段落本身超过 max_chars：
        # 在该段落内部回退为字符级切分。
        if first_span.char_count > config.max_chars:
            source_start_char = (
                first_span.start_char
            )

            while (
                source_start_char
                < first_span.end_char
            ):
                source_end_char = min(
                    source_start_char
                    + config.max_chars,
                    first_span.end_char,
                )

                chunks.append(
                    _build_chunk_record(
                        page=page,
                        report=report,
                        page_dataset_manifest=(
                            page_dataset_manifest
                        ),
                        chunk_dataset_id=(
                            chunk_dataset_id
                        ),
                        config=config,
                        chunk_index=chunk_index,
                        source_start_char=(
                            source_start_char
                        ),
                        source_end_char=(
                            source_end_char
                        ),
                        paragraph_start_index=(
                            first_span
                            .paragraph_index
                        ),
                        paragraph_end_index=(
                            first_span
                            .paragraph_index
                        ),
                    )
                )

                chunk_index += 1

                if (
                    source_end_char
                    == first_span.end_char
                ):
                    break

                source_start_char += (
                    config
                    .long_paragraph_step_chars
                )

            paragraph_position += 1
            continue

        # 尽可能将相邻完整段落装入同一个 Chunk。
        final_position = paragraph_position

        while (
            final_position + 1
            < len(paragraph_spans)
        ):
            candidate_end = (
                paragraph_spans[
                    final_position + 1
                ].end_char
            )

            candidate_length = (
                candidate_end
                - first_span.start_char
            )

            if candidate_length > config.max_chars:
                break

            final_position += 1

        final_span = paragraph_spans[
            final_position
        ]

        chunks.append(
            _build_chunk_record(
                page=page,
                report=report,
                page_dataset_manifest=(
                    page_dataset_manifest
                ),
                chunk_dataset_id=(
                    chunk_dataset_id
                ),
                config=config,
                chunk_index=chunk_index,
                source_start_char=(
                    first_span.start_char
                ),
                source_end_char=(
                    final_span.end_char
                ),
                paragraph_start_index=(
                    first_span.paragraph_index
                ),
                paragraph_end_index=(
                    final_span.paragraph_index
                ),
            )
        )

        chunk_index += 1

        if (
            final_position
            == len(paragraph_spans) - 1
        ):
            break

        next_position = (
            final_position
            + 1
            - config.overlap_paragraphs
        )

        # 即使 overlap_paragraphs 很大，
        # 也必须至少前进一个段落，避免死循环。
        paragraph_position = max(
            paragraph_position + 1,
            next_position,
        )

    return tuple(chunks)


def build_paragraph_chunks_for_pages(
    *,
    pages: Iterable[ParsedPage],
    report: Report,
    page_dataset_manifest: (
        PageDatasetManifest
    ),
    config: ParagraphChunkingConfig,
) -> tuple[Chunk, ...]:
    """逐页执行段落切分，禁止跨页组合。"""

    pages_tuple = tuple(pages)

    actual_pdf_pages = tuple(
        page.pdf_page
        for page in pages_tuple
    )

    expected_pdf_pages = tuple(
        sorted(set(actual_pdf_pages))
    )

    if actual_pdf_pages != expected_pdf_pages:
        raise ChunkSourceMismatchError(
            "页面必须按 PDF 页码升序排列，"
            "并且不能重复"
        )

    chunk_dataset_id = (
        build_chunk_dataset_id(
            report=report,
            page_dataset_manifest=(
                page_dataset_manifest
            ),
            config=config,
        )
    )

    chunks: list[Chunk] = []

    for page in pages_tuple:
        chunks.extend(
            build_paragraph_chunks(
                page=page,
                report=report,
                page_dataset_manifest=(
                    page_dataset_manifest
                ),
                chunk_dataset_id=(
                    chunk_dataset_id
                ),
                config=config,
            )
        )

    return tuple(chunks)


def build_chunks_for_pages(
    *,
    pages: Iterable[ParsedPage],
    report: Report,
    page_dataset_manifest: (
        PageDatasetManifest
    ),
    config: ChunkingConfig,
) -> tuple[Chunk, ...]:
    """按照显式策略选择确定性 Chunker。"""

    if isinstance(
        config,
        FixedLengthChunkingConfig,
    ):
        return (
            build_fixed_length_chunks_for_pages(
                pages=pages,
                report=report,
                page_dataset_manifest=(
                    page_dataset_manifest
                ),
                config=config,
            )
        )

    if isinstance(
        config,
        SectionParagraphChunkingConfig,
    ):
        return (
            build_section_paragraph_chunks_for_pages(
                pages=pages,
                report=report,
                page_dataset_manifest=(
                    page_dataset_manifest
                ),
                config=config,
            )
        )

    if isinstance(
        config,
        ParagraphChunkingConfig,
    ):
        return build_paragraph_chunks_for_pages(
            pages=pages,
            report=report,
            page_dataset_manifest=(
                page_dataset_manifest
            ),
            config=config,
        )

    raise ChunkingError(
        "不支持的 ChunkingConfig 类型："
        f"{type(config).__name__}"
    )


def detect_section_headings(
    source_text: str,
    *,
    max_heading_chars: int,
) -> tuple[SectionHeading, ...]:
    """按行识别企业年报中的保守章节标题。"""

    headings: list[SectionHeading] = []

    cursor = 0

    for raw_line in source_text.splitlines(
        keepends=True
    ):
        line_without_newline = raw_line.rstrip(
            "\r\n"
        )

        stripped = line_without_newline.strip()

        leading_whitespace_count = (
            len(line_without_newline)
            - len(line_without_newline.lstrip())
        )

        start_char = (
            cursor + leading_whitespace_count
        )

        end_char = (
            start_char + len(stripped)
        )

        cursor += len(raw_line)

        if not stripped:
            continue

        if len(stripped) > max_heading_chars:
            continue

        if stripped.endswith(
            ("。", "；", "，", ".", ";", ",")
        ):
            continue

        level: int | None = None

        if _SECTION_LEVEL_1_PATTERN.match(
            stripped
        ):
            level = 1

        elif _SECTION_LEVEL_2_PATTERN.match(
            stripped
        ):
            level = 2

        elif _SECTION_LEVEL_3_PATTERN.match(
            stripped
        ):
            level = 3

        if level is None:
            continue

        headings.append(
            SectionHeading(
                title=stripped,
                level=level,
                start_char=start_char,
                end_char=end_char,
            )
        )

    return tuple(headings)


SectionState = dict[
    int,
    tuple[str, str],
]


def _apply_section_heading(
    *,
    state: SectionState,
    heading: SectionHeading,
    page_id: str,
) -> None:
    """应用标题并清除同级及更低级旧标题。"""

    obsolete_levels = [
        level
        for level in state
        if level >= heading.level
    ]

    for level in obsolete_levels:
        del state[level]

    state[heading.level] = (
        heading.title,
        page_id,
    )


def _resolve_section_metadata(
    *,
    initial_state: SectionState,
    headings: tuple[
        SectionHeading,
        ...,
    ],
    position: int,
    page_id: str,
) -> tuple[
    tuple[str, ...],
    str | None,
    bool,
]:
    """解析指定字符位置生效的章节上下文。"""

    state = dict(initial_state)

    for heading in headings:
        if heading.start_char > position:
            break

        _apply_section_heading(
            state=state,
            heading=heading,
            page_id=page_id,
        )

    if not state:
        return (), None, False

    ordered_levels = sorted(state)

    section_path = tuple(
        state[level][0]
        for level in ordered_levels
    )

    deepest_level = ordered_levels[-1]

    source_page_id = state[
        deepest_level
    ][1]

    return (
        section_path,
        source_page_id,
        source_page_id != page_id,
    )


def _build_section_paragraph_chunks_for_page(
    *,
    page: ParsedPage,
    report: Report,
    page_dataset_manifest: (
        PageDatasetManifest
    ),
    chunk_dataset_id: str,
    config: SectionParagraphChunkingConfig,
    initial_section_state: SectionState,
) -> tuple[
    tuple[Chunk, ...],
    SectionState,
]:
    """为一个页面生成带章节元数据的段落 Chunk。"""

    paragraph_config = (
        ParagraphChunkingConfig(
            max_chars=config.max_chars,
            overlap_paragraphs=(
                config.overlap_paragraphs
            ),
            long_paragraph_overlap_chars=(
                config
                .long_paragraph_overlap_chars
            ),
            include_content_types=(
                config.include_content_types
            ),
        )
    )

    paragraph_dataset_id = (
        build_chunk_dataset_id(
            report=report,
            page_dataset_manifest=(
                page_dataset_manifest
            ),
            config=paragraph_config,
        )
    )

    template_chunks = build_paragraph_chunks(
        page=page,
        report=report,
        page_dataset_manifest=(
            page_dataset_manifest
        ),
        chunk_dataset_id=(
            paragraph_dataset_id
        ),
        config=paragraph_config,
    )

    headings = detect_section_headings(
        page.normalized_text,
        max_heading_chars=(
            config.max_heading_chars
        ),
    )

    chunks: list[Chunk] = []

    chunk_index = 0

    for template_chunk in template_chunks:
        boundaries = {
            template_chunk.source_start_char,
            template_chunk.source_end_char,
        }

        for heading in headings:
            if (
                template_chunk.source_start_char
                < heading.start_char
                < template_chunk.source_end_char
            ):
                boundaries.add(
                    heading.start_char
                )

        ordered_boundaries = sorted(
            boundaries
        )

        for boundary_index in range(
            len(ordered_boundaries) - 1
        ):
            source_start_char = (
                ordered_boundaries[
                    boundary_index
                ]
            )

            source_end_char = (
                ordered_boundaries[
                    boundary_index + 1
                ]
            )

            if (
                source_end_char
                <= source_start_char
            ):
                continue

            (
                section_path,
                section_source_page_id,
                section_inherited,
            ) = _resolve_section_metadata(
                initial_state=(
                    initial_section_state
                ),
                headings=headings,
                position=source_start_char,
                page_id=page.page_id,
            )

            chunks.append(
                _build_chunk_record(
                    page=page,
                    report=report,
                    page_dataset_manifest=(
                        page_dataset_manifest
                    ),
                    chunk_dataset_id=(
                        chunk_dataset_id
                    ),
                    config=config,
                    chunk_index=chunk_index,
                    source_start_char=(
                        source_start_char
                    ),
                    source_end_char=(
                        source_end_char
                    ),
                    paragraph_start_index=(
                        template_chunk
                        .paragraph_start_index
                    ),
                    paragraph_end_index=(
                        template_chunk
                        .paragraph_end_index
                    ),
                    section_path=section_path,
                    section_source_page_id=(
                        section_source_page_id
                    ),
                    section_inherited=(
                        section_inherited
                    ),
                )
            )

            chunk_index += 1

    final_state = dict(
        initial_section_state
    )

    for heading in headings:
        _apply_section_heading(
            state=final_state,
            heading=heading,
            page_id=page.page_id,
        )

    return tuple(chunks), final_state


def build_section_paragraph_chunks_for_pages(
    *,
    pages: Iterable[ParsedPage],
    report: Report,
    page_dataset_manifest: (
        PageDatasetManifest
    ),
    config: SectionParagraphChunkingConfig,
) -> tuple[Chunk, ...]:
    """按页面顺序生成可继承章节上下文的 Chunk。"""

    pages_tuple = tuple(pages)

    actual_pdf_pages = tuple(
        page.pdf_page
        for page in pages_tuple
    )

    expected_pdf_pages = tuple(
        sorted(set(actual_pdf_pages))
    )

    if actual_pdf_pages != expected_pdf_pages:
        raise ChunkSourceMismatchError(
            "页面必须按 PDF 页码升序排列，"
            "并且不能重复"
        )

    chunk_dataset_id = (
        build_chunk_dataset_id(
            report=report,
            page_dataset_manifest=(
                page_dataset_manifest
            ),
            config=config,
        )
    )

    all_chunks: list[Chunk] = []

    section_state: SectionState = {}

    for page in pages_tuple:
        (
            page_chunks,
            final_state,
        ) = (
            _build_section_paragraph_chunks_for_page(
                page=page,
                report=report,
                page_dataset_manifest=(
                    page_dataset_manifest
                ),
                chunk_dataset_id=(
                    chunk_dataset_id
                ),
                config=config,
                initial_section_state=(
                    section_state
                ),
            )
        )

        all_chunks.extend(page_chunks)

        if config.inherit_section_across_pages:
            section_state = final_state
        else:
            section_state = {}

    return tuple(all_chunks)