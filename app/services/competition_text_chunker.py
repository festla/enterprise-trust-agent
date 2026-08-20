from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from app.schemas.competition_chunk import (
    CompetitionChunkSourceSpan,
    CompetitionTextChunk,
)
from app.schemas.competition_text import (
    CompetitionTextBlock,
    CompetitionTextDocument,
)
from app.services.competition_regulatory_context import (
    CompetitionRegulatoryContext,
    CompetitionRegulatoryContextTracker,
)


DEFAULT_MAX_CHARS = 900

MIN_SPLIT_RATIO = 0.55


# ============================================================
# Internal Text Unit
#
# 一个 Unit 是 Chunker 的最小输入：
#
# Word:
#     一个 Paragraph
#
# PDF:
#     Page Block 中的一行
#
# 长 Unit 还会进一步按照标点切分。
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class _TextUnit:
    text: str

    block_id: str
    block_index: int

    start_char: int
    end_char: int

    context: CompetitionRegulatoryContext

    # 当前 Unit 是否直接包含：
    #
    # 第十二条……
    #
    # 而不是仅仅从前文继承 Article。
    explicit_article: bool = False

    page: int | None = None

    paragraph_index: int | None = None


# ============================================================
# Chunk Buffer
# ============================================================


@dataclass(
    slots=True,
)
class _ChunkBuffer:
    units: list[_TextUnit]

    def __init__(
        self,
    ) -> None:
        self.units = []

    @property
    def empty(
        self,
    ) -> bool:
        return not self.units

    @property
    def text(
        self,
    ) -> str:
        return "\n".join(
            unit.text
            for unit in self.units
        )

    @property
    def char_count(
        self,
    ) -> int:
        return len(
            self.text
        )

    @property
    def context(
        self,
    ) -> (
        CompetitionRegulatoryContext
        | None
    ):
        if not self.units:
            return None

        return self.units[0].context

    def clear(
        self,
    ) -> None:
        self.units.clear()


# ============================================================
# Helpers
# ============================================================


def _sha256_text(
    text: str,
) -> str:
    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def _build_chunk_id(
    *,
    doc_id: str,
    chunk_index: int,
) -> str:
    return (
        f"chunk:{doc_id}:"
        f"{chunk_index:05d}"
    )


def _context_key(
    context: CompetitionRegulatoryContext,
) -> tuple[
    tuple[str, ...],
    str | None,
    tuple[str, ...],
]:
    """
    article_inherited 不参与上下文分组。

    原因：

    第十二条 ……        explicit article
    下一段正文          inherited article

    二者仍然属于同一个 Article，
    应允许合并进一个 Chunk。
    """

    return (
        context.section_path,
        context.article,
        context.item_path,
    )


def _same_context(
    left: CompetitionRegulatoryContext,
    right: CompetitionRegulatoryContext,
) -> bool:
    return (
        _context_key(left)
        == _context_key(right)
    )


# ============================================================
# PDF Line Span
# ============================================================


def _iter_non_empty_line_spans(
    text: str,
):
    """
    将 PDF page_text 切成行，同时保留它在原 Page Block
    中的精确字符位置。

    输出：

        line_text
        start_char
        end_char

    start/end 对应原 CompetitionTextBlock.text。
    """

    cursor = 0

    for raw_line in (
        text.splitlines(
            keepends=True
        )
    ):
        # 去掉换行符，
        # 但暂时不改变原始正文。
        without_newline = (
            raw_line.rstrip(
                "\r\n"
            )
        )

        leading_count = (
            len(without_newline)
            - len(
                without_newline.lstrip()
            )
        )

        trailing_stripped = (
            without_newline.rstrip()
        )

        if not trailing_stripped.strip():
            cursor += len(
                raw_line
            )
            continue

        start_char = (
            cursor
            + leading_count
        )

        end_char = (
            cursor
            + len(
                trailing_stripped
            )
        )

        line_text = text[
            start_char:end_char
        ]

        if line_text.strip():
            yield (
                line_text,
                start_char,
                end_char,
            )

        cursor += len(
            raw_line
        )

    # splitlines(keepends=True) 对没有任何换行的字符串
    # 正常也会产生一行，因此无需额外 fallback。


# ============================================================
# Oversized Unit Split
# ============================================================


_SENTENCE_BOUNDARY_PATTERN = re.compile(
    r"[。！？；!?;]"
)


def _choose_split_position(
    text: str,
    *,
    max_chars: int,
) -> int:
    """
    尽量在中文/英文句末标点切分。

    如果 max_chars 前完全没有合理标点，
    才进行硬切。
    """

    if len(text) <= max_chars:
        return len(
            text
        )

    minimum = int(
        max_chars
        * MIN_SPLIT_RATIO
    )

    candidate = None

    for match in (
        _SENTENCE_BOUNDARY_PATTERN
        .finditer(
            text,
            0,
            max_chars,
        )
    ):
        position = (
            match.end()
        )

        if position >= minimum:
            candidate = (
                position
            )

    if candidate is not None:
        return candidate

    return max_chars


def _split_unit(
    unit: _TextUnit,
    *,
    max_chars: int,
) -> tuple[
    _TextUnit,
    ...,
]:
    """
    单个 Paragraph / PDF line 本身超过 max_chars 时，
    保守拆分。

    Source span 会同步缩小，
    所以可追溯性不会丢。
    """

    if len(
        unit.text
    ) <= max_chars:
        return (
            unit,
        )

    result: list[
        _TextUnit
    ] = []

    local_start = 0

    first_piece = True

    while (
        local_start
        < len(unit.text)
    ):
        remaining = (
            unit.text[
                local_start:
            ]
        )

        split_length = (
            _choose_split_position(
                remaining,
                max_chars=max_chars,
            )
        )

        local_end = (
            local_start
            + split_length
        )

        piece_text = (
            unit.text[
                local_start:
                local_end
            ]
        )

        result.append(
            _TextUnit(
                text=piece_text,
                block_id=(
                    unit.block_id
                ),
                block_index=(
                    unit.block_index
                ),
                start_char=(
                    unit.start_char
                    + local_start
                ),
                end_char=(
                    unit.start_char
                    + local_end
                ),
                context=(
                    unit.context
                ),

                # 如果原 Unit 是 Article 起始行，
                # 只有第一片真正包含显式 Article。
                explicit_article=(
                    unit.explicit_article
                    and first_piece
                ),
                page=unit.page,
                paragraph_index=(
                    unit.paragraph_index
                ),
            )
        )

        first_piece = False

        local_start = (
            local_end
        )

    return tuple(
        result
    )


# ============================================================
# Word Paragraph -> Unit
# ============================================================


def _word_block_to_units(
    block: CompetitionTextBlock,
    *,
    tracker: CompetitionRegulatoryContextTracker,
    max_chars: int,
) -> tuple[
    _TextUnit,
    ...,
]:
    update = tracker.consume(
        block.text,
        outline_level=(
            block.outline_level
        ),
    )

    # ========================================================
    # 纯结构标题：
    #
    # 信用风险
    # 第一章 总则
    #
    # 本身不产生正文 Chunk，
    # 只更新 Tracker。
    # ========================================================

    if (
        update.is_structure_only
    ):
        return ()

    marker = update.marker

    explicit_article = bool(
        marker is not None
        and marker.marker_type
        == "article"
    )

    unit = _TextUnit(
        text=block.text,
        block_id=block.block_id,
        block_index=(
            block.block_index
        ),
        start_char=0,
        end_char=len(
            block.text
        ),
        context=update.context,
        explicit_article=(
            explicit_article
        ),
        paragraph_index=(
            block.paragraph_index
        ),
    )

    return _split_unit(
        unit,
        max_chars=max_chars,
    )


# ============================================================
# PDF Page -> Units
# ============================================================


def _pdf_block_to_units(
    block: CompetitionTextBlock,
    *,
    tracker: CompetitionRegulatoryContextTracker,
    max_chars: int,
) -> tuple[
    _TextUnit,
    ...,
]:
    result: list[
        _TextUnit
    ] = []

    for (
        line_text,
        start_char,
        end_char,
    ) in (
        _iter_non_empty_line_spans(
            block.text
        )
    ):
        update = (
            tracker.consume(
                line_text
            )
        )

        if (
            update.is_structure_only
        ):
            continue

        marker = update.marker

        explicit_article = bool(
            marker is not None
            and marker.marker_type
            == "article"
        )

        unit = _TextUnit(
            text=line_text,
            block_id=(
                block.block_id
            ),
            block_index=(
                block.block_index
            ),
            start_char=(
                start_char
            ),
            end_char=(
                end_char
            ),
            context=(
                update.context
            ),
            explicit_article=(
                explicit_article
            ),
            page=block.page,
        )

        result.extend(
            _split_unit(
                unit,
                max_chars=max_chars,
            )
        )

    return tuple(
        result
    )


# ============================================================
# Buffer -> CompetitionTextChunk
# ============================================================


def _build_chunk_from_buffer(
    *,
    buffer: _ChunkBuffer,
    document: CompetitionTextDocument,
    chunk_index: int,
) -> CompetitionTextChunk:
    if buffer.empty:
        raise ValueError(
            "不能从空 Buffer 构造 Chunk"
        )

    units = tuple(
        buffer.units
    )

    text = "\n".join(
        unit.text
        for unit in units
    )

    context = units[
        0
    ].context

    source_spans = tuple(
        CompetitionChunkSourceSpan(
            block_id=unit.block_id,
            block_index=(
                unit.block_index
            ),
            start_char=(
                unit.start_char
            ),
            end_char=(
                unit.end_char
            ),
        )
        for unit in units
    )

    contains_explicit_article = any(
        unit.explicit_article
        for unit
        in units
    )

    # ========================================================
    # Location
    # ========================================================

    page_values = [
        unit.page
        for unit
        in units
        if unit.page is not None
    ]

    paragraph_values = [
        unit.paragraph_index
        for unit
        in units
        if (
            unit.paragraph_index
            is not None
        )
    ]

    if (
        document.source
        .source_type
        == "pdf"
    ):
        if not page_values:
            raise ValueError(
                "PDF Chunk 缺少 page"
            )

        page_start = min(
            page_values
        )

        page_end = max(
            page_values
        )

        paragraph_start_index = None
        paragraph_end_index = None

    else:
        if not paragraph_values:
            raise ValueError(
                "Word text Chunk "
                "缺少 paragraph_index"
            )

        page_start = None
        page_end = None

        paragraph_start_index = min(
            paragraph_values
        )

        paragraph_end_index = max(
            paragraph_values
        )

    return CompetitionTextChunk(
        chunk_id=_build_chunk_id(
            doc_id=(
                document.source.doc_id
            ),
            chunk_index=(
                chunk_index
            ),
        ),
        source_id=(
            document.source.source_id
        ),
        doc_id=(
            document.source.doc_id
        ),
        source_type=(
            document.source.source_type
        ),
        chunk_index=chunk_index,
        chunk_type="text",
        source_spans=(
            source_spans
        ),
        text=text,
        char_count=len(
            text
        ),
        text_sha256=(
            _sha256_text(
                text
            )
        ),
        section_path=(
            context.section_path
        ),
        article=(
            context.article
        ),
        item_path=(
            context.item_path
        ),

        # 有 Article 但当前 Chunk 没包含 Article 起始行，
        # 才叫 inherited。
        article_inherited=(
            context.article
            is not None
            and not (
                contains_explicit_article
            )
        ),
        page_start=(
            page_start
        ),
        page_end=(
            page_end
        ),
        paragraph_start_index=(
            paragraph_start_index
        ),
        paragraph_end_index=(
            paragraph_end_index
        ),
    )


# ============================================================
# Public API
# ============================================================


def build_competition_text_chunks(
    document: CompetitionTextDocument,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> tuple[
    CompetitionTextChunk,
    ...,
]:
    """
    Competition 文本 Chunk Builder V1。

    当前处理：

        DOCX paragraph
        PDF page_text

    当前跳过：

        DOCX table

    Table 在 Step 3C.5c 单独实现。

    核心规则：

    1. Regulatory Context 不同 -> flush
    2. Table boundary -> flush
    3. 超过 max_chars -> flush
    4. 连续相同 Context -> 合并
    """

    if max_chars <= 0:
        raise ValueError(
            "max_chars 必须大于 0"
        )

    tracker = (
        CompetitionRegulatoryContextTracker()
    )

    buffer = _ChunkBuffer()

    chunks: list[
        CompetitionTextChunk
    ] = []

    def flush() -> None:
        if buffer.empty:
            return

        chunk = (
            _build_chunk_from_buffer(
                buffer=buffer,
                document=document,
                chunk_index=len(
                    chunks
                ),
            )
        )

        chunks.append(
            chunk
        )

        buffer.clear()

    for block in (
        document.blocks
    ):
        # ====================================================
        # Table
        #
        # 当前 Table 不进入 Text Chunk，
        # 但必须作为一个明确边界。
        #
        # 否则：
        #
        # 表格前正文
        # TABLE
        # 表格后填写说明
        #
        # 可能错误合成一个 Text Chunk。
        # ====================================================

        if (
            block.block_type
            == "table"
        ):
            flush()
            continue

        # ====================================================
        # Word
        # ====================================================

        if (
            block.block_type
            == "paragraph"
        ):
            units = (
                _word_block_to_units(
                    block,
                    tracker=tracker,
                    max_chars=(
                        max_chars
                    ),
                )
            )

        # ====================================================
        # PDF
        # ====================================================

        elif (
            block.block_type
            == "page_text"
        ):
            units = (
                _pdf_block_to_units(
                    block,
                    tracker=tracker,
                    max_chars=(
                        max_chars
                    ),
                )
            )

        else:
            raise ValueError(
                "Unsupported TextBlock type: "
                f"{block.block_type}"
            )

        for unit in units:
            if buffer.empty:
                buffer.units.append(
                    unit
                )

                continue

            current_context = (
                buffer.context
            )

            if current_context is None:
                raise RuntimeError(
                    "非空 Buffer 缺少 context"
                )

            # =================================================
            # Context Boundary
            # =================================================

            if not _same_context(
                current_context,
                unit.context,
            ):
                flush()

                buffer.units.append(
                    unit
                )

                continue

            # =================================================
            # Size Boundary
            #
            # +1 是两个 Unit 中间添加的 "\n"。
            # =================================================

            candidate_size = (
                buffer.char_count
                + 1
                + len(
                    unit.text
                )
            )

            if (
                candidate_size
                > max_chars
            ):
                flush()

            buffer.units.append(
                unit
            )

    flush()

    return tuple(
        chunks
    )