from __future__ import annotations
import re
from dataclasses import dataclass


from app.schemas.competition_text import (
    CompetitionTextBlock,
)
from app.services.competition_regulatory_context import (
    CompetitionRegulatoryContext,
)


@dataclass(
    frozen=True,
    slots=True,
)
class CompetitionTableContext:
    """
    表格上下文。

    表格本身只有二维数据，
    Context 负责回答：

    这个表是什么？
    属于哪里？
    描述什么？
    """

    section_path: tuple[str, ...]

    article: str | None

    item_path: tuple[str, ...]

    title: str | None

    unit: str | None

    frequency: str | None

    purpose: str | None

    content: str | None

    scope: str | None

    format: str | None

MAX_METADATA_CHARS = 500
MAX_SHORT_METADATA_CHARS = 120


def _normalize_metadata_text(
    text: str,
) -> str:
    return " ".join(
        text.split()
    ).strip()

_EXPLICIT_TABLE_TITLE_PATTERN = re.compile(
    r"^(?:[（(]"
    r"[一二三四五六七八九十百零〇0-9]+"
    r"[）)]\s*)?"
    r"表(?:格)?\s*"
    r"[A-Za-z0-9一二三四五六七八九十]+"
    r"(?:[-./]"
    r"[A-Za-z0-9一二三四五六七八九十]+)*"
    r"\s*[：:]\s*"
    r"\S.+$"
)


def extract_explicit_table_title(
    text: str,
) -> str | None:
    """
    识别具有明确结构的表格标题，例如：

        （一）表格KM1：监管并表关键审慎监管指标
        （二）表格 CR1：资产质量
        表CR3：资本充足率

    普通正文、概览标题和表内字段不会被识别为表格标题。
    """

    normalized = (
        _normalize_metadata_text(
            text
        )
    )

    if (
        not normalized
        or len(normalized) > 200
    ):
        return None

    if (
        _EXPLICIT_TABLE_TITLE_PATTERN
        .fullmatch(normalized)
        is None
    ):
        return None

    return normalized

def _collect_metadata_candidates(
    *,
    table_block: CompetitionTextBlock,
    nearby_text: tuple[str, ...],
) -> tuple[str, ...]:
    """
    元数据候选来源：

    1. 表格中的独立单元格；
    2. 表格附近的独立段落。

    不使用table_block.text整体进行元数据匹配，
    避免把整张表保存进purpose等字段。
    """

    result: list[str] = []
    seen: set[str] = set()

    for row in table_block.table_rows:
        for cell in row:
            normalized = (
                _normalize_metadata_text(
                    cell
                )
            )

            if (
                not normalized
                or normalized in seen
            ):
                continue

            seen.add(normalized)
            result.append(normalized)

    for text in nearby_text:
        normalized = (
            _normalize_metadata_text(
                text
            )
        )

        if (
            not normalized
            or normalized in seen
        ):
            continue

        seen.add(normalized)
        result.append(normalized)

    return tuple(result)


def _extract_labeled_value(
    candidates: tuple[str, ...],
    *,
    labels: tuple[str, ...],
    max_chars: int,
) -> tuple[str, str] | None:
    """
    返回：

        完整标签文本
        去掉标签后的值

    例如：

        频率：季度。
            ->
        ("频率：季度。", "季度")
    """

    label_pattern = "|".join(
        re.escape(label)
        for label in labels
    )

    pattern = re.compile(
        rf"^(?:{label_pattern})"
        r"\s*[：:]\s*(.+)$"
    )

    for candidate in candidates:
        normalized = (
            _normalize_metadata_text(
                candidate
            )
        )

        if (
            not normalized
            or len(normalized)
            > max_chars
        ):
            continue

        match = pattern.match(
            normalized
        )

        if match is None:
            continue

        value = (
            _normalize_metadata_text(
                match.group(1)
            )
            .rstrip("。；; ")
        )

        if not value:
            continue

        return (
            normalized,
            value,
        )

    return None


def _canonicalize_frequency(
    value: str,
) -> str:
    """
    table_frequency保留可过滤的标准类别。

    更详细的更新条件仍保留在原始表格正文中。
    """

    for prefix, canonical in (
        ("半年度", "半年"),
        ("半年", "半年"),
        ("季度", "季度"),
        ("年度", "年度"),
        ("月度", "月度"),
    ):
        if value.startswith(prefix):
            return canonical

    return value


def _canonicalize_format(
    value: str,
) -> str:
    """
    将：

        固定。如有其他分类……
        可变。

    规范为：

        固定
        可变
    """

    if value.startswith("固定"):
        return "固定"

    if value.startswith("可变"):
        return "可变"

    return value


def _resolve_table_title(
    *,
    table_block: CompetitionTextBlock,
    title_hint: str | None,
) -> str | None:
    """
    标题来源优先级：

    1. 表格自身就是一个独立标题；
    2. 文档遍历过程中继承的最近明确表格标题。

    不从表格后方的 nearby_text 获取标题，
    避免错误绑定到下一张表的标题。
    """

    direct_title = (
        extract_explicit_table_title(
            table_block.text
        )
    )

    if direct_title is not None:
        return direct_title

    if title_hint is None:
        return None

    return extract_explicit_table_title(
        title_hint
    )

def build_table_context(
    *,
    table_block: CompetitionTextBlock,
    context: CompetitionRegulatoryContext,
    nearby_text: tuple[str, ...] = (),
    title_hint: str | None = None,
) -> CompetitionTableContext:
    """
    从明确的表格标签和法规上下文构造Table Context。

    V2规则：

    1. 只从独立单元格或附近段落提取元数据；
    2. 标签必须位于文本开头；
    3. 不扫描整张表格关键词；
    4. 没有明确证据时保留None；
    5. frequency和format保存标准类别。
    """

    candidates = (
        _collect_metadata_candidates(
            table_block=table_block,
            nearby_text=nearby_text,
        )
    )

    unit_match = (
        _extract_labeled_value(
            candidates,
            labels=("单位",),
            max_chars=(
                MAX_SHORT_METADATA_CHARS
            ),
        )
    )

    frequency_match = (
        _extract_labeled_value(
            candidates,
            labels=("频率",),
            max_chars=(
                MAX_SHORT_METADATA_CHARS
            ),
        )
    )

    purpose_match = (
        _extract_labeled_value(
            candidates,
            labels=("目的",),
            max_chars=MAX_METADATA_CHARS,
        )
    )

    content_match = (
        _extract_labeled_value(
            candidates,
            labels=("内容",),
            max_chars=MAX_METADATA_CHARS,
        )
    )

    scope_match = (
        _extract_labeled_value(
            candidates,
            labels=(
                "适用范围",
                "范围",
            ),
            max_chars=MAX_METADATA_CHARS,
        )
    )

    format_match = (
        _extract_labeled_value(
            candidates,
            labels=("格式",),
            max_chars=MAX_METADATA_CHARS,
        )
    )

    unit = (
        unit_match[0]
        if unit_match is not None
        else None
    )

    frequency = (
        _canonicalize_frequency(
            frequency_match[1]
        )
        if frequency_match is not None
        else None
    )

    purpose = (
        purpose_match[0]
        if purpose_match is not None
        else None
    )

    content = (
        content_match[0]
        if content_match is not None
        else None
    )

    scope = (
        scope_match[0]
        if scope_match is not None
        else None
    )

    table_format = (
        _canonicalize_format(
            format_match[1]
        )
        if format_match is not None
        else "table"
    )

    return CompetitionTableContext(
        section_path=(
            context.section_path
        ),
        article=context.article,
        item_path=context.item_path,
        title=(
            _resolve_table_title(
                table_block=table_block,
                title_hint=title_hint,
            )
        ),
        unit=unit,
        frequency=frequency,
        purpose=purpose,
        content=content,
        scope=scope,
        format=table_format,
    )