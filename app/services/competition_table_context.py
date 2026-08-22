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


def _extract_short_table_title(
    *,
    table_block: CompetitionTextBlock,
    nearby_text: tuple[str, ...],
) -> str | None:
    """
    暂时保留V1的独立短标题能力。

    跨主表标题继承将在4C3实现。
    """

    candidates = (
        table_block.text,
    ) + nearby_text

    for candidate in candidates:
        # 整张表格正文不能作为标题。
        if (
            "\n" in candidate
            or "\r" in candidate
        ):
            continue

        normalized = (
            _normalize_metadata_text(
                candidate
            )
        )

        if (
            normalized.startswith("表")
            and len(normalized) < 80
        ):
            return normalized

    return None

def build_table_context(
    *,
    table_block: CompetitionTextBlock,
    context: CompetitionRegulatoryContext,
    nearby_text: tuple[str, ...] = (),
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
            _extract_short_table_title(
                table_block=table_block,
                nearby_text=nearby_text,
            )
        ),
        unit=unit,
        frequency=frequency,
        purpose=purpose,
        content=content,
        scope=scope,
        format=table_format,
    )