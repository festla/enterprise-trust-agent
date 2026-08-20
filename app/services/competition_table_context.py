from __future__ import annotations

from dataclasses import dataclass

from app.schemas.competition_chunk import (
    CompetitionTableChunk,
)
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


def build_table_context(
    *,
    table_block: CompetitionTextBlock,
    context: CompetitionRegulatoryContext,
    nearby_text: tuple[str, ...] = (),
) -> CompetitionTableContext:
    """
    根据：

    1. 当前 Regulatory Context
    2. 表格自身信息
    3. 表格附近文本

    构造 Table Context。

    V1:

    - context 来自 tracker
    - metadata 先从 nearby text 简单提取

    后续可以增强为专门 extractor。
    """

    title = None

    unit = None

    frequency = None

    purpose = None

    content = None

    scope = None

    table_format = "table"


    candidates = (
        table_block.text,
    ) + nearby_text


    for text in candidates:

        normalized = (
            text.strip()
        )

        if not normalized:
            continue


        # -------------------------
        # 标题
        # -------------------------

        if (
            normalized.startswith(
                "表"
            )
            and len(normalized)
            < 80
        ):
            title = normalized


        # -------------------------
        # 单位
        # -------------------------

        if (
            "单位" in normalized
        ):
            unit = normalized


        # -------------------------
        # 频率
        # -------------------------

        for keyword in (
            "季度",
            "年度",
            "月度",
            "半年度",
        ):
            if keyword in normalized:
                frequency = keyword


        # -------------------------
        # 用途
        # -------------------------

        if (
            normalized.startswith(
                "目的"
            )
        ):
            purpose = normalized


        # -------------------------
        # 内容
        # -------------------------

        if (
            normalized.startswith(
                "内容"
            )
        ):
            content = normalized


        # -------------------------
        # 范围
        # -------------------------

        if (
            normalized.startswith(
                "范围"
            )
        ):
            scope = normalized


    return CompetitionTableContext(

        section_path=(
            context.section_path
        ),

        article=(
            context.article
        ),

        item_path=(
            context.item_path
        ),

        title=title,

        unit=unit,

        frequency=frequency,

        purpose=purpose,

        content=content,

        scope=scope,

        format=table_format,
    )