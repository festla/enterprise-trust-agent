from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.competition_regulatory import (
    CompetitionRegulatoryMarker,
)
from app.services.competition_regulatory_structure import (
    detect_regulatory_marker,
)


# ============================================================
# Context Snapshot
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class CompetitionRegulatoryContext:
    """
    某一个文本位置对应的监管结构上下文。

    section_path:
        Word outline / chapter / section

        例如：
            信用风险
            风险管理

    article:
        第十二条

    item_path:
        编号条目层级。

        这里保存完整结构行，而不只是编号：

            一、披露表格
            （一）表格 CR1：资产质量
            1.定义

        这样后续 Retrieval 能获得更完整的上下文。
    """

    section_path: tuple[
        str,
        ...,
    ] = ()

    article: str | None = None

    item_path: tuple[
        str,
        ...,
    ] = ()

    article_inherited: bool = False


# ============================================================
# Update Result
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class CompetitionRegulatoryContextUpdate:
    """
    Context Tracker 处理一行文本后的结果。

    marker:
        当前行是否属于显式编号结构。

    is_structure_only:
        当前行是否主要用于建立结构，而不是普通正文。

    context:
        处理当前行后得到的结构上下文。
    """

    context: CompetitionRegulatoryContext

    marker: (
        CompetitionRegulatoryMarker
        | None
    ) = None

    is_structure_only: bool = False


# ============================================================
# Mutable Tracker
# ============================================================


@dataclass(
    slots=True,
)
class CompetitionRegulatoryContextTracker:
    """
    监管文档结构状态机。

    Tracker 本身是 mutable 的：

        consume line 1
            ↓
        update state
            ↓
        consume line 2
            ↓
        inherit state

    但向外返回的 Context Snapshot 是 immutable。
    """

    # Word outline level -> heading text
    #
    # 例如：
    #
    # 0 -> 信用风险
    # 1 -> 风险管理
    #
    _outline_sections: dict[
        int,
        str,
    ] = field(
        default_factory=dict
    )

    # chapter / section 也放进独立层级。
    _chapter: str | None = None

    _section: str | None = None

    _article: str | None = None

    _item_l1: str | None = None

    _item_l2: str | None = None

    _item_l3: str | None = None

    # ========================================================
    # Public
    # ========================================================

    def reset(
        self,
    ) -> None:
        """
        开始新文档时清空所有结构状态。
        """

        self._outline_sections.clear()

        self._chapter = None
        self._section = None
        self._article = None

        self._item_l1 = None
        self._item_l2 = None
        self._item_l3 = None

    def snapshot(
        self,
        *,
        article_inherited: bool = False,
    ) -> CompetitionRegulatoryContext:
        """
        返回当前不可变 Context。
        """

        return CompetitionRegulatoryContext(
            section_path=(
                self._build_section_path()
            ),
            article=self._article,
            item_path=(
                self._build_item_path()
            ),
            article_inherited=(
                article_inherited
            ),
        )

    def consume(
        self,
        text: str,
        *,
        outline_level: (
            int | None
        ) = None,
    ) -> CompetitionRegulatoryContextUpdate:
        """
        消费一个 Paragraph 或 PDF 文本行。

        判断优先级：

            1. 显式监管编号结构
            2. Word outline heading
            3. 普通正文

        这样：

            一、披露内容

        即使它同时具有 Word outline，
        仍然优先作为 item_l1，
        不会被误当成普通 semantic heading。
        """

        normalized = text.strip()

        if not normalized:
            return CompetitionRegulatoryContextUpdate(
                context=self.snapshot(),
                marker=None,
                is_structure_only=False,
            )

        marker = (
            detect_regulatory_marker(
                normalized
            )
        )

        if marker is not None:
            return (
                self._consume_marker(
                    marker
                )
            )

        if outline_level is not None:
            return (
                self._consume_outline_heading(
                    text=normalized,
                    outline_level=(
                        outline_level
                    ),
                )
            )

        # ====================================================
        # 普通正文
        #
        # 当前 article 如果来自前面的 Article，
        # 那么正文使用的是“继承 Article”。
        # ====================================================

        return CompetitionRegulatoryContextUpdate(
            context=self.snapshot(
                article_inherited=(
                    self._article
                    is not None
                )
            ),
            marker=None,
            is_structure_only=False,
        )

    # ========================================================
    # Explicit Marker
    # ========================================================

    def _consume_marker(
        self,
        marker: CompetitionRegulatoryMarker,
    ) -> CompetitionRegulatoryContextUpdate:
        marker_type = (
            marker.marker_type
        )

        # ====================================================
        # Chapter
        # ====================================================

        if marker_type == "chapter":
            self._chapter = (
                marker.raw_text
            )

            self._section = None

            self._article = None

            self._clear_items()

            return (
                CompetitionRegulatoryContextUpdate(
                    context=self.snapshot(),
                    marker=marker,
                    is_structure_only=True,
                )
            )

        # ====================================================
        # Section
        # ====================================================

        if marker_type == "section":
            self._section = (
                marker.raw_text
            )

            self._article = None

            self._clear_items()

            return (
                CompetitionRegulatoryContextUpdate(
                    context=self.snapshot(),
                    marker=marker,
                    is_structure_only=True,
                )
            )

        # ====================================================
        # Article
        # ====================================================

        if marker_type == "article":
            self._article = (
                marker.marker
            )

            self._clear_items()

            # 第十二条本身不是“继承 Article”，
            # 它就是 Article 的来源。
            return (
                CompetitionRegulatoryContextUpdate(
                    context=self.snapshot(
                        article_inherited=False
                    ),
                    marker=marker,

                    # 如果 Article 行后面本身还有正文，
                    # 这行不能被当成纯标题丢掉。
                    is_structure_only=(
                        not bool(
                            marker.content
                        )
                    ),
                )
            )

        # ====================================================
        # Item L1
        #
        # 一、披露内容
        # ====================================================

        if marker_type == "item_l1":
            self._item_l1 = (
                marker.raw_text
            )

            self._item_l2 = None
            self._item_l3 = None

            return (
                CompetitionRegulatoryContextUpdate(
                    context=self.snapshot(
                        article_inherited=(
                            self._article
                            is not None
                        )
                    ),
                    marker=marker,

                    # item 行通常同时有语义内容，
                    # 所以保守地不视为 structure-only。
                    is_structure_only=False,
                )
            )

        # ====================================================
        # Item L2
        #
        # （一）表格 CR1：资产质量
        # ====================================================

        if marker_type == "item_l2":
            self._item_l2 = (
                marker.raw_text
            )

            self._item_l3 = None

            return (
                CompetitionRegulatoryContextUpdate(
                    context=self.snapshot(
                        article_inherited=(
                            self._article
                            is not None
                        )
                    ),
                    marker=marker,
                    is_structure_only=False,
                )
            )

        # ====================================================
        # Item L3
        #
        # 1.定义
        # ====================================================

        if marker_type == "item_l3":
            self._item_l3 = (
                marker.raw_text
            )

            return (
                CompetitionRegulatoryContextUpdate(
                    context=self.snapshot(
                        article_inherited=(
                            self._article
                            is not None
                        )
                    ),
                    marker=marker,
                    is_structure_only=False,
                )
            )

        raise ValueError(
            "Unsupported regulatory marker: "
            f"{marker_type}"
        )

    # ========================================================
    # Word Outline Heading
    # ========================================================

    def _consume_outline_heading(
        self,
        *,
        text: str,
        outline_level: int,
    ) -> CompetitionRegulatoryContextUpdate:
        """
        Word Outline Heading。

        例如：

            信用风险
                level 0

            风险管理
                level 1

        新的同级或高级 heading 出现时，
        清除其下所有旧 heading。
        """

        if outline_level < 0:
            raise ValueError(
                "outline_level "
                "不能小于 0"
            )

        # 删除当前级别及所有更低层级。
        stale_levels = [
            level
            for level
            in self._outline_sections
            if level >= outline_level
        ]

        for level in stale_levels:
            del self._outline_sections[
                level
            ]

        self._outline_sections[
            outline_level
        ] = text

        # ====================================================
        # 新 semantic section 开始，
        # 老 Article / Items 不能泄漏进新章节。
        # ====================================================

        self._article = None

        self._clear_items()

        return CompetitionRegulatoryContextUpdate(
            context=self.snapshot(),
            marker=None,
            is_structure_only=True,
        )

    # ========================================================
    # Helpers
    # ========================================================

    def _clear_items(
        self,
    ) -> None:
        self._item_l1 = None
        self._item_l2 = None
        self._item_l3 = None

    def _build_item_path(
        self,
    ) -> tuple[
        str,
        ...,
    ]:
        result = []

        if self._item_l1:
            result.append(
                self._item_l1
            )

        if self._item_l2:
            result.append(
                self._item_l2
            )

        if self._item_l3:
            result.append(
                self._item_l3
            )

        return tuple(
            result
        )

    def _build_section_path(
        self,
    ) -> tuple[
        str,
        ...,
    ]:
        result: list[str] = []

        # Word semantic headings
        for level in sorted(
            self._outline_sections
        ):
            result.append(
                self._outline_sections[
                    level
                ]
            )

        # 法规式 chapter / section
        if self._chapter:
            result.append(
                self._chapter
            )

        if self._section:
            result.append(
                self._section
            )

        return tuple(
            result
        )