from __future__ import annotations

import re

from app.schemas.competition_regulatory import (
    CompetitionRegulatoryMarker,
)


# ============================================================
# Chinese Number
#
# 支持：
#
# 一
# 十二
# 一百零二
# 1
# 12
#
# 暂时不试图解析成 int。
# Detector 只负责识别结构。
# ============================================================

_CN_NUMBER = (
    r"[零〇一二三四五六七八九十百千万两]+"
)

_ARABIC_NUMBER = (
    r"[0-9]+"
)

_NUMBER = (
    rf"(?:{_CN_NUMBER}|{_ARABIC_NUMBER})"
)


# ============================================================
# Chapter
#
# 第一章 总则
# 第2章 监督管理
#
# 保守规则：
# - 必须位于行首
# - 整行不能太长
#
# 避免普通正文：
#
# “按照第一章规定……”
#
# 被误判。
# ============================================================

_CHAPTER_PATTERN = re.compile(
    rf"^(?P<marker>第{_NUMBER}章)"
    rf"(?:[\s　]+(?P<title>.+))?$"
)


# ============================================================
# Section
#
# 第一节 基本原则
# 第2节 信息披露
# ============================================================

_SECTION_PATTERN = re.compile(
    rf"^(?P<marker>第{_NUMBER}节)"
    rf"(?:[\s　]+(?P<title>.+))?$"
)


# ============================================================
# Article
#
# 第十二条 商业银行应当……
#
# 与 chapter / section 不同：
# article 后面通常直接跟正文。
#
# 因此只提取：
#
# marker = 第十二条
# content = 商业银行应当……
# ============================================================

_ARTICLE_PATTERN = re.compile(
    rf"^(?P<marker>第{_NUMBER}条)"
    rf"[\s　]*(?P<content>.*)$"
)


# ============================================================
# Item Level 1
#
# 一、总体要求
# 二、适用范围
#
# 必须使用顿号。
#
# 不识别：
#
# 第一，……
#
# 因为这种句式在普通正文中很常见，
# 误判风险更高。
# ============================================================

_ITEM_L1_PATTERN = re.compile(
    rf"^(?P<marker>{_CN_NUMBER}、)"
    rf"[\s　]*(?P<content>.+)$"
)


# ============================================================
# Item Level 2
#
# （一）基本要求
# (二) 基本要求
#
# 同时兼容中英文括号。
# ============================================================

_ITEM_L2_PATTERN = re.compile(
    rf"^(?P<marker>"
    rf"[（(]{_NUMBER}[）)]"
    rf")"
    rf"[\s　]*(?P<content>.+)$"
)


# ============================================================
# Item Level 3
#
# 1. 内容
# 1、内容
#
# 当前不识别：
#
# 1)
# (1)
#
# 避免规则一下扩得太宽。
# 后面根据真实 Dev 文档再决定。
# ============================================================

_ITEM_L3_PATTERN = re.compile(
    r"^(?P<marker>[0-9]+[\.、])"
    r"[\s　]*(?P<content>.+)$"
)


MAX_HEADING_CHARS = 100


def _normalize_structure_text(
    text: str,
) -> str:
    """
    结构识别只做非常低风险的规范化：

    - 去掉首尾空白
    - 全角空格转普通空格
    - 不修改正文内部标点
    """

    return (
        text
        .replace(
            "\u3000",
            " ",
        )
        .strip()
    )


def _detect_chapter(
    text: str,
) -> (
    CompetitionRegulatoryMarker
    | None
):
    if len(text) > MAX_HEADING_CHARS:
        return None

    match = _CHAPTER_PATTERN.fullmatch(
        text
    )

    if match is None:
        return None

    return CompetitionRegulatoryMarker(
        marker_type="chapter",
        marker=match.group(
            "marker"
        ),
        raw_text=text,
        title=text,
        content=None,
    )


def _detect_section(
    text: str,
) -> (
    CompetitionRegulatoryMarker
    | None
):
    if len(text) > MAX_HEADING_CHARS:
        return None

    match = _SECTION_PATTERN.fullmatch(
        text
    )

    if match is None:
        return None

    return CompetitionRegulatoryMarker(
        marker_type="section",
        marker=match.group(
            "marker"
        ),
        raw_text=text,
        title=text,
        content=None,
    )


def _detect_article(
    text: str,
) -> (
    CompetitionRegulatoryMarker
    | None
):
    match = _ARTICLE_PATTERN.fullmatch(
        text
    )

    if match is None:
        return None

    content = (
        match.group(
            "content"
        )
        .strip()
    )

    return CompetitionRegulatoryMarker(
        marker_type="article",
        marker=match.group(
            "marker"
        ),
        raw_text=text,
        title=None,
        content=(
            content
            if content
            else None
        ),
    )


def _detect_item_l1(
    text: str,
) -> (
    CompetitionRegulatoryMarker
    | None
):
    match = (
        _ITEM_L1_PATTERN
        .fullmatch(
            text
        )
    )

    if match is None:
        return None

    return CompetitionRegulatoryMarker(
        marker_type="item_l1",
        marker=match.group(
            "marker"
        ),
        raw_text=text,
        title=None,
        content=(
            match.group(
                "content"
            ).strip()
        ),
    )


def _detect_item_l2(
    text: str,
) -> (
    CompetitionRegulatoryMarker
    | None
):
    match = (
        _ITEM_L2_PATTERN
        .fullmatch(
            text
        )
    )

    if match is None:
        return None

    return CompetitionRegulatoryMarker(
        marker_type="item_l2",
        marker=match.group(
            "marker"
        ),
        raw_text=text,
        title=None,
        content=(
            match.group(
                "content"
            ).strip()
        ),
    )


def _detect_item_l3(
    text: str,
) -> (
    CompetitionRegulatoryMarker
    | None
):
    match = (
        _ITEM_L3_PATTERN
        .fullmatch(
            text
        )
    )

    if match is None:
        return None

    return CompetitionRegulatoryMarker(
        marker_type="item_l3",
        marker=match.group(
            "marker"
        ),
        raw_text=text,
        title=None,
        content=(
            match.group(
                "content"
            ).strip()
        ),
    )


def detect_regulatory_marker(
    text: str,
) -> (
    CompetitionRegulatoryMarker
    | None
):
    """
    识别一行监管制度文本中的结构标记。

    优先级非常重要：

        chapter
        section
        article
        item_l1
        item_l2
        item_l3

    当前采用 conservative matching：

    宁可少识别，
    不轻易把普通正文误识别成结构。
    """

    normalized = (
        _normalize_structure_text(
            text
        )
    )

    if not normalized:
        return None

    detectors = (
        _detect_chapter,
        _detect_section,
        _detect_article,
        _detect_item_l1,
        _detect_item_l2,
        _detect_item_l3,
    )

    for detector in detectors:
        marker = detector(
            normalized
        )

        if marker is not None:
            return marker

    return None