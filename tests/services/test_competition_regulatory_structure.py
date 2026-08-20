from __future__ import annotations

import pytest

from app.services.competition_regulatory_structure import (
    detect_regulatory_marker,
)


@pytest.mark.parametrize(
    (
        "text",
        "marker_type",
        "marker",
    ),
    [
        (
            "第一章 总则",
            "chapter",
            "第一章",
        ),
        (
            "第2章 监督管理",
            "chapter",
            "第2章",
        ),
        (
            "第二节 风险管理",
            "section",
            "第二节",
        ),
        (
            "第3节 信息披露",
            "section",
            "第3节",
        ),
    ],
)
def test_detect_chapter_and_section(
    text: str,
    marker_type: str,
    marker: str,
) -> None:
    result = (
        detect_regulatory_marker(
            text
        )
    )

    assert result is not None

    assert (
        result.marker_type
        == marker_type
    )

    assert (
        result.marker
        == marker
    )

    assert (
        result.title
        == text
    )


@pytest.mark.parametrize(
    (
        "text",
        "marker",
        "content",
    ),
    [
        (
            (
                "第十二条 "
                "商业银行应当建立"
                "风险管理制度。"
            ),
            "第十二条",
            (
                "商业银行应当建立"
                "风险管理制度。"
            ),
        ),
        (
            "第20条 本办法自公布之日起施行。",
            "第20条",
            "本办法自公布之日起施行。",
        ),
        (
            "第三条",
            "第三条",
            None,
        ),
    ],
)
def test_detect_article(
    text: str,
    marker: str,
    content: str | None,
) -> None:
    result = (
        detect_regulatory_marker(
            text
        )
    )

    assert result is not None

    assert (
        result.marker_type
        == "article"
    )

    assert (
        result.marker
        == marker
    )

    assert (
        result.content
        == content
    )


@pytest.mark.parametrize(
    (
        "text",
        "marker_type",
        "marker",
        "content",
    ),
    [
        (
            "一、总体要求",
            "item_l1",
            "一、",
            "总体要求",
        ),
        (
            "（一）适用范围",
            "item_l2",
            "（一）",
            "适用范围",
        ),
        (
            "(二) 信息报送",
            "item_l2",
            "(二)",
            "信息报送",
        ),
        (
            "1. 建立内部控制制度",
            "item_l3",
            "1.",
            "建立内部控制制度",
        ),
        (
            "2、加强风险管理",
            "item_l3",
            "2、",
            "加强风险管理",
        ),
    ],
)
def test_detect_items(
    text: str,
    marker_type: str,
    marker: str,
    content: str,
) -> None:
    result = (
        detect_regulatory_marker(
            text
        )
    )

    assert result is not None

    assert (
        result.marker_type
        == marker_type
    )

    assert (
        result.marker
        == marker
    )

    assert (
        result.content
        == content
    )


@pytest.mark.parametrize(
    "text",
    [
        (
            "商业银行应当按照"
            "第一章有关规定执行。"
        ),
        "第一，应当加强风险管理。",
        "资本充足率不得低于监管要求。",
        "2024年银行业运行情况",
        "",
        "   ",
    ],
)
def test_normal_body_text_is_not_marker(
    text: str,
) -> None:
    assert (
        detect_regulatory_marker(
            text
        )
        is None
    )


def test_full_width_spaces_are_normalized(
) -> None:
    result = (
        detect_regulatory_marker(
            "　第一章　总则　"
        )
    )

    assert result is not None

    assert (
        result.marker_type
        == "chapter"
    )

    assert (
        result.marker
        == "第一章"
    )

    assert (
        result.title
        == "第一章 总则"
    )


def test_long_sentence_is_not_chapter_heading(
) -> None:
    text = (
        "第一章 "
        + (
            "这是普通正文内容"
            * 30
        )
    )

    assert (
        detect_regulatory_marker(
            text
        )
        is None
    )