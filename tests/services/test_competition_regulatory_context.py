from __future__ import annotations

from app.services.competition_regulatory_context import (
    CompetitionRegulatoryContextTracker,
)


def test_outline_heading_builds_section_path(
) -> None:
    tracker = (
        CompetitionRegulatoryContextTracker()
    )

    first = tracker.consume(
        "信用风险",
        outline_level=0,
    )

    assert (
        first.context.section_path
        == (
            "信用风险",
        )
    )

    assert (
        first.is_structure_only
        is True
    )

    second = tracker.consume(
        "风险管理",
        outline_level=1,
    )

    assert (
        second.context.section_path
        == (
            "信用风险",
            "风险管理",
        )
    )


def test_new_outline_heading_removes_lower_levels(
) -> None:
    tracker = (
        CompetitionRegulatoryContextTracker()
    )

    tracker.consume(
        "信用风险",
        outline_level=0,
    )

    tracker.consume(
        "内部评级",
        outline_level=1,
    )

    result = tracker.consume(
        "交易对手信用风险",
        outline_level=0,
    )

    assert (
        result.context.section_path
        == (
            "交易对手信用风险",
        )
    )


def test_numbered_item_hierarchy_is_inherited(
) -> None:
    tracker = (
        CompetitionRegulatoryContextTracker()
    )

    tracker.consume(
        "信用风险",
        outline_level=0,
    )

    tracker.consume(
        "二、披露表格"
    )

    tracker.consume(
        "（一）表格 CR1：资产质量"
    )

    result = tracker.consume(
        "目的：披露商业银行资产质量信息。"
    )

    assert (
        result.context.section_path
        == (
            "信用风险",
        )
    )

    assert (
        result.context.item_path
        == (
            "二、披露表格",
            "（一）表格 CR1：资产质量",
        )
    )

    assert (
        result.is_structure_only
        is False
    )


def test_item_l1_resets_lower_item_levels(
) -> None:
    tracker = (
        CompetitionRegulatoryContextTracker()
    )

    tracker.consume(
        "一、披露内容"
    )

    tracker.consume(
        "（一）表格 CRA：信用风险定性信息"
    )

    tracker.consume(
        "1.风险管理目标"
    )

    result = tracker.consume(
        "二、披露表格"
    )

    assert (
        result.context.item_path
        == (
            "二、披露表格",
        )
    )


def test_article_is_inherited_by_following_body(
) -> None:
    tracker = (
        CompetitionRegulatoryContextTracker()
    )

    article = tracker.consume(
        "第十二条 商业银行应当建立风险管理制度。"
    )

    assert (
        article.context.article
        == "第十二条"
    )

    assert (
        article.context.article_inherited
        is False
    )

    body = tracker.consume(
        "风险管理制度应覆盖全部业务。"
    )

    assert (
        body.context.article
        == "第十二条"
    )

    assert (
        body.context.article_inherited
        is True
    )


def test_new_outline_heading_clears_old_article(
) -> None:
    tracker = (
        CompetitionRegulatoryContextTracker()
    )

    tracker.consume(
        "第十二条 商业银行应当……"
    )

    result = tracker.consume(
        "信用风险",
        outline_level=0,
    )

    assert (
        result.context.article
        is None
    )

    assert (
        result.context.item_path
        == ()
    )


def test_explicit_item_has_priority_over_outline_level(
) -> None:
    tracker = (
        CompetitionRegulatoryContextTracker()
    )

    tracker.consume(
        "信用风险",
        outline_level=0,
    )

    # 真实 DOCX 中这种 paragraph
    # 可能自己也具有 outline_level。
    result = tracker.consume(
        "一、披露内容",
        outline_level=1,
    )

    # 它仍然应该进入 item_path，
    # 而不是 section_path。
    assert (
        result.context.section_path
        == (
            "信用风险",
        )
    )

    assert (
        result.context.item_path
        == (
            "一、披露内容",
        )
    )


def test_item_l2_can_exist_without_item_l1(
) -> None:
    tracker = (
        CompetitionRegulatoryContextTracker()
    )

    result = tracker.consume(
        "（一）适用范围"
    )

    assert (
        result.context.item_path
        == (
            "（一）适用范围",
        )
    )


def test_reset_clears_all_context(
) -> None:
    tracker = (
        CompetitionRegulatoryContextTracker()
    )

    tracker.consume(
        "信用风险",
        outline_level=0,
    )

    tracker.consume(
        "一、披露内容"
    )

    tracker.reset()

    context = tracker.snapshot()

    assert (
        context.section_path
        == ()
    )

    assert (
        context.article
        is None
    )

    assert (
        context.item_path
        == ()
    )