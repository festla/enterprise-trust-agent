from __future__ import annotations

from typing import (
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


CompetitionRegulatoryMarkerType = Literal[
    "chapter",
    "section",
    "article",
    "item_l1",
    "item_l2",
    "item_l3",
]


class CompetitionRegulatoryMarker(
    BaseModel
):
    """
    从监管制度文本中识别出的一个结构标记。

    例如：

        第一章 总则
            ->
        marker_type="chapter"
        marker="第一章"
        title="第一章 总则"

        第十二条 商业银行应当……
            ->
        marker_type="article"
        marker="第十二条"
        content="商业银行应当……"
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    marker_type: (
        CompetitionRegulatoryMarkerType
    )

    # 结构标记本身：
    #
    # 第一章
    # 第二节
    # 第十二条
    # 一、
    # （一）
    # 1.
    marker: str = Field(
        min_length=1,
    )

    # 原始完整文本。
    raw_text: str = Field(
        min_length=1,
    )

    # chapter / section 使用。
    #
    # 第一章 总则
    #
    # title 就保存整行。
    title: str | None = None

    # article / item 使用。
    #
    # 第十二条 商业银行应当……
    #
    # content =
    # 商业银行应当……
    content: str | None = None