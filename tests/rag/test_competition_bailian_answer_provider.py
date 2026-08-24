from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.rag.bailian_answer_provider import (
    BailianAnswerProviderConfigError,
    BailianAnswerProviderResponseError,
)
from app.rag.competition_bailian_answer_provider import (
    CompetitionBailianOpenAIAnswerProvider,
)
from app.schemas.answer_provider import (
    BailianAnswerProviderConfig,
)


class FakeCompletionCreator:
    def __init__(
        self,
        *,
        content: str,
    ) -> None:
        self.content = content
        self.call_count = 0

        self.last_kwargs: dict[
            str,
            Any,
        ] | None = None

    def create(
        self,
        **kwargs: Any,
    ) -> Any:
        self.call_count += 1
        self.last_kwargs = kwargs

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=(
                        SimpleNamespace(
                            content=(
                                self.content
                            )
                        )
                    )
                )
            ]
        )


def _config(
) -> BailianAnswerProviderConfig:
    return BailianAnswerProviderConfig(
        api_key="test-api-key",
        api_base=(
            "https://example.com/"
            "compatible-mode/v1"
        ),
        model_name="qwen3.7-plus",
        enable_thinking=False,
    )


def test_provider_generates_competition_answer(
) -> None:
    creator = FakeCompletionCreator(
        content=json.dumps(
            {
                "answer": "B",
                "answer_text": (
                    "答案为 B。"
                    "核心一级资本工具"
                    "应直接发行且实缴。[E1]"
                ),
                "citation_ids": [
                    "E1"
                ],
            },
            ensure_ascii=False,
        )
    )

    provider = (
        CompetitionBailianOpenAIAnswerProvider(
            config=_config(),
            completion_creator=creator,
        )
    )

    result = provider.generate(
        question=(
            "核心一级资本工具"
            "应满足哪项要求？"
        ),
        options={
            "A": "可以间接发行",
            "B": "直接发行且实缴",
            "C": "必须设置到期日",
            "D": "可以由关联方担保",
        },
        generation_context=(
            "[E1]\n"
            "（一）直接发行且实缴的。"
        ),
        allowed_citation_ids=(
            "E1",
        ),
    )

    assert result.answer == "B"

    assert result.citation_ids == (
        "E1",
    )

    assert provider.request_count == 1

    assert (
        creator.call_count
        == 1
    )

    assert (
        creator.last_kwargs
        is not None
    )

    assert (
        creator.last_kwargs[
            "temperature"
        ]
        == 0
    )

    assert (
        creator.last_kwargs[
            "response_format"
        ]
        == {
            "type": "json_object"
        }
    )

    assert (
        creator.last_kwargs[
            "extra_body"
        ]
        == {
            "enable_thinking": False
        }
    )


def test_prompt_contains_all_options_and_evidence(
) -> None:
    creator = FakeCompletionCreator(
        content=json.dumps(
            {
                "answer": "B",
                "answer_text": (
                    "答案为 B。[E1]"
                ),
                "citation_ids": [
                    "E1"
                ],
            }
        )
    )

    provider = (
        CompetitionBailianOpenAIAnswerProvider(
            config=_config(),
            completion_creator=creator,
        )
    )

    provider.generate(
        question="请选择正确选项",
        options={
            "A": "选项一",
            "B": "选项二",
            "C": "选项三",
            "D": "选项四",
        },
        generation_context=(
            "[E1] 测试证据"
        ),
        allowed_citation_ids=(
            "E1",
        ),
    )

    assert (
        creator.last_kwargs
        is not None
    )

    messages = (
        creator.last_kwargs[
            "messages"
        ]
    )

    user_prompt = (
        messages[1]["content"]
    )

    assert "A. 选项一" in user_prompt
    assert "B. 选项二" in user_prompt
    assert "C. 选项三" in user_prompt
    assert "D. 选项四" in user_prompt

    assert (
        "测试证据"
        in user_prompt
    )

    assert (
        "allowed_citation_ids"
        in user_prompt
    )


def test_provider_rejects_invalid_json(
) -> None:
    creator = FakeCompletionCreator(
        content="不是 JSON"
    )

    provider = (
        CompetitionBailianOpenAIAnswerProvider(
            config=_config(),
            completion_creator=creator,
        )
    )

    with pytest.raises(
        BailianAnswerProviderResponseError,
        match="合法 JSON",
    ):
        provider.generate(
            question="请选择",
            options={
                "A": "一",
                "B": "二",
                "C": "三",
                "D": "四",
            },
            generation_context=(
                "[E1] evidence"
            ),
            allowed_citation_ids=(
                "E1",
            ),
        )


def test_provider_rejects_invalid_answer_schema(
) -> None:
    creator = FakeCompletionCreator(
        content=json.dumps(
            {
                "answer": "E",
                "answer_text": (
                    "错误答案。[E1]"
                ),
                "citation_ids": [
                    "E1"
                ],
            }
        )
    )

    provider = (
        CompetitionBailianOpenAIAnswerProvider(
            config=_config(),
            completion_creator=creator,
        )
    )

    with pytest.raises(
        BailianAnswerProviderResponseError,
        match="Answer Schema",
    ):
        provider.generate(
            question="请选择",
            options={
                "A": "一",
                "B": "二",
                "C": "三",
                "D": "四",
            },
            generation_context=(
                "[E1] evidence"
            ),
            allowed_citation_ids=(
                "E1",
            ),
        )


def test_provider_rejects_missing_option(
) -> None:
    creator = FakeCompletionCreator(
        content="{}"
    )

    provider = (
        CompetitionBailianOpenAIAnswerProvider(
            config=_config(),
            completion_creator=creator,
        )
    )

    with pytest.raises(
        BailianAnswerProviderConfigError,
        match="A/B/C/D",
    ):
        provider.generate(
            question="请选择",
            options={
                "A": "一",
                "B": "二",
                "C": "三",
            },
            generation_context=(
                "[E1] evidence"
            ),
            allowed_citation_ids=(
                "E1",
            ),
        )