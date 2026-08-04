from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.rag.bailian_answer_provider import (
    BailianAnswerProviderConfigError,
    BailianAnswerProviderResponseError,
    BailianOpenAIAnswerProvider,
    load_bailian_answer_provider_config,
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
                    message=SimpleNamespace(
                        content=self.content
                    )
                )
            ]
        )


def build_config(
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


def test_load_config_from_environment_mapping(
) -> None:
    config = (
        load_bailian_answer_provider_config(
            environ={
                "DASHSCOPE_API_KEY": (
                    "test-key"
                ),
                "DASHSCOPE_API_BASE": (
                    "https://example.com/v1"
                ),
                "QWEN_MODEL": (
                    "qwen3.7-plus"
                ),
            }
        )
    )

    assert (
        config.api_key.get_secret_value()
        == "test-key"
    )

    assert (
        config.model_name
        == "qwen3.7-plus"
    )

    assert config.enable_thinking is False


def test_missing_environment_variable_fails(
) -> None:
    with pytest.raises(
        BailianAnswerProviderConfigError,
        match="DASHSCOPE_API_BASE",
    ):
        load_bailian_answer_provider_config(
            environ={
                "DASHSCOPE_API_KEY": (
                    "test-key"
                ),
                "QWEN_MODEL": (
                    "qwen3.7-plus"
                ),
            }
        )


def test_provider_generates_structured_answer(
) -> None:
    creator = FakeCompletionCreator(
        content=json.dumps(
            {
                "answer_text": (
                    "美的集团2024年度营业收入"
                    "为407,149,600千元。[E1]"
                ),
                "citation_ids": ["E1"],
            },
            ensure_ascii=False,
        )
    )

    provider = BailianOpenAIAnswerProvider(
        config=build_config(),
        completion_creator=creator,
    )

    answer = provider.generate(
        question=(
            "美的集团2024年营业收入"
            "是多少？"
        ),
        metric_name="营业收入",
        generation_context=(
            "[E1] 合并利润表 "
            "营业收入 407,149,600 千元"
        ),
        allowed_citation_ids=("E1",),
    )

    assert answer.citation_ids == ("E1",)

    assert "[E1]" in answer.answer_text

    assert provider.request_count == 1

    assert creator.last_kwargs is not None

    assert creator.last_kwargs[
        "response_format"
    ] == {
        "type": "json_object",
    }

    assert creator.last_kwargs[
        "extra_body"
    ] == {
        "enable_thinking": False,
    }


def test_provider_rejects_invalid_json(
) -> None:
    creator = FakeCompletionCreator(
        content="这不是 JSON"
    )

    provider = BailianOpenAIAnswerProvider(
        config=build_config(),
        completion_creator=creator,
    )

    with pytest.raises(
        BailianAnswerProviderResponseError,
        match="合法 JSON",
    ):
        provider.generate(
            question="营业收入是多少？",
            metric_name="营业收入",
            generation_context=(
                "[E1] 营业收入 "
                "407,149,600 千元"
            ),
            allowed_citation_ids=("E1",),
        )


def test_provider_rejects_invalid_schema(
) -> None:
    creator = FakeCompletionCreator(
        content=json.dumps(
            {
                "answer_text": "没有引用",
                "citation_ids": [],
            },
            ensure_ascii=False,
        )
    )

    provider = BailianOpenAIAnswerProvider(
        config=build_config(),
        completion_creator=creator,
    )

    with pytest.raises(
        BailianAnswerProviderResponseError,
        match="回答 Schema",
    ):
        provider.generate(
            question="营业收入是多少？",
            metric_name="营业收入",
            generation_context=(
                "[E1] 营业收入 "
                "407,149,600 千元"
            ),
            allowed_citation_ids=("E1",),
        )