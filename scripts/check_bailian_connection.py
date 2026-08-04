from __future__ import annotations

import json
import os

from openai import OpenAI, OpenAIError


class BailianConnectionCheckError(
    RuntimeError
):
    """百炼连通性检查失败。"""


def _require_environment(
    name: str,
) -> str:
    value = os.getenv(name)

    if value is None or not value.strip():
        raise BailianConnectionCheckError(
            f"缺少环境变量：{name}"
        )

    return value.strip()


def main() -> None:
    api_key = _require_environment(
        "DASHSCOPE_API_KEY"
    )

    base_url = _require_environment(
        "DASHSCOPE_API_BASE"
    )

    model_name = _require_environment(
        "QWEN_MODEL"
    )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=30.0,
        max_retries=1,
    )

    try:
        completion = (
            client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你正在执行 API 连通性检查。"
                            "只返回 JSON 对象，不要输出 "
                            "Markdown 或额外解释。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            '请返回：{"status":"ok"}'
                        ),
                    },
                ],
                temperature=0,
                response_format={
                    "type": "json_object",
                },
            )
        )
    except OpenAIError as exc:
        raise BailianConnectionCheckError(
            "百炼 OpenAI 兼容接口调用失败"
        ) from exc

    if not completion.choices:
        raise BailianConnectionCheckError(
            "百炼响应没有 choices"
        )

    content = (
        completion
        .choices[0]
        .message
        .content
    )

    if content is None or not content.strip():
        raise BailianConnectionCheckError(
            "百炼响应正文为空"
        )

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise BailianConnectionCheckError(
            "百炼没有返回合法 JSON："
            f"{content!r}"
        ) from exc

    if payload.get("status") != "ok":
        raise BailianConnectionCheckError(
            "百炼返回了非预期内容："
            f"{payload!r}"
        )

    print(
        "provider=bailian_openai_compatible"
    )

    print(
        f"model={model_name}"
    )

    print(
        "structured_output_valid=True"
    )

    print(
        "status=ok"
    )


if __name__ == "__main__":
    main()