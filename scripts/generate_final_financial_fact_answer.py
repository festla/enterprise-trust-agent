from __future__ import annotations

import argparse
from pathlib import Path

from app.rag.answer_generation import (
    generate_financial_fact_answer,
)
from app.rag.bailian_answer_provider import (
    BailianOpenAIAnswerProvider,
)
from app.schemas.financial_fact_answer import (
    FinancialFactAnswerPacket,
)


class FinalAnswerBuildError(
    RuntimeError
):
    """最终财务事实回答构建异常。"""


def _load_packet(
    path: Path,
) -> FinancialFactAnswerPacket:
    if not path.is_file():
        raise FinalAnswerBuildError(
            f"Answer Packet 不存在：{path}"
        )

    return (
        FinancialFactAnswerPacket
        .model_validate_json(
            path.read_bytes()
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "使用百炼 Provider 构建最终"
            "财务事实回答或拒答结果"
        )
    )

    parser.add_argument(
        "--packet-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    packet = _load_packet(
        args.packet_path
    )

    provider = (
        BailianOpenAIAnswerProvider
        .from_environment()
    )

    result = generate_financial_fact_answer(
        packet=packet,
        provider=provider,
    )

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output_path.write_text(
        result.model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )

    print(f"status={result.status}")
    print(
        f"generator_id={result.generator_id}"
    )
    print(
        f"citation_ids={result.citation_ids}"
    )
    print(
        "used_chunk_ids="
        f"{result.used_chunk_ids}"
    )
    print(
        "provider_request_count="
        f"{provider.request_count}"
    )
    print(
        f"answer_text={result.answer_text}"
    )
    print(
        f"output_path={args.output_path}"
    )


if __name__ == "__main__":
    main()