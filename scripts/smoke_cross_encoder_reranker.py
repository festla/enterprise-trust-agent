from __future__ import annotations

import argparse
from pathlib import Path

from app.rag.rerankers import (
    SentenceTransformerCrossEncoderProvider,
)
from app.schemas.reranker import (
    RerankerRuntimeConfig,
    RerankerSpec,
)


MODEL_NAME = (
    "BAAI/bge-reranker-base"
)

MODEL_REVISION = (
    "2cfc18c9415c912f9d8155881c133215df768a70"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "运行 BGE Cross-Encoder "
            "中文相关性 Smoke Test"
        )
    )

    parser.add_argument(
        "--cache-folder",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--local-files-only",
        action="store_true",
    )

    args = parser.parse_args()

    spec = RerankerSpec(
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        max_length=512,
    )

    runtime_config = (
        RerankerRuntimeConfig(
            batch_size=args.batch_size,
            device=args.device,
            local_files_only=(
                args.local_files_only
            ),
            rerank_candidate_count=4,
            return_count=4,
        )
    )

    provider = (
        SentenceTransformerCrossEncoderProvider(
            spec=spec,
            runtime_config=runtime_config,
            cache_folder=args.cache_folder,
            show_progress_bar=True,
        )
    )

    query = (
        "美的集团2024年"
        "营业收入是多少？"
    )

    passages = (
        (
            "合并利润表显示，"
            "2024年度营业收入为"
            "407,149,600,000元。"
        ),
        (
            "公司固定资产采用年限平均法"
            "计提折旧。"
        ),
        (
            "报告期内公司持续推进"
            "海外业务战略布局。"
        ),
        (
            "合并资产负债表显示，"
            "期末资产总计有所增长。"
        ),
    )

    pairs = tuple(
        (
            query,
            passage,
        )
        for passage in passages
    )

    scores = provider.score_pairs(
        pairs
    )

    ranked = sorted(
        zip(
            passages,
            scores,
            strict=True,
        ),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    print(
        f"provider={provider.spec.provider}"
    )

    print(
        f"model_name={provider.spec.model_name}"
    )

    print(
        "model_revision="
        f"{provider.spec.model_revision}"
    )

    print(
        f"max_length={provider.spec.max_length}"
    )

    print(
        "score_type="
        f"{provider.spec.score_type}"
    )

    print(
        "local_files_only="
        f"{str(runtime_config.local_files_only).lower()}"
    )

    print("-" * 80)

    for rank, (
        passage,
        score,
    ) in enumerate(
        ranked,
        start=1,
    ):
        print(
            f"rank={rank} "
            f"score={score:.6f} "
            f"text={passage}"
        )

    if len(scores) != len(passages):
        raise RuntimeError(
            "Smoke Test 输出数量异常"
        )

    if ranked[0][0] != passages[0]:
        raise RuntimeError(
            "明显相关的营业收入证据"
            "没有排在第一位"
        )

    print("-" * 80)
    print("smoke_status=passed")


if __name__ == "__main__":
    main()