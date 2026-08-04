from __future__ import annotations

import argparse
from pathlib import Path

from app.rag.tokenization import (
    DeterministicChineseBigramTokenizer,
)
from app.schemas.bm25 import BM25Config
from app.services.bm25_index import (
    build_bm25_index,
    load_bm25_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "为一份 ChunkDataset 构建并验证 "
            "持久化 BM25 索引"
        )
    )

    parser.add_argument(
        "--chunk-dataset-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--k1",
        type=float,
        default=1.2,
    )

    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
    )

    args = parser.parse_args()

    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    config = BM25Config(
        k1=args.k1,
        b=args.b,
    )

    result = build_bm25_index(
        chunk_dataset_directory=(
            args.chunk_dataset_dir
        ),
        output_root=args.output_root,
        tokenizer=tokenizer,
        config=config,
    )

    # 构建完成后再走一次正式加载流程，
    # 验证落盘文件能够通过全部完整性检查。
    verified_result = load_bm25_index(
        result.index_directory
    )

    manifest = verified_result.manifest

    print(f"index_id={manifest.index_id}")

    print(
        "index_directory="
        f"{verified_result.index_directory}"
    )

    print(
        "chunk_dataset_id="
        f"{manifest.chunk_dataset_id}"
    )

    print(f"report_id={manifest.report_id}")

    print(
        "chunk_strategy="
        f"{manifest.chunk_strategy.value}"
    )

    print(
        "document_count="
        f"{manifest.document_count}"
    )

    print(
        "metadata_record_count="
        f"{manifest.metadata_record_count}"
    )

    print(
        "vocabulary_size="
        f"{manifest.vocabulary_size}"
    )

    print(
        "total_token_count="
        f"{manifest.total_token_count}"
    )

    print(
        "average_document_length="
        f"{manifest.average_document_length:.6f}"
    )

    print(
        "tokenizer_version="
        f"{manifest.tokenizer_spec.tokenizer_version}"
    )

    print(
        f"bm25_k1={manifest.bm25_config.k1}"
    )

    print(
        f"bm25_b={manifest.bm25_config.b}"
    )

    print(
        "quality_gate_passed="
        f"{str(manifest.quality_gate_passed).lower()}"
    )

    print(
        "created_at="
        f"{manifest.created_at.isoformat()}"
    )

    print("load_verification_passed=true")


if __name__ == "__main__":
    main()