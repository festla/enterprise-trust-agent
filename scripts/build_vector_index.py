from __future__ import annotations

import argparse
from pathlib import Path

from app.rag.embedders import (
    BGE_SMALL_ZH_V15_REVISION,
    SentenceTransformerEmbeddingProvider,
    build_bge_small_zh_v15_spec,
)
from app.services.vector_index import (
    build_vector_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "使用中文 BGE 模型构建精确向量索引"
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
        "--model-revision",
        default=(
            BGE_SMALL_ZH_V15_REVISION
        ),
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--cache-folder",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--local-files-only",
        action="store_true",
    )

    parser.add_argument(
        "--show-progress-bar",
        action="store_true",
    )

    args = parser.parse_args()

    spec = build_bge_small_zh_v15_spec(
        model_revision=args.model_revision
    )

    provider = (
        SentenceTransformerEmbeddingProvider(
            spec=spec,
            batch_size=args.batch_size,
            device=args.device,
            cache_folder=args.cache_folder,
            local_files_only=(
                args.local_files_only
            ),
            show_progress_bar=(
                args.show_progress_bar
            ),
        )
    )

    result = build_vector_index(
        chunk_dataset_directory=(
            args.chunk_dataset_dir
        ),
        output_root=args.output_root,
        provider=provider,
    )

    manifest = result.manifest

    print(
        f"index_id={manifest.index_id}"
    )

    print(
        "chunk_dataset_id="
        f"{manifest.chunk_dataset_id}"
    )

    print(
        "model_name="
        f"{manifest.embedding_spec.model_name}"
    )

    print(
        "model_revision="
        f"{manifest.embedding_spec.model_version}"
    )

    print(
        "vector_count="
        f"{manifest.vector_count}"
    )

    print(
        "vector_dimension="
        f"{manifest.vector_dimension}"
    )

    print(
        f"created={result.created}"
    )

    print(
        "index_directory="
        f"{result.index_directory}"
    )


if __name__ == "__main__":
    main()