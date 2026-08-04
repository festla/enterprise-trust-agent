from __future__ import annotations

import argparse
from pathlib import Path

from app.rag.embedders import (
    BGE_SMALL_ZH_V15_REVISION,
    SentenceTransformerEmbeddingProvider,
    build_bge_small_zh_v15_spec,
)
from app.rag.hybrid_retriever import (
    HybridRetriever,
)
from app.rag.tokenization import (
    DeterministicChineseBigramTokenizer,
)
from app.schemas.hybrid_retrieval import (
    RRFConfig,
)
from app.schemas.retrieval import (
    RetrievalFilter,
)
from app.services.bm25_index import (
    load_bm25_index,
)
from app.services.vector_index import (
    build_vector_index,
)


def _build_preview(
    text: str,
    *,
    max_chars: int,
) -> str:
    normalized = " ".join(
        text.split()
    )

    if len(normalized) <= max_chars:
        return normalized

    return (
        normalized[:max_chars]
        + "..."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "对真实 Dense 与 BM25 索引执行 "
            "Hybrid RRF 检索"
        )
    )

    parser.add_argument(
        "--chunk-dataset-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--vector-index-output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--bm25-index-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--query",
        required=True,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--dense-candidate-count",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--bm25-candidate-count",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--rank-constant",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--model-revision",
        default=(
            BGE_SMALL_ZH_V15_REVISION
        ),
    )

    parser.add_argument(
        "--local-files-only",
        action="store_true",
    )

    parser.add_argument(
        "--pdf-pages",
        type=int,
        nargs="*",
        default=(),
    )

    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=220,
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
            local_files_only=(
                args.local_files_only
            ),
        )
    )

    # 当前 Vector Service 没有公开的 load 函数。
    # build_vector_index 是幂等入口：
    # 已存在时会验证并复用已有索引。
    dense_result = build_vector_index(
        chunk_dataset_directory=(
            args.chunk_dataset_dir
        ),
        output_root=(
            args.vector_index_output_root
        ),
        provider=provider,
    )

    bm25_result = load_bm25_index(
        args.bm25_index_dir
    )

    dense_manifest = (
        dense_result.manifest
    )

    bm25_manifest = (
        bm25_result.manifest
    )

    if (
        dense_manifest.chunk_dataset_id
        != bm25_manifest.chunk_dataset_id
    ):
        raise ValueError(
            "Dense 与 BM25 绑定了不同的 "
            "chunk_dataset_id"
        )

    if (
        dense_manifest.report_id
        != bm25_manifest.report_id
    ):
        raise ValueError(
            "Dense 与 BM25 的 "
            "report_id 不一致"
        )

    tokenizer = (
        DeterministicChineseBigramTokenizer(
            spec=(
                bm25_manifest.tokenizer_spec
            )
        )
    )

    config = RRFConfig(
        rank_constant=(
            args.rank_constant
        ),
        dense_candidate_count=(
            args.dense_candidate_count
        ),
        bm25_candidate_count=(
            args.bm25_candidate_count
        ),
    )

    retriever = HybridRetriever(
        dense_index=dense_result.index,
        bm25_index=bm25_result.index,
        provider=provider,
        tokenizer=tokenizer,
        config=config,
    )

    filters = RetrievalFilter(
        company_ids=(
            dense_manifest.company_id,
        ),
        report_ids=(
            dense_manifest.report_id,
        ),
        fiscal_years=(
            dense_manifest.fiscal_year,
        ),
        report_types=(
            dense_manifest.report_type,
        ),
        document_ids=(
            dense_manifest.document_id,
        ),
        pdf_pages=tuple(
            args.pdf_pages
        ),
    )

    hits = retriever.search(
        query=args.query,
        top_k=args.top_k,
        filters=filters,
    )

    print(
        "dense_index_id="
        f"{dense_manifest.index_id}"
    )

    print(
        "dense_index_created="
        f"{str(dense_result.created).lower()}"
    )

    print(
        "bm25_index_id="
        f"{bm25_manifest.index_id}"
    )

    print(
        "chunk_dataset_id="
        f"{dense_manifest.chunk_dataset_id}"
    )

    print(
        f"report_id="
        f"{dense_manifest.report_id}"
    )

    print(f"query={args.query}")

    print(
        "rank_constant="
        f"{config.rank_constant}"
    )

    print(
        "dense_candidate_count="
        f"{config.dense_candidate_count}"
    )

    print(
        "bm25_candidate_count="
        f"{config.bm25_candidate_count}"
    )

    print(
        "returned_hit_count="
        f"{len(hits)}"
    )

    for hit in hits:
        print("-" * 88)

        dense_rank = (
            "-"
            if hit.dense_rank is None
            else str(hit.dense_rank)
        )

        bm25_rank = (
            "-"
            if hit.bm25_rank is None
            else str(hit.bm25_rank)
        )

        print(
            f"hybrid_rank={hit.rank} "
            f"dense_rank={dense_rank} "
            f"bm25_rank={bm25_rank} "
            f"rrf_score="
            f"{hit.rrf_score:.8f}"
        )

        print(
            f"pdf_page={hit.pdf_page} "
            f"printed_page="
            f"{hit.printed_page}"
        )

        print(
            "source_retrievers="
            + ",".join(
                hit.source_retrievers
            )
        )

        print(
            f"chunk_id={hit.chunk_id}"
        )

        if hit.section_path:
            print(
                "section_path="
                + " > ".join(
                    hit.section_path
                )
            )

        print(
            "text="
            + _build_preview(
                hit.text,
                max_chars=(
                    args.max_text_chars
                ),
            )
        )


if __name__ == "__main__":
    main()