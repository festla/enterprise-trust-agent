from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from app.rag.embedders import (
    BGE_SMALL_ZH_V15_REVISION,
    SentenceTransformerEmbeddingProvider,
    build_bge_small_zh_v15_spec,
)
from app.rag.hybrid_retriever import (
    HybridRetriever,
)
from app.rag.query_planning import (
    build_financial_fact_query_plan,
)
from app.rag.reranking import (
    rerank_hybrid_hits,
)
from app.rag.rerankers import (
    SentenceTransformerCrossEncoderProvider,
)
from app.rag.tokenization import (
    DeterministicChineseBigramTokenizer,
)
from app.schemas.hybrid_retrieval import (
    RRFConfig,
)
from app.schemas.reranker import (
    RerankerRuntimeConfig,
    RerankerSpec,
)
from app.services.bm25_index import (
    load_bm25_index,
)
from app.services.retrieval_eval_dataset import (
    load_financial_fact_retrieval_cases,
)
from app.services.vector_index import (
    build_vector_index,
)


RERANKER_MODEL_NAME = (
    "BAAI/bge-reranker-base"
)

RERANKER_MODEL_REVISION = (
    "2cfc18c9415c912f9d8155881c133215df768a70"
)


def _validate_index_identity(
    *,
    dense_manifest: object,
    bm25_manifest: object,
) -> None:
    """确认 Dense 与 BM25 绑定同一份 ChunkDataset。"""

    compared_fields = (
        "company_id",
        "report_id",
        "fiscal_year",
        "report_type",
        "document_id",
        "chunk_dataset_id",
        "chunk_strategy",
    )

    for field_name in compared_fields:
        dense_value = getattr(
            dense_manifest,
            field_name,
        )

        bm25_value = getattr(
            bm25_manifest,
            field_name,
        )

        if dense_value != bm25_value:
            raise ValueError(
                "Dense 与 BM25 Index "
                "来源身份不一致："
                f"field={field_name}, "
                f"dense={dense_value!r}, "
                f"bm25={bm25_value!r}"
            )


def _first_relevant_rank(
    *,
    hits: tuple[object, ...],
    gold_pdf_pages: tuple[int, ...],
) -> int | None:
    """查找首个 Gold 页面对应的候选排名。"""

    gold_pages = set(
        gold_pdf_pages
    )

    for hit in hits:
        pdf_page = getattr(
            hit,
            "pdf_page",
        )

        rank = getattr(
            hit,
            "rank",
        )

        if pdf_page in gold_pages:
            return int(rank)

    return None


def _format_optional_rank(
    rank: int | None,
) -> str:
    return (
        "-"
        if rank is None
        else str(rank)
    )


def _text_preview(
    text: str,
    *,
    max_chars: int = 180,
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
            "使用真实 Hybrid Top-N 与 "
            "BGE Cross-Encoder 运行重排 Smoke"
        )
    )

    parser.add_argument(
        "--cases-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--case-ids",
        nargs="+",
        required=True,
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
        "--rerank-candidate-count",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--rank-constant",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--print-top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--embedding-model-revision",
        default=(
            BGE_SMALL_ZH_V15_REVISION
        ),
    )

    parser.add_argument(
        "--local-files-only",
        action="store_true",
    )

    args = parser.parse_args()

    if args.print_top_k < 1:
        raise ValueError(
            "print_top_k 必须大于等于 1"
        )

    if args.rerank_candidate_count < 1:
        raise ValueError(
            "rerank_candidate_count "
            "必须大于等于 1"
        )

    cases = (
        load_financial_fact_retrieval_cases(
            args.cases_path
        )
    )

    cases_by_id = {
        case.case_id: case
        for case in cases
    }

    selected_cases = []

    for case_id in args.case_ids:
        case = cases_by_id.get(
            case_id
        )

        if case is None:
            available_ids = ", ".join(
                sorted(cases_by_id)
            )

            raise ValueError(
                "找不到评测题："
                f"case_id={case_id}; "
                f"available={available_ids}"
            )

        selected_cases.append(case)

    embedding_spec = (
        build_bge_small_zh_v15_spec(
            model_revision=(
                args.embedding_model_revision
            )
        )
    )

    embedding_provider = (
        SentenceTransformerEmbeddingProvider(
            spec=embedding_spec,
            batch_size=(
                args.embedding_batch_size
            ),
            device=args.device,
            local_files_only=(
                args.local_files_only
            ),
        )
    )

    dense_result = build_vector_index(
        chunk_dataset_directory=(
            args.chunk_dataset_dir
        ),
        output_root=(
            args.vector_index_output_root
        ),
        provider=embedding_provider,
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

    _validate_index_identity(
        dense_manifest=dense_manifest,
        bm25_manifest=bm25_manifest,
    )

    tokenizer = (
        DeterministicChineseBigramTokenizer(
            spec=(
                bm25_manifest.tokenizer_spec
            )
        )
    )

    rrf_config = RRFConfig(
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

    hybrid_retriever = HybridRetriever(
        dense_index=dense_result.index,
        bm25_index=bm25_result.index,
        provider=embedding_provider,
        tokenizer=tokenizer,
        config=rrf_config,
    )

    reranker_spec = RerankerSpec(
        model_name=(
            RERANKER_MODEL_NAME
        ),
        model_revision=(
            RERANKER_MODEL_REVISION
        ),
        max_length=512,
    )

    # Smoke 时返回全部重排候选，
    # 便于观察 Gold 页面重排后的完整位置。
    reranker_runtime_config = (
        RerankerRuntimeConfig(
            batch_size=(
                args.reranker_batch_size
            ),
            device=args.device,
            local_files_only=(
                args.local_files_only
            ),
            rerank_candidate_count=(
                args.rerank_candidate_count
            ),
            return_count=(
                args.rerank_candidate_count
            ),
        )
    )

    reranker_provider = (
        SentenceTransformerCrossEncoderProvider(
            spec=reranker_spec,
            runtime_config=(
                reranker_runtime_config
            ),
            show_progress_bar=True,
        )
    )

    print(
        f"dense_index_id="
        f"{dense_manifest.index_id}"
    )

    print(
        "dense_index_created="
        f"{str(dense_result.created).lower()}"
    )

    print(
        f"bm25_index_id="
        f"{bm25_manifest.index_id}"
    )

    print(
        "chunk_dataset_id="
        f"{dense_manifest.chunk_dataset_id}"
    )

    print(
        "reranker_model="
        f"{reranker_spec.model_name}"
    )

    print(
        "reranker_revision="
        f"{reranker_spec.model_revision}"
    )

    print(
        "rerank_candidate_count="
        f"{args.rerank_candidate_count}"
    )

    print("=" * 100)

    for case in selected_cases:
        if (
            case.company_id
            != dense_manifest.company_id
            or case.report_id
            != dense_manifest.report_id
            or case.fiscal_year
            != dense_manifest.fiscal_year
            or case.report_type
            != dense_manifest.report_type
        ):
            raise ValueError(
                "评测题与当前索引报告身份不一致："
                f"{case.case_id}"
            )

        plan = (
            build_financial_fact_query_plan(
                original_query=case.question,
                metric_name=case.metric_name,
                fiscal_year=case.fiscal_year,
                company_id=case.company_id,
                report_id=case.report_id,
                report_type=case.report_type,
                statement_type=(
                    case.statement_type
                ),
                statement_scope=(
                    case.statement_scope
                ),
            )
        )

        hybrid_started = perf_counter()

        hybrid_hits = (
            hybrid_retriever.search(
                query=plan.semantic_query,
                top_k=(
                    args.rerank_candidate_count
                ),
                filters=plan.filters,
            )
        )

        hybrid_elapsed_ms = (
            perf_counter()
            - hybrid_started
        ) * 1000

        reranker_started = perf_counter()

        reranked_hits = rerank_hybrid_hits(
            query=plan.semantic_query,
            hits=hybrid_hits,
            provider=reranker_provider,
            config=reranker_runtime_config,
        )

        reranker_elapsed_ms = (
            perf_counter()
            - reranker_started
        ) * 1000

        hybrid_rank = _first_relevant_rank(
            hits=hybrid_hits,
            gold_pdf_pages=(
                case.gold_pdf_pages
            ),
        )

        reranker_rank = _first_relevant_rank(
            hits=reranked_hits,
            gold_pdf_pages=(
                case.gold_pdf_pages
            ),
        )

        print(
            f"case_id={case.case_id}"
        )

        print(
            f"question={case.question}"
        )

        print(
            "semantic_query="
            f"{plan.semantic_query}"
        )

        print(
            "gold_pdf_pages="
            + ",".join(
                str(page)
                for page
                in case.gold_pdf_pages
            )
        )

        print(
            "first_relevant_rank:"
            f" rrf={_format_optional_rank(hybrid_rank)}"
            f" -> reranker="
            f"{_format_optional_rank(reranker_rank)}"
        )

        print(
            "latency_ms:"
            f" hybrid={hybrid_elapsed_ms:.2f}"
            f" reranker={reranker_elapsed_ms:.2f}"
        )

        print("-" * 100)
        print("RRF TOP RESULTS")

        for hit in hybrid_hits[
            :args.print_top_k
        ]:
            is_gold = (
                hit.pdf_page
                in case.gold_pdf_pages
            )

            print(
                f"rrf_rank={hit.rank} "
                f"gold={str(is_gold).lower()} "
                f"pdf_page={hit.pdf_page} "
                f"dense_rank="
                f"{_format_optional_rank(hit.dense_rank)} "
                f"bm25_rank="
                f"{_format_optional_rank(hit.bm25_rank)} "
                f"rrf_score={hit.rrf_score:.8f}"
            )

            print(
                f"chunk_id={hit.chunk_id}"
            )

            print(
                f"text={_text_preview(hit.text)}"
            )

        print("-" * 100)
        print("RERANKER TOP RESULTS")

        for hit in reranked_hits[
            :args.print_top_k
        ]:
            is_gold = (
                hit.pdf_page
                in case.gold_pdf_pages
            )

            print(
                f"reranker_rank={hit.rank} "
                f"gold={str(is_gold).lower()} "
                f"pdf_page={hit.pdf_page} "
                f"reranker_score="
                f"{hit.reranker_score:.6f} "
                f"rrf_rank={hit.rrf_rank} "
                f"dense_rank="
                f"{_format_optional_rank(hit.dense_rank)} "
                f"bm25_rank="
                f"{_format_optional_rank(hit.bm25_rank)}"
            )

            print(
                f"chunk_id={hit.chunk_id}"
            )

            print(
                f"text={_text_preview(hit.text)}"
            )

        print("=" * 100)


if __name__ == "__main__":
    main()