from __future__ import annotations

import argparse
import hashlib
import json
from math import ceil
from pathlib import Path
from statistics import mean
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
from app.rag.retrieval_evaluation import (
    evaluate_financial_fact_retrieval,
    summarize_retrieval_results,
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


def _canonical_json_bytes(
    value: object,
) -> bytes:
    """生成稳定、确定性的 JSON 字节。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _build_hybrid_run_id(
    *,
    report_id: str,
    dense_index_id: str,
    bm25_index_id: str,
    config: RRFConfig,
) -> str:
    """重建当前 RRF 配置对应的稳定运行 ID。"""

    payload = {
        "schema_version": 1,
        "retriever_type": (
            "hybrid_rrf"
        ),
        "dense_index_id": (
            dense_index_id
        ),
        "bm25_index_id": (
            bm25_index_id
        ),
        "rrf_config": config.model_dump(
            mode="json"
        ),
    }

    digest = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()[:24]

    return (
        f"hybrid_run_{report_id}_"
        f"{digest}"
    )


def _build_reranker_run_id(
    *,
    report_id: str,
    evaluation_set_id: str,
    hybrid_run_id: str,
    spec: RerankerSpec,
    runtime_config: (
        RerankerRuntimeConfig
    ),
) -> str:
    """生成稳定的 Reranker 实验运行 ID。"""

    payload = {
        "schema_version": 1,
        "retriever_type": (
            "hybrid_reranker"
        ),
        "evaluation_set_id": (
            evaluation_set_id
        ),
        "hybrid_run_id": (
            hybrid_run_id
        ),
        "reranker_spec": (
            spec.model_dump(
                mode="json"
            )
        ),
        "runtime_config": (
            runtime_config.model_dump(
                mode="json"
            )
        ),
    }

    digest = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()[:24]

    return (
        f"reranker_run_{report_id}_"
        f"{digest}"
    )


def _validate_index_identity(
    *,
    dense_manifest: object,
    bm25_manifest: object,
) -> None:
    """保证 Dense 和 BM25 绑定同一份 ChunkDataset。"""

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


def _format_optional_rank(
    rank: int | None,
) -> str:
    return (
        "-"
        if rank is None
        else str(rank)
    )


def _nearest_rank_percentile(
    values: list[float],
    percentile: float,
) -> float:
    """计算简单、确定性的 nearest-rank 分位数。"""

    if not values:
        return 0.0

    ordered = sorted(values)

    rank = ceil(
        percentile
        * len(ordered)
    )

    index = max(
        0,
        min(
            rank - 1,
            len(ordered) - 1,
        ),
    )

    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "在 Hybrid RRF 候选上运行 "
            "BGE Cross-Encoder 正式检索评测"
        )
    )

    parser.add_argument(
        "--cases-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--evaluation-set-id",
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
        "--evaluation-output-dir",
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

    cases = (
        load_financial_fact_retrieval_cases(
            args.cases_path
        )
    )

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

    # 正式评测必须保留全部 50 条重排结果。
    # evaluate_financial_fact_retrieval 会自行
    # 只保存最终 Top-5，但使用完整列表计算排名。
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

    hybrid_run_id = _build_hybrid_run_id(
        report_id=(
            dense_manifest.report_id
        ),
        dense_index_id=(
            dense_manifest.index_id
        ),
        bm25_index_id=(
            bm25_manifest.index_id
        ),
        config=rrf_config,
    )

    reranker_run_id = (
        _build_reranker_run_id(
            report_id=(
                dense_manifest.report_id
            ),
            evaluation_set_id=(
                args.evaluation_set_id
            ),
            hybrid_run_id=(
                hybrid_run_id
            ),
            spec=reranker_spec,
            runtime_config=(
                reranker_runtime_config
            ),
        )
    )

    results = []

    case_timings: list[
        dict[str, object]
    ] = []

    hybrid_latencies_ms: list[
        float
    ] = []

    reranker_latencies_ms: list[
        float
    ] = []

    total_latencies_ms: list[
        float
    ] = []

    for case in cases:
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
                "评测题与当前索引的"
                "报告身份不一致："
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

        total_started = perf_counter()

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
            config=(
                reranker_runtime_config
            ),
        )

        reranker_elapsed_ms = (
            perf_counter()
            - reranker_started
        ) * 1000

        total_elapsed_ms = (
            perf_counter()
            - total_started
        ) * 1000

        result = (
            evaluate_financial_fact_retrieval(
                case=case,
                plan=plan,
                strategy=(
                    dense_manifest.chunk_strategy
                ),
                chunk_dataset_id=(
                    dense_manifest
                    .chunk_dataset_id
                ),
                index_id=(
                    reranker_run_id
                ),
                retriever_type=(
                    "hybrid_reranker"
                ),
                hits=reranked_hits,
            )
        )

        results.append(result)

        hybrid_latencies_ms.append(
            hybrid_elapsed_ms
        )

        reranker_latencies_ms.append(
            reranker_elapsed_ms
        )

        total_latencies_ms.append(
            total_elapsed_ms
        )

        case_timings.append(
            {
                "case_id": case.case_id,
                "hybrid_latency_ms": (
                    hybrid_elapsed_ms
                ),
                "reranker_latency_ms": (
                    reranker_elapsed_ms
                ),
                "total_latency_ms": (
                    total_elapsed_ms
                ),
                "hybrid_hit_count": (
                    len(hybrid_hits)
                ),
                "reranked_hit_count": (
                    len(reranked_hits)
                ),
            }
        )

        pages = ",".join(
            str(page)
            for page
            in result.top_pdf_pages
        )

        scores = ",".join(
            f"{score:.6f}"
            for score
            in result.top_scores
        )

        source_ranks = ";".join(
            (
                f"p{hit.pdf_page}:"
                f"rrf={hit.rrf_rank},"
                f"d={_format_optional_rank(hit.dense_rank)},"
                f"b={_format_optional_rank(hit.bm25_rank)}"
            )
            for hit
            in result.top_hits
        )

        print(
            f"case_id={result.case_id} "
            f"first_relevant_rank="
            f"{result.first_relevant_rank} "
            f"top5_pdf_pages={pages} "
            f"top5_scores={scores} "
            f"top5_source_ranks="
            f"{source_ranks} "
            f"hybrid_latency_ms="
            f"{hybrid_elapsed_ms:.2f} "
            f"reranker_latency_ms="
            f"{reranker_elapsed_ms:.2f}"
        )

    summary = summarize_retrieval_results(
        evaluation_set_id=(
            args.evaluation_set_id
        ),
        results=results,
    )

    args.evaluation_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        args.evaluation_output_dir
        / "results.jsonl"
    )

    summary_path = (
        args.evaluation_output_dir
        / "summary.json"
    )

    run_manifest_path = (
        args.evaluation_output_dir
        / "reranker_run.json"
    )

    results_text = (
        "\n".join(
            json.dumps(
                result.model_dump(
                    mode="json"
                ),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for result in results
        )
        + "\n"
    )

    results_path.write_text(
        results_text,
        encoding="utf-8",
    )

    summary_path.write_text(
        summary.model_dump_json(
            indent=2
        )
        + "\n",
        encoding="utf-8",
    )

    run_manifest = {
        "schema_version": 1,
        "evaluation_set_id": (
            args.evaluation_set_id
        ),
        "retriever_type": (
            "hybrid_reranker"
        ),
        "reranker_run_id": (
            reranker_run_id
        ),
        "hybrid_run_id": (
            hybrid_run_id
        ),
        "dense_index_id": (
            dense_manifest.index_id
        ),
        "bm25_index_id": (
            bm25_manifest.index_id
        ),
        "chunk_dataset_id": (
            dense_manifest
            .chunk_dataset_id
        ),
        "report_id": (
            dense_manifest.report_id
        ),
        "embedding_spec": (
            embedding_provider
            .spec
            .model_dump(
                mode="json"
            )
        ),
        "tokenizer_spec": (
            tokenizer
            .spec
            .model_dump(
                mode="json"
            )
        ),
        "rrf_config": (
            rrf_config.model_dump(
                mode="json"
            )
        ),
        "reranker_spec": (
            reranker_spec.model_dump(
                mode="json"
            )
        ),
        "reranker_runtime_config": (
            reranker_runtime_config
            .model_dump(
                mode="json"
            )
        ),
        "dense_index_created": (
            dense_result.created
        ),
        "latency_summary_ms": {
            "hybrid_mean": mean(
                hybrid_latencies_ms
            ),
            "hybrid_p50": (
                _nearest_rank_percentile(
                    hybrid_latencies_ms,
                    0.50,
                )
            ),
            "hybrid_p95": (
                _nearest_rank_percentile(
                    hybrid_latencies_ms,
                    0.95,
                )
            ),
            "reranker_mean": mean(
                reranker_latencies_ms
            ),
            "reranker_p50": (
                _nearest_rank_percentile(
                    reranker_latencies_ms,
                    0.50,
                )
            ),
            "reranker_p95": (
                _nearest_rank_percentile(
                    reranker_latencies_ms,
                    0.95,
                )
            ),
            "total_mean": mean(
                total_latencies_ms
            ),
            "total_p50": (
                _nearest_rank_percentile(
                    total_latencies_ms,
                    0.50,
                )
            ),
            "total_p95": (
                _nearest_rank_percentile(
                    total_latencies_ms,
                    0.95,
                )
            ),
        },
        "case_timings": case_timings,
    }

    run_manifest_path.write_text(
        json.dumps(
            run_manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("-" * 100)

    print(
        "evaluation_set_id="
        f"{summary.evaluation_set_id}"
    )

    print(
        "reranker_run_id="
        f"{reranker_run_id}"
    )

    print(
        "hybrid_run_id="
        f"{hybrid_run_id}"
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
        "retriever_type="
        f"{summary.retriever_type}"
    )

    print(
        "strategy="
        f"{summary.strategy.value}"
    )

    print(
        f"case_count="
        f"{summary.case_count}"
    )

    print(
        f"hit_at_1_count="
        f"{summary.hit_at_1_count}"
    )

    print(
        f"hit_at_3_count="
        f"{summary.hit_at_3_count}"
    )

    print(
        f"hit_at_5_count="
        f"{summary.hit_at_5_count}"
    )

    print(
        f"recall_at_1="
        f"{summary.recall_at_1:.6f}"
    )

    print(
        f"recall_at_3="
        f"{summary.recall_at_3:.6f}"
    )

    print(
        f"recall_at_5="
        f"{summary.recall_at_5:.6f}"
    )

    print(
        f"mrr={summary.mrr:.6f}"
    )

    print(
        f"ndcg_at_1="
        f"{summary.ndcg_at_1:.6f}"
    )

    print(
        f"ndcg_at_3="
        f"{summary.ndcg_at_3:.6f}"
    )

    print(
        f"ndcg_at_5="
        f"{summary.ndcg_at_5:.6f}"
    )

    print(
        "reranker_latency_mean_ms="
        f"{mean(reranker_latencies_ms):.2f}"
    )

    print(
        "reranker_latency_p50_ms="
        f"{_nearest_rank_percentile(reranker_latencies_ms, 0.50):.2f}"
    )

    print(
        "reranker_latency_p95_ms="
        f"{_nearest_rank_percentile(reranker_latencies_ms, 0.95):.2f}"
    )

    print(
        f"results_path={results_path}"
    )

    print(
        f"summary_path={summary_path}"
    )

    print(
        "run_manifest_path="
        f"{run_manifest_path}"
    )


if __name__ == "__main__":
    main()