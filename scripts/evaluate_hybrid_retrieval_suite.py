from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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
from app.services.bm25_index import (
    load_bm25_index,
)
from app.services.retrieval_eval_dataset import (
    load_financial_fact_retrieval_cases,
)
from app.services.vector_index import (
    build_vector_index,
)


def _canonical_json_bytes(
    value: object,
) -> bytes:
    """生成稳定、跨运行一致的 JSON 字节。"""

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
    """根据两套索引和 RRF 配置生成稳定运行 ID。"""

    payload = {
        "schema_version": 1,
        "retriever_type": "hybrid_rrf",
        "dense_index_id": dense_index_id,
        "bm25_index_id": bm25_index_id,
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


def _validate_index_identity(
    *,
    dense_manifest: object,
    bm25_manifest: object,
) -> None:
    """保证两套索引绑定同一份 ChunkDataset。"""

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "在持久化 Dense 与 BM25 索引上"
            "运行 Hybrid RRF 财务事实检索评测"
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

    args = parser.parse_args()

    cases = (
        load_financial_fact_retrieval_cases(
            args.cases_path
        )
    )

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

    # 当前 Vector Service 的幂等入口：
    # 已有索引时进行验证并直接复用。
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
        config=config,
    )

    # RRF 最多产生两路候选数量之和。
    # 使用全部融合候选评测 first_relevant_rank，
    # 而不是只评测最终 Top-5。
    hybrid_evaluation_top_k = (
        config.dense_candidate_count
        + config.bm25_candidate_count
    )

    results = []

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
                "评测题与当前 Hybrid Index "
                "的报告身份不一致："
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

        hits = retriever.search(
            query=plan.semantic_query,
            top_k=(
                hybrid_evaluation_top_k
            ),
            filters=plan.filters,
        )

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
                index_id=hybrid_run_id,
                retriever_type=(
                    "hybrid_rrf"
                ),
                hits=hits,
            )
        )

        results.append(result)

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
        / "hybrid_run.json"
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
        "hybrid_run_id": hybrid_run_id,
        "evaluation_set_id": (
            args.evaluation_set_id
        ),
        "retriever_type": (
            "hybrid_rrf"
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
            provider.spec.model_dump(
                mode="json"
            )
        ),
        "tokenizer_spec": (
            tokenizer.spec.model_dump(
                mode="json"
            )
        ),
        "rrf_config": config.model_dump(
            mode="json"
        ),
        "hybrid_evaluation_top_k": (
            hybrid_evaluation_top_k
        ),
        "dense_index_created": (
            dense_result.created
        ),
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

    for result in results:
        pages = ",".join(
            str(page)
            for page in result.top_pdf_pages
        )

        scores = ",".join(
            f"{score:.6f}"
            for score in result.top_scores
        )

        source_ranks = ";".join(
            (
                f"p{hit.pdf_page}:"
                f"d={_format_optional_rank(hit.dense_rank)},"
                f"b={_format_optional_rank(hit.bm25_rank)}"
            )
            for hit in result.top_hits[:5]
        )

        print(
            f"case_id={result.case_id} "
            f"first_relevant_rank="
            f"{result.first_relevant_rank} "
            f"top5_pdf_pages={pages} "
            f"top5_scores={scores} "
            f"top5_source_ranks="
            f"{source_ranks}"
        )

    print("-" * 80)

    print(
        "evaluation_set_id="
        f"{summary.evaluation_set_id}"
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
        f"retriever_type="
        f"{summary.retriever_type}"
    )

    print(
        f"strategy="
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