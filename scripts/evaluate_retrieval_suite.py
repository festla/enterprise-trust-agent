from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.rag.embedders import (
    BGE_SMALL_ZH_V15_REVISION,
    SentenceTransformerEmbeddingProvider,
    build_bge_small_zh_v15_spec,
)
from app.rag.query_planning import (
    build_financial_fact_query_plan,
)
from app.rag.retrieval_evaluation import (
    evaluate_financial_fact_retrieval,
    summarize_retrieval_results,
)
from app.services.retrieval_eval_dataset import (
    load_financial_fact_retrieval_cases,
)
from app.services.vector_index import (
    build_vector_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "在同一向量索引上批量运行财务事实"
            "检索评测"
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
        "--index-output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--evaluation-output-dir",
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

    index_result = build_vector_index(
        chunk_dataset_directory=(
            args.chunk_dataset_dir
        ),
        output_root=args.index_output_root,
        provider=provider,
    )

    manifest = index_result.manifest

    results = []

    for case in cases:
        if (
            case.company_id
            != manifest.company_id
            or case.report_id
            != manifest.report_id
            or case.fiscal_year
            != manifest.fiscal_year
            or case.report_type
            != manifest.report_type
        ):
            raise ValueError(
                "评测题与当前 VectorIndex "
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

        hits = index_result.index.search(
            query=plan.semantic_query,
            provider=provider,
            top_k=manifest.vector_count,
            filters=plan.filters,
        )

        result = evaluate_financial_fact_retrieval(
            case=case,
            plan=plan,
            strategy=manifest.chunk_strategy,
            chunk_dataset_id=(
                manifest.chunk_dataset_id
            ),
            index_id=manifest.index_id,
            retriever_type="dense",
            hits=hits,
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
        summary.model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )

    print(
        "evaluation_set_id="
        f"{summary.evaluation_set_id}"
    )

    print(
        f"strategy={summary.strategy.value}"
    )

    print(
        f"case_count={summary.case_count}"
    )

    print(
        "hit_at_1_count="
        f"{summary.hit_at_1_count}"
    )

    print(
        "hit_at_3_count="
        f"{summary.hit_at_3_count}"
    )

    print(
        "hit_at_5_count="
        f"{summary.hit_at_5_count}"
    )

    print(
        f"recall_at_1={summary.recall_at_1:.6f}"
    )

    print(
        f"recall_at_3={summary.recall_at_3:.6f}"
    )

    print(
        f"recall_at_5={summary.recall_at_5:.6f}"
    )

    print(f"mrr={summary.mrr:.6f}")

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


if __name__ == "__main__":
    main()