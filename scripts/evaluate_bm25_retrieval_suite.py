from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from app.services.bm25_index import (
    load_bm25_index,
)
from app.services.retrieval_eval_dataset import (
    load_financial_fact_retrieval_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "在持久化 BM25 索引上运行"
            "财务事实检索评测"
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
        "--index-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--evaluation-output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    cases = (
        load_financial_fact_retrieval_cases(
            args.cases_path
        )
    )

    index_result = load_bm25_index(
        args.index_dir
    )

    manifest = index_result.manifest

    tokenizer = (
        DeterministicChineseBigramTokenizer(
            spec=manifest.tokenizer_spec
        )
    )

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
                "评测题与当前 BM25 Index "
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
            tokenizer=tokenizer,
            top_k=manifest.document_count,
            filters=plan.filters,
        )

        result = (
            evaluate_financial_fact_retrieval(
                case=case,
                plan=plan,
                strategy=(
                    manifest.chunk_strategy
                ),
                chunk_dataset_id=(
                    manifest.chunk_dataset_id
                ),
                index_id=manifest.index_id,
                retriever_type="bm25",
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

    for result in results:
        pages = ",".join(
            str(page)
            for page in result.top_pdf_pages
        )

        scores = ",".join(
            f"{score:.6f}"
            for score in result.top_scores
        )

        print(
            f"case_id={result.case_id} "
            f"first_relevant_rank="
            f"{result.first_relevant_rank} "
            f"top5_pdf_pages={pages} "
            f"top5_scores={scores}"
        )

    print("-" * 80)

    print(
        "evaluation_set_id="
        f"{summary.evaluation_set_id}"
    )

    print(
        f"retriever_type="
        f"{summary.retriever_type}"
    )

    print(
        f"strategy={summary.strategy.value}"
    )

    print(
        f"case_count={summary.case_count}"
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

    print(f"results_path={results_path}")
    print(f"summary_path={summary_path}")


if __name__ == "__main__":
    main()