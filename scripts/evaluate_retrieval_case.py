from __future__ import annotations

import argparse
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
)
from app.schemas.enums import (
    StatementScope,
    StatementType,
)
from app.schemas.retrieval_eval import (
    FinancialFactRetrievalEvalCase,
)
from app.services.vector_index import (
    build_vector_index,
)


def _parse_pdf_pages(
    value: str,
) -> tuple[int, ...]:
    try:
        pages = tuple(
            sorted(
                {
                    int(item.strip())
                    for item in value.split(",")
                    if item.strip()
                }
            )
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "PDF 页码必须是整数"
        ) from exc

    if (
        not pages
        or any(page < 1 for page in pages)
    ):
        raise argparse.ArgumentTypeError(
            "至少提供一个合法 Gold PDF 页码"
        )

    return pages


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "运行一条财务事实检索评测并保存结果"
        )
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
        "--evaluation-output-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--case-id",
        required=True,
    )

    parser.add_argument(
        "--query",
        required=True,
    )

    parser.add_argument(
        "--metric-name",
        required=True,
    )

    parser.add_argument(
        "--gold-pdf-pages",
        type=_parse_pdf_pages,
        required=True,
    )

    parser.add_argument(
        "--statement-type",
        choices=[
            item.value
            for item in StatementType
        ],
        default=(
            StatementType
            .INCOME_STATEMENT
            .value
        ),
    )

    parser.add_argument(
        "--statement-scope",
        choices=[
            item.value
            for item in StatementScope
        ],
        default=(
            StatementScope
            .CONSOLIDATED
            .value
        ),
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

    case = FinancialFactRetrievalEvalCase(
        case_id=args.case_id,
        question=args.query,
        metric_name=args.metric_name,
        company_id=manifest.company_id,
        report_id=manifest.report_id,
        fiscal_year=manifest.fiscal_year,
        report_type=manifest.report_type,
        statement_type=StatementType(
            args.statement_type
        ),
        statement_scope=StatementScope(
            args.statement_scope
        ),
        gold_pdf_pages=(
            args.gold_pdf_pages
        ),
    )

    plan = build_financial_fact_query_plan(
        original_query=case.question,
        metric_name=case.metric_name,
        fiscal_year=case.fiscal_year,
        company_id=case.company_id,
        report_id=case.report_id,
        report_type=case.report_type,
        statement_type=case.statement_type,
        statement_scope=case.statement_scope,
    )

    hits = index_result.index.search(
        query=plan.semantic_query,
        provider=provider,
        top_k=manifest.vector_count,
        filters=plan.filters,
    )

    evaluation = (
        evaluate_financial_fact_retrieval(
            case=case,
            plan=plan,
            strategy=(
                manifest.chunk_strategy
            ),
            chunk_dataset_id=(
                manifest.chunk_dataset_id
            ),
            vector_index_id=(
                manifest.index_id
            ),
            hits=hits,
        )
    )

    args.evaluation_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.evaluation_output_path.write_text(
        evaluation.model_dump_json(
            indent=2
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"case_id={evaluation.case_id}"
    )

    print(
        f"strategy={evaluation.strategy.value}"
    )

    print(
        "semantic_query="
        f"{evaluation.semantic_query}"
    )

    print(
        "evaluated_hit_count="
        f"{evaluation.evaluated_hit_count}"
    )

    print(
        "first_relevant_rank="
        f"{evaluation.first_relevant_rank}"
    )

    print(
        f"recall_at_1={evaluation.recall_at_1}"
    )

    print(
        f"recall_at_3={evaluation.recall_at_3}"
    )

    print(
        f"recall_at_5={evaluation.recall_at_5}"
    )

    print(
        "evaluation_output_path="
        f"{args.evaluation_output_path}"
    )


if __name__ == "__main__":
    main()