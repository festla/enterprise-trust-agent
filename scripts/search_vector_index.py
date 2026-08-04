from __future__ import annotations

import argparse
from pathlib import Path

from app.rag.embedders import (
    BGE_SMALL_ZH_V15_REVISION,
    SentenceTransformerEmbeddingProvider,
    build_bge_small_zh_v15_spec,
)
from app.schemas.retrieval import (
    RetrievalFilter,
)
from app.services.vector_index import (
    build_vector_index,
)
from app.rag.query_planning import (
    build_financial_fact_query_plan,
)
from app.schemas.enums import (
    StatementScope,
    StatementType,
)

def _parse_pdf_pages(
    value: str,
) -> tuple[int, ...]:
    pages = tuple(
        sorted(
            {
                int(item.strip())
                for item in value.split(",")
                if item.strip()
            }
        )
    )

    if any(page < 1 for page in pages):
        raise argparse.ArgumentTypeError(
            "PDF 页码必须大于等于 1"
        )

    return pages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="查询持久化精确向量索引"
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
        "--query",
        required=True,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--pdf-pages",
        type=_parse_pdf_pages,
        default=(),
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
        "--target-pdf-page",
        type=int,
        default=None,
        help=(
            "诊断指定 PDF 页在全部检索结果中的"
            "最佳全局排名，不输出全部结果"
        ),
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
        "--metric-name",
        default=None,
        help=(
            "填写后启用 financial_fact "
            "结构化 Query Planning"
        ),
    )

    parser.add_argument(
        "--statement-type",
        choices=[
            StatementType.BALANCE_SHEET.value,
            StatementType.INCOME_STATEMENT.value,
            StatementType.CASH_FLOW_STATEMENT.value,
            (
                StatementType
                .STATEMENT_OF_CHANGES_IN_EQUITY
                .value
            ),
            StatementType.FINANCIAL_SUMMARY.value,
            StatementType.NOTE.value,
        ],
        default=(
            StatementType.INCOME_STATEMENT.value
        ),
    )

    parser.add_argument(
        "--statement-scope",
        choices=[
            StatementScope.CONSOLIDATED.value,
            StatementScope.PARENT_COMPANY.value,
            StatementScope.GROUP.value,
        ],
        default=(
            StatementScope.CONSOLIDATED.value
        ),
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
        )
    )

    result = build_vector_index(
        chunk_dataset_directory=(
            args.chunk_dataset_dir
        ),
        output_root=args.output_root,
        provider=provider,
    )

    
    effective_top_k = (
        result.manifest.vector_count
        if args.target_pdf_page is not None
        else args.top_k
    )

    if args.metric_name is not None:
        query_plan = (
            build_financial_fact_query_plan(
                original_query=args.query,
                metric_name=args.metric_name,
                fiscal_year=(
                    result.manifest.fiscal_year
                ),
                company_id=(
                    result.manifest.company_id
                ),
                report_id=(
                    result.manifest.report_id
                ),
                report_type=(
                    result.manifest.report_type
                ),
                statement_type=StatementType(
                    args.statement_type
                ),
                statement_scope=StatementScope(
                    args.statement_scope
                ),
                pdf_pages=args.pdf_pages,
            )
        )

        semantic_query = (
            query_plan.semantic_query
        )

        filters = query_plan.filters

        query_mode = "financial_fact_plan"

    else:
        semantic_query = args.query

        filters = RetrievalFilter(
            pdf_pages=args.pdf_pages
        )

        query_mode = "raw"


    hits = result.index.search(
        query=semantic_query,
        provider=provider,
        top_k=effective_top_k,
        filters=filters,
    )

    print(
        f"query_mode={query_mode}"
    )

    print(
        f"original_query={args.query}"
    )

    print(
        f"semantic_query={semantic_query}"
    )

    print(
        f"index_id={result.manifest.index_id}"
    )

    print(
        f"candidate_result_count={len(hits)}"
    )


    if args.target_pdf_page is not None:
        target_hits = tuple(
            hit
            for hit in hits
            if (
                hit.pdf_page
                == args.target_pdf_page
            )
        )

        print(
            "target_pdf_page="
            f"{args.target_pdf_page}"
        )

        if not target_hits:
            print(
                "target_page_found=False"
            )
            return

        best_hit = target_hits[0]

        print(
            "target_page_found=True"
        )

        print(
            "target_best_rank="
            f"{best_hit.rank}"
        )

        print(
            "target_best_score="
            f"{best_hit.score:.6f}"
        )

        print(
            "target_chunk_count="
            f"{len(target_hits)}"
        )

        print("-" * 80)

        print(
            f"chunk_id={best_hit.chunk_id}"
        )

        print(
            f"printed_page="
            f"{best_hit.printed_page}"
        )

        preview = best_hit.text.replace(
            "\n",
            " ",
        )[:500]

        print(
            f"text={preview}"
        )

        return

    for hit in hits:
        preview = hit.text.replace(
            "\n",
            " ",
        )[:240]

        section = (
            " > ".join(hit.section_path)
            if hit.section_path
            else "-"
        )

        print("-" * 80)

        print(
            f"rank={hit.rank}"
        )

        print(
            f"score={hit.score:.6f}"
        )

        print(
            f"chunk_id={hit.chunk_id}"
        )

        print(
            f"pdf_page={hit.pdf_page}"
        )

        print(
            "printed_page="
            f"{hit.printed_page}"
        )

        print(
            f"section={section}"
        )

        print(
            f"text={preview}"
        )


if __name__ == "__main__":
    main()