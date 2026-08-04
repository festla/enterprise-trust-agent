from __future__ import annotations

import argparse
from pathlib import Path

from app.rag.tokenization import (
    DeterministicChineseBigramTokenizer,
)
from app.schemas.retrieval import (
    RetrievalFilter,
)
from app.services.bm25_index import (
    load_bm25_index,
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

    return normalized[:max_chars] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "加载持久化 BM25 索引并执行检索"
        )
    )

    parser.add_argument(
        "--index-dir",
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
        "--pdf-pages",
        type=int,
        nargs="*",
        default=(),
    )

    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    result = load_bm25_index(
        args.index_dir
    )

    manifest = result.manifest

    tokenizer = (
        DeterministicChineseBigramTokenizer(
            spec=manifest.tokenizer_spec
        )
    )

    filters = RetrievalFilter(
        company_ids=(
            manifest.company_id,
        ),
        report_ids=(
            manifest.report_id,
        ),
        fiscal_years=(
            manifest.fiscal_year,
        ),
        report_types=(
            manifest.report_type,
        ),
        document_ids=(
            manifest.document_id,
        ),
        pdf_pages=tuple(
            args.pdf_pages
        ),
    )

    hits = result.index.search(
        query=args.query,
        tokenizer=tokenizer,
        top_k=args.top_k,
        filters=filters,
    )

    query_tokens = tokenizer.tokenize(
        args.query
    )

    print(f"index_id={manifest.index_id}")
    print(f"report_id={manifest.report_id}")
    print(f"query={args.query}")

    print(
        "query_tokens="
        + " | ".join(query_tokens)
    )

    print(f"returned_hit_count={len(hits)}")

    for hit in hits:
        print("-" * 80)

        print(
            f"rank={hit.rank} "
            f"score={hit.score:.6f} "
            f"pdf_page={hit.pdf_page} "
            f"printed_page={hit.printed_page}"
        )

        print(f"chunk_id={hit.chunk_id}")

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