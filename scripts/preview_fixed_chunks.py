from __future__ import annotations

import argparse
from pathlib import Path

from app.rag.chunking import (
    build_chunk_dataset_id,
    build_fixed_length_chunks_for_pages,
)
from app.schemas.chunk import (
    FixedLengthChunkingConfig,
)
from app.schemas.page import ParsedPage
from app.schemas.page_dataset import (
    PageDatasetManifest,
)
from app.services.registry_loader import (
    load_reports,
)


def _parse_pdf_pages(
    value: str,
) -> tuple[int, ...]:
    try:
        pages = tuple(
            sorted(
                {
                    int(item.strip())
                    for item
                    in value.split(",")
                    if item.strip()
                }
            )
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--pdf-pages 必须是逗号分隔的整数，"
            "例如 34,35"
        ) from exc

    if (
        not pages
        or any(page < 1 for page in pages)
    ):
        raise argparse.ArgumentTypeError(
            "--pdf-pages 必须至少包含一个正整数"
        )

    return pages


def _load_selected_pages(
    *,
    pages_path: Path,
    selected_pdf_pages: tuple[int, ...],
) -> tuple[ParsedPage, ...]:
    selected_set = set(
        selected_pdf_pages
    )

    selected_pages: list[
        ParsedPage
    ] = []

    for line in pages_path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        page = ParsedPage.model_validate_json(
            line
        )

        if page.pdf_page in selected_set:
            selected_pages.append(page)

    found_pages = {
        page.pdf_page
        for page in selected_pages
    }

    missing_pages = sorted(
        selected_set - found_pages
    )

    if missing_pages:
        raise ValueError(
            "pages.jsonl 中缺少 PDF 页码："
            f"{missing_pages}"
        )

    return tuple(
        sorted(
            selected_pages,
            key=lambda page: page.pdf_page,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reports-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--page-dataset-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--report-id",
        required=True,
    )

    parser.add_argument(
        "--pdf-pages",
        type=_parse_pdf_pages,
        required=True,
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=800,
    )

    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=120,
    )

    args = parser.parse_args()

    report_registry, _ = load_reports(
        args.reports_path
    )

    report = report_registry.require(
        args.report_id
    )

    manifest_path = (
        args.page_dataset_dir
        / "dataset_manifest.json"
    )

    pages_path = (
        args.page_dataset_dir
        / "pages.jsonl"
    )

    manifest = (
        PageDatasetManifest
        .model_validate_json(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    )

    pages = _load_selected_pages(
        pages_path=pages_path,
        selected_pdf_pages=(
            args.pdf_pages
        ),
    )

    config = FixedLengthChunkingConfig(
        max_chars=args.max_chars,
        overlap_chars=(
            args.overlap_chars
        ),
    )

    chunk_dataset_id = (
        build_chunk_dataset_id(
            report=report,
            page_dataset_manifest=manifest,
            config=config,
        )
    )

    chunks = (
        build_fixed_length_chunks_for_pages(
            pages=pages,
            report=report,
            page_dataset_manifest=manifest,
            config=config,
        )
    )

    print(
        f"report_id={report.report_id}"
    )

    print(
        "page_dataset_id="
        f"{manifest.dataset_id}"
    )

    print(
        "chunk_dataset_id="
        f"{chunk_dataset_id}"
    )

    print(
        f"selected_page_count={len(pages)}"
    )

    print(
        f"chunk_count={len(chunks)}"
    )

    for chunk in chunks:
        preview = chunk.text.replace(
            "\n",
            " ",
        )[:120]

        print("-" * 70)

        print(
            {
                "chunk_id": chunk.chunk_id,
                "pdf_page": chunk.pdf_page,
                "printed_page": (
                    chunk.printed_page
                ),
                "page_id": chunk.page_id,
                "chunk_index": (
                    chunk.chunk_index
                ),
                "source_range": (
                    chunk.source_start_char,
                    chunk.source_end_char,
                ),
                "char_count": (
                    chunk.char_count
                ),
                "preview": preview,
            }
        )


if __name__ == "__main__":
    main()