from __future__ import annotations

import argparse
from pathlib import Path

from app.schemas.chunk import (
    FixedLengthChunkingConfig,
    ParagraphChunkingConfig,
    SectionParagraphChunkingConfig,
)
from app.schemas.enums import ChunkStrategy

from app.services.chunk_dataset import (
    build_chunk_dataset,
)
from app.services.registry_loader import (
    load_reports,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "从页面数据集构建固定长度 "
            "Chunk 数据集"
        )
    )

    parser.add_argument(
        "--reports-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--report-id",
        required=True,
    )

    parser.add_argument(
        "--page-dataset-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
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

    parser.add_argument(
        "--strategy",
        choices=[
            ChunkStrategy.FIXED_LENGTH.value,
            ChunkStrategy.PARAGRAPH.value,
            ChunkStrategy.SECTION_PARAGRAPH.value,
        ],
        default=(
            ChunkStrategy.FIXED_LENGTH.value
        ),
    )

    parser.add_argument(
        "--overlap-paragraphs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--long-paragraph-overlap-chars",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--max-heading-chars",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--no-inherit-section-across-pages",
        action="store_true",
    )

    args = parser.parse_args()

    report_registry, _ = load_reports(
        args.reports_path
    )

    report = report_registry.require(
        args.report_id
    )

    if (
        args.strategy
        == ChunkStrategy.FIXED_LENGTH.value
    ):
        config = FixedLengthChunkingConfig(
            max_chars=args.max_chars,
            overlap_chars=(
                args.overlap_chars
            ),
        )

    elif (
        args.strategy
        == ChunkStrategy.PARAGRAPH.value
    ):
        config = ParagraphChunkingConfig(
            max_chars=args.max_chars,
            overlap_paragraphs=(
                args.overlap_paragraphs
            ),
            long_paragraph_overlap_chars=(
                args
                .long_paragraph_overlap_chars
            ),
        )

    else:
        config = (
            SectionParagraphChunkingConfig(
                max_chars=args.max_chars,
                overlap_paragraphs=(
                    args.overlap_paragraphs
                ),
                long_paragraph_overlap_chars=(
                    args
                    .long_paragraph_overlap_chars
                ),
                max_heading_chars=(
                    args.max_heading_chars
                ),
                inherit_section_across_pages=(
                    not args
                    .no_inherit_section_across_pages
                ),
            )
        )
    result = build_chunk_dataset(
        report=report,
        page_dataset_directory=(
            args.page_dataset_dir
        ),
        output_root=args.output_root,
        config=config,
    )

    manifest = result.manifest

    print(
        f"report_id={manifest.report_id}"
    )

    print(
        "page_dataset_id="
        f"{manifest.page_dataset_id}"
    )

    print(
        "chunk_dataset_id="
        f"{manifest.dataset_id}"
    )

    print(
        f"strategy={manifest.strategy.value}"
    )

    print(
        "max_chars="
        f"{manifest.chunking_config.max_chars}"
    )

    if isinstance(
        manifest.chunking_config,
        FixedLengthChunkingConfig,
    ):
        print(
            "overlap_chars="
            f"{manifest.chunking_config.overlap_chars}"
        )

    if isinstance(
        manifest.chunking_config,
        (
            ParagraphChunkingConfig,
            SectionParagraphChunkingConfig,
        ),
    ):
        print(
            "overlap_paragraphs="
            f"{manifest.chunking_config.overlap_paragraphs}"
        )

        print(
            "long_paragraph_overlap_chars="
            f"{manifest.chunking_config.long_paragraph_overlap_chars}"
        )

    if isinstance(
        manifest.chunking_config,
        SectionParagraphChunkingConfig,
    ):
        print(
            "max_heading_chars="
            f"{manifest.chunking_config.max_heading_chars}"
        )

        print(
            "inherit_section_across_pages="
            f"{manifest.chunking_config.inherit_section_across_pages}"
        )

    print(
        "input_page_count="
        f"{manifest.input_page_count}"
    )

    print(
        "eligible_page_count="
        f"{manifest.eligible_page_count}"
    )

    print(
        "skipped_page_count="
        f"{manifest.skipped_page_count}"
    )

    print(
        "chunk_record_count="
        f"{manifest.chunk_record_count}"
    )

    print(
        "quality_gate_passed="
        f"{manifest.quality_gate_passed}"
    )

    print(
        f"created={result.created}"
    )

    print(
        "dataset_directory="
        f"{result.dataset_directory}"
    )


if __name__ == "__main__":
    main()