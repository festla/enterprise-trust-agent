from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from app.schemas.competition import (
    CompetitionQaCase,
    CompetitionSourceRecord,
)
from app.schemas.competition_chunk import (
    CompetitionTextChunk,
)
from app.schemas.competition_text import (
    CompetitionTextDocument,
)
from app.services.competition_dataset import (
    build_competition_question,
    load_competition_qa_excel,
)
from app.services.competition_source_resolver import (
    CompetitionSourceResolver,
    build_competition_source_manifest,
)
from app.services.competition_text_chunker import (
    build_competition_text_chunks,
)
from app.services.competition_text_parser import (
    parse_competition_text_document,
)


PdfSourcePair = tuple[
    CompetitionQaCase,
    CompetitionSourceRecord,
]


@dataclass(frozen=True)
class PdfTableChunkSmokeSummary:
    source_count: int
    success_count: int
    failure_count: int
    table_source_count: int
    table_block_count: int
    table_chunk_count: int
    failures: tuple[
        tuple[str, str],
        ...
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse and chunk QA-used PDF tables "
            "without printing questions, answers, "
            "evidence, or document content."
        )
    )

    parser.add_argument(
        "--qa",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--attachments",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--expected-sources",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--minimum-table-sources",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--minimum-table-blocks",
        type=int,
        default=37,
    )

    return parser.parse_args()


def collect_qa_used_pdf_sources(
    *,
    cases: tuple[
        CompetitionQaCase,
        ...
    ],
    source_manifest: tuple[
        CompetitionSourceRecord,
        ...
    ],
) -> dict[
    str,
    PdfSourcePair,
]:
    resolver = CompetitionSourceResolver(
        source_manifest
    )

    source_by_id = {
        source.source_id: source
        for source in source_manifest
    }

    result: dict[
        str,
        PdfSourcePair,
    ] = {}

    for case in cases:
        if case.source_type != "pdf":
            continue

        resolution = resolver.resolve(
            case
        )

        source = source_by_id[
            resolution.source_id
        ]

        if (
            source.extension.casefold()
            != ".pdf"
        ):
            continue

        result.setdefault(
            source.source_id,
            (
                case,
                source,
            ),
        )

    return dict(
        sorted(
            result.items()
        )
    )


def validate_pdf_table_chunks(
    *,
    document: CompetitionTextDocument,
    chunks: tuple[
        CompetitionTextChunk,
        ...
    ],
) -> list[str]:
    errors: list[str] = []

    if (
        document.source.source_type
        != "pdf"
    ):
        errors.append(
            "document source_type is not pdf"
        )

    chunk_ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    if (
        len(chunk_ids)
        != len(set(chunk_ids))
    ):
        errors.append(
            "duplicate chunk IDs"
        )

    chunk_indexes = [
        chunk.chunk_index
        for chunk in chunks
    ]

    if chunk_indexes != list(
        range(len(chunks))
    ):
        errors.append(
            "chunk indexes are not contiguous"
        )

    table_blocks = {
        block.block_id: block
        for block in document.blocks
        if block.block_type == "table"
    }

    table_chunks = [
        chunk
        for chunk in chunks
        if chunk.chunk_type == "table"
    ]

    chunks_by_block: dict[
        str,
        list[CompetitionTextChunk],
    ] = {}

    for chunk in table_chunks:
        if chunk.source_type != "pdf":
            errors.append(
                f"{chunk.chunk_id}: "
                "table chunk source_type is not pdf"
            )

        if len(chunk.source_spans) != 1:
            errors.append(
                f"{chunk.chunk_id}: "
                "table chunk must have one source span"
            )

            continue

        block_id = (
            chunk.source_spans[0]
            .block_id
        )

        if block_id not in table_blocks:
            errors.append(
                f"{chunk.chunk_id}: "
                "table chunk references unknown block"
            )

            continue

        chunks_by_block.setdefault(
            block_id,
            [],
        ).append(
            chunk
        )

    for block_id, block in (
        table_blocks.items()
    ):
        related = chunks_by_block.get(
            block_id,
            [],
        )

        if not related:
            errors.append(
                f"{block_id}: missing table chunks"
            )

            continue

        related.sort(
            key=lambda chunk: (
                chunk.table_row_start
                if chunk.table_row_start
                is not None
                else -1
            )
        )

        expected_row_start = 0
        restored_rows: list[
            tuple[str, ...]
        ] = []

        for chunk in related:
            if (
                chunk.table_row_start
                != expected_row_start
            ):
                errors.append(
                    f"{block_id}: "
                    "table row ranges are not contiguous"
                )

            if chunk.table_row_end is None:
                errors.append(
                    f"{block_id}: "
                    "table_row_end is missing"
                )

                continue

            if (
                chunk.table_index
                != block.table_index
            ):
                errors.append(
                    f"{block_id}: "
                    "table_index changed"
                )

            if (
                chunk.page_start
                != block.page
                or chunk.page_end
                != block.page
            ):
                errors.append(
                    f"{block_id}: "
                    "page location changed"
                )

            if (
                chunk.pdf_bbox
                != block.pdf_bbox
            ):
                errors.append(
                    f"{block_id}: "
                    "pdf_bbox changed"
                )

            restored_rows.extend(
                chunk.table_rows
            )

            expected_row_start = (
                chunk.table_row_end + 1
            )

        if (
            tuple(restored_rows)
            != block.table_rows
        ):
            errors.append(
                f"{block_id}: "
                "table rows were lost or duplicated"
            )

    return errors


def run_pdf_table_chunk_smoke(
    *,
    cases: tuple[
        CompetitionQaCase,
        ...
    ],
    source_manifest: tuple[
        CompetitionSourceRecord,
        ...
    ],
    attachments_root: Path,
) -> PdfTableChunkSmokeSummary:
    sources = collect_qa_used_pdf_sources(
        cases=cases,
        source_manifest=source_manifest,
    )

    success_count = 0
    table_source_count = 0
    table_block_count = 0
    table_chunk_count = 0

    failures: list[
        tuple[str, str]
    ] = []

    print(
        "=== QA-used PDF Table Chunk Smoke ==="
    )
    print(
        "Unique PDF sources:",
        len(sources),
    )
    print()

    for source_id, (
        case,
        source,
    ) in sources.items():
        try:
            question = (
                build_competition_question(
                    case
                )
            )

            document = (
                parse_competition_text_document(
                    question=question,
                    source=source,
                    attachments_root=(
                        attachments_root
                    ),
                )
            )

            chunks = (
                build_competition_text_chunks(
                    document
                )
            )

            table_blocks = [
                block
                for block in document.blocks
                if block.block_type
                == "table"
            ]

            table_chunks = [
                chunk
                for chunk in chunks
                if chunk.chunk_type
                == "table"
            ]

            errors = (
                validate_pdf_table_chunks(
                    document=document,
                    chunks=chunks,
                )
            )

            if errors:
                failures.append(
                    (
                        source_id,
                        "; ".join(errors),
                    )
                )

                print(
                    f"[FAIL] {source_id} "
                    f"errors={len(errors)}"
                )

                continue

            success_count += 1

            if table_blocks:
                table_source_count += 1

            table_block_count += len(
                table_blocks
            )

            table_chunk_count += len(
                table_chunks
            )

            print(
                f"[OK]   {source_id} "
                f"blocks={len(document.blocks)} "
                f"chunks={len(chunks)} "
                f"table_blocks={len(table_blocks)} "
                f"table_chunks={len(table_chunks)}"
            )

        except Exception as exc:
            failures.append(
                (
                    source_id,
                    type(exc).__name__,
                )
            )

            print(
                f"[FAIL] {source_id} "
                f"error={type(exc).__name__}"
            )

    return PdfTableChunkSmokeSummary(
        source_count=len(sources),
        success_count=success_count,
        failure_count=len(failures),
        table_source_count=(
            table_source_count
        ),
        table_block_count=(
            table_block_count
        ),
        table_chunk_count=(
            table_chunk_count
        ),
        failures=tuple(failures),
    )


def collect_gate_errors(
    *,
    summary: PdfTableChunkSmokeSummary,
    expected_source_count: int,
    minimum_table_sources: int,
    minimum_table_blocks: int,
) -> list[str]:
    errors: list[str] = []

    if (
        summary.source_count
        != expected_source_count
    ):
        errors.append(
            "PDF source count mismatch: "
            f"expected={expected_source_count}, "
            f"actual={summary.source_count}"
        )

    if summary.failure_count:
        errors.append(
            "PDF table chunk failures: "
            f"{summary.failure_count}"
        )

    if (
        summary.success_count
        != summary.source_count
    ):
        errors.append(
            "PDF success count does not "
            "match source count"
        )

    if (
        summary.table_source_count
        < minimum_table_sources
    ):
        errors.append(
            "Too few PDF table sources: "
            f"minimum={minimum_table_sources}, "
            f"actual={summary.table_source_count}"
        )

    if (
        summary.table_block_count
        < minimum_table_blocks
    ):
        errors.append(
            "Too few PDF table blocks: "
            f"minimum={minimum_table_blocks}, "
            f"actual={summary.table_block_count}"
        )

    if (
        summary.table_chunk_count
        < summary.table_block_count
    ):
        errors.append(
            "PDF table chunk count is lower "
            "than table block count"
        )

    return errors


def main() -> None:
    args = parse_args()

    cases = load_competition_qa_excel(
        args.qa
    )

    source_manifest = (
        build_competition_source_manifest(
            args.attachments
        )
    )

    summary = run_pdf_table_chunk_smoke(
        cases=cases,
        source_manifest=source_manifest,
        attachments_root=(
            args.attachments
        ),
    )

    print()
    print("=== Summary ===")
    print(
        "Sources:",
        summary.source_count,
    )
    print(
        "Success:",
        summary.success_count,
    )
    print(
        "Failure:",
        summary.failure_count,
    )
    print(
        "Sources with tables:",
        summary.table_source_count,
    )
    print(
        "Table blocks:",
        summary.table_block_count,
    )
    print(
        "Table chunks:",
        summary.table_chunk_count,
    )

    errors = collect_gate_errors(
        summary=summary,
        expected_source_count=(
            args.expected_sources
        ),
        minimum_table_sources=(
            args.minimum_table_sources
        ),
        minimum_table_blocks=(
            args.minimum_table_blocks
        ),
    )

    print()
    print("=== Smoke Result ===")
    print(
        "Errors:",
        len(errors),
    )

    if errors:
        for error in errors:
            print(error)

        raise SystemExit(1)

    print(
        "Competition PDF table chunk "
        "smoke passed."
    )


if __name__ == "__main__":
    main()