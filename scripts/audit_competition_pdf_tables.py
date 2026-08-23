from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.schemas.competition import (
    CompetitionQaCase,
    CompetitionSourceRecord,
)
from app.services.competition_dataset import (
    load_competition_qa_excel,
)
from app.services.competition_source_catalog import (
    resolve_competition_source_path,
)
from app.services.competition_source_resolver import (
    CompetitionSourceResolver,
    build_competition_source_manifest,
)


@dataclass(frozen=True)
class PdfTableSourceAudit:
    source_id: str
    page_count: int
    table_pages: tuple[int, ...]
    table_count: int
    extracted_table_count: int
    detection_failure_count: int
    extraction_failure_count: int
    invalid_table_count: int
    total_rows: int
    total_cells: int
    non_empty_cells: int


@dataclass(frozen=True)
class PdfTableCorpusAudit:
    source_count: int
    source_audits: tuple[
        PdfTableSourceAudit,
        ...
    ]
    failures: tuple[
        tuple[str, str],
        ...
    ]

    @property
    def success_count(self) -> int:
        return len(
            self.source_audits
        )

    @property
    def failure_count(self) -> int:
        return len(
            self.failures
        )

    @property
    def table_source_count(self) -> int:
        return sum(
            audit.table_count > 0
            for audit in self.source_audits
        )

    @property
    def page_count(self) -> int:
        return sum(
            audit.page_count
            for audit in self.source_audits
        )

    @property
    def table_page_count(self) -> int:
        return sum(
            len(audit.table_pages)
            for audit in self.source_audits
        )

    @property
    def table_count(self) -> int:
        return sum(
            audit.table_count
            for audit in self.source_audits
        )

    @property
    def extracted_table_count(self) -> int:
        return sum(
            audit.extracted_table_count
            for audit in self.source_audits
        )

    @property
    def detection_failure_count(self) -> int:
        return sum(
            audit.detection_failure_count
            for audit in self.source_audits
        )

    @property
    def extraction_failure_count(self) -> int:
        return sum(
            audit.extraction_failure_count
            for audit in self.source_audits
        )

    @property
    def invalid_table_count(self) -> int:
        return sum(
            audit.invalid_table_count
            for audit in self.source_audits
        )

    @property
    def total_rows(self) -> int:
        return sum(
            audit.total_rows
            for audit in self.source_audits
        )

    @property
    def total_cells(self) -> int:
        return sum(
            audit.total_cells
            for audit in self.source_audits
        )

    @property
    def non_empty_cells(self) -> int:
        return sum(
            audit.non_empty_cells
            for audit in self.source_audits
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit table structures in QA-used PDFs "
            "without printing document content, "
            "questions, answers, or evidence."
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
        "--minimum-tables",
        type=int,
        default=2,
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
    CompetitionSourceRecord,
]:
    resolver = CompetitionSourceResolver(
        source_manifest
    )

    source_by_id = {
        source.source_id: source
        for source in source_manifest
    }

    pdf_sources: dict[
        str,
        CompetitionSourceRecord,
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

        pdf_sources.setdefault(
            source.source_id,
            source,
        )

    return dict(
        sorted(
            pdf_sources.items()
        )
    )


def _is_valid_bbox(
    bbox,
) -> bool:
    try:
        values = tuple(
            float(value)
            for value in bbox
        )
    except (
        TypeError,
        ValueError,
    ):
        return False

    if len(values) != 4:
        return False

    if not all(
        math.isfinite(value)
        for value in values
    ):
        return False

    x0, y0, x1, y1 = values

    return (
        x1 > x0
        and y1 > y0
    )


def _normalize_cell(
    value: object,
) -> str:
    if value is None:
        return ""

    return str(value).strip()


def audit_pdf_table_source(
    *,
    source_id: str,
    path: Path,
) -> PdfTableSourceAudit:
    document = None

    try:
        document = pymupdf.open(
            path
        )

        if not document.is_pdf:
            raise RuntimeError(
                "Source is not a PDF"
            )

        if document.needs_pass:
            raise RuntimeError(
                "PDF is encrypted"
            )

        table_pages: list[int] = []

        table_count = 0
        extracted_table_count = 0
        detection_failure_count = 0
        extraction_failure_count = 0
        invalid_table_count = 0

        total_rows = 0
        total_cells = 0
        non_empty_cells = 0

        for page_index in range(
            document.page_count
        ):
            page = document.load_page(
                page_index
            )

            try:
                finder = page.find_tables()
                tables = tuple(
                    finder.tables
                )
            except Exception:
                detection_failure_count += 1
                continue

            if tables:
                table_pages.append(
                    page_index + 1
                )

            for table in tables:
                table_count += 1

                row_count = int(
                    table.row_count
                )

                col_count = int(
                    table.col_count
                )

                if (
                    row_count <= 0
                    or col_count <= 0
                    or not _is_valid_bbox(
                        table.bbox
                    )
                ):
                    invalid_table_count += 1

                try:
                    rows = table.extract()
                except Exception:
                    extraction_failure_count += 1
                    continue

                if rows is None:
                    extraction_failure_count += 1
                    continue

                extracted_table_count += 1
                total_rows += len(
                    rows
                )

                for row in rows:
                    if row is None:
                        continue

                    total_cells += len(
                        row
                    )

                    non_empty_cells += sum(
                        bool(
                            _normalize_cell(
                                cell
                            )
                        )
                        for cell in row
                    )

        return PdfTableSourceAudit(
            source_id=source_id,
            page_count=document.page_count,
            table_pages=tuple(
                table_pages
            ),
            table_count=table_count,
            extracted_table_count=(
                extracted_table_count
            ),
            detection_failure_count=(
                detection_failure_count
            ),
            extraction_failure_count=(
                extraction_failure_count
            ),
            invalid_table_count=(
                invalid_table_count
            ),
            total_rows=total_rows,
            total_cells=total_cells,
            non_empty_cells=(
                non_empty_cells
            ),
        )

    finally:
        if document is not None:
            document.close()


def run_pdf_table_corpus_audit(
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
) -> PdfTableCorpusAudit:
    pdf_sources = (
        collect_qa_used_pdf_sources(
            cases=cases,
            source_manifest=source_manifest,
        )
    )

    source_audits: list[
        PdfTableSourceAudit
    ] = []

    failures: list[
        tuple[str, str]
    ] = []

    print(
        "=== QA-used PDF Table Corpus Audit ==="
    )

    print(
        "Unique PDF sources:",
        len(pdf_sources),
    )

    print()

    for source_id in sorted(
        pdf_sources
    ):
        source = pdf_sources[
            source_id
        ]

        try:
            path = (
                resolve_competition_source_path(
                    attachments_root=(
                        attachments_root
                    ),
                    source=source,
                )
            )

            audit = audit_pdf_table_source(
                source_id=source_id,
                path=path,
            )

        except Exception as exc:
            failures.append(
                (
                    source_id,
                    type(exc).__name__,
                )
            )

            # 不打印文件名、正文或异常内容。
            print(
                f"[FAIL] "
                f"{source_id} "
                f"error={type(exc).__name__}"
            )

            continue

        source_audits.append(
            audit
        )

        print(
            f"[OK]   "
            f"{source_id} "
            f"pages={audit.page_count} "
            f"table_pages="
            f"{len(audit.table_pages)} "
            f"tables={audit.table_count} "
            f"extracted="
            f"{audit.extracted_table_count} "
            f"rows={audit.total_rows} "
            f"cells={audit.total_cells} "
            f"non_empty="
            f"{audit.non_empty_cells}"
        )

    return PdfTableCorpusAudit(
        source_count=len(
            pdf_sources
        ),
        source_audits=tuple(
            source_audits
        ),
        failures=tuple(
            failures
        ),
    )


def collect_gate_errors(
    *,
    audit: PdfTableCorpusAudit,
    expected_source_count: int,
    minimum_table_sources: int,
    minimum_tables: int,
) -> list[str]:
    errors: list[str] = []

    if (
        audit.source_count
        != expected_source_count
    ):
        errors.append(
            "PDF source count mismatch: "
            f"expected={expected_source_count}, "
            f"actual={audit.source_count}"
        )

    if audit.failure_count:
        errors.append(
            "PDF source audit failures: "
            f"{audit.failure_count}"
        )

    if (
        audit.success_count
        != audit.source_count
    ):
        errors.append(
            "PDF audit success count "
            "does not match source count"
        )

    if audit.detection_failure_count:
        errors.append(
            "PDF table detection failures: "
            f"{audit.detection_failure_count}"
        )

    if audit.extraction_failure_count:
        errors.append(
            "PDF table extraction failures: "
            f"{audit.extraction_failure_count}"
        )

    if audit.invalid_table_count:
        errors.append(
            "Invalid PDF tables: "
            f"{audit.invalid_table_count}"
        )

    if (
        audit.table_source_count
        < minimum_table_sources
    ):
        errors.append(
            "Too few PDF table sources: "
            f"minimum={minimum_table_sources}, "
            f"actual={audit.table_source_count}"
        )

    if audit.table_count < minimum_tables:
        errors.append(
            "Too few PDF tables: "
            f"minimum={minimum_tables}, "
            f"actual={audit.table_count}"
        )

    if (
        audit.table_count > 0
        and audit.non_empty_cells <= 0
    ):
        errors.append(
            "Detected PDF tables contain "
            "no non-empty cells"
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

    audit = run_pdf_table_corpus_audit(
        cases=cases,
        source_manifest=source_manifest,
        attachments_root=args.attachments,
    )

    print()
    print(
        "=== Summary ==="
    )

    print(
        "Sources:",
        audit.source_count,
    )

    print(
        "Success:",
        audit.success_count,
    )

    print(
        "Failure:",
        audit.failure_count,
    )

    print(
        "Pages:",
        audit.page_count,
    )

    print(
        "Sources with tables:",
        audit.table_source_count,
    )

    print(
        "Table pages:",
        audit.table_page_count,
    )

    print(
        "Tables:",
        audit.table_count,
    )

    print(
        "Extracted tables:",
        audit.extracted_table_count,
    )

    print(
        "Rows:",
        audit.total_rows,
    )

    print(
        "Cells:",
        audit.total_cells,
    )

    print(
        "Non-empty cells:",
        audit.non_empty_cells,
    )

    errors = collect_gate_errors(
        audit=audit,
        expected_source_count=(
            args.expected_sources
        ),
        minimum_table_sources=(
            args.minimum_table_sources
        ),
        minimum_tables=(
            args.minimum_tables
        ),
    )

    print()
    print(
        "=== Audit Result ==="
    )

    print(
        "Errors:",
        len(errors),
    )

    if errors:
        for error in errors:
            print(
                error
            )

        raise SystemExit(1)

    print(
        "Competition PDF table "
        "corpus audit passed."
    )


if __name__ == "__main__":
    main()