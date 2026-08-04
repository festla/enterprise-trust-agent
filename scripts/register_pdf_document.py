from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.schemas.enums import (
    DocumentValidationStatus,
)
from app.services.document_ingestion import (
    DocumentIngestionError,
    register_pdf_document,
)
from app.services.registry import RegistryError
from app.services.registry_loader import (
    RegistryLoaderError,
    load_reports,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORTS_YAML_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "registries"
    / "reports.yaml"
)

DOCUMENT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "documents"
)


def build_argument_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser(
        description="登记并检查一个实际 PDF 文档版本"
    )

    parser.add_argument(
        "--report-id",
        required=True,
        help="reports.yaml 中的 report_id",
    )

    parser.add_argument(
        "--pdf-path",
        required=True,
        type=Path,
        help="相对于项目根目录的 PDF 路径",
    )

    return parser


def main() -> int:
    """CLI 主入口。"""

    parser = build_argument_parser()
    args = parser.parse_args()

    try:
        report_registry, _ = load_reports(
            REPORTS_YAML_PATH
        )

        report = report_registry.require(
            args.report_id
        )

        result = register_pdf_document(
            report=report,
            pdf_path=args.pdf_path,
            project_root=PROJECT_ROOT,
            output_root=DOCUMENT_OUTPUT_ROOT,
        )

    except (
        RegistryLoaderError,
        RegistryError,
        DocumentIngestionError,
    ) as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 1

    manifest = result.manifest

    print("Document registration completed")
    print(f"created: {result.created}")
    print(f"report_id: {manifest.report_id}")
    print(f"document_id: {manifest.document_id}")
    print(f"sha256: {manifest.sha256}")
    print(
        "file_size_bytes: "
        f"{manifest.file_size_bytes}"
    )
    print(
        "pdf_page_count: "
        f"{manifest.pdf_page_count}"
    )
    print(
        "expected_pdf_page_count: "
        f"{manifest.expected_pdf_page_count}"
    )
    print(
        "page_count_status: "
        f"{manifest.page_count_status.value}"
    )
    print(
        "validation_status: "
        f"{manifest.validation_status.value}"
    )
    print(
        "manifest_path: "
        f"{result.manifest_path.relative_to(PROJECT_ROOT)}"
    )

    if (
        manifest.validation_status
        is DocumentValidationStatus.BLOCKED
    ):
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())