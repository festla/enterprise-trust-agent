from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from app.schemas.document import DocumentManifest
from app.services.page_mapping import (
    PageMappingError,
    PageMappingResolver,
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


class PageMappingAuditInputError(ValueError):
    """映射审计输入文件无效。"""


def build_argument_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "检查 DocumentManifest 对应的 "
            "整份 PDF 页码映射"
        )
    )

    parser.add_argument(
        "--manifest-path",
        action="append",
        required=True,
        type=Path,
        help=(
            "DocumentManifest JSON 路径；"
            "可重复提供该参数"
        ),
    )

    return parser


def load_manifest(
    path: Path,
) -> DocumentManifest:
    """读取并校验 Manifest。"""

    candidate = path

    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate

    try:
        content = candidate.read_text(
            encoding="utf-8"
        )
    except OSError as exc:
        raise PageMappingAuditInputError(
            f"Manifest 无法读取：{candidate}"
        ) from exc

    try:
        return DocumentManifest.model_validate_json(
            content
        )
    except ValidationError as exc:
        raise PageMappingAuditInputError(
            f"Manifest 结构无效：{candidate}"
        ) from exc


def print_audit(
    resolver: PageMappingResolver,
) -> None:
    """打印单份文档的审计结果。"""

    audit = resolver.audit()

    print("=" * 60)
    print(f"report_id: {audit.report_id}")
    print(f"document_id: {audit.document_id}")
    print(
        f"total_pdf_pages: "
        f"{audit.total_pdf_pages}"
    )
    print(
        f"mapped_pages: "
        f"{audit.mapped_page_count}"
    )
    print(
        "unmapped_pdf_pages: "
        f"{list(audit.unmapped_pdf_pages)}"
    )

    if audit.duplicate_printed_pages:
        print("duplicate_printed_pages:")

        for (
            printed_page,
            pdf_pages,
        ) in audit.duplicate_printed_pages.items():
            print(
                f"  printed {printed_page}"
                f" -> PDF {list(pdf_pages)}"
            )
    else:
        print("duplicate_printed_pages: none")

    print("result: PASSED")


def main() -> int:
    """CLI 主入口。"""

    parser = build_argument_parser()
    args = parser.parse_args()

    try:
        (
            report_registry,
            page_mappings,
        ) = load_reports(
            REPORTS_YAML_PATH
        )

        for manifest_path in args.manifest_path:
            manifest = load_manifest(
                manifest_path
            )

            report = report_registry.require(
                manifest.report_id
            )

            if (
                report.expected_pdf_page_count
                != manifest.expected_pdf_page_count
            ):
                raise PageMappingAuditInputError(
                    "Manifest 的预期页数与 "
                    "当前 Report Registry 不一致："
                    f"{manifest.report_id}"
                )

            resolver = PageMappingResolver(
                manifest=manifest,
                page_mappings=page_mappings,
            )

            print_audit(resolver)

    except (
        RegistryLoaderError,
        RegistryError,
        PageMappingError,
        PageMappingAuditInputError,
    ) as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())