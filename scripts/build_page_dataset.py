from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from app.schemas.document import DocumentManifest
from app.services.page_dataset import (
    PageDatasetError,
    build_page_dataset,
)
from app.services.page_mapping import (
    PageMappingError,
)
from app.services.page_parser import (
    PageParserError,
)
from app.services.registry import (
    RegistryError,
)
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

PAGE_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pages"
)


class PageDatasetInputError(ValueError):
    """页面数据集脚本输入无效。"""


def build_argument_parser() -> argparse.ArgumentParser:
    """创建 CLI 参数。"""

    parser = argparse.ArgumentParser(
        description="构建完整页面 JSONL 数据集"
    )

    parser.add_argument(
        "--manifest-path",
        action="append",
        required=True,
        type=Path,
        help=(
            "DocumentManifest 路径，"
            "可以重复提供"
        ),
    )

    return parser


def load_document_manifest(
    path: Path,
) -> DocumentManifest:
    """读取 DocumentManifest。"""

    candidate_path = path

    if not candidate_path.is_absolute():
        candidate_path = (
            PROJECT_ROOT / candidate_path
        )

    try:
        content = candidate_path.read_text(
            encoding="utf-8"
        )
    except OSError as exc:
        raise PageDatasetInputError(
            f"Manifest 无法读取：{candidate_path}"
        ) from exc

    try:
        return DocumentManifest.model_validate_json(
            content
        )
    except ValidationError as exc:
        raise PageDatasetInputError(
            f"Manifest 结构无效：{candidate_path}"
        ) from exc


def print_result(result) -> None:
    """打印页面数据集构建结果。"""

    manifest = result.manifest

    print("=" * 70)
    print(f"created: {result.created}")
    print(f"report_id: {manifest.report_id}")
    print(f"dataset_id: {manifest.dataset_id}")
    print(
        f"page_record_count: "
        f"{manifest.page_record_count}"
    )
    print(
        f"mapped_page_count: "
        f"{manifest.mapped_page_count}"
    )
    print(
        f"unmapped_page_count: "
        f"{manifest.unmapped_page_count}"
    )

    print("content_type_counts:")

    for content_type, count in (
        manifest.content_type_counts.items()
    ):
        print(
            f"  {content_type.value}: {count}"
        )

    print("parse_status_counts:")

    for parse_status, count in (
        manifest.parse_status_counts.items()
    ):
        print(
            f"  {parse_status.value}: {count}"
        )

    print(
        f"quality_gate_passed: "
        f"{manifest.quality_gate_passed}"
    )

    for warning in manifest.quality_warnings:
        print(f"WARNING: {warning}")

    for error in manifest.quality_gate_errors:
        print(f"ERROR: {error}")

    print(
        "pages_path: "
        f"{result.pages_path.relative_to(PROJECT_ROOT)}"
    )

    print(
        "manifest_path: "
        f"{result.manifest_path.relative_to(PROJECT_ROOT)}"
    )


def main() -> int:
    """CLI 主入口。"""

    parser = build_argument_parser()
    args = parser.parse_args()

    quality_failed = False

    try:
        (
            report_registry,
            page_mappings,
        ) = load_reports(
            REPORTS_YAML_PATH
        )

        for manifest_path in args.manifest_path:
            source_manifest = (
                load_document_manifest(
                    manifest_path
                )
            )

            report = report_registry.require(
                source_manifest.report_id
            )

            if (
                report.expected_pdf_page_count
                != source_manifest.pdf_page_count
            ):
                raise PageDatasetInputError(
                    "Report Registry、Manifest "
                    "与实际页数不一致："
                    f"{source_manifest.report_id}"
                )

            result = build_page_dataset(
                source_manifest=source_manifest,
                page_mappings=page_mappings,
                project_root=PROJECT_ROOT,
                output_root=PAGE_OUTPUT_ROOT,
            )

            print_result(result)

            if not result.manifest.quality_gate_passed:
                quality_failed = True

    except (
        RegistryLoaderError,
        RegistryError,
        PageMappingError,
        PageParserError,
        PageDatasetError,
        PageDatasetInputError,
    ) as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 1

    if quality_failed:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())