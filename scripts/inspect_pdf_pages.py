from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from app.schemas.document import DocumentManifest
from app.services.page_mapping import (
    PageMappingError,
)
from app.services.page_parser import (
    PageParserError,
    parse_pdf_pages,
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


class PageInspectionInputError(ValueError):
    """页面检查脚本输入无效。"""


def build_argument_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser(
        description="抽样检查 PDF 页面解析结果"
    )

    parser.add_argument(
        "--manifest-path",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--pdf-page",
        action="append",
        required=True,
        type=int,
        help=(
            "需要检查的 1-based PDF 页码；"
            "可重复提供"
        ),
    )

    return parser


def load_manifest(
    path: Path,
) -> DocumentManifest:
    """读取并校验 DocumentManifest。"""

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
        raise PageInspectionInputError(
            f"Manifest 无法读取：{candidate_path}"
        ) from exc

    try:
        return DocumentManifest.model_validate_json(
            content
        )
    except ValidationError as exc:
        raise PageInspectionInputError(
            f"Manifest 结构无效：{candidate_path}"
        ) from exc


def main() -> int:
    """CLI 主入口。"""

    parser = build_argument_parser()
    args = parser.parse_args()

    try:
        manifest = load_manifest(
            args.manifest_path
        )

        _, page_mappings = load_reports(
            REPORTS_YAML_PATH
        )

        parsed_pages = parse_pdf_pages(
            manifest=manifest,
            page_mappings=page_mappings,
            project_root=PROJECT_ROOT,
            pdf_pages=args.pdf_page,
        )

    except (
        RegistryLoaderError,
        PageMappingError,
        PageParserError,
        PageInspectionInputError,
    ) as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 1

    print("=" * 70)
    print(f"report_id: {manifest.report_id}")
    print(f"document_id: {manifest.document_id}")

    for page in parsed_pages:
        preview = " ".join(
            page.normalized_text.split()
        )[:120]

        print("-" * 70)

        print(
            {
                "pdf_page": page.pdf_page,
                "printed_page": page.printed_page,
                "mapping_status": (
                    page.mapping_status.value
                ),
                "content_type": (
                    page.content_type.value
                ),
                "parse_status": (
                    page.parse_status.value
                ),
                "raw_char_count": (
                    page.raw_char_count
                ),
                "normalized_char_count": (
                    page.normalized_char_count
                ),
                "text_block_count": (
                    page.text_block_count
                ),
                "image_block_count": (
                    page.image_block_count
                ),
                "embedded_image_count": (
                    page.embedded_image_count
                ),
                "max_image_area_ratio": round(
                    page.max_image_area_ratio,
                    4,
                ),
                "parse_error": page.parse_error,
                "preview": preview,
            }
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())