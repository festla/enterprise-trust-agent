from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class AuditPageExportError(ValueError):
    """审计页面导出异常。"""


def _load_page_records(
    pages_path: Path,
) -> dict[int, dict[str, Any]]:
    if not pages_path.is_file():
        raise AuditPageExportError(
            f"pages.jsonl 不存在：{pages_path}"
        )

    try:
        text = pages_path.read_text(
            encoding="utf-8"
        )
    except UnicodeDecodeError as exc:
        raise AuditPageExportError(
            "pages.jsonl 不是合法 UTF-8"
        ) from exc
    except OSError as exc:
        raise AuditPageExportError(
            f"无法读取：{pages_path}"
        ) from exc

    records: dict[int, dict[str, Any]] = {}

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuditPageExportError(
                "页面 JSON 解析失败："
                f"line={line_number}"
            ) from exc

        try:
            pdf_page = int(
                payload["pdf_page"]
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise AuditPageExportError(
                "页面记录缺少合法 pdf_page："
                f"line={line_number}"
            ) from exc

        if pdf_page in records:
            raise AuditPageExportError(
                f"出现重复 PDF 页码：{pdf_page}"
            )

        records[pdf_page] = payload

    if not records:
        raise AuditPageExportError(
            "pages.jsonl 没有有效页面记录"
        )

    return records


def _render_page(
    payload: dict[str, Any],
) -> str:
    normalized_text = payload.get(
        "normalized_text"
    )

    if not isinstance(
        normalized_text,
        str,
    ):
        raise AuditPageExportError(
            "页面缺少 normalized_text："
            f"pdf_page={payload.get('pdf_page')}"
        )

    header = "\n".join(
        [
            f"report_id={payload.get('report_id')}",
            f"document_id={payload.get('document_id')}",
            f"page_id={payload.get('page_id')}",
            f"pdf_page={payload.get('pdf_page')}",
            (
                "printed_page="
                f"{payload.get('printed_page')}"
            ),
            (
                "parse_status="
                f"{payload.get('parse_status')}"
            ),
            "",
            "=" * 80,
            "NORMALIZED TEXT",
            "=" * 80,
            "",
        ]
    )

    return (
        header
        + normalized_text
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "从 PageDataset 导出指定页面，"
            "用于人工核验 Retrieval Gold Evidence"
        )
    )

    parser.add_argument(
        "--page-dataset-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--pdf-page",
        type=int,
        action="append",
        required=True,
        dest="pdf_pages",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    pages_path = (
        args.page_dataset_dir
        / "pages.jsonl"
    )

    records = _load_page_records(
        pages_path
    )

    requested_pages = tuple(
        sorted(set(args.pdf_pages))
    )

    missing_pages = tuple(
        page
        for page in requested_pages
        if page not in records
    )

    if missing_pages:
        raise AuditPageExportError(
            "PageDataset 中不存在请求页面："
            f"{missing_pages}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for pdf_page in requested_pages:
        payload = records[pdf_page]

        output_path = (
            args.output_dir
            / f"pdf_page_{pdf_page:04d}.txt"
        )

        output_path.write_text(
            _render_page(payload),
            encoding="utf-8",
        )

        print(
            f"pdf_page={pdf_page} "
            f"printed_page="
            f"{payload.get('printed_page')} "
            f"output={output_path}"
        )

    print(
        f"exported_page_count="
        f"{len(requested_pages)}"
    )


if __name__ == "__main__":
    main()