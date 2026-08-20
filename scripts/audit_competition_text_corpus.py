from __future__ import annotations

import argparse
import json
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pymupdf

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
from app.services.page_parser import (
    normalize_page_text,
)


WORD_NS = (
    "http://schemas.openxmlformats.org/"
    "wordprocessingml/2006/main"
)

W_P = f"{{{WORD_NS}}}p"
W_T = f"{{{WORD_NS}}}t"
W_TBL = f"{{{WORD_NS}}}tbl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Competition PDF / Word corpus "
            "without printing document content."
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
        "--split",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def _audit_pdf(
    path: Path,
) -> dict[str, int | str]:
    document = None

    try:
        document = pymupdf.open(
            path
        )

        if not document.is_pdf:
            return {
                "status": "not_pdf",
            }

        if document.needs_pass:
            return {
                "status": "encrypted",
            }

        page_count = (
            document.page_count
        )

        text_pages = 0
        empty_text_pages = 0
        scanned_like_pages = 0
        parse_error_pages = 0
        normalized_chars = 0

        for page_index in range(
            page_count
        ):
            try:
                page = (
                    document.load_page(
                        page_index
                    )
                )

                raw_text = (
                    page.get_text(
                        "text"
                    )
                )

                normalized_text = (
                    normalize_page_text(
                        raw_text
                    )
                )

                normalized_chars += len(
                    normalized_text
                )

                if normalized_text:
                    text_pages += 1
                    continue

                empty_text_pages += 1

                images = (
                    page.get_images(
                        full=True
                    )
                )

                if images:
                    scanned_like_pages += 1

            except Exception:
                parse_error_pages += 1

        return {
            "status": "ok",
            "page_count":
                page_count,
            "text_pages":
                text_pages,
            "empty_text_pages":
                empty_text_pages,
            "scanned_like_pages":
                scanned_like_pages,
            "parse_error_pages":
                parse_error_pages,
            "normalized_chars":
                normalized_chars,
        }

    except Exception:
        return {
            "status": "open_error",
        }

    finally:
        if document is not None:
            document.close()


def _audit_docx(
    path: Path,
) -> dict[str, int | str]:
    """
    不引入 python-docx。

    这里只使用 ZIP + XML 验证 DOCX
    是否能够正常读取，并统计最基本结构。
    """

    try:
        with zipfile.ZipFile(
            path
        ) as archive:
            try:
                xml_bytes = (
                    archive.read(
                        "word/document.xml"
                    )
                )
            except KeyError:
                return {
                    "status":
                        "missing_document_xml",
                }

    except zipfile.BadZipFile:
        return {
            "status": "invalid_docx",
        }

    try:
        root = ET.fromstring(
            xml_bytes
        )
    except ET.ParseError:
        return {
            "status": "invalid_xml",
        }

    paragraph_count = 0
    non_empty_paragraph_count = 0
    table_count = 0
    text_chars = 0

    for paragraph in root.iter(
        W_P
    ):
        paragraph_count += 1

        texts = [
            node.text or ""
            for node
            in paragraph.iter(
                W_T
            )
        ]

        text = "".join(
            texts
        ).strip()

        if text:
            non_empty_paragraph_count += 1
            text_chars += len(
                text
            )

    table_count = sum(
        1
        for _ in root.iter(
            W_TBL
        )
    )

    return {
        "status": "ok",
        "paragraph_count":
            paragraph_count,
        "non_empty_paragraph_count":
            non_empty_paragraph_count,
        "table_count":
            table_count,
        "text_chars":
            text_chars,
    }

def _detect_legacy_word_format(
    path: Path,
) -> str:
    """
    只检查文件签名，不读取或打印正文。

    用于确认 .doc 是否真的是：
    - OLE Compound Binary Word
    - DOCX/ZIP 错误后缀
    - RTF
    - HTML
    """

    with path.open(
        "rb"
    ) as file:
        prefix = file.read(
            512
        )

    # 传统 Word .doc / OLE Compound File
    if prefix.startswith(
        bytes.fromhex(
            "D0CF11E0A1B11AE1"
        )
    ):
        return "ole_binary_doc"

    # DOCX 本质是 ZIP
    if prefix.startswith(
        b"PK\x03\x04"
    ):
        return "zip_docx_like"

    stripped = (
        prefix.lstrip()
    )

    if stripped.startswith(
        b"{\\rtf"
    ):
        return "rtf"

    lowered = (
        stripped.lower()
    )

    if (
        b"<html" in lowered
        or b"<!doctype html"
        in lowered
    ):
        return "html"

    return "unknown"

def _print_case_distribution(
    *,
    name: str,
    cases,
) -> None:
    counter = Counter(
        (
            case.source_type,
            case.qa_type,
        )
        for case in cases
    )

    print()
    print(
        f"=== {name} QA Distribution ==="
    )

    for key in sorted(
        counter
    ):
        print(
            f"{key}: "
            f"{counter[key]}"
        )


def main() -> None:
    args = parse_args()

    cases = (
        load_competition_qa_excel(
            args.qa
        )
    )

    split = json.loads(
        args.split.read_text(
            encoding="utf-8"
        )
    )

    dev_ids = set(
        split["dev_case_ids"]
    )

    test_ids = set(
        split["test_case_ids"]
    )

    text_cases = [
        case
        for case in cases
        if case.source_type
        in {
            "word",
            "pdf",
        }
    ]

    dev_cases = [
        case
        for case in text_cases
        if case.case_id
        in dev_ids
    ]

    test_cases = [
        case
        for case in text_cases
        if case.case_id
        in test_ids
    ]

    manifest = (
        build_competition_source_manifest(
            args.attachments
        )
    )

    resolver = (
        CompetitionSourceResolver(
            manifest
        )
    )

    source_by_id = {
        source.source_id:
            source
        for source in manifest
    }

    # ========================================================
    # 所有附件中的 Word / PDF 格式分布
    # ========================================================

    all_text_sources = [
        source
        for source in manifest
        if source.source_type
        in {
            "word",
            "pdf",
        }
    ]

    all_extension_counts = Counter(
        source.extension
        for source
        in all_text_sources
    )

    # ========================================================
    # QA 真正引用到的 Source。
    # ========================================================

    used_sources = {}

    used_dev_source_ids = set()
    used_test_source_ids = set()

    for case in text_cases:
        resolution = (
            resolver.resolve(
                case
            )
        )

        source = source_by_id[
            resolution.source_id
        ]

        used_sources[
            source.source_id
        ] = source

        if case.case_id in dev_ids:
            used_dev_source_ids.add(
                source.source_id
            )

        if case.case_id in test_ids:
            used_test_source_ids.add(
                source.source_id
            )

    used_extension_counts = Counter(
        source.extension
        for source
        in used_sources.values()
    )

    used_type_counts = Counter(
        source.source_type
        for source
        in used_sources.values()
    )

    # ========================================================
    # 基础统计
    # ========================================================

    print(
        "=== Competition Text Corpus Audit ==="
    )

    print(
        "Text QA cases:",
        len(text_cases),
    )

    print(
        "Dev text QA cases:",
        len(dev_cases),
    )

    print(
        "Test text QA cases:",
        len(test_cases),
    )

    print()

    print(
        "All Word/PDF attachment files:",
        len(all_text_sources),
    )

    print(
        "All extensions:",
        dict(
            all_extension_counts
        ),
    )

    print()

    print(
        "QA-used unique sources:",
        len(used_sources),
    )

    print(
        "QA-used source types:",
        dict(
            used_type_counts
        ),
    )

    print(
        "QA-used extensions:",
        dict(
            used_extension_counts
        ),
    )

    print(
        "Dev unique sources:",
        len(
            used_dev_source_ids
        ),
    )

    print(
        "Test unique sources:",
        len(
            used_test_source_ids
        ),
    )

    print(
        "Dev/Test source overlap:",
        len(
            used_dev_source_ids
            & used_test_source_ids
        ),
    )

    print(
        "Dev unique sources:",
        len(
            used_dev_source_ids
        ),
    )

    print(
        "Test unique sources:",
        len(
            used_test_source_ids
        ),
    )

    dev_extension_counts = Counter(
        source_by_id[
            source_id
        ].extension
        for source_id
        in used_dev_source_ids
    )

    test_extension_counts = Counter(
        source_by_id[
            source_id
        ].extension
        for source_id
        in used_test_source_ids
    )

    print(
        "Dev used extensions:",
        dict(
            dev_extension_counts
        ),
    )

    print(
        "Test used extensions:",
        dict(
            test_extension_counts
        ),
    )

    _print_case_distribution(
        name="Overall",
        cases=text_cases,
    )

    _print_case_distribution(
        name="Dev",
        cases=dev_cases,
    )

    _print_case_distribution(
        name="Test",
        cases=test_cases,
    )

    # ========================================================
    # 只深入解析 QA 真正使用的文档。
    #
    # 当前阶段没必要为了比赛把全部 111 个
    # Word/PDF 附件都进行重解析。
    # ========================================================

    pdf_statuses = Counter()
    pdf_totals = Counter()

    word_statuses = Counter()
    word_totals = Counter()

    legacy_doc_formats = Counter()

    legacy_doc_split = Counter()

    for source in (
        used_sources.values()
    ):
        path = (
            resolve_competition_source_path(
                attachments_root=(
                    args.attachments
                ),
                source=source,
            )
        )

        if source.source_type == "pdf":
            audit = _audit_pdf(
                path
            )

            status = str(
                audit["status"]
            )

            pdf_statuses[
                status
            ] += 1

            if status == "ok":
                for key in (
                    "page_count",
                    "text_pages",
                    "empty_text_pages",
                    "scanned_like_pages",
                    "parse_error_pages",
                    "normalized_chars",
                ):
                    pdf_totals[
                        key
                    ] += int(
                        audit[key]
                    )

            continue

        if source.source_type == "word":
            if (
                source.extension
                == ".docx"
            ):
                audit = (
                    _audit_docx(
                        path
                    )
                )

                status = str(
                    audit["status"]
                )

                word_statuses[
                    status
                ] += 1

                if status == "ok":
                    for key in (
                        "paragraph_count",
                        "non_empty_paragraph_count",
                        "table_count",
                        "text_chars",
                    ):
                        word_totals[
                            key
                        ] += int(
                            audit[key]
                        )

            elif (
                source.extension
                == ".doc"
            ):
                word_statuses[
                    "legacy_doc"
                ] += 1

                legacy_format = (
                    _detect_legacy_word_format(
                        path
                    )
                )

                legacy_doc_formats[
                    legacy_format
                ] += 1

                if (
                    source.source_id
                    in used_dev_source_ids
                ):
                    legacy_doc_split[
                        "dev"
                    ] += 1

                if (
                    source.source_id
                    in used_test_source_ids
                ):
                    legacy_doc_split[
                        "test"
                    ] += 1

            else:
                word_statuses[
                    "unsupported_extension"
                ] += 1

    print()
    print(
        "=== QA-used PDF Compatibility ==="
    )

    print(
        "Status:",
        dict(
            pdf_statuses
        ),
    )

    print(
        "Totals:",
        dict(
            pdf_totals
        ),
    )

    print(
        "Legacy DOC formats:",
        dict(
            legacy_doc_formats
        ),
    )

    print(
        "Legacy DOC split:",
        dict(
            legacy_doc_split
        ),
    )

    print()
    print(
        "=== QA-used Word Compatibility ==="
    )

    print(
        "Status:",
        dict(
            word_statuses
        ),
    )

    print(
        "Totals:",
        dict(
            word_totals
        ),
    )


if __name__ == "__main__":
    main()