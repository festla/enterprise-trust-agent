from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from app.services.competition_dataset import (
    build_competition_question,
    load_competition_qa_excel,
)
from app.services.competition_regulatory_structure import (
    detect_regulatory_marker,
)
from app.services.competition_source_resolver import (
    CompetitionSourceResolver,
    build_competition_source_manifest,
)
from app.services.competition_text_parser import (
    parse_competition_text_document,
)


# ============================================================
# 这些不是正式 Detector。
#
# 只是 Audit 时统计“看起来可能像结构标题，
# 但当前 Detector 没识别出来”的形式。
#
# 不打印正文，只打印数量。
# ============================================================

_POSSIBLE_PART_PATTERN = re.compile(
    r"^第[一二三四五六七八九十百零〇0-9]+部分"
)

_POSSIBLE_ATTACHMENT_PATTERN = re.compile(
    r"^(?:附件|附表)[：:\s]*[0-9一二三四五六七八九十]*"
)

_POSSIBLE_ARABIC_FULLWIDTH_DOT_PATTERN = re.compile(
    r"^[0-9]+．"
)

_POSSIBLE_ARABIC_RIGHT_PAREN_PATTERN = re.compile(
    r"^[0-9]+[）)]"
)

_POSSIBLE_PAREN_ARABIC_PATTERN = re.compile(
    r"^[（(][0-9]+[）)]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit regulatory structure markers "
            "on Frozen Dev text sources only."
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


def _iter_structure_lines(
    text: str,
):
    """
    Detector 是“一行一个结构判断”。

    PDF page_text 是整页文本，
    因此必须按换行扫描。

    DOCX Paragraph 中也可能有手动换行，
    所以统一 splitlines。
    """

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        yield line


def _classify_unmatched_candidate(
    line: str,
) -> str | None:
    """
    只用于发现当前 Detector 可能遗漏的
    常见编号格式。
    """

    if _POSSIBLE_PART_PATTERN.match(
        line
    ):
        return "part"

    if _POSSIBLE_ATTACHMENT_PATTERN.match(
        line
    ):
        return "attachment"

    if _POSSIBLE_ARABIC_FULLWIDTH_DOT_PATTERN.match(
        line
    ):
        return "arabic_fullwidth_dot"

    if _POSSIBLE_ARABIC_RIGHT_PAREN_PATTERN.match(
        line
    ):
        return "arabic_right_paren"

    if _POSSIBLE_PAREN_ARABIC_PATTERN.match(
        line
    ):
        return "paren_arabic"

    return None


def main() -> None:
    args = parse_args()

    # ========================================================
    # 1. Load QA + Frozen Dev split
    # ========================================================

    cases = (
        load_competition_qa_excel(
            args.qa
        )
    )

    split_payload = json.loads(
        args.split.read_text(
            encoding="utf-8"
        )
    )

    dev_case_ids = set(
        split_payload[
            "dev_case_ids"
        ]
    )

    # ========================================================
    # 2. Source manifest / resolver
    # ========================================================

    source_manifest = (
        build_competition_source_manifest(
            args.attachments
        )
    )

    source_by_id = {
        source.source_id: source
        for source
        in source_manifest
    }

    resolver = (
        CompetitionSourceResolver(
            source_manifest
        )
    )

    # ========================================================
    # 3. Frozen Dev text sources
    #
    # 一个 Source 只解析一次。
    # ========================================================

    dev_sources: dict[
        str,
        tuple,
    ] = {}

    for case in cases:
        if (
            case.case_id
            not in dev_case_ids
        ):
            continue

        if (
            case.source_type
            not in {
                "pdf",
                "word",
            }
        ):
            continue

        resolution = (
            resolver.resolve(
                case
            )
        )

        source = source_by_id[
            resolution.source_id
        ]

        if (
            source.source_id
            in dev_sources
        ):
            continue

        dev_sources[
            source.source_id
        ] = (
            case,
            source,
        )

    # ========================================================
    # 4. Audit
    # ========================================================

    total_marker_counts = Counter()

    total_unmatched_candidates = Counter()

    total_block_types = Counter()

    total_lines = 0

    total_detected_lines = 0

    print(
        "=== Frozen Dev Regulatory Structure Audit ==="
    )

    print(
        "Unique Dev text sources:",
        len(
            dev_sources
        ),
    )

    print()

    for source_id in sorted(
        dev_sources
    ):
        (
            case,
            source,
        ) = dev_sources[
            source_id
        ]

        question = (
            build_competition_question(
                case
            )
        )

        parsed = (
            parse_competition_text_document(
                question=question,
                source=source,
                attachments_root=(
                    args.attachments
                ),
            )
        )

        local_marker_counts = Counter()

        local_unmatched_candidates = (
            Counter()
        )

        local_block_types = Counter(
            block.block_type
            for block
            in parsed.blocks
        )

        local_lines = 0
        local_detected_lines = 0

        for block in parsed.blocks:
            # ================================================
            # Table 暂时不参与监管标题 Detector。
            #
            # 原因：
            # 表格中的“第一、第二、1.”很容易是普通行号，
            # 现在先避免误识别。
            # ================================================

            if block.block_type == "table":
                continue

            for line in (
                _iter_structure_lines(
                    block.text
                )
            ):
                local_lines += 1

                marker = (
                    detect_regulatory_marker(
                        line
                    )
                )

                if marker is not None:
                    local_detected_lines += 1

                    local_marker_counts[
                        marker.marker_type
                    ] += 1

                    continue

                # ============================================
                # Detector 没识别，但统计一些潜在遗漏格式。
                # ============================================

                candidate_type = (
                    _classify_unmatched_candidate(
                        line
                    )
                )

                if (
                    candidate_type
                    is not None
                ):
                    local_unmatched_candidates[
                        candidate_type
                    ] += 1

        total_marker_counts.update(
            local_marker_counts
        )

        total_unmatched_candidates.update(
            local_unmatched_candidates
        )

        total_block_types.update(
            local_block_types
        )

        total_lines += (
            local_lines
        )

        total_detected_lines += (
            local_detected_lines
        )

        print(
            f"[OK] {source_id} "
            f"type={source.source_type} "
            f"ext={source.extension} "
            f"lines={local_lines} "
            f"markers="
            f"{dict(local_marker_counts)} "
            f"unmatched_candidates="
            f"{dict(local_unmatched_candidates)}"
        )

    # ========================================================
    # 5. Summary
    # ========================================================

    print()
    print(
        "=== Summary ==="
    )

    print(
        "Block types:",
        dict(
            total_block_types
        ),
    )

    print(
        "Scanned non-empty lines:",
        total_lines,
    )

    print(
        "Detected marker lines:",
        total_detected_lines,
    )

    detection_ratio = (
        (
            total_detected_lines
            / total_lines
        )
        if total_lines
        else 0.0
    )

    print(
        "Detected line ratio:",
        f"{detection_ratio:.4f}",
    )

    print()
    print(
        "Marker counts:",
        dict(
            total_marker_counts
        ),
    )

    print(
        "Unmatched candidate counts:",
        dict(
            total_unmatched_candidates
        ),
    )

    print()
    print(
        "Frozen Dev regulatory "
        "structure audit completed."
    )


if __name__ == "__main__":
    main()