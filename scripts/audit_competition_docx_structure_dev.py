from __future__ import annotations

import argparse
import json
import re
from collections import Counter, deque
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

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


# ============================================================
# Heading Style
# ============================================================

_BUILTIN_HEADING_STYLE_PATTERN = re.compile(
    r"^(?:Heading\s*[1-9]|标题\s*[1-9])$",
    re.IGNORECASE,
)


# ============================================================
# Table Unit
#
# 例如：
#
# （一）表格 KM1：监管并表关键审慎监管指标
# （二）表格 CR1：资产质量
# 表格 LI1：……
# ============================================================

_CN_NUMBER = (
    "零〇一二三四五六七八九十百千万两"
)

_TABLE_LABEL_PATTERN = re.compile(
    rf"^(?:"
    rf"[（(][{_CN_NUMBER}0-9]+[）)]"
    rf"\s*"
    rf")?"
    rf"表格\s*"
    rf"[A-Za-z][A-Za-z0-9\-]*"
    rf"\s*[:：]"
)


_CONTEXT_FIELDS = {
    "purpose": (
        "目的：",
        "目的:",
    ),
    "scope": (
        "适用范围：",
        "适用范围:",
    ),
    "content": (
        "内容：",
        "内容:",
    ),
    "frequency": (
        "频率：",
        "频率:",
    ),
    "format": (
        "格式：",
        "格式:",
    ),
}

_INTERNAL_TABLE_CONTEXT_FIELDS = {
    "purpose": (
        "目的：",
        "目的:",
    ),
    "scope": (
        "适用范围：",
        "适用范围:",
    ),
    "content": (
        "内容：",
        "内容:",
    ),
    "frequency": (
        "频率：",
        "频率:",
    ),
    "format": (
        "格式：",
        "格式:",
    ),
    "supplement": (
        "补充说明：",
        "补充说明:",
    ),
}

TABLE_CONTEXT_WINDOW = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit DOCX heading styles, "
            "outline levels and table context "
            "on Frozen Dev sources only."
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

def _normalize_table_cell_text(
    value: str,
) -> str:
    return (
        value
        .replace(
            "\u3000",
            " ",
        )
        .strip()
    )


def _table_first_rows_text(
    table: Table,
    *,
    max_rows: int = 12,
) -> tuple[str, ...]:
    """
    只读取表格前若干行，用于结构审计。

    不打印实际内容。
    """

    lines: list[str] = []

    for row in table.rows[
        :max_rows
    ]:
        values = []

        for cell in row.cells:
            text = (
                _normalize_table_cell_text(
                    cell.text
                )
            )

            if text:
                values.append(
                    text
                )

        if values:
            lines.append(
                " ".join(
                    values
                )
            )

    return tuple(
        lines
    )


def _detect_internal_context_fields(
    table: Table,
) -> set[str]:
    lines = (
        _table_first_rows_text(
            table
        )
    )

    matches: set[str] = set()

    for line in lines:
        compact = (
            line.replace(
                " ",
                "",
            )
        )

        for (
            field_name,
            prefixes,
        ) in (
            _INTERNAL_TABLE_CONTEXT_FIELDS
            .items()
        ):
            for prefix in prefixes:
                compact_prefix = (
                    prefix.replace(
                        " ",
                        "",
                    )
                )

                if (
                    compact.startswith(
                        compact_prefix
                    )
                    or compact_prefix
                    in compact
                ):
                    matches.add(
                        field_name
                    )

                    break

    return matches

# ============================================================
# Paragraph / Style Helpers
# ============================================================


def _paragraph_style_name(
    paragraph: Paragraph,
) -> str:
    style = paragraph.style

    if style is None:
        return "<none>"

    return (
        style.name
        or style.style_id
        or "<unnamed>"
    )


def _read_outline_level_from_ppr(
    ppr,
) -> int | None:
    """
    读取 OOXML 中：

        <w:outlineLvl w:val="1"/>

    这是比单纯 style.name 更可靠的
    Word 大纲层级信号。
    """

    if ppr is None:
        return None

    outline = ppr.find(
        qn(
            "w:outlineLvl"
        )
    )

    if outline is None:
        return None

    value = outline.get(
        qn(
            "w:val"
        )
    )

    if value is None:
        return None

    try:
        return int(
            value
        )
    except ValueError:
        return None


def _effective_outline_level(
    paragraph: Paragraph,
) -> int | None:
    """
    Word 的大纲级别可能来自：

    1. Paragraph 自己的 direct formatting
    2. Paragraph Style
    3. Style 的 base_style

    因此不能只检查 style.name。
    """

    # ========================================================
    # 1. Paragraph Direct Formatting
    # ========================================================

    direct_level = (
        _read_outline_level_from_ppr(
            paragraph._p.pPr
        )
    )

    if direct_level is not None:
        return direct_level

    # ========================================================
    # 2. Style / Base Style Chain
    # ========================================================

    style = paragraph.style

    visited: set[
        str
    ] = set()

    while style is not None:
        style_id = (
            style.style_id
            or str(
                id(style)
            )
        )

        if style_id in visited:
            break

        visited.add(
            style_id
        )

        style_element = (
            style._element
        )

        style_level = (
            _read_outline_level_from_ppr(
                style_element.pPr
            )
        )

        if style_level is not None:
            return style_level

        style = (
            style.base_style
        )

    return None


def _is_builtin_heading_style(
    style_name: str,
) -> bool:
    return bool(
        _BUILTIN_HEADING_STYLE_PATTERN.fullmatch(
            style_name.strip()
        )
    )


# ============================================================
# Table Context
# ============================================================


def _is_table_label(
    text: str,
) -> bool:
    return bool(
        _TABLE_LABEL_PATTERN.match(
            text.strip()
        )
    )


def _context_field_matches(
    paragraphs: list[
        tuple[
            str,
            str,
            int | None,
        ]
    ],
) -> set[str]:
    """
    返回 Table 前若干 Paragraph 中出现了哪些：

        目的
        适用范围
        内容
        频率
        格式
    """

    matches: set[
        str
    ] = set()

    for (
        text,
        _style_name,
        _outline_level,
    ) in paragraphs:
        stripped = (
            text.strip()
        )

        for (
            field_name,
            prefixes,
        ) in (
            _CONTEXT_FIELDS
            .items()
        ):
            if any(
                stripped.startswith(
                    prefix
                )
                for prefix
                in prefixes
            ):
                matches.add(
                    field_name
                )

    return matches


def _table_row_bucket(
    row_count: int,
) -> str:
    if row_count <= 5:
        return "1-5"

    if row_count <= 10:
        return "6-10"

    if row_count <= 25:
        return "11-25"

    if row_count <= 50:
        return "26-50"

    return "51+"


def _table_col_bucket(
    col_count: int,
) -> str:
    if col_count <= 3:
        return "1-3"

    if col_count <= 6:
        return "4-6"

    if col_count <= 10:
        return "7-10"

    return "11+"


# ============================================================
# Single DOCX Audit
# ============================================================


def _audit_docx(
    path: Path,
) -> dict[str, object]:
    document = Document(
        path
    )

    style_counts = Counter()

    builtin_heading_styles = (
        Counter()
    )

    outline_counts = Counter()

    outline_style_counts = (
        Counter()
    )

    raw_paragraph_count = 0

    non_empty_paragraph_count = 0

    table_count = 0

    # ========================================================
    # Paragraph Style Audit
    # ========================================================

    for paragraph in (
        document.paragraphs
    ):
        raw_paragraph_count += 1

        text = (
            paragraph.text
            .strip()
        )

        if not text:
            continue

        non_empty_paragraph_count += 1

        style_name = (
            _paragraph_style_name(
                paragraph
            )
        )

        style_counts[
            style_name
        ] += 1

        if (
            _is_builtin_heading_style(
                style_name
            )
        ):
            builtin_heading_styles[
                style_name
            ] += 1

        outline_level = (
            _effective_outline_level(
                paragraph
            )
        )

        if outline_level is not None:
            outline_counts[
                outline_level
            ] += 1

            outline_style_counts[
                (
                    style_name,
                    outline_level,
                )
            ] += 1

    # ========================================================
    # Table + Nearby Context Audit
    #
    # iter_inner_content() 保留 Paragraph / Table
    # 在 Word 正文中的真实顺序。
    # ========================================================

    recent_paragraphs = deque(
        maxlen=(
            TABLE_CONTEXT_WINDOW
        )
    )

    table_label_nearby_count = 0

    table_outline_nearby_count = 0

    table_context_3plus_count = 0

    table_context_all5_count = 0

    context_field_counts = (
        Counter()
    )

    internal_context_field_counts = Counter()

    table_internal_context_3plus_count = 0

    table_internal_context_all5_count = 0

    preceding_style_counts = (
        Counter()
    )

    table_row_buckets = (
        Counter()
    )

    table_col_buckets = (
        Counter()
    )

    max_table_rows = 0
    max_table_cols = 0

    for item in (
        document.iter_inner_content()
    ):
        # ====================================================
        # Paragraph
        # ====================================================

        if isinstance(
            item,
            Paragraph,
        ):
            text = (
                item.text
                .strip()
            )

            if not text:
                continue

            style_name = (
                _paragraph_style_name(
                    item
                )
            )

            outline_level = (
                _effective_outline_level(
                    item
                )
            )

            recent_paragraphs.append(
                (
                    text,
                    style_name,
                    outline_level,
                )
            )

            continue

        # ====================================================
        # Table
        # ====================================================

        if not isinstance(
            item,
            Table,
        ):
            continue

        internal_fields = (
            _detect_internal_context_fields(
                item
            )
        )

        internal_context_field_counts.update(
            internal_fields
        )

        core_internal_fields = (
            internal_fields
            & {
                "purpose",
                "scope",
                "content",
                "frequency",
                "format",
            }
        )

        if (
            len(
                core_internal_fields
            )
            >= 3
        ):
            table_internal_context_3plus_count += 1

        if (
            len(
                core_internal_fields
            )
            == 5
        ):
            table_internal_context_all5_count += 1

        table_count += 1

        recent = list(
            recent_paragraphs
        )

        # ====================================================
        # Table Shape
        # ====================================================

        row_count = len(
            item.rows
        )

        col_count = max(
            (
                len(
                    row.cells
                )
                for row
                in item.rows
            ),
            default=0,
        )

        max_table_rows = max(
            max_table_rows,
            row_count,
        )

        max_table_cols = max(
            max_table_cols,
            col_count,
        )

        table_row_buckets[
            _table_row_bucket(
                row_count
            )
        ] += 1

        table_col_buckets[
            _table_col_bucket(
                col_count
            )
        ] += 1

        # ====================================================
        # Immediate Preceding Style
        # ====================================================

        if recent:
            (
                _previous_text,
                previous_style,
                _previous_outline,
            ) = recent[-1]

            preceding_style_counts[
                previous_style
            ] += 1

        # ====================================================
        # Nearby Table Label
        # ====================================================

        if any(
            _is_table_label(
                text
            )
            for (
                text,
                _style,
                _outline,
            )
            in recent
        ):
            table_label_nearby_count += 1

        # ====================================================
        # Nearby Outline Heading
        # ====================================================

        if any(
            outline_level
            is not None
            for (
                _text,
                _style,
                outline_level,
            )
            in recent
        ):
            table_outline_nearby_count += 1

        # ====================================================
        # Purpose / Scope / Content /
        # Frequency / Format
        # ====================================================

        field_matches = (
            _context_field_matches(
                recent
            )
        )

        context_field_counts.update(
            field_matches
        )

        if len(
            field_matches
        ) >= 3:
            table_context_3plus_count += 1

        if len(
            field_matches
        ) == len(
            _CONTEXT_FIELDS
        ):
            table_context_all5_count += 1

    return {
        "raw_paragraph_count":
            raw_paragraph_count,
        "non_empty_paragraph_count":
            non_empty_paragraph_count,
        "table_count":
            table_count,
        "style_counts":
            style_counts,
        "builtin_heading_styles":
            builtin_heading_styles,
        "outline_counts":
            outline_counts,
        "outline_style_counts":
            outline_style_counts,
        "table_label_nearby_count":
            table_label_nearby_count,
        "table_outline_nearby_count":
            table_outline_nearby_count,
        "table_context_3plus_count":
            table_context_3plus_count,
        "table_context_all5_count":
            table_context_all5_count,
        "context_field_counts":
            context_field_counts,
        "preceding_style_counts":
            preceding_style_counts,
        "table_row_buckets":
            table_row_buckets,
        "table_col_buckets":
            table_col_buckets,
        "max_table_rows":
            max_table_rows,
        "max_table_cols":
            max_table_cols,
        "internal_context_field_counts":
            internal_context_field_counts,

        "table_internal_context_3plus_count":
            table_internal_context_3plus_count,

        "table_internal_context_all5_count":
            table_internal_context_all5_count,
    }


def _counter_to_dict(
    counter: Counter,
) -> dict:
    return dict(
        counter.most_common()
    )


# ============================================================
# Main
# ============================================================


def main() -> None:
    args = parse_args()

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

    source_manifest = (
        build_competition_source_manifest(
            args.attachments
        )
    )

    source_by_id = {
        source.source_id:
            source
        for source
        in source_manifest
    }

    resolver = (
        CompetitionSourceResolver(
            source_manifest
        )
    )

    # ========================================================
    # Frozen Dev DOCX Sources
    # ========================================================

    dev_docx_sources = {}

    for case in cases:
        if (
            case.case_id
            not in dev_case_ids
        ):
            continue

        if (
            case.source_type
            != "word"
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
            source.extension
            != ".docx"
        ):
            continue

        dev_docx_sources[
            source.source_id
        ] = source

    print(
        "=== Frozen Dev DOCX "
        "Heading / Table Context Audit ==="
    )

    print(
        "Unique Dev DOCX sources:",
        len(
            dev_docx_sources
        ),
    )

    print()

    # ========================================================
    # Aggregate
    # ========================================================

    aggregate_styles = Counter()

    aggregate_builtin_headings = (
        Counter()
    )

    aggregate_outline_levels = (
        Counter()
    )

    aggregate_outline_styles = (
        Counter()
    )

    aggregate_context_fields = (
        Counter()
    )

    aggregate_preceding_styles = (
        Counter()
    )

    aggregate_row_buckets = (
        Counter()
    )

    aggregate_col_buckets = (
        Counter()
    )

    total_tables = 0

    total_table_labels = 0

    total_outline_nearby = 0

    total_context_3plus = 0

    total_context_all5 = 0

    max_table_rows = 0
    max_table_cols = 0

    # ========================================================
    # Per Document
    # ========================================================

    for source_id in sorted(
        dev_docx_sources
    ):
        source = (
            dev_docx_sources[
                source_id
            ]
        )

        path = (
            resolve_competition_source_path(
                attachments_root=(
                    args.attachments
                ),
                source=source,
            )
        )

        result = (
            _audit_docx(
                path
            )
        )

        table_count = int(
            result[
                "table_count"
            ]
        )

        print(
            f"[DOCX] {source_id}"
        )

        print(
            "  paragraphs:",
            result[
                "raw_paragraph_count"
            ],
        )

        print(
            "  non_empty_paragraphs:",
            result[
                "non_empty_paragraph_count"
            ],
        )

        print(
            "  tables:",
            table_count,
        )

        print(
            "  top_styles:",
            dict(
                result[
                    "style_counts"
                ].most_common(
                    10
                )
            ),
        )

        print(
            "  builtin_heading_styles:",
            _counter_to_dict(
                result[
                    "builtin_heading_styles"
                ]
            ),
        )

        print(
            "  outline_levels:",
            _counter_to_dict(
                result[
                    "outline_counts"
                ]
            ),
        )

        print(
            "  styles_with_outline:",
            {
                (
                    f"{style_name}"
                    f"@level{level}"
                ):
                    count
                for (
                    (
                        style_name,
                        level,
                    ),
                    count,
                )
                in result[
                    "outline_style_counts"
                ].most_common(
                    10
                )
            },
        )

        print(
            "  table_row_buckets:",
            _counter_to_dict(
                result[
                    "table_row_buckets"
                ]
            ),
        )

        print(
            "  table_col_buckets:",
            _counter_to_dict(
                result[
                    "table_col_buckets"
                ]
            ),
        )

        print(
            "  max_table_rows:",
            result[
                "max_table_rows"
            ],
        )

        print(
            "  max_table_cols:",
            result[
                "max_table_cols"
            ],
        )

        print(
            "  table_label_nearby:",
            (
                f"{result['table_label_nearby_count']}"
                f"/{table_count}"
            ),
        )

        print(
            "  outline_heading_nearby:",
            (
                f"{result['table_outline_nearby_count']}"
                f"/{table_count}"
            ),
        )

        print(
            "  context_fields:",
            _counter_to_dict(
                result[
                    "context_field_counts"
                ]
            ),
        )

        print(
            "  context_3plus_fields:",
            (
                f"{result['table_context_3plus_count']}"
                f"/{table_count}"
            ),
        )

        print(
            "  context_all5_fields:",
            (
                f"{result['table_context_all5_count']}"
                f"/{table_count}"
            ),
        )

        print(
            "  immediate_preceding_styles:",
            dict(
                result[
                    "preceding_style_counts"
                ].most_common(
                    5
                )
            ),
        )

        print(
            "  internal_context_fields:",
            _counter_to_dict(
                result[
                    "internal_context_field_counts"
                ]
            ),
        )

        print(
            "  internal_context_3plus:",
            (
                f"{result['table_internal_context_3plus_count']}"
                f"/{table_count}"
            ),
        )

        print(
            "  internal_context_all5:",
            (
                f"{result['table_internal_context_all5_count']}"
                f"/{table_count}"
            ),
        )

        print()

        # ====================================================
        # Aggregate
        # ====================================================

        aggregate_styles.update(
            result[
                "style_counts"
            ]
        )

        aggregate_builtin_headings.update(
            result[
                "builtin_heading_styles"
            ]
        )

        aggregate_outline_levels.update(
            result[
                "outline_counts"
            ]
        )

        aggregate_outline_styles.update(
            result[
                "outline_style_counts"
            ]
        )

        aggregate_context_fields.update(
            result[
                "context_field_counts"
            ]
        )

        aggregate_preceding_styles.update(
            result[
                "preceding_style_counts"
            ]
        )

        aggregate_row_buckets.update(
            result[
                "table_row_buckets"
            ]
        )

        aggregate_col_buckets.update(
            result[
                "table_col_buckets"
            ]
        )

        total_tables += table_count

        total_table_labels += int(
            result[
                "table_label_nearby_count"
            ]
        )

        total_outline_nearby += int(
            result[
                "table_outline_nearby_count"
            ]
        )

        total_context_3plus += int(
            result[
                "table_context_3plus_count"
            ]
        )

        total_context_all5 += int(
            result[
                "table_context_all5_count"
            ]
        )

        max_table_rows = max(
            max_table_rows,
            int(
                result[
                    "max_table_rows"
                ]
            ),
        )

        max_table_cols = max(
            max_table_cols,
            int(
                result[
                    "max_table_cols"
                ]
            ),
        )

    # ========================================================
    # Summary
    # ========================================================

    print(
        "=== Aggregate Summary ==="
    )

    print(
        "Top styles:",
        dict(
            aggregate_styles
            .most_common(
                15
            )
        ),
    )

    print(
        "Built-in heading styles:",
        _counter_to_dict(
            aggregate_builtin_headings
        ),
    )

    print(
        "Outline levels:",
        _counter_to_dict(
            aggregate_outline_levels
        ),
    )

    print(
        "Styles with outline:",
        {
            (
                f"{style_name}"
                f"@level{level}"
            ):
                count
            for (
                (
                    style_name,
                    level,
                ),
                count,
            )
            in aggregate_outline_styles
            .most_common(
                15
            )
        },
    )

    print()

    print(
        "Total tables:",
        total_tables,
    )

    print(
        "Table row buckets:",
        _counter_to_dict(
            aggregate_row_buckets
        ),
    )

    print(
        "Table col buckets:",
        _counter_to_dict(
            aggregate_col_buckets
        ),
    )

    print(
        "Max table rows:",
        max_table_rows,
    )

    print(
        "Max table cols:",
        max_table_cols,
    )

    print()

    print(
        "Tables with nearby table label:",
        f"{total_table_labels}/{total_tables}",
    )

    print(
        "Tables with nearby outline heading:",
        f"{total_outline_nearby}/{total_tables}",
    )

    print(
        "Table context field coverage:",
        _counter_to_dict(
            aggregate_context_fields
        ),
    )

    print(
        "Tables with >=3 context fields:",
        f"{total_context_3plus}/{total_tables}",
    )

    print(
        "Tables with all 5 context fields:",
        f"{total_context_all5}/{total_tables}",
    )

    print(
        "Immediate preceding styles:",
        dict(
            aggregate_preceding_styles
            .most_common(
                10
            )
        ),
    )

    print()

    print(
        "Frozen Dev DOCX structure "
        "audit completed."
    )


if __name__ == "__main__":
    main()