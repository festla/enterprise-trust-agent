from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


SUPPORTED_SUFFIXES = {
    ".json",
    ".jsonl",
    ".csv",
}

# JSON manifest 有时不是直接一个 list，
# 而是 {"records": [...]}
RECORD_CONTAINER_KEYS = (
    "records",
    "items",
    "data",
    "files",
    "documents",
    "attachments",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit competition manifest metadata "
            "without printing sensitive record values."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help=(
            "Competition private data root, e.g. "
            "data/competition/private"
        ),
    )

    return parser.parse_args()


def _is_non_empty(
    value: Any,
) -> bool:
    if value is None:
        return False

    if isinstance(
        value,
        str,
    ):
        return bool(
            value.strip()
        )

    if isinstance(
        value,
        (
            list,
            tuple,
            dict,
            set,
        ),
    ):
        return bool(value)

    return True


def _extract_json_records(
    payload: Any,
) -> list[
    dict[str, Any]
]:
    """
    支持：

    1. [
         {...},
         {...}
       ]

    2. {
         "records": [...]
       }

    3. 单条 dict
    """

    if isinstance(
        payload,
        list,
    ):
        return [
            item
            for item in payload
            if isinstance(
                item,
                dict,
            )
        ]

    if isinstance(
        payload,
        dict,
    ):
        for key in (
            RECORD_CONTAINER_KEYS
        ):
            value = payload.get(
                key
            )

            if isinstance(
                value,
                list,
            ):
                records = [
                    item
                    for item in value
                    if isinstance(
                        item,
                        dict,
                    )
                ]

                if records:
                    return records

        # 如果整个 JSON 本身就是一条记录，
        # 也允许审计。
        return [
            payload
        ]

    return []


def _load_json(
    path: Path,
) -> list[
    dict[str, Any]
]:
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        payload = json.load(
            file
        )

    return _extract_json_records(
        payload
    )


def _load_jsonl(
    path: Path,
) -> list[
    dict[str, Any]
]:
    records = []

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(
                    line
                )
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path}: "
                    f"JSONL 第 {line_number} 行解析失败"
                ) from exc

            if isinstance(
                item,
                dict,
            ):
                records.append(
                    item
                )

    return records


def _load_csv(
    path: Path,
) -> list[
    dict[str, Any]
]:
    # 优先 UTF-8。
    # 如果真实 manifest 是 GBK/GB18030，
    # 再做 fallback。
    encodings = (
        "utf-8-sig",
        "gb18030",
    )

    last_error: Exception | None = None

    for encoding in encodings:
        try:
            with path.open(
                "r",
                encoding=encoding,
                newline="",
            ) as file:
                reader = csv.DictReader(
                    file
                )

                return [
                    dict(row)
                    for row in reader
                ]

        except UnicodeDecodeError as exc:
            last_error = exc

    raise RuntimeError(
        f"无法识别 CSV 编码: {path}"
    ) from last_error


def _load_records(
    path: Path,
) -> list[
    dict[str, Any]
]:
    suffix = path.suffix.lower()

    if suffix == ".json":
        return _load_json(
            path
        )

    if suffix == ".jsonl":
        return _load_jsonl(
            path
        )

    if suffix == ".csv":
        return _load_csv(
            path
        )

    raise ValueError(
        f"Unsupported manifest format: "
        f"{suffix}"
    )


def _find_manifest_candidates(
    root: Path,
) -> list[Path]:
    """
    只找文件名中带 manifest 的 JSON / JSONL / CSV。

    避免误把 QA 数据、评测结果等普通 JSON/CSV
    当成 manifest。
    """

    candidates = []

    for path in root.rglob(
        "*"
    ):
        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            not in SUPPORTED_SUFFIXES
        ):
            continue

        if (
            "manifest"
            not in path.name.lower()
        ):
            continue

        candidates.append(
            path
        )

    return sorted(
        candidates
    )


def _collect_field_stats(
    records: list[
        dict[str, Any]
    ],
) -> tuple[
    list[str],
    Counter[str],
]:
    all_fields = set()

    non_empty_counts: Counter[
        str
    ] = Counter()

    for record in records:
        for key, value in (
            record.items()
        ):
            key = str(
                key
            )

            all_fields.add(
                key
            )

            if _is_non_empty(
                value
            ):
                non_empty_counts[
                    key
                ] += 1

    return (
        sorted(
            all_fields
        ),
        non_empty_counts,
    )


def _print_manifest_audit(
    *,
    root: Path,
    path: Path,
    records: list[
        dict[str, Any]
    ],
) -> None:
    relative_path = (
        path.relative_to(
            root
        )
    )

    print()
    print(
        "=" * 72
    )

    print(
        "Manifest:",
        relative_path,
    )

    print(
        "Format:",
        path.suffix.lower(),
    )

    print(
        "Records:",
        len(records),
    )

    if not records:
        print(
            "Fields: none"
        )
        return

    (
        fields,
        non_empty_counts,
    ) = _collect_field_stats(
        records
    )

    print()
    print(
        "Fields:"
    )

    for field in fields:
        count = (
            non_empty_counts[
                field
            ]
        )

        ratio = (
            count / len(records)
        )

        print(
            f"  {field:<30} "
            f"{count:>5}/"
            f"{len(records):<5} "
            f"({ratio:>6.1%})"
        )


def main() -> None:
    args = parse_args()

    root = args.root.resolve()

    if not root.exists():
        raise FileNotFoundError(
            f"Root 不存在: {root}"
        )

    candidates = (
        _find_manifest_candidates(
            root
        )
    )

    print(
        "=== Competition Manifest Audit ==="
    )

    print(
        "Root:",
        root,
    )

    print(
        "Manifest candidates:",
        len(candidates),
    )

    if not candidates:
        print()
        print(
            "没有找到文件名包含 "
            "'manifest' 的 "
            "JSON / JSONL / CSV 文件。"
        )

        print(
            "当前脚本不会自动扫描所有 "
            "JSON/CSV，以避免误读 QA "
            "或评测文件。"
        )

        return

    success_count = 0
    failure_count = 0

    for path in candidates:
        try:
            records = (
                _load_records(
                    path
                )
            )

            _print_manifest_audit(
                root=root,
                path=path,
                records=records,
            )

            success_count += 1

        except Exception as exc:
            failure_count += 1

            print()
            print(
                "=" * 72
            )

            print(
                "Manifest:",
                path.relative_to(
                    root
                ),
            )

            print(
                "ERROR:",
                type(exc).__name__,
                str(exc),
            )

    print()
    print(
        "=" * 72
    )

    print(
        "Summary:"
    )

    print(
        "  Candidates:",
        len(candidates),
    )

    print(
        "  Parsed:",
        success_count,
    )

    print(
        "  Failed:",
        failure_count,
    )


if __name__ == "__main__":
    main()