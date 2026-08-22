from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


CHUNK_FILE = Path(
    "data/competition/processed/"
    "competition_text_chunks_dev.jsonl"
)

MAX_REASONABLE_CHARS = 900

# Frozen Dev 数据集基线。
EXPECTED_UNIQUE_DOCS = 5
EXPECTED_UNIQUE_TABLES = 140

Chunk = dict[str, Any]


def load_chunks(
    path: Path = CHUNK_FILE,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as reader:
        for line_number, line in enumerate(
            reader,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                chunks.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Chunk JSONL 解析失败: "
                    f"line={line_number}; "
                    f"error={exc}"
                ) from exc

    return chunks


def _text_sha256(
    text: str,
) -> str:
    return sha256(
        text.encode("utf-8")
    ).hexdigest()


def _render_table_rows(
    rows: list[list[str]],
) -> str:
    return "\n".join(
        "\t".join(row)
        for row in rows
    )


def _is_allowed_oversized_table(
    chunk: Chunk,
) -> bool:
    rows = chunk.get("table_rows")

    return (
        chunk.get("chunk_type") == "table"
        and isinstance(rows, list)
        and len(rows) == 1
        and chunk.get("table_row_start")
        == chunk.get("table_row_end")
    )


def collect_audit_errors(
    chunks: list[Chunk],
    *,
    expected_doc_count: int | None = None,
    expected_table_count: int | None = None,
) -> list[str]:
    errors: list[str] = []

    if not chunks:
        return ["Chunk 文件为空"]

    required_fields = (
        "chunk_id",
        "source_id",
        "doc_id",
        "source_type",
        "chunk_index",
        "chunk_type",
        "text",
        "char_count",
        "text_sha256",
        "source_spans",
    )

    chunk_id_counter: Counter[str] = Counter()
    chunks_by_doc: dict[
        str,
        list[Chunk],
    ] = defaultdict(list)

    table_groups: dict[
        tuple[str, int],
        list[Chunk],
    ] = defaultdict(list)

    for position, chunk in enumerate(chunks):
        chunk_id = chunk.get(
            "chunk_id",
            f"<record:{position}>",
        )

        missing_fields = [
            field
            for field in required_fields
            if field not in chunk
        ]

        if missing_fields:
            errors.append(
                f"{chunk_id}: 缺少字段 "
                f"{missing_fields}"
            )
            continue

        actual_chunk_id = chunk["chunk_id"]
        doc_id = chunk["doc_id"]
        chunk_index = chunk["chunk_index"]
        chunk_type = chunk["chunk_type"]
        source_type = chunk["source_type"]
        text = chunk["text"]
        char_count = chunk["char_count"]

        if isinstance(actual_chunk_id, str):
            chunk_id_counter[
                actual_chunk_id
            ] += 1
        else:
            errors.append(
                f"{chunk_id}: chunk_id 不是字符串"
            )

        if isinstance(doc_id, str):
            chunks_by_doc[doc_id].append(
                chunk
            )
        else:
            errors.append(
                f"{chunk_id}: doc_id 不是字符串"
            )

        if (
            not isinstance(text, str)
            or not text.strip()
        ):
            errors.append(
                f"{chunk_id}: text 为空"
            )
            continue

        if (
            not isinstance(char_count, int)
            or char_count != len(text)
        ):
            errors.append(
                f"{chunk_id}: char_count 不一致; "
                f"stored={char_count}; "
                f"actual={len(text)}"
            )

        expected_hash = _text_sha256(text)

        if chunk.get(
            "text_sha256"
        ) != expected_hash:
            errors.append(
                f"{chunk_id}: text_sha256 不一致"
            )

        source_spans = chunk.get(
            "source_spans"
        )

        if (
            not isinstance(source_spans, list)
            or not source_spans
        ):
            errors.append(
                f"{chunk_id}: 缺少 source_spans"
            )

        if (
            isinstance(char_count, int)
            and char_count
            > MAX_REASONABLE_CHARS
            and not _is_allowed_oversized_table(
                chunk
            )
        ):
            errors.append(
                f"{chunk_id}: 非法超长 Chunk; "
                f"chars={char_count}"
            )

        if chunk_type == "text":
            if source_type not in {
                "word",
                "pdf",
            }:
                errors.append(
                    f"{chunk_id}: Text Chunk "
                    f"来源类型非法: {source_type}"
                )

            continue

        if chunk_type != "table":
            errors.append(
                f"{chunk_id}: 未知 chunk_type: "
                f"{chunk_type}"
            )
            continue

        if source_type != "word":
            errors.append(
                f"{chunk_id}: Table Chunk "
                "只能来自 Word"
            )

        table_index = chunk.get(
            "table_index"
        )
        row_start = chunk.get(
            "table_row_start"
        )
        row_end = chunk.get(
            "table_row_end"
        )
        rows = chunk.get(
            "table_rows"
        )
        table_format = chunk.get(
            "table_format"
        )

        if not isinstance(table_index, int):
            errors.append(
                f"{chunk_id}: table_index 非法"
            )
            continue

        if (
            not isinstance(row_start, int)
            or not isinstance(row_end, int)
            or row_start < 0
            or row_end < row_start
        ):
            errors.append(
                f"{chunk_id}: 表格行范围非法"
            )
            continue

        if (
            not isinstance(rows, list)
            or not rows
            or not all(
                isinstance(row, list)
                and all(
                    isinstance(cell, str)
                    for cell in row
                )
                for row in rows
            )
        ):
            errors.append(
                f"{chunk_id}: table_rows 非法"
            )
            continue

        expected_row_count = (
            row_end - row_start + 1
        )

        if len(rows) != expected_row_count:
            errors.append(
                f"{chunk_id}: table_rows 数量"
                "与行范围不一致; "
                f"rows={len(rows)}; "
                f"range={expected_row_count}"
            )

        rendered_text = _render_table_rows(
            rows
        )

        if rendered_text != text:
            errors.append(
                f"{chunk_id}: table_rows "
                "无法还原 text"
            )

        if (
            not isinstance(table_format, str)
            or not table_format.strip()
        ):
            errors.append(
                f"{chunk_id}: 缺少 table_format"
            )

        table_groups[
            (
                doc_id,
                table_index,
            )
        ].append(chunk)

    duplicate_ids = [
        chunk_id
        for chunk_id, count
        in chunk_id_counter.items()
        if count > 1
    ]

    for chunk_id in duplicate_ids:
        errors.append(
            f"{chunk_id}: 重复 Chunk ID"
        )

    for doc_id, doc_chunks in (
        chunks_by_doc.items()
    ):
        indices = sorted(
            chunk.get("chunk_index")
            for chunk in doc_chunks
            if isinstance(
                chunk.get("chunk_index"),
                int,
            )
        )

        expected_indices = list(
            range(len(doc_chunks))
        )

        if indices != expected_indices:
            errors.append(
                f"{doc_id}: chunk_index "
                "不连续或存在重复"
            )

        for chunk in doc_chunks:
            chunk_index = chunk.get(
                "chunk_index"
            )

            if not isinstance(
                chunk_index,
                int,
            ):
                continue

            expected_chunk_id = (
                f"chunk:{doc_id}:"
                f"{chunk_index:05d}"
            )

            if (
                chunk.get("chunk_id")
                != expected_chunk_id
            ):
                errors.append(
                    f"{chunk.get('chunk_id')}: "
                    "Chunk ID 与索引不一致"
                )

    for table_key, table_chunks in (
        table_groups.items()
    ):
        expected_start = 0

        for chunk in sorted(
            table_chunks,
            key=lambda item: (
                item["table_row_start"]
            ),
        ):
            actual_start = chunk[
                "table_row_start"
            ]

            if actual_start != expected_start:
                errors.append(
                    f"{table_key}: 表格行范围"
                    "存在断裂或重叠; "
                    f"expected={expected_start}; "
                    f"actual={actual_start}"
                )

            expected_start = (
                chunk["table_row_end"]
                + 1
            )

    if (
        expected_doc_count is not None
        and len(chunks_by_doc)
        != expected_doc_count
    ):
        errors.append(
            "Dev 文档数量不符合冻结基线; "
            f"expected={expected_doc_count}; "
            f"actual={len(chunks_by_doc)}"
        )

    if (
        expected_table_count is not None
        and len(table_groups)
        != expected_table_count
    ):
        errors.append(
            "Dev 表格数量不符合冻结基线; "
            f"expected={expected_table_count}; "
            f"actual={len(table_groups)}"
        )

    return errors


def print_report(
    chunks: list[Chunk],
) -> None:
    lengths = [
        chunk["char_count"]
        for chunk in chunks
        if isinstance(
            chunk.get("char_count"),
            int,
        )
    ]

    source_and_type = Counter(
        (
            chunk.get("source_type"),
            chunk.get("chunk_type"),
        )
        for chunk in chunks
    )

    doc_ids = {
        chunk.get("doc_id")
        for chunk in chunks
        if chunk.get("doc_id")
    }

    table_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("chunk_type")
        == "table"
    ]

    table_groups = {
        (
            chunk.get("doc_id"),
            chunk.get("table_index"),
        )
        for chunk in table_chunks
    }

    oversized = [
        chunk
        for chunk in chunks
        if isinstance(
            chunk.get("char_count"),
            int,
        )
        and chunk["char_count"]
        > MAX_REASONABLE_CHARS
    ]

    print(
        "=== Competition Dev Chunk Audit ==="
    )
    print("Total chunks:", len(chunks))
    print("Unique doc IDs:", len(doc_ids))
    print(
        "Unique table blocks:",
        len(table_groups),
    )
    print(
        "Table chunks:",
        len(table_chunks),
    )

    if lengths:
        print(
            "Average chars:",
            sum(lengths) / len(lengths),
        )
        print("Max chars:", max(lengths))
        print("Min chars:", min(lengths))

    print(
        "Allowed oversized table rows:",
        sum(
            _is_allowed_oversized_table(
                chunk
            )
            for chunk in oversized
        ),
    )

    print("\nChunk distribution:")

    for key, count in sorted(
        source_and_type.items(),
        key=lambda item: str(item[0]),
    ):
        print(
            f"  {key}: {count}"
        )

    print("\nTable metadata coverage:")

    for field in (
        "table_title",
        "table_unit",
        "table_frequency",
        "table_purpose",
        "table_content",
        "table_scope",
        "table_format",
    ):
        count = sum(
            bool(chunk.get(field))
            for chunk in table_chunks
        )
        print(
            f"  {field}: "
            f"{count}/{len(table_chunks)}"
        )


def main() -> None:
    chunks = load_chunks()

    print_report(chunks)

    errors = collect_audit_errors(
        chunks,
        expected_doc_count=(
            EXPECTED_UNIQUE_DOCS
        ),
        expected_table_count=(
            EXPECTED_UNIQUE_TABLES
        ),
    )

    print("\n===== Audit Result =====")
    print("Errors:", len(errors))

    for error in errors[:50]:
        print("[ERROR]", error)

    if errors:
        raise SystemExit(1)

    print(
        "Competition Dev chunk audit passed."
    )


if __name__ == "__main__":
    main()