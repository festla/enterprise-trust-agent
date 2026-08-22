from hashlib import sha256

from scripts.audit_competition_chunk_dev import (
    collect_audit_errors,
)


def _hash(text: str) -> str:
    return sha256(
        text.encode("utf-8")
    ).hexdigest()


def _chunk(
    *,
    chunk_index: int,
    text: str,
    chunk_type: str = "text",
    row_start: int | None = None,
    row_end: int | None = None,
    rows: list[list[str]] | None = None,
) -> dict:
    doc_id = "doc_test"

    return {
        "chunk_id": (
            f"chunk:{doc_id}:"
            f"{chunk_index:05d}"
        ),
        "source_id": "src_test",
        "doc_id": doc_id,
        "source_type": "word",
        "chunk_index": chunk_index,
        "chunk_type": chunk_type,
        "text": text,
        "char_count": len(text),
        "text_sha256": _hash(text),
        "source_spans": [
            {
                "block_id": "block_test",
                "block_index": 0,
                "start_char": 0,
                "end_char": len(text),
            }
        ],
        "table_index": (
            0
            if chunk_type == "table"
            else None
        ),
        "table_row_start": row_start,
        "table_row_end": row_end,
        "table_rows": rows,
        "table_format": (
            "table"
            if chunk_type == "table"
            else None
        ),
    }


def test_valid_text_and_table_chunks_pass(
) -> None:
    chunks = [
        _chunk(
            chunk_index=0,
            text="监管正文。",
        ),
        _chunk(
            chunk_index=1,
            chunk_type="table",
            text="指标\t数值",
            row_start=0,
            row_end=0,
            rows=[
                ["指标", "数值"],
            ],
        ),
    ]

    assert collect_audit_errors(
        chunks
    ) == []


def test_single_oversized_table_row_is_allowed(
) -> None:
    text = "超长单元格" * 200

    chunks = [
        _chunk(
            chunk_index=0,
            chunk_type="table",
            text=text,
            row_start=0,
            row_end=0,
            rows=[[text]],
        )
    ]

    assert len(text) > 900

    assert collect_audit_errors(
        chunks
    ) == []


def test_oversized_multirow_table_fails(
) -> None:
    first = "甲" * 500
    second = "乙" * 500
    text = f"{first}\n{second}"

    chunks = [
        _chunk(
            chunk_index=0,
            chunk_type="table",
            text=text,
            row_start=0,
            row_end=1,
            rows=[
                [first],
                [second],
            ],
        )
    ]

    errors = collect_audit_errors(
        chunks
    )

    assert any(
        "非法超长 Chunk" in error
        for error in errors
    )


def test_table_row_gap_fails(
) -> None:
    chunks = [
        _chunk(
            chunk_index=0,
            chunk_type="table",
            text="第一行",
            row_start=0,
            row_end=0,
            rows=[["第一行"]],
        ),
        _chunk(
            chunk_index=1,
            chunk_type="table",
            text="第三行",
            row_start=2,
            row_end=2,
            rows=[["第三行"]],
        ),
    ]

    errors = collect_audit_errors(
        chunks
    )

    assert any(
        "断裂或重叠" in error
        for error in errors
    )