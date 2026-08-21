from __future__ import annotations

from collections import Counter
from pathlib import Path
import json


CHUNK_FILE = Path(
    "data/competition/processed/competition_text_chunks_dev.jsonl"
)


MAX_REASONABLE_CHARS = 900


def load_chunks():
    chunks = []

    with CHUNK_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                chunks.append(
                    json.loads(line)
                )

    return chunks


def audit_basic_stats(chunks):

    lengths = [
        c["char_count"]
        for c in chunks
    ]

    print("\n===== Chunk Statistics =====")

    print(
        "Total chunks:",
        len(chunks)
    )

    print(
        "Average chars:",
        sum(lengths) / len(lengths)
    )

    print(
        "Max chars:",
        max(lengths)
    )

    print(
        "Min chars:",
        min(lengths)
    )


def audit_source_type(chunks):

    counter = Counter(
        c["source_type"]
        for c in chunks
    )

    print(
        "\n===== Source Type ====="
    )

    for k, v in counter.items():
        print(
            k,
            ":",
            v
        )


def audit_metadata(chunks):

    print(
        "\n===== Metadata Check ====="
    )

    missing = []

    for c in chunks:

        required = [
            "chunk_id",
            "source_id",
            "doc_id",
            "text",
            "source_spans",
        ]

        for key in required:

            if key not in c:
                missing.append(
                    (
                        c.get(
                            "chunk_id"
                        ),
                        key,
                    )
                )


    if missing:

        print(
            "Missing fields:"
        )

        for item in missing[:20]:
            print(item)

    else:

        print(
            "All required fields exist"
        )


def audit_length(chunks):

    print(
        "\n===== Length Check ====="
    )

    oversized = []

    tiny = []

    for c in chunks:

        length = c["char_count"]

        if length > MAX_REASONABLE_CHARS:
            oversized.append(
                c
            )

        if length < 50:
            tiny.append(
                c
            )


    print(
        "Oversized chunks:",
        len(oversized)
    )

    print(
        "Tiny chunks:",
        len(tiny)
    )


    if oversized:

        print(
            "\nExample oversized:"
        )

        for c in oversized[:3]:

            print(
                c["chunk_id"],
                c["char_count"]
            )


def audit_source_span(chunks):

    print(
        "\n===== Source Span Check ====="
    )

    invalid = []

    for c in chunks:

        spans = c.get(
            "source_spans",
            []
        )

        if not spans:

            invalid.append(
                c["chunk_id"]
            )


    if invalid:

        print(
            "Missing source spans:",
            len(invalid)
        )

    else:

        print(
            "All chunks have source spans"
        )


def main():

    chunks = load_chunks()

    audit_basic_stats(
        chunks
    )

    audit_source_type(
        chunks
    )

    audit_metadata(
        chunks
    )

    audit_length(
        chunks
    )

    audit_source_span(
        chunks
    )


if __name__ == "__main__":

    main()