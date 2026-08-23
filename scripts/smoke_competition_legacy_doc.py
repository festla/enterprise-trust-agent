from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.schemas.competition import (
    CompetitionQaCase,
    CompetitionSourceRecord,
)
from app.services.competition_dataset import (
    build_competition_question,
    load_competition_qa_excel,
)
from app.services.competition_source_catalog import (
    resolve_competition_source_path,
)
from app.services.competition_source_resolver import (
    CompetitionSourceResolver,
    build_competition_source_manifest,
)
from app.services.competition_text_parser import (
    parse_competition_text_document,
)
from app.services.document_ingestion import (
    calculate_file_sha256,
)


LegacyDocSource = tuple[
    CompetitionQaCase,
    CompetitionSourceRecord,
]


class CompetitionLegacyDocSmokeError(
    RuntimeError
):
    """Legacy DOC compatibility gate failed."""


@dataclass(frozen=True)
class LegacyDocSmokeSummary:
    source_count: int
    success_count: int
    failure_count: int
    total_blocks: int
    total_chars: int
    block_type_counts: dict[str, int]
    failures: tuple[
        tuple[str, str],
        ...
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test QA-used legacy DOC files "
            "without printing document content, "
            "questions, answers, or evidence."
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
        "--expected-sources",
        type=int,
        default=2,
    )

    return parser.parse_args()


def collect_qa_used_legacy_doc_sources(
    *,
    cases: tuple[
        CompetitionQaCase,
        ...
    ],
    source_manifest: tuple[
        CompetitionSourceRecord,
        ...
    ],
) -> dict[
    str,
    LegacyDocSource,
]:
    """
    Collect unique legacy DOC sources used by QA cases.

    Only source resolution fields are used. Gold answer,
    answer_text and evidence are never passed to the parser.
    """

    source_by_id = {
        source.source_id: source
        for source in source_manifest
    }

    resolver = CompetitionSourceResolver(
        source_manifest
    )

    legacy_sources: dict[
        str,
        LegacyDocSource,
    ] = {}

    for case in cases:
        if case.source_type != "word":
            continue

        resolution = resolver.resolve(
            case
        )

        source = source_by_id[
            resolution.source_id
        ]

        if (
            source.extension.casefold()
            != ".doc"
        ):
            continue

        # 同一份文档可能对应多道 QA。
        # 这里只保存第一道题，用于构建不含 Gold
        # 字段的 CompetitionQuestion。
        legacy_sources.setdefault(
            source.source_id,
            (
                case,
                source,
            ),
        )

    return dict(
        sorted(
            legacy_sources.items()
        )
    )


def run_legacy_doc_smoke(
    *,
    cases: tuple[
        CompetitionQaCase,
        ...
    ],
    source_manifest: tuple[
        CompetitionSourceRecord,
        ...
    ],
    attachments_root: Path,
) -> LegacyDocSmokeSummary:
    legacy_sources = (
        collect_qa_used_legacy_doc_sources(
            cases=cases,
            source_manifest=source_manifest,
        )
    )

    block_type_counts: Counter[
        str
    ] = Counter()

    total_blocks = 0
    total_chars = 0
    success_count = 0

    failures: list[
        tuple[str, str]
    ] = []

    print(
        "=== QA-used Legacy DOC Parser Smoke ==="
    )

    print(
        "Unique legacy DOC sources:",
        len(legacy_sources),
    )

    print()

    for source_id in sorted(
        legacy_sources
    ):
        (
            case,
            source,
        ) = legacy_sources[
            source_id
        ]

        try:
            source_path = (
                resolve_competition_source_path(
                    attachments_root=(
                        attachments_root
                    ),
                    source=source,
                )
            )

            expected_sha256 = (
                calculate_file_sha256(
                    source_path
                )
            )

            # build_competition_question() 会移除：
            # answer、answer_text、evidence、difficulty。
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
                        attachments_root
                    ),
                )
            )

            if not parsed.blocks:
                raise (
                    CompetitionLegacyDocSmokeError(
                        "Parser returned no blocks"
                    )
                )

            if not any(
                block.text.strip()
                for block in parsed.blocks
            ):
                raise (
                    CompetitionLegacyDocSmokeError(
                        "Parser returned no "
                        "non-empty blocks"
                    )
                )

            if (
                parsed.source.source_id
                != source.source_id
            ):
                raise (
                    CompetitionLegacyDocSmokeError(
                        "source_id changed "
                        "after conversion"
                    )
                )

            if (
                parsed.source.relative_path
                != source.relative_path
            ):
                raise (
                    CompetitionLegacyDocSmokeError(
                        "relative_path changed "
                        "after conversion"
                    )
                )

            if (
                parsed.source.source_type
                != "word"
            ):
                raise (
                    CompetitionLegacyDocSmokeError(
                        "source_type changed "
                        "after conversion"
                    )
                )

            if (
                parsed.source.sha256
                != expected_sha256
            ):
                raise (
                    CompetitionLegacyDocSmokeError(
                        "SHA-256 does not match "
                        "the original DOC"
                    )
                )

            local_block_counts = Counter(
                block.block_type
                for block in parsed.blocks
            )

            local_chars = sum(
                len(block.text)
                for block in parsed.blocks
            )

        except Exception as exc:
            failures.append(
                (
                    source_id,
                    type(exc).__name__,
                )
            )

            # 不打印文件名、正文或异常消息，
            # 避免将 Test 文档内容写入日志。
            print(
                f"[FAIL] "
                f"{source_id} "
                f"error={type(exc).__name__}"
            )

            continue

        success_count += 1
        total_blocks += len(
            parsed.blocks
        )
        total_chars += local_chars

        block_type_counts.update(
            local_block_counts
        )

        print(
            f"[OK]   "
            f"{source_id} "
            f"ext=.doc "
            f"blocks={len(parsed.blocks)} "
            f"chars={local_chars} "
            f"block_types="
            f"{dict(local_block_counts)}"
        )

    return LegacyDocSmokeSummary(
        source_count=len(
            legacy_sources
        ),
        success_count=success_count,
        failure_count=len(
            failures
        ),
        total_blocks=total_blocks,
        total_chars=total_chars,
        block_type_counts=dict(
            block_type_counts
        ),
        failures=tuple(
            failures
        ),
    )


def collect_gate_errors(
    *,
    summary: LegacyDocSmokeSummary,
    expected_source_count: int,
) -> list[str]:
    errors: list[str] = []

    if (
        summary.source_count
        != expected_source_count
    ):
        errors.append(
            "Legacy DOC source count mismatch: "
            f"expected={expected_source_count}, "
            f"actual={summary.source_count}"
        )

    if summary.failure_count:
        errors.append(
            "Legacy DOC parse failures: "
            f"{summary.failure_count}"
        )

    if (
        summary.success_count
        != summary.source_count
    ):
        errors.append(
            "Legacy DOC success count "
            "does not match source count"
        )

    if (
        summary.source_count > 0
        and summary.total_blocks <= 0
    ):
        errors.append(
            "Legacy DOC parser produced "
            "no blocks"
        )

    if (
        summary.source_count > 0
        and summary.total_chars <= 0
    ):
        errors.append(
            "Legacy DOC parser produced "
            "no text"
        )

    return errors


def main() -> None:
    args = parse_args()

    cases = load_competition_qa_excel(
        args.qa
    )

    source_manifest = (
        build_competition_source_manifest(
            args.attachments
        )
    )

    summary = run_legacy_doc_smoke(
        cases=cases,
        source_manifest=source_manifest,
        attachments_root=args.attachments,
    )

    print()
    print(
        "=== Summary ==="
    )

    print(
        "Sources:",
        summary.source_count,
    )

    print(
        "Success:",
        summary.success_count,
    )

    print(
        "Failure:",
        summary.failure_count,
    )

    print(
        "Total blocks:",
        summary.total_blocks,
    )

    print(
        "Block types:",
        summary.block_type_counts,
    )

    print(
        "Total chars:",
        summary.total_chars,
    )

    errors = collect_gate_errors(
        summary=summary,
        expected_source_count=(
            args.expected_sources
        ),
    )

    if errors:
        print()
        print(
            "=== Gate Errors ==="
        )

        for error in errors:
            print(
                error
            )

        raise SystemExit(1)

    print()
    print(
        "Legacy DOC parser smoke passed."
    )


if __name__ == "__main__":
    main()