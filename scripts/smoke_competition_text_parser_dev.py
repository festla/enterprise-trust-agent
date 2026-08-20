from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from app.services.competition_dataset import (
    build_competition_question,
    load_competition_qa_excel,
)
from app.services.competition_source_resolver import (
    CompetitionSourceResolver,
    build_competition_source_manifest,
)
from app.services.competition_text_parser import (
    parse_competition_text_document,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test Competition text parser "
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


def main() -> None:
    args = parse_args()

    # ========================================================
    # 1. 加载 QA 与 Frozen Split
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
    # 2. 构造附件 Source Manifest / Resolver
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
    # 3. 找到 Frozen Dev 真正使用的
    #    PDF / Word Source。
    #
    # 每个 Source 只解析一次。
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

        # 保存第一个对应 QA。
        #
        # Parser 真正获得的是后面构造的
        # CompetitionQuestion，
        # 不会获得 answer / evidence / difficulty。
        dev_sources[
            source.source_id
        ] = (
            case,
            source,
        )

    # ========================================================
    # 4. Smoke Parse
    # ========================================================

    source_type_counts = Counter()

    extension_counts = Counter()

    block_type_counts = Counter()

    total_blocks = 0
    total_chars = 0

    success_count = 0
    failure_count = 0

    failures: list[
        tuple[
            str,
            str,
        ]
    ] = []

    print(
        "=== Frozen Dev Text Parser Smoke ==="
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

        source_type_counts[
            source.source_type
        ] += 1

        extension_counts[
            source.extension
        ] += 1

        try:
            parsed = (
                parse_competition_text_document(
                    question=question,
                    source=source,
                    attachments_root=(
                        args.attachments
                    ),
                )
            )

        except Exception as exc:
            failure_count += 1

            failures.append(
                (
                    source_id,
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                )
            )

            print(
                f"[FAIL] "
                f"{source_id} "
                f"type={source.source_type} "
                f"ext={source.extension} "
                f"error="
                f"{type(exc).__name__}"
            )

            continue

        success_count += 1

        local_block_counts = (
            Counter(
                block.block_type
                for block
                in parsed.blocks
            )
        )

        local_chars = sum(
            len(
                block.text
            )
            for block
            in parsed.blocks
        )

        total_blocks += len(
            parsed.blocks
        )

        total_chars += (
            local_chars
        )

        block_type_counts.update(
            local_block_counts
        )

        # ====================================================
        # 这里故意不打印：
        #
        # - 文档标题
        # - 文件名
        # - 正文
        # - QA
        # - Gold Evidence
        #
        # 只打印非敏感结构统计。
        # ====================================================

        print(
            f"[OK]   "
            f"{source_id} "
            f"type={source.source_type} "
            f"ext={source.extension} "
            f"blocks={len(parsed.blocks)} "
            f"chars={local_chars} "
            f"block_types="
            f"{dict(local_block_counts)}"
        )

    # ========================================================
    # 5. Summary
    # ========================================================

    print()
    print(
        "=== Summary ==="
    )

    print(
        "Source types:",
        dict(
            source_type_counts
        ),
    )

    print(
        "Extensions:",
        dict(
            extension_counts
        ),
    )

    print(
        "Success:",
        success_count,
    )

    print(
        "Failure:",
        failure_count,
    )

    print(
        "Total blocks:",
        total_blocks,
    )

    print(
        "Block types:",
        dict(
            block_type_counts
        ),
    )

    print(
        "Total chars:",
        total_chars,
    )

    # ========================================================
    # Frozen Dev 当前应该只有 PDF + DOCX。
    #
    # 如果出现其他格式，说明前面的 Corpus Audit
    # 与当前 Source Resolution 状态不一致。
    # ========================================================

    unexpected_extensions = (
        set(
            extension_counts
        )
        - {
            ".pdf",
            ".docx",
        }
    )

    if unexpected_extensions:
        raise RuntimeError(
            "Frozen Dev 出现未预期文本格式: "
            f"{sorted(unexpected_extensions)}"
        )

    if failures:
        print()
        print(
            "=== Failures ==="
        )

        for (
            source_id,
            error,
        ) in failures:
            print(
                f"{source_id}: "
                f"{error}"
            )

        raise SystemExit(
            1
        )

    if (
        success_count
        != len(
            dev_sources
        )
    ):
        raise RuntimeError(
            "Parser success count "
            "与 Dev source count 不一致"
        )

    print()
    print(
        "Frozen Dev text parser "
        "smoke passed."
    )


if __name__ == "__main__":
    main()