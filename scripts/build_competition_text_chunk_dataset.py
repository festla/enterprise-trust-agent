from __future__ import annotations

import json
from pathlib import Path


from app.services.competition_dataset import (
    load_competition_qa_excel,
)

from app.services.competition_source_resolver import (
    build_competition_source_manifest,
    CompetitionSourceResolver,
)

from app.services.competition_text_parser import (
    parse_competition_text_document,
)

from app.services.competition_text_chunker import (
    build_competition_text_chunks,
)


QA_FILE = Path(
    "data/competition/private/qa/QA数据.xlsx"
)


SPLIT_FILE = Path(
    "data/competition/processed/competition_eval_split_v1.json"
)


ATTACHMENTS_ROOT = Path(
    "data/competition/private/attachments"
)


OUTPUT_FILE = Path(
    "data/competition/processed/competition_text_chunks_dev.jsonl"
)

def load_dev_case_ids():

    with SPLIT_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        split = json.load(f)

    return set(
        split["dev_case_ids"]
    )

def main():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ======================================
    # 1. Load QA cases
    # ======================================

    cases = load_competition_qa_excel(
        QA_FILE
    )

    dev_case_ids = load_dev_case_ids()


    cases = [
        case
        for case in cases
        if case.case_id in dev_case_ids
    ]

    # ======================================
    # 2. Build attachment manifest
    # ======================================

    manifest = (
        build_competition_source_manifest(
            ATTACHMENTS_ROOT
        )
    )


    resolver = CompetitionSourceResolver(
        manifest
    )


    total_chunks = 0
    failed_cases = []


    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as writer:


        for case in cases:


            # 只处理 PDF / Word
            if case.source_type not in {
                "pdf",
                "word",
            }:
                continue


            try:

                # --------------------------
                # Question
                # --------------------------

                from app.services.competition_dataset import (
                    build_competition_question,
                )

                question = (
                    build_competition_question(
                        case
                    )
                )


                # --------------------------
                # Resolve source
                # --------------------------

                resolution = (
                    resolver.resolve(
                        case
                    )
                )


                source = next(
                    item
                    for item
                    in manifest
                    if (
                        item.source_id
                        ==
                        resolution.source_id
                    )
                )


                # --------------------------
                # Parse document
                # --------------------------

                document = (
                    parse_competition_text_document(
                        question=question,
                        source=source,
                        attachments_root=(
                            ATTACHMENTS_ROOT
                        ),
                    )
                )


                # --------------------------
                # Chunk
                # --------------------------

                chunks = (
                    build_competition_text_chunks(
                        document
                    )
                )


                for chunk in chunks:

                    writer.write(
                        json.dumps(
                            chunk.model_dump(),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    total_chunks += 1


            except RuntimeError as exc:

                failed_cases.append(
                    {
                        "case_id": case.case_id,
                        "error": str(exc),
                    }
                )


    print(
        "Processed cases:",
        len(cases)
    )

    print(
        "Total chunks:",
        total_chunks
    )


    print(
        "Failed cases:",
        len(failed_cases)
    )


    for item in failed_cases[:10]:

        print(
            item
        )


if __name__ == "__main__":
    main()