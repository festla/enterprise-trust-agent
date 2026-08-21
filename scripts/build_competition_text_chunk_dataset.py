from __future__ import annotations

import json
from pathlib import Path


from app.services.competition_dataset import (
    build_competition_question,
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

def main() -> None:
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ======================================
    # 1. Load Dev QA cases
    # ======================================

    all_cases = load_competition_qa_excel(
        QA_FILE
    )

    dev_case_ids = load_dev_case_ids()

    dev_cases = [
        case
        for case in all_cases
        if case.case_id in dev_case_ids
    ]

    text_cases = [
        case
        for case in dev_cases
        if case.source_type in {
            "pdf",
            "word",
        }
    ]

    # ======================================
    # 2. Build attachment manifest
    # ======================================

    manifest = (
        build_competition_source_manifest(
            ATTACHMENTS_ROOT
        )
    )

    manifest_by_source_id = {
        source.source_id: source
        for source in manifest
    }

    resolver = CompetitionSourceResolver(
        manifest
    )

    # ======================================
    # 3. Resolve and deduplicate sources
    #
    # 同一份文档可能对应多个 QA，
    # 但这里只保留一个代表 case。
    # ======================================

    unique_sources = {}

    resolution_failures = []

    for case in sorted(
        text_cases,
        key=lambda item: item.case_id,
    ):
        try:
            resolution = resolver.resolve(
                case
            )

            source = manifest_by_source_id.get(
                resolution.source_id
            )

            if source is None:
                raise RuntimeError(
                    "Manifest 中找不到已解析的数据源: "
                    f"{resolution.source_id}"
                )

            unique_sources.setdefault(
                source.source_id,
                (case, source),
            )

        except RuntimeError as exc:
            resolution_failures.append(
                {
                    "case_id": case.case_id,
                    "error": str(exc),
                }
            )

    # ======================================
    # 4. Parse and chunk unique sources
    # ======================================

    total_chunks = 0
    processed_sources = 0

    failed_sources = []

    seen_doc_ids = set()
    seen_chunk_ids = set()

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as writer:
        for (
            source_id,
            representative,
        ) in sorted(
            unique_sources.items()
        ):
            case, source = representative

            try:
                question = (
                    build_competition_question(
                        case
                    )
                )

                document = (
                    parse_competition_text_document(
                        question=question,
                        source=source,
                        attachments_root=(
                            ATTACHMENTS_ROOT
                        ),
                    )
                )

                if (
                    document.source.source_id
                    != source_id
                ):
                    raise RuntimeError(
                        "Parsed document source_id "
                        "与 Manifest 不一致: "
                        f"expected={source_id}; "
                        "actual="
                        f"{document.source.source_id}"
                    )

                if (
                    document.source.doc_id
                    in seen_doc_ids
                ):
                    raise RuntimeError(
                        "检测到重复 doc_id: "
                        f"{document.source.doc_id}"
                    )

                chunks = (
                    build_competition_text_chunks(
                        document
                    )
                )

                if not chunks:
                    raise RuntimeError(
                        "文档没有生成任何 Chunk: "
                        f"{document.source.doc_id}"
                    )

                current_chunk_ids = {
                    chunk.chunk_id
                    for chunk in chunks
                }

                if (
                    len(current_chunk_ids)
                    != len(chunks)
                ):
                    raise RuntimeError(
                        "同一文档内部存在重复 "
                        "chunk_id: "
                        f"{document.source.doc_id}"
                    )

                duplicated_chunk_ids = (
                    seen_chunk_ids.intersection(
                        current_chunk_ids
                    )
                )

                if duplicated_chunk_ids:
                    duplicate_example = min(
                        duplicated_chunk_ids
                    )

                    raise RuntimeError(
                        "不同文档之间存在重复 "
                        "chunk_id: "
                        f"{duplicate_example}"
                    )

                # 全部检查通过后再写入，
                # 避免失败文档只写入一部分 Chunk。
                for chunk in chunks:
                    writer.write(
                        json.dumps(
                            chunk.model_dump(),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                seen_doc_ids.add(
                    document.source.doc_id
                )

                seen_chunk_ids.update(
                    current_chunk_ids
                )

                total_chunks += len(
                    chunks
                )

                processed_sources += 1

            except (
                RuntimeError,
                ValueError,
            ) as exc:
                failed_sources.append(
                    {
                        "source_id": source_id,
                        "case_id": case.case_id,
                        "filename": (
                            source.actual_filename
                        ),
                        "error": str(exc),
                    }
                )

    # ======================================
    # 5. Report
    # ======================================

    print(
        "Dev QA cases:",
        len(dev_cases),
    )

    print(
        "Dev text QA cases:",
        len(text_cases),
    )

    print(
        "Unique text sources:",
        len(unique_sources),
    )

    print(
        "Processed sources:",
        processed_sources,
    )

    print(
        "Unique doc IDs:",
        len(seen_doc_ids),
    )

    print(
        "Total chunks:",
        total_chunks,
    )

    print(
        "Resolution failures:",
        len(resolution_failures),
    )

    print(
        "Failed sources:",
        len(failed_sources),
    )

    for item in resolution_failures[:10]:
        print(item)

    for item in failed_sources[:10]:
        print(item)

    # Dev 语料不应该静默生成不完整结果。
    if (
        resolution_failures
        or failed_sources
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
