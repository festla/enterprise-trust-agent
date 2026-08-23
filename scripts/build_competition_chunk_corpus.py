from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

from app.schemas.competition_chunk import (
    CompetitionChunkDocument,
)
from app.services.competition_corpus import (
    CompetitionCorpusBuildResult,
    build_competition_chunk_corpus,
)
from app.services.competition_dataset import (
    build_competition_question,
    load_competition_qa_excel,
)
from app.services.competition_source_resolver import (
    CompetitionSourceResolver,
    build_competition_source_manifest,
)
from app.services.competition_text_chunker import (
    build_competition_text_chunks,
)
from app.services.competition_text_parser import (
    parse_competition_text_document,
)


DEFAULT_QA_FILE = Path(
    "data/competition/private/qa/QA数据.xlsx"
)

DEFAULT_ATTACHMENTS_ROOT = Path(
    "data/competition/private/attachments"
)

DEFAULT_OUTPUT_ROOT = Path(
    "data/competition/processed/corpora"
)

DEFAULT_MAX_CHARS = 900

TEXT_SOURCE_TYPES = frozenset(
    {
        "pdf",
        "word",
    }
)


@dataclass(frozen=True)
class CompetitionCorpusPipelineResult:
    qa_case_count: int
    text_qa_case_count: int
    resolved_source_count: int
    corpus: CompetitionCorpusBuildResult


def resolve_qa_used_text_sources(
    *,
    cases: Sequence[object],
    manifest: Sequence[object],
) -> tuple[
    int,
    tuple[
        tuple[object, object],
        ...,
    ],
]:
    manifest_by_source_id = {
        source.source_id: source
        for source in manifest
    }

    resolver = CompetitionSourceResolver(
        manifest
    )

    unique_sources: dict[
        str,
        tuple[object, object],
    ] = {}

    text_qa_case_count = 0

    for case in sorted(
        cases,
        key=lambda item: item.case_id,
    ):
        if (
            case.source_type
            not in TEXT_SOURCE_TYPES
        ):
            continue

        text_qa_case_count += 1

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

        # 同一文档可能服务多道 QA。
        # 选择 case_id 最小的一道作为解析代表。
        unique_sources.setdefault(
            source.source_id,
            (
                case,
                source,
            ),
        )

    if not unique_sources:
        raise RuntimeError(
            "没有解析到任何 QA-used 文本来源"
        )

    resolved_sources = tuple(
        unique_sources[source_id]
        for source_id in sorted(
            unique_sources
        )
    )

    return (
        text_qa_case_count,
        resolved_sources,
    )


def build_chunk_documents(
    *,
    resolved_sources: Sequence[
        tuple[object, object]
    ],
    attachments_root: Path,
    max_chars: int,
) -> tuple[
    CompetitionChunkDocument,
    ...,
]:
    if max_chars <= 0:
        raise ValueError(
            "max_chars 必须大于 0"
        )

    documents: list[
        CompetitionChunkDocument
    ] = []

    seen_doc_ids: set[str] = set()
    seen_chunk_ids: set[str] = set()

    for case, source in resolved_sources:
        question = build_competition_question(
            case
        )

        parsed_document = (
            parse_competition_text_document(
                question=question,
                source=source,
                attachments_root=(
                    attachments_root
                ),
            )
        )

        if (
            parsed_document.source.source_id
            != source.source_id
        ):
            raise RuntimeError(
                "解析结果 source_id 与 "
                "Manifest 不一致: "
                f"expected={source.source_id}; "
                "actual="
                f"{parsed_document.source.source_id}"
            )

        doc_id = (
            parsed_document.source.doc_id
        )

        if doc_id in seen_doc_ids:
            raise RuntimeError(
                "检测到重复 doc_id: "
                f"{doc_id}"
            )

        chunks = (
            build_competition_text_chunks(
                parsed_document,
                max_chars=max_chars,
            )
        )

        if not chunks:
            raise RuntimeError(
                "文档没有生成任何 Chunk: "
                f"{doc_id}"
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
                f"chunk_id: {doc_id}"
            )

        duplicated_chunk_ids = (
            seen_chunk_ids.intersection(
                current_chunk_ids
            )
        )

        if duplicated_chunk_ids:
            raise RuntimeError(
                "不同文档之间存在重复 "
                "chunk_id: "
                f"{min(duplicated_chunk_ids)}"
            )

        documents.append(
            CompetitionChunkDocument(
                source=(
                    parsed_document.source
                ),
                chunks=chunks,
            )
        )

        seen_doc_ids.add(
            doc_id
        )

        seen_chunk_ids.update(
            current_chunk_ids
        )

    return tuple(
        sorted(
            documents,
            key=lambda document: (
                document.source.source_id
            ),
        )
    )


def build_qa_used_text_corpus(
    *,
    qa_file: Path,
    attachments_root: Path,
    output_root: Path,
    max_chars: int = (
        DEFAULT_MAX_CHARS
    ),
) -> CompetitionCorpusPipelineResult:
    cases = tuple(
        load_competition_qa_excel(
            qa_file
        )
    )

    manifest = tuple(
        build_competition_source_manifest(
            attachments_root
        )
    )

    (
        text_qa_case_count,
        resolved_sources,
    ) = resolve_qa_used_text_sources(
        cases=cases,
        manifest=manifest,
    )

    documents = build_chunk_documents(
        resolved_sources=resolved_sources,
        attachments_root=attachments_root,
        max_chars=max_chars,
    )

    if (
        len(documents)
        != len(resolved_sources)
    ):
        raise RuntimeError(
            "处理文档数量与唯一来源数量不一致"
        )

    corpus = build_competition_chunk_corpus(
        documents=documents,
        output_root=output_root,
        max_chars=max_chars,
    )

    if (
        corpus.manifest.source_count
        != len(resolved_sources)
    ):
        raise RuntimeError(
            "Corpus Manifest 来源数量不完整"
        )

    return CompetitionCorpusPipelineResult(
        qa_case_count=len(cases),
        text_qa_case_count=(
            text_qa_case_count
        ),
        resolved_source_count=len(
            resolved_sources
        ),
        corpus=corpus,
    )


def print_pipeline_result(
    result: CompetitionCorpusPipelineResult,
) -> None:
    manifest = result.corpus.manifest

    print(
        "=== Competition QA-used "
        "Text Chunk Corpus ==="
    )
    print(
        "QA cases:",
        result.qa_case_count,
    )
    print(
        "Text QA cases:",
        result.text_qa_case_count,
    )
    print(
        "Unique text sources:",
        result.resolved_source_count,
    )
    print(
        "Processed documents:",
        manifest.doc_count,
    )
    print(
        "Total chunks:",
        manifest.chunk_count,
    )
    print(
        "Text chunks:",
        manifest.text_chunk_count,
    )
    print(
        "Table chunks:",
        manifest.table_chunk_count,
    )
    print(
        "Source types:",
        dict(
            sorted(
                manifest
                .source_type_counts
                .items()
            )
        ),
    )
    print(
        "Chunk types:",
        dict(
            sorted(
                manifest
                .chunk_type_counts
                .items()
            )
        ),
    )
    print(
        "Corpus ID:",
        manifest.corpus_id,
    )
    print(
        "Chunks SHA256:",
        manifest.chunks_jsonl_sha256,
    )
    print(
        "Saved:",
        result.corpus.corpus_directory,
    )


def build_argument_parser() -> (
    argparse.ArgumentParser
):
    parser = argparse.ArgumentParser(
        description=(
            "构建全部 QA-used Word/PDF "
            "Competition Chunk Corpus"
        )
    )

    parser.add_argument(
        "--qa",
        type=Path,
        default=DEFAULT_QA_FILE,
    )

    parser.add_argument(
        "--attachments",
        type=Path,
        default=DEFAULT_ATTACHMENTS_ROOT,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    arguments = (
        build_argument_parser()
        .parse_args(argv)
    )

    result = build_qa_used_text_corpus(
        qa_file=arguments.qa,
        attachments_root=(
            arguments.attachments
        ),
        output_root=(
            arguments.output_root
        ),
        max_chars=arguments.max_chars,
    )

    print_pipeline_result(
        result
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )