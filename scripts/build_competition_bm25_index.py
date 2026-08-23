from __future__ import annotations

import argparse
from collections.abc import (
    Sequence,
)
from dataclasses import dataclass
from pathlib import Path

from app.rag.tokenization import (
    DeterministicChineseBigramTokenizer,
)
from app.schemas.bm25 import (
    BM25Config,
)
from app.schemas.competition_retrieval import (
    CompetitionBM25Hit,
)
from app.services.competition_bm25_index import (
    CompetitionBM25IndexResult,
    build_competition_bm25_index,
    load_competition_bm25_index,
)
from app.services.competition_corpus import (
    CompetitionCorpusBuildResult,
    load_competition_chunk_corpus,
)


DEFAULT_CORPUS_ROOT = Path(
    "data/competition/processed/corpora"
)

DEFAULT_OUTPUT_ROOT = Path(
    "data/competition/processed/indexes/bm25"
)

DEFAULT_SMOKE_QUERY = "资本充足率"
DEFAULT_TOP_K = 5


@dataclass(frozen=True, slots=True)
class CompetitionBM25PipelineResult:
    corpus: CompetitionCorpusBuildResult
    index_result: CompetitionBM25IndexResult
    smoke_query: str
    smoke_hits: tuple[
        CompetitionBM25Hit,
        ...,
    ]


def resolve_corpus_directory(
    *,
    corpus_directory: Path | None,
    corpus_root: Path,
) -> Path:
    if corpus_directory is not None:
        if not corpus_directory.is_dir():
            raise FileNotFoundError(
                "指定的 Competition Corpus "
                "目录不存在："
                f"{corpus_directory}"
            )

        return corpus_directory

    if not corpus_root.is_dir():
        raise FileNotFoundError(
            "Competition Corpus 根目录不存在："
            f"{corpus_root}"
        )

    candidates = tuple(
        sorted(
            path
            for path in corpus_root.iterdir()
            if (
                path.is_dir()
                and path.name.startswith(
                    "competition_corpus_"
                )
                and (
                    path
                    / "corpus_manifest.json"
                ).is_file()
                and (
                    path
                    / "chunks.jsonl"
                ).is_file()
            )
        )
    )

    if not candidates:
        raise RuntimeError(
            "没有找到可用的 Competition Corpus"
        )

    if len(candidates) > 1:
        candidate_names = ", ".join(
            path.name
            for path in candidates
        )

        raise RuntimeError(
            "发现多个 Competition Corpus，"
            "请通过 --corpus 明确指定："
            f"{candidate_names}"
        )

    return candidates[0]


def build_competition_bm25_pipeline(
    *,
    corpus_directory: Path,
    output_root: Path,
    smoke_query: str = (
        DEFAULT_SMOKE_QUERY
    ),
    top_k: int = DEFAULT_TOP_K,
) -> CompetitionBM25PipelineResult:
    if not smoke_query.strip():
        raise ValueError(
            "smoke_query 不能为空"
        )

    if top_k < 1:
        raise ValueError(
            "top_k 必须大于等于 1"
        )

    corpus = (
        load_competition_chunk_corpus(
            corpus_directory
        )
    )

    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    index_result = (
        build_competition_bm25_index(
            corpus_directory=(
                corpus_directory
            ),
            output_root=output_root,
            tokenizer=tokenizer,
            config=BM25Config(),
        )
    )

    # 再从磁盘完整加载一次，保证当前返回的
    # 不是仅存在于内存中的临时索引。
    loaded = (
        load_competition_bm25_index(
            index_result.index_directory,
            corpus_directory=(
                corpus_directory
            ),
        )
    )

    if (
        loaded.manifest.document_count
        != corpus.manifest.chunk_count
    ):
        raise RuntimeError(
            "BM25 文档数量与 Corpus "
            "Chunk 数量不一致"
        )

    if (
        loaded.manifest
        .metadata_record_count
        != corpus.manifest.chunk_count
    ):
        raise RuntimeError(
            "BM25 元数据数量与 Corpus "
            "Chunk 数量不一致"
        )

    if (
        loaded.index.chunks
        != corpus.chunks
    ):
        raise RuntimeError(
            "BM25 Chunk 与 Corpus 不一致"
        )

    if (
        loaded.manifest.vocabulary_size
        < 1
    ):
        raise RuntimeError(
            "BM25 词表不能为空"
        )

    smoke_hits = loaded.index.search(
        query=smoke_query,
        tokenizer=tokenizer,
        top_k=top_k,
    )

    if not smoke_hits:
        raise RuntimeError(
            "BM25 Smoke Query 没有命中："
            f"{smoke_query}"
        )

    return CompetitionBM25PipelineResult(
        corpus=corpus,
        index_result=loaded,
        smoke_query=smoke_query,
        smoke_hits=smoke_hits,
    )


def print_pipeline_result(
    result: CompetitionBM25PipelineResult,
) -> None:
    corpus_manifest = (
        result.corpus.manifest
    )

    index_manifest = (
        result.index_result.manifest
    )

    top_hit = result.smoke_hits[0]

    print(
        "=== Competition QA-used "
        "BM25 Index ==="
    )

    print(
        "Corpus ID:",
        corpus_manifest.corpus_id,
    )

    print(
        "Corpus sources:",
        corpus_manifest.source_count,
    )

    print(
        "Corpus documents:",
        corpus_manifest.doc_count,
    )

    print(
        "Corpus chunks:",
        corpus_manifest.chunk_count,
    )

    print(
        "Index ID:",
        index_manifest.index_id,
    )

    print(
        "Indexed documents:",
        index_manifest.document_count,
    )

    print(
        "Metadata records:",
        index_manifest
        .metadata_record_count,
    )

    print(
        "Vocabulary size:",
        index_manifest.vocabulary_size,
    )

    print(
        "Total tokens:",
        index_manifest.total_token_count,
    )

    print(
        "Average document length:",
        index_manifest
        .average_document_length,
    )

    print(
        "Tokenizer:",
        index_manifest
        .tokenizer_spec
        .tokenizer_name,
    )

    print(
        "BM25 config:",
        index_manifest
        .bm25_config
        .model_dump(
            mode="json"
        ),
    )

    print(
        "Tokenized corpus SHA256:",
        index_manifest
        .tokenized_corpus_sha256,
    )

    print(
        "Smoke query:",
        result.smoke_query,
    )

    print(
        "Smoke hits:",
        len(result.smoke_hits),
    )

    print(
        "Top hit:",
        {
            "chunk_id": top_hit.chunk_id,
            "source_id": top_hit.source_id,
            "chunk_type": (
                top_hit.chunk_type
            ),
            "score": round(
                top_hit.score,
                6,
            ),
        },
    )

    print(
        "Saved:",
        result.index_result
        .index_directory,
    )


def build_argument_parser() -> (
    argparse.ArgumentParser
):
    parser = argparse.ArgumentParser(
        description=(
            "为 Competition QA-used "
            "Text Chunk Corpus 构建 BM25 索引"
        )
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help=(
            "指定 competition_corpus_* "
            "目录；不指定时自动发现唯一 Corpus"
        ),
    )

    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=DEFAULT_CORPUS_ROOT,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--smoke-query",
        default=DEFAULT_SMOKE_QUERY,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    arguments = (
        build_argument_parser()
        .parse_args(argv)
    )

    corpus_directory = (
        resolve_corpus_directory(
            corpus_directory=(
                arguments.corpus
            ),
            corpus_root=(
                arguments.corpus_root
            ),
        )
    )

    result = (
        build_competition_bm25_pipeline(
            corpus_directory=(
                corpus_directory
            ),
            output_root=(
                arguments.output_root
            ),
            smoke_query=(
                arguments.smoke_query
            ),
            top_k=arguments.top_k,
        )
    )

    print_pipeline_result(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())