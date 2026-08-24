from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import (
    asdict,
    dataclass,
)
import hashlib
import json
from pathlib import Path

from app.rag.competition_dense import (
    CompetitionExactDenseIndex,
)
from app.rag.embedders.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
    build_bge_small_zh_v15_spec,
)
from app.schemas.competition import (
    CompetitionQaCase,
    CompetitionSourceResolution,
)
from app.schemas.competition_gold import (
    CompetitionGoldChunkDataset,
    CompetitionGoldFactRecord,
)
from app.schemas.competition_retrieval import (
    CompetitionDenseHit,
    CompetitionRetrievalFilter,
)
from app.services.competition_corpus import (
    CompetitionCorpusBuildResult,
    load_competition_chunk_corpus,
)
from app.services.competition_dataset import (
    load_competition_qa_excel,
)
from app.services.competition_gold_retrieval_eval import (
    CompetitionGoldFactRetrievalResult,
    CompetitionGoldRetrievalEvalSummary,
    evaluate_competition_gold_fact,
    summarize_competition_gold_retrieval,
)
from app.services.competition_source_resolver import (
    CompetitionSourceResolver,
    build_competition_source_manifest,
)
from scripts.evaluate_competition_bm25_dev import (
    CompetitionBM25QueryMode,
    build_bm25_query,
    load_dev_case_ids,
)
from scripts.evaluate_competition_bm25_gold_dev import (
    load_competition_gold_dataset,
)


DEFAULT_QA_FILE = Path(
    "data/competition/private/qa/QA数据.xlsx"
)

DEFAULT_ATTACHMENTS_ROOT = Path(
    "data/competition/private/attachments"
)

DEFAULT_SPLIT_FILE = Path(
    "data/competition/processed/"
    "competition_eval_split_v1.json"
)

DEFAULT_GOLD_FILE = Path(
    "data/competition/processed/eval/"
    "competition_dev_gold_chunks_v1.json"
)

DEFAULT_OUTPUT_FILE = Path(
    "data/competition/processed/eval/"
    "competition_dense_gold_dev_v1.json"
)

DEFAULT_CUTOFFS = (
    1,
    3,
    5,
    10,
)

DEFAULT_TOP_K = 10

DEFAULT_EXPECTED_TEXT_DEV_CASES = 69
DEFAULT_EXPECTED_FACTS = 97


@dataclass(
    frozen=True,
    slots=True,
)
class CompetitionDenseGoldCaseResult:
    case_id: str
    query: str
    expected_source_id: str
    resolution_strategy: str

    hits: tuple[
        CompetitionDenseHit,
        ...,
    ]

    fact_results: tuple[
        CompetitionGoldFactRetrievalResult,
        ...,
    ]


@dataclass(
    frozen=True,
    slots=True,
)
class CompetitionDenseGoldModeResult:
    query_mode: CompetitionBM25QueryMode

    case_results: tuple[
        CompetitionDenseGoldCaseResult,
        ...,
    ]

    fact_results: tuple[
        CompetitionGoldFactRetrievalResult,
        ...,
    ]

    summary: (
        CompetitionGoldRetrievalEvalSummary
    )


@dataclass(
    frozen=True,
    slots=True,
)
class CompetitionDenseGoldDevResult:
    dev_case_count: int
    text_dev_case_count: int
    fact_count: int

    qa_sha256: str
    split_sha256: str
    gold_sha256: str

    gold_dataset: (
        CompetitionGoldChunkDataset
    )

    corpus: (
        CompetitionCorpusBuildResult
    )

    mode_results: tuple[
        CompetitionDenseGoldModeResult,
        ...,
    ]

    cutoffs: tuple[int, ...]
    top_k: int

    embedding_spec: object

    output_path: Path


def _sha256_file(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _group_gold_records(
    records: tuple[
        CompetitionGoldFactRecord,
        ...,
    ],
) -> dict[
    str,
    tuple[
        CompetitionGoldFactRecord,
        ...,
    ],
]:
    groups: dict[
        str,
        list[
            CompetitionGoldFactRecord
        ],
    ] = {}

    for record in records:
        groups.setdefault(
            record.case_id,
            [],
        ).append(record)

    return {
        case_id: tuple(
            sorted(
                case_records,
                key=lambda record: (
                    record.fact_index
                ),
            )
        )
        for case_id, case_records
        in sorted(groups.items())
    }


def _validate_binding(
    *,
    gold_dataset: (
        CompetitionGoldChunkDataset
    ),
    cases: tuple[
        CompetitionQaCase,
        ...,
    ],
    resolutions: tuple[
        CompetitionSourceResolution,
        ...,
    ],
    corpus: CompetitionCorpusBuildResult,
) -> dict[
    str,
    tuple[
        CompetitionGoldFactRecord,
        ...,
    ],
]:
    if (
        gold_dataset.corpus_id
        != corpus.manifest.corpus_id
    ):
        raise RuntimeError(
            "Gold 与 Dense Corpus 的 "
            "corpus_id 不一致"
        )

    if len(cases) != len(resolutions):
        raise RuntimeError(
            "Cases 与 Resolutions 数量不一致"
        )

    records_by_case = (
        _group_gold_records(
            gold_dataset.records
        )
    )

    case_ids = {
        case.case_id
        for case in cases
    }

    gold_case_ids = set(
        records_by_case
    )

    if case_ids != gold_case_ids:
        raise RuntimeError(
            "Dense Dev Case 与 Gold Case "
            "集合不一致"
        )

    resolution_by_case = {
        resolution.case_id: resolution
        for resolution in resolutions
    }

    for case in cases:
        resolution = (
            resolution_by_case.get(
                case.case_id
            )
        )

        if resolution is None:
            raise RuntimeError(
                "缺少 Source Resolution："
                f"{case.case_id}"
            )

        for record in (
            records_by_case[
                case.case_id
            ]
        ):
            if (
                record.expected_source_id
                != resolution.source_id
            ):
                raise RuntimeError(
                    "Gold expected_source_id "
                    "与 Source Resolution "
                    "不一致："
                    f"{record.case_id}#"
                    f"{record.fact_index}"
                )

    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunk in corpus.chunks
    }

    for record in (
        gold_dataset.records
    ):
        for reference in (
            record.gold_chunks
        ):
            chunk = chunks_by_id.get(
                reference.chunk_id
            )

            if chunk is None:
                raise RuntimeError(
                    "Gold Chunk 不存在于 "
                    "Frozen Corpus："
                    f"{record.case_id}#"
                    f"{record.fact_index} "
                    f"{reference.chunk_id}"
                )

            if (
                chunk.source_id
                != record.expected_source_id
            ):
                raise RuntimeError(
                    "Gold Chunk source_id 与 "
                    "expected_source_id "
                    "不一致："
                    f"{record.case_id}#"
                    f"{record.fact_index} "
                    f"{reference.chunk_id}"
                )

    return records_by_case


def _evaluate_mode(
    *,
    query_mode: (
        CompetitionBM25QueryMode
    ),
    cases: tuple[
        CompetitionQaCase,
        ...,
    ],
    resolutions: tuple[
        CompetitionSourceResolution,
        ...,
    ],
    records_by_case: dict[
        str,
        tuple[
            CompetitionGoldFactRecord,
            ...,
        ],
    ],
    index: CompetitionExactDenseIndex,
    provider: (
        SentenceTransformerEmbeddingProvider
    ),
    top_k: int,
    cutoffs: tuple[int, ...],
    resolved_source_only: bool,
) -> CompetitionDenseGoldModeResult:
    case_results = []
    all_fact_results = []

    for case, resolution in zip(
        cases,
        resolutions,
        strict=True,
    ):
        if (
            case.case_id
            != resolution.case_id
        ):
            raise RuntimeError(
                "Case 与 Resolution "
                "case_id 不一致"
            )

        # 故意复用 BM25 Query Builder，
        # 保证 Dense / Sparse 使用完全相同
        # 的 question_only / options 输入。
        query = build_bm25_query(
            case=case,
            mode=query_mode,
        )

        filters = (
            CompetitionRetrievalFilter(
                source_ids=(
                    resolution.source_id,
                ),
            )
            if resolved_source_only
            else None
        )

        hits = index.search(
            query=query,
            provider=provider,
            top_k=top_k,
            filters=filters,
        )

        fact_results = tuple(
            evaluate_competition_gold_fact(
                record=record,
                hits=hits,
            )
            for record
            in records_by_case[
                case.case_id
            ]
        )

        case_results.append(
            CompetitionDenseGoldCaseResult(
                case_id=case.case_id,
                query=query,
                expected_source_id=(
                    resolution.source_id
                ),
                resolution_strategy=(
                    resolution.strategy
                ),
                hits=hits,
                fact_results=(
                    fact_results
                ),
            )
        )

        all_fact_results.extend(
            fact_results
        )

    fact_result_tuple = tuple(
        all_fact_results
    )

    return CompetitionDenseGoldModeResult(
        query_mode=query_mode,
        case_results=tuple(
            case_results
        ),
        fact_results=(
            fact_result_tuple
        ),
        summary=(
            summarize_competition_gold_retrieval(
                fact_result_tuple,
                cutoffs=cutoffs,
            )
        ),
    )


def _summary_payload(
    summary: (
        CompetitionGoldRetrievalEvalSummary
    ),
) -> dict[str, object]:
    return asdict(summary)


def _build_evidence_mode_breakdowns(
    *,
    results: tuple[
        CompetitionGoldFactRetrievalResult,
        ...,
    ],
    cutoffs: tuple[int, ...],
) -> dict[str, object]:
    groups: dict[
        str,
        list[
            CompetitionGoldFactRetrievalResult
        ],
    ] = {}

    for result in results:
        groups.setdefault(
            result.evidence_mode,
            [],
        ).append(result)

    return {
        evidence_mode: (
            _summary_payload(
                summarize_competition_gold_retrieval(
                    tuple(group_results),
                    cutoffs=cutoffs,
                )
            )
        )
        for (
            evidence_mode,
            group_results,
        )
        in sorted(groups.items())
    }


def _hard_cases_payload(
    *,
    results: tuple[
        CompetitionGoldFactRetrievalResult,
        ...,
    ],
    cutoff: int,
) -> list[
    dict[str, object]
]:
    hard_cases = []

    for result in results:
        if (
            result.complete_gold_hit_at(
                cutoff
            )
        ):
            continue

        missing_chunks = [
            {
                "chunk_id": match.chunk_id,
                "role": match.role,
            }
            for match
            in result.chunk_matches
            if not match.hit_at(cutoff)
        ]

        hard_cases.append(
            {
                "case_id": (
                    result.case_id
                ),
                "fact_index": (
                    result.fact_index
                ),
                "fact": result.fact,
                "evidence_mode": (
                    result.evidence_mode
                ),
                "expected_source_id": (
                    result.expected_source_id
                ),
                "direct_hit": (
                    result.direct_hit_at(
                        cutoff
                    )
                ),
                "complete_gold": False,
                "gold_recall": (
                    result.gold_chunk_recall_at(
                        cutoff
                    )
                ),
                "missing_gold_chunks": (
                    missing_chunks
                ),
            }
        )

    return hard_cases


def _case_payload(
    result: (
        CompetitionDenseGoldCaseResult
    ),
) -> dict[str, object]:
    return {
        "case_id": result.case_id,
        "query": result.query,
        "expected_source_id": (
            result.expected_source_id
        ),
        "resolution_strategy": (
            result.resolution_strategy
        ),
        "retrieved_chunks": [
            {
                "rank": hit.rank,
                "score": hit.score,
                "chunk_id": (
                    hit.chunk_id
                ),
                "source_id": (
                    hit.source_id
                ),
                "doc_id": hit.doc_id,
                "chunk_type": (
                    hit.chunk_type
                ),
            }
            for hit in result.hits
        ],
        "facts": [
            asdict(fact_result)
            for fact_result
            in result.fact_results
        ],
    }


def _build_output_payload(
    result: (
        CompetitionDenseGoldDevResult
    ),
) -> dict[str, object]:
    modes = {}

    for mode_result in (
        result.mode_results
    ):
        modes[
            mode_result.query_mode
        ] = {
            "summary": (
                _summary_payload(
                    mode_result.summary
                )
            ),
            "breakdowns": {
                "evidence_mode": (
                    _build_evidence_mode_breakdowns(
                        results=(
                            mode_result
                            .fact_results
                        ),
                        cutoffs=(
                            result.cutoffs
                        ),
                    )
                )
            },
            "hard_cases_at_top_k": (
                _hard_cases_payload(
                    results=(
                        mode_result
                        .fact_results
                    ),
                    cutoff=(
                        result.top_k
                    ),
                )
            ),
            "cases": [
                _case_payload(
                    case_result
                )
                for case_result
                in mode_result.case_results
            ],
        }

    return {
        "schema_version": 1,
        "evaluation_version": (
            "competition_dense_"
            "gold_dev_v1"
        ),
        "qa_sha256": (
            result.qa_sha256
        ),
        "split_sha256": (
            result.split_sha256
        ),
        "gold_sha256": (
            result.gold_sha256
        ),
        "gold_version": (
            result.gold_dataset
            .gold_version
        ),
        "corpus_id": (
            result.corpus
            .manifest
            .corpus_id
        ),
        "corpus_chunk_count": (
            len(result.corpus.chunks)
        ),
        "embedding_spec": (
            result.embedding_spec
            .model_dump(
                mode="json"
            )
        ),
        "dev_case_count": (
            result.dev_case_count
        ),
        "text_dev_case_count": (
            result.text_dev_case_count
        ),
        "fact_count": (
            result.fact_count
        ),
        "evidence_mode_counts": dict(
            sorted(
                Counter(
                    record.evidence_mode
                    for record
                    in result
                    .gold_dataset
                    .records
                ).items()
            )
        ),
        "cutoffs": list(
            result.cutoffs
        ),
        "top_k": result.top_k,
        "query_uses_gold_fact_text": False,
        "modes": modes,
    }


def run_competition_dense_gold_dev(
    *,
    qa_file: Path,
    attachments_root: Path,
    split_file: Path,
    gold_file: Path,
    corpus_directory: Path,
    output_path: Path,
    device: str = "cpu",
    batch_size: int = 32,
    cache_folder: Path | None = None,
    local_files_only: bool = False,
    show_progress_bar: bool = False,
    top_k: int = DEFAULT_TOP_K,
    cutoffs: tuple[
        int,
        ...,
    ] = DEFAULT_CUTOFFS,
    expected_text_case_count: (
        int | None
    ) = DEFAULT_EXPECTED_TEXT_DEV_CASES,
    expected_fact_count: (
        int | None
    ) = DEFAULT_EXPECTED_FACTS,
    resolved_source_only: bool = False,
) -> CompetitionDenseGoldDevResult:
    normalized_cutoffs = tuple(
        sorted(
            set(cutoffs)
        )
    )

    if not normalized_cutoffs:
        raise ValueError(
            "cutoffs 不能为空"
        )

    if (
        top_k
        < max(normalized_cutoffs)
    ):
        raise ValueError(
            "top_k 不能小于最大 cutoff"
        )

    cases = tuple(
        load_competition_qa_excel(
            qa_file
        )
    )

    case_by_id = {
        case.case_id: case
        for case in cases
    }

    dev_case_ids = (
        load_dev_case_ids(
            split_file
        )
    )

    unknown_case_ids = tuple(
        case_id
        for case_id
        in dev_case_ids
        if case_id not in case_by_id
    )

    if unknown_case_ids:
        raise RuntimeError(
            "Frozen Dev Split "
            "包含未知 QA："
            f"{unknown_case_ids}"
        )

    dev_cases = tuple(
        sorted(
            (
                case_by_id[
                    case_id
                ]
                for case_id
                in dev_case_ids
            ),
            key=lambda case: (
                case.case_id
            ),
        )
    )

    text_cases = tuple(
        case
        for case in dev_cases
        if case.source_type
        in {
            "word",
            "pdf",
        }
    )

    if (
        expected_text_case_count
        is not None
        and len(text_cases)
        != expected_text_case_count
    ):
        raise RuntimeError(
            "Frozen Dev 文本 QA "
            "数量改变："
            f"expected="
            f"{expected_text_case_count}; "
            f"actual={len(text_cases)}"
        )

    gold_dataset = (
        load_competition_gold_dataset(
            gold_file,
            expected_fact_count=(
                expected_fact_count
            ),
        )
    )

    corpus = (
        load_competition_chunk_corpus(
            corpus_directory
        )
    )

    source_manifest = (
        build_competition_source_manifest(
            attachments_root
        )
    )

    resolver = (
        CompetitionSourceResolver(
            source_manifest
        )
    )

    resolutions = tuple(
        resolver.resolve(case)
        for case in text_cases
    )

    records_by_case = (
        _validate_binding(
            gold_dataset=(
                gold_dataset
            ),
            cases=text_cases,
            resolutions=(
                resolutions
            ),
            corpus=corpus,
        )
    )

    spec = (
        build_bge_small_zh_v15_spec()
    )

    provider = (
        SentenceTransformerEmbeddingProvider(
            spec=spec,
            batch_size=batch_size,
            device=device,
            cache_folder=(
                cache_folder
            ),
            local_files_only=(
                local_files_only
            ),
            show_progress_bar=(
                show_progress_bar
            ),
        )
    )

    print(
        "Building Dense Index:",
        len(corpus.chunks),
        "chunks",
    )

    index = (
        CompetitionExactDenseIndex.build(
            chunks=corpus.chunks,
            provider=provider,
        )
    )

    print(
        "Dense Index ready:",
        index.vectors.shape,
    )

    mode_results = tuple(
        _evaluate_mode(
            query_mode=query_mode,
            cases=text_cases,
            resolutions=(
                resolutions
            ),
            records_by_case=(
                records_by_case
            ),
            index=index,
            provider=provider,
            top_k=top_k,
            cutoffs=(
                normalized_cutoffs
            ),
            resolved_source_only=(
                resolved_source_only
            ),
        )
        for query_mode in (
            "question_only",
            "question_with_options",
        )
    )

    result = (
        CompetitionDenseGoldDevResult(
            dev_case_count=(
                len(dev_cases)
            ),
            text_dev_case_count=(
                len(text_cases)
            ),
            fact_count=(
                len(
                    gold_dataset.records
                )
            ),
            qa_sha256=(
                _sha256_file(
                    qa_file
                )
            ),
            split_sha256=(
                _sha256_file(
                    split_file
                )
            ),
            gold_sha256=(
                _sha256_file(
                    gold_file
                )
            ),
            gold_dataset=(
                gold_dataset
            ),
            corpus=corpus,
            mode_results=(
                mode_results
            ),
            cutoffs=(
                normalized_cutoffs
            ),
            top_k=top_k,
            embedding_spec=spec,
            output_path=(
                output_path
            ),
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            _build_output_payload(
                result
            ),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return result


def _format_rate(
    value: float,
) -> str:
    return f"{value:.4f}"


def print_dense_gold_result(
    result: (
        CompetitionDenseGoldDevResult
    ),
) -> None:
    print()
    print(
        "=== Competition Frozen Dev "
        "Dense Gold Evaluation ==="
    )

    print(
        "Dev QA cases:",
        result.dev_case_count,
    )

    print(
        "Dev text QA cases:",
        result.text_dev_case_count,
    )

    print(
        "Gold facts:",
        result.fact_count,
    )

    print(
        "Corpus chunks:",
        len(result.corpus.chunks),
    )

    print(
        "Embedding:",
        (
            result.embedding_spec
            .model_name
        ),
    )

    print(
        "Embedding revision:",
        (
            result.embedding_spec
            .model_version
        ),
    )

    for mode_result in (
        result.mode_results
    ):
        print()
        print(
            "=====",
            mode_result.query_mode,
            "=====",
        )

        for metric in (
            mode_result
            .summary
            .metrics
        ):
            print(
                f"@{metric.cutoff}",
                "DirectHit=",
                _format_rate(
                    metric
                    .fact_direct_hit_rate
                ),
                "AnyGold=",
                _format_rate(
                    metric
                    .fact_any_gold_hit_rate
                ),
                "CompleteGold=",
                _format_rate(
                    metric
                    .fact_complete_gold_hit_rate
                ),
                "GoldRecall=",
                _format_rate(
                    metric
                    .mean_gold_chunk_recall
                ),
                "DirectMRR=",
                _format_rate(
                    metric
                    .mean_direct_reciprocal_rank
                ),
            )

        print(
            f"Breakdown @{result.top_k}:"
        )

        evidence_modes = sorted(
            {
                fact_result
                .evidence_mode
                for fact_result
                in mode_result
                .fact_results
            }
        )

        for evidence_mode in (
            evidence_modes
        ):
            group = tuple(
                fact_result
                for fact_result
                in mode_result
                .fact_results
                if (
                    fact_result
                    .evidence_mode
                    == evidence_mode
                )
            )

            metric = (
                summarize_competition_gold_retrieval(
                    group,
                    cutoffs=(
                        result.top_k,
                    ),
                )
                .at(result.top_k)
            )

            print(
                " ",
                f"{evidence_mode}",
                f"facts={len(group)}",
                "DirectHit=",
                _format_rate(
                    metric
                    .fact_direct_hit_rate
                ),
                "CompleteGold=",
                _format_rate(
                    metric
                    .fact_complete_gold_hit_rate
                ),
                "GoldRecall=",
                _format_rate(
                    metric
                    .mean_gold_chunk_recall
                ),
            )

        hard_cases = tuple(
            fact_result
            for fact_result
            in mode_result.fact_results
            if not (
                fact_result
                .complete_gold_hit_at(
                    result.top_k
                )
            )
        )

        direct_misses = sum(
            not fact_result.direct_hit_at(
                result.top_k
            )
            for fact_result
            in mode_result.fact_results
        )

        print(
            "Incomplete facts @"
            f"{result.top_k}:",
            len(hard_cases),
        )

        print(
            "Direct misses @"
            f"{result.top_k}:",
            direct_misses,
        )

    print()
    print(
        "Saved:",
        result.output_path,
    )


def build_argument_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Competition Frozen Dev "
            "Dense Gold Baseline"
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
        default=(
            DEFAULT_ATTACHMENTS_ROOT
        ),
    )

    parser.add_argument(
        "--split",
        type=Path,
        default=DEFAULT_SPLIT_FILE,
    )

    parser.add_argument(
        "--gold",
        type=Path,
        default=DEFAULT_GOLD_FILE,
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--cache-folder",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--local-files-only",
        action="store_true",
    )

    parser.add_argument(
        "--show-progress-bar",
        action="store_true",
    )

    parser.add_argument(
        "--resolved-source-only",
        action="store_true",
        help=(
            "仅在 Source Resolver "
            "定位的附件内执行 Dense 检索"
        ),
    )

    return parser


def main(
    argv: (
        Sequence[str]
        | None
    ) = None,
) -> int:
    arguments = (
        build_argument_parser()
        .parse_args(argv)
    )

    result = (
        run_competition_dense_gold_dev(
            qa_file=arguments.qa,
            attachments_root=(
                arguments.attachments
            ),
            split_file=(
                arguments.split
            ),
            gold_file=(
                arguments.gold
            ),
            corpus_directory=(
                arguments.corpus
            ),
            output_path=(
                arguments.output
            ),
            device=(
                arguments.device
            ),
            batch_size=(
                arguments.batch_size
            ),
            cache_folder=(
                arguments.cache_folder
            ),
            local_files_only=(
                arguments
                .local_files_only
            ),
            show_progress_bar=(
                arguments
                .show_progress_bar
            ),
            top_k=(
                arguments.top_k
            ),
            resolved_source_only=(
                arguments.resolved_source_only
            ),
        )
    )

    print_dense_gold_result(
        result
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )