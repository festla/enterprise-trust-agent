from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import (
    asdict,
    dataclass,
)
import hashlib
import json
from pathlib import Path
from typing import Literal

from app.rag.tokenization import (
    DeterministicChineseBigramTokenizer,
)
from app.schemas.competition import (
    CompetitionQaCase,
    CompetitionSourceResolution,
)
from app.services.competition_bm25_index import (
    CompetitionBM25IndexResult,
    load_competition_bm25_index,
)
from app.services.competition_dataset import (
    load_competition_qa_excel,
)
from app.services.competition_retrieval_eval import (
    CompetitionRetrievalCaseResult,
    CompetitionRetrievalEvalSummary,
    summarize_competition_retrieval,
    evaluate_competition_retrieval_case,
)
from app.services.competition_source_resolver import (
    CompetitionSourceResolver,
    build_competition_source_manifest,
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

DEFAULT_OUTPUT_FILE = Path(
    "data/competition/processed/eval/"
    "competition_bm25_dev_baseline_v1.json"
)

DEFAULT_CUTOFFS = (
    1,
    3,
    5,
    10,
)

DEFAULT_TOP_K = 10
DEFAULT_EXPECTED_TEXT_DEV_CASES = 69

CompetitionBM25QueryMode = Literal[
    "question_only",
    "question_with_options",
]


@dataclass(frozen=True, slots=True)
class CompetitionBM25ModeResult:
    query_mode: CompetitionBM25QueryMode
    queries: tuple[str, ...]
    resolutions: tuple[
        CompetitionSourceResolution,
        ...,
    ]
    case_results: tuple[
        CompetitionRetrievalCaseResult,
        ...,
    ]
    summary: CompetitionRetrievalEvalSummary


@dataclass(frozen=True, slots=True)
class CompetitionBM25DevResult:
    dev_case_count: int
    text_dev_case_count: int
    qa_sha256: str
    split_sha256: str
    index_result: CompetitionBM25IndexResult
    mode_results: tuple[
        CompetitionBM25ModeResult,
        ...,
    ]
    cutoffs: tuple[int, ...]
    top_k: int
    output_path: Path


def _sha256_file(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_dev_case_ids(
    split_path: Path,
) -> tuple[str, ...]:
    try:
        payload = json.loads(
            split_path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError(
            "无法读取 Frozen Dev Split："
            f"{split_path}"
        ) from exc

    case_ids = payload.get(
        "dev_case_ids"
    )

    if (
        not isinstance(case_ids, list)
        or not case_ids
        or any(
            not isinstance(case_id, str)
            or not case_id
            for case_id in case_ids
        )
    ):
        raise RuntimeError(
            "Frozen Dev Split 中的 "
            "dev_case_ids 无效"
        )

    if (
        len(case_ids)
        != len(set(case_ids))
    ):
        raise RuntimeError(
            "Frozen Dev Split 包含重复 case_id"
        )

    return tuple(sorted(case_ids))


def build_bm25_query(
    *,
    case: CompetitionQaCase,
    mode: CompetitionBM25QueryMode,
) -> str:
    if mode == "question_only":
        return case.question

    if mode == "question_with_options":
        option_lines = tuple(
            f"{option}. {text}"
            for option, text
            in case.options.items()
        )

        return "\n".join(
            (
                case.question,
                *option_lines,
            )
        )

    raise ValueError(
        f"不支持的 BM25 Query Mode: {mode}"
    )


def _evaluate_mode(
    *,
    query_mode: CompetitionBM25QueryMode,
    cases: tuple[
        CompetitionQaCase,
        ...,
    ],
    resolutions: tuple[
        CompetitionSourceResolution,
        ...,
    ],
    index_result: CompetitionBM25IndexResult,
    top_k: int,
    cutoffs: tuple[int, ...],
) -> CompetitionBM25ModeResult:
    if len(cases) != len(resolutions):
        raise RuntimeError(
            "Cases 与 Resolutions 数量不一致"
        )

    tokenizer = (
        DeterministicChineseBigramTokenizer(
            spec=(
                index_result
                .manifest
                .tokenizer_spec
            )
        )
    )

    queries = []
    case_results = []

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

        query = build_bm25_query(
            case=case,
            mode=query_mode,
        )

        hits = index_result.index.search(
            query=query,
            tokenizer=tokenizer,
            top_k=top_k,
        )

        evaluation = (
            evaluate_competition_retrieval_case(
                case=case,
                expected_source_id=(
                    resolution.source_id
                ),
                hits=hits,
            )
        )

        queries.append(query)
        case_results.append(evaluation)

    result_tuple = tuple(case_results)

    return CompetitionBM25ModeResult(
        query_mode=query_mode,
        queries=tuple(queries),
        resolutions=resolutions,
        case_results=result_tuple,
        summary=(
            summarize_competition_retrieval(
                result_tuple,
                cutoffs=cutoffs,
            )
        ),
    )


def _summary_payload(
    summary: CompetitionRetrievalEvalSummary,
) -> dict[str, object]:
    return asdict(summary)


def _build_breakdowns(
    *,
    results: tuple[
        CompetitionRetrievalCaseResult,
        ...,
    ],
    cutoffs: tuple[int, ...],
) -> dict[str, object]:
    breakdowns: dict[str, object] = {}

    for field_name in (
        "source_type",
        "qa_type",
        "difficulty",
    ):
        groups: dict[
            str,
            list[
                CompetitionRetrievalCaseResult
            ],
        ] = {}

        for result in results:
            value = str(
                getattr(
                    result,
                    field_name,
                )
            )

            groups.setdefault(
                value,
                [],
            ).append(result)

        breakdowns[field_name] = {
            value: _summary_payload(
                summarize_competition_retrieval(
                    tuple(group_results),
                    cutoffs=cutoffs,
                )
            )
            for value, group_results
            in sorted(groups.items())
        }

    return breakdowns


def _build_output_payload(
    result: CompetitionBM25DevResult,
) -> dict[str, object]:
    manifest = (
        result.index_result.manifest
    )

    modes: dict[str, object] = {}

    for mode_result in (
        result.mode_results
    ):
        case_records = []

        for (
            query,
            resolution,
            case_result,
        ) in zip(
            mode_result.queries,
            mode_result.resolutions,
            mode_result.case_results,
            strict=True,
        ):
            case_records.append(
                {
                    "case_id": (
                        case_result.case_id
                    ),
                    "query": query,
                    "expected_source_id": (
                        resolution.source_id
                    ),
                    "resolution_strategy": (
                        resolution.strategy
                    ),
                    "evaluation": (
                        asdict(case_result)
                    ),
                }
            )

        modes[
            mode_result.query_mode
        ] = {
            "summary": (
                _summary_payload(
                    mode_result.summary
                )
            ),
            "breakdowns": (
                _build_breakdowns(
                    results=(
                        mode_result
                        .case_results
                    ),
                    cutoffs=(
                        result.cutoffs
                    ),
                )
            ),
            "cases": case_records,
        }

    return {
        "schema_version": 1,
        "baseline_version": (
            "competition_bm25_dev_v1"
        ),
        "qa_sha256": result.qa_sha256,
        "split_sha256": (
            result.split_sha256
        ),
        "corpus_id": manifest.corpus_id,
        "index_id": manifest.index_id,
        "tokenizer_spec": (
            manifest.tokenizer_spec
            .model_dump(mode="json")
        ),
        "bm25_config": (
            manifest.bm25_config
            .model_dump(mode="json")
        ),
        "dev_case_count": (
            result.dev_case_count
        ),
        "text_dev_case_count": (
            result.text_dev_case_count
        ),
        "cutoffs": list(
            result.cutoffs
        ),
        "top_k": result.top_k,
        "modes": modes,
    }


def run_competition_bm25_dev(
    *,
    qa_file: Path,
    attachments_root: Path,
    split_file: Path,
    corpus_directory: Path,
    index_directory: Path,
    output_path: Path,
    top_k: int = DEFAULT_TOP_K,
    cutoffs: tuple[int, ...] = (
        DEFAULT_CUTOFFS
    ),
    expected_text_case_count: (
        int | None
    ) = (
        DEFAULT_EXPECTED_TEXT_DEV_CASES
    ),
) -> CompetitionBM25DevResult:
    if not cutoffs:
        raise ValueError(
            "cutoffs 不能为空"
        )

    if top_k < max(cutoffs):
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

    dev_case_ids = load_dev_case_ids(
        split_file
    )

    unknown_case_ids = tuple(
        case_id
        for case_id in dev_case_ids
        if case_id not in case_by_id
    )

    if unknown_case_ids:
        raise RuntimeError(
            "Frozen Dev Split 包含未知 QA："
            f"{unknown_case_ids}"
        )

    dev_cases = tuple(
        sorted(
            (
                case_by_id[case_id]
                for case_id in dev_case_ids
            ),
            key=lambda case: case.case_id,
        )
    )

    text_cases = tuple(
        case
        for case in dev_cases
        if case.source_type in {
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
            "Frozen Dev 文本 QA 数量改变："
            f"expected="
            f"{expected_text_case_count}; "
            f"actual={len(text_cases)}"
        )

    source_manifest = (
        build_competition_source_manifest(
            attachments_root
        )
    )

    resolver = CompetitionSourceResolver(
        source_manifest
    )

    resolutions = tuple(
        resolver.resolve(case)
        for case in text_cases
    )

    index_result = (
        load_competition_bm25_index(
            index_directory,
            corpus_directory=(
                corpus_directory
            ),
        )
    )

    indexed_source_ids = {
        chunk.source_id
        for chunk in (
            index_result.index.chunks
        )
    }

    missing_source_ids = tuple(
        sorted(
            {
                resolution.source_id
                for resolution
                in resolutions
                if (
                    resolution.source_id
                    not in indexed_source_ids
                )
            }
        )
    )

    if missing_source_ids:
        raise RuntimeError(
            "Frozen Dev QA 来源没有进入 "
            "BM25 Index："
            f"{missing_source_ids}"
        )

    mode_results = tuple(
        _evaluate_mode(
            query_mode=mode,
            cases=text_cases,
            resolutions=resolutions,
            index_result=index_result,
            top_k=top_k,
            cutoffs=cutoffs,
        )
        for mode in (
            "question_only",
            "question_with_options",
        )
    )

    result = CompetitionBM25DevResult(
        dev_case_count=len(
            dev_cases
        ),
        text_dev_case_count=len(
            text_cases
        ),
        qa_sha256=_sha256_file(
            qa_file
        ),
        split_sha256=_sha256_file(
            split_file
        ),
        index_result=index_result,
        mode_results=mode_results,
        cutoffs=tuple(
            sorted(set(cutoffs))
        ),
        top_k=top_k,
        output_path=output_path,
    )

    payload = _build_output_payload(
        result
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            payload,
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


def print_dev_result(
    result: CompetitionBM25DevResult,
) -> None:
    print(
        "=== Competition Frozen Dev "
        "BM25 Baseline ==="
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
        "Corpus ID:",
        result.index_result
        .manifest.corpus_id,
    )
    print(
        "Index ID:",
        result.index_result
        .manifest.index_id,
    )

    for mode_result in (
        result.mode_results
    ):
        print()
        print(
            f"===== {mode_result.query_mode} ====="
        )

        for metric in (
            mode_result.summary.metrics
        ):
            print(
                f"@{metric.cutoff}",
                "SourceHit=",
                _format_rate(
                    metric.source_hit_rate
                ),
                "AnyFact=",
                _format_rate(
                    metric.any_fact_hit_rate
                ),
                "AllFacts=",
                _format_rate(
                    metric.all_facts_hit_rate
                ),
                "FactRecall=",
                _format_rate(
                    metric.mean_fact_recall
                ),
                "MRR=",
                _format_rate(
                    metric.mean_reciprocal_rank
                ),
            )

        print(
            f"Breakdown @{result.top_k}:"
        )

        for field_name in (
            "source_type",
            "qa_type",
            "difficulty",
        ):
            values = sorted(
                {
                    str(
                        getattr(
                            case_result,
                            field_name,
                        )
                    )
                    for case_result
                    in mode_result
                    .case_results
                }
            )

            for value in values:
                group = tuple(
                    case_result
                    for case_result
                    in mode_result
                    .case_results
                    if (
                        str(
                            getattr(
                                case_result,
                                field_name,
                            )
                        )
                        == value
                    )
                )

                summary = (
                    summarize_competition_retrieval(
                        group,
                        cutoffs=(
                            result.top_k,
                        ),
                    )
                ).at(
                    result.top_k
                )

                print(
                    f"  {field_name}={value}",
                    f"cases={len(group)}",
                    "SourceHit=",
                    _format_rate(
                        summary.source_hit_rate
                    ),
                    "AllFacts=",
                    _format_rate(
                        summary.all_facts_hit_rate
                    ),
                    "FactRecall=",
                    _format_rate(
                        summary.mean_fact_recall
                    ),
                )

    print()
    print(
        "Saved:",
        result.output_path,
    )


def build_argument_parser() -> (
    argparse.ArgumentParser
):
    parser = argparse.ArgumentParser(
        description=(
            "运行 Competition Frozen Dev "
            "BM25 检索基线"
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
        "--split",
        type=Path,
        default=DEFAULT_SPLIT_FILE,
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--index",
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

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    arguments = (
        build_argument_parser()
        .parse_args(argv)
    )

    result = run_competition_bm25_dev(
        qa_file=arguments.qa,
        attachments_root=(
            arguments.attachments
        ),
        split_file=arguments.split,
        corpus_directory=(
            arguments.corpus
        ),
        index_directory=(
            arguments.index
        ),
        output_path=arguments.output,
        top_k=arguments.top_k,
    )

    print_dev_result(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())