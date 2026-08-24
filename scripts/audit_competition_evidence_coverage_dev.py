from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from app.schemas.competition import (
    CompetitionSourceResolution,
)
from app.services.competition_corpus import (
    CompetitionCorpusBuildResult,
    load_competition_chunk_corpus,
)
from app.services.competition_dataset import (
    load_competition_qa_excel,
)
from app.services.competition_evidence_coverage import (
    CompetitionEvidenceCaseCoverage,
    CompetitionEvidenceCoverageSummary,
    audit_competition_evidence_coverage_case,
    summarize_competition_evidence_coverage,
)
from app.services.competition_source_resolver import (
    CompetitionSourceResolver,
    build_competition_source_manifest,
)
from scripts.evaluate_competition_bm25_dev import (
    load_dev_case_ids,
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
    "competition_evidence_coverage_dev_v1.json"
)

DEFAULT_EXPECTED_TEXT_DEV_CASES = 69


@dataclass(frozen=True, slots=True)
class CompetitionEvidenceCoverageDevResult:
    dev_case_count: int
    text_dev_case_count: int
    qa_sha256: str
    split_sha256: str
    corpus: CompetitionCorpusBuildResult
    resolutions: tuple[
        CompetitionSourceResolution,
        ...,
    ]
    case_results: tuple[
        CompetitionEvidenceCaseCoverage,
        ...,
    ]
    summary: CompetitionEvidenceCoverageSummary
    output_path: Path


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _summary_payload(
    summary: CompetitionEvidenceCoverageSummary,
) -> dict[str, object]:
    payload = asdict(summary)

    payload.update(
        {
            "single_chunk_fact_coverage": (
                summary.single_chunk_fact_coverage
            ),
            "two_chunk_fact_coverage": (
                summary.two_chunk_fact_coverage
            ),
            "document_fact_coverage": (
                summary.document_fact_coverage
            ),
            "all_facts_single_chunk_case_rate": (
                summary
                .all_facts_single_chunk_case_rate
            ),
            "all_facts_two_chunk_case_rate": (
                summary
                .all_facts_two_chunk_case_rate
            ),
            "all_facts_document_case_rate": (
                summary
                .all_facts_document_case_rate
            ),
        }
    )

    return payload


def _build_breakdowns(
    results: tuple[
        CompetitionEvidenceCaseCoverage,
        ...,
    ],
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
                CompetitionEvidenceCaseCoverage
            ],
        ] = {}

        for result in results:
            value = str(
                getattr(result, field_name)
            )

            groups.setdefault(
                value,
                [],
            ).append(result)

        breakdowns[field_name] = {
            value: _summary_payload(
                summarize_competition_evidence_coverage(
                    tuple(group_results)
                )
            )
            for value, group_results
            in sorted(groups.items())
        }

    return breakdowns


def _case_payload(
    *,
    result: CompetitionEvidenceCaseCoverage,
    resolution: CompetitionSourceResolution,
) -> dict[str, object]:
    facts = []

    for fact in result.fact_coverages:
        payload = asdict(fact)

        payload.update(
            {
                "single_chunk_covered": (
                    fact.single_chunk_covered
                ),
                "two_chunk_window_covered": (
                    fact.two_chunk_window_covered
                ),
                "boundary_only": (
                    fact.boundary_only
                ),
                "document_only": (
                    fact.document_only
                ),
                "missing": fact.missing,
            }
        )

        facts.append(payload)

    return {
        "case_id": result.case_id,
        "source_type": result.source_type,
        "qa_type": result.qa_type,
        "difficulty": result.difficulty,
        "expected_source_id": (
            result.expected_source_id
        ),
        "resolution_strategy": (
            resolution.strategy
        ),
        "source_chunk_count": (
            result.source_chunk_count
        ),
        "fact_count": result.fact_count,
        "single_chunk_covered_fact_count": (
            result
            .single_chunk_covered_fact_count
        ),
        "two_chunk_covered_fact_count": (
            result
            .two_chunk_covered_fact_count
        ),
        "document_covered_fact_count": (
            result
            .document_covered_fact_count
        ),
        "boundary_only_fact_count": (
            result.boundary_only_fact_count
        ),
        "document_only_fact_count": (
            result.document_only_fact_count
        ),
        "missing_fact_count": (
            result.missing_fact_count
        ),
        "fact_coverages": facts,
    }


def _build_output_payload(
    result: CompetitionEvidenceCoverageDevResult,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "audit_version": (
            "competition_evidence_coverage_dev_v1"
        ),
        "qa_sha256": result.qa_sha256,
        "split_sha256": result.split_sha256,
        "corpus_id": (
            result.corpus.manifest.corpus_id
        ),
        "dev_case_count": (
            result.dev_case_count
        ),
        "text_dev_case_count": (
            result.text_dev_case_count
        ),
        "summary": _summary_payload(
            result.summary
        ),
        "breakdowns": _build_breakdowns(
            result.case_results
        ),
        "cases": [
            _case_payload(
                result=case_result,
                resolution=resolution,
            )
            for case_result, resolution in zip(
                result.case_results,
                result.resolutions,
                strict=True,
            )
        ],
    }


def run_competition_evidence_coverage_dev(
    *,
    qa_file: Path,
    attachments_root: Path,
    split_file: Path,
    corpus_directory: Path,
    output_path: Path,
    expected_text_case_count: int | None = (
        DEFAULT_EXPECTED_TEXT_DEV_CASES
    ),
) -> CompetitionEvidenceCoverageDevResult:
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
        expected_text_case_count is not None
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

    corpus = load_competition_chunk_corpus(
        corpus_directory
    )

    chunks_by_source: dict[
        str,
        list,
    ] = {}

    for chunk in corpus.chunks:
        chunks_by_source.setdefault(
            chunk.source_id,
            [],
        ).append(chunk)

    missing_source_ids = tuple(
        sorted(
            {
                resolution.source_id
                for resolution in resolutions
                if (
                    resolution.source_id
                    not in chunks_by_source
                )
            }
        )
    )

    if missing_source_ids:
        raise RuntimeError(
            "Frozen Dev QA 来源没有进入 Corpus："
            f"{missing_source_ids}"
        )

    case_results = tuple(
        audit_competition_evidence_coverage_case(
            case=case,
            expected_source_id=(
                resolution.source_id
            ),
            chunks=tuple(
                chunks_by_source[
                    resolution.source_id
                ]
            ),
        )
        for case, resolution in zip(
            text_cases,
            resolutions,
            strict=True,
        )
    )

    result = CompetitionEvidenceCoverageDevResult(
        dev_case_count=len(dev_cases),
        text_dev_case_count=len(
            text_cases
        ),
        qa_sha256=_sha256_file(
            qa_file
        ),
        split_sha256=_sha256_file(
            split_file
        ),
        corpus=corpus,
        resolutions=resolutions,
        case_results=case_results,
        summary=(
            summarize_competition_evidence_coverage(
                case_results
            )
        ),
        output_path=output_path,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            _build_output_payload(result),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return result


def _format_rate(value: float) -> str:
    return f"{value:.4f}"


def _print_summary(
    summary: CompetitionEvidenceCoverageSummary,
) -> None:
    print(
        "Cases:",
        summary.case_count,
        "Facts:",
        summary.fact_count,
    )

    print(
        "Fact coverage:",
        "SingleChunk=",
        _format_rate(
            summary.single_chunk_fact_coverage
        ),
        "TwoChunk=",
        _format_rate(
            summary.two_chunk_fact_coverage
        ),
        "Document=",
        _format_rate(
            summary.document_fact_coverage
        ),
    )

    print(
        "Special facts:",
        "BoundaryOnly=",
        summary.boundary_only_fact_count,
        "DocumentOnly=",
        summary.document_only_fact_count,
        "Missing=",
        summary.missing_fact_count,
    )

    print(
        "All-facts case rate:",
        "SingleChunk=",
        _format_rate(
            summary
            .all_facts_single_chunk_case_rate
        ),
        "TwoChunk=",
        _format_rate(
            summary
            .all_facts_two_chunk_case_rate
        ),
        "Document=",
        _format_rate(
            summary
            .all_facts_document_case_rate
        ),
    )


def print_audit_result(
    result: CompetitionEvidenceCoverageDevResult,
) -> None:
    print(
        "=== Competition Frozen Dev "
        "Evidence Coverage Audit ==="
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
        result.corpus.manifest.corpus_id,
    )

    print()
    print("===== Overall =====")
    _print_summary(result.summary)

    breakdowns = _build_breakdowns(
        result.case_results
    )

    print()
    print("===== Breakdowns =====")

    for field_name in (
        "source_type",
        "qa_type",
        "difficulty",
    ):
        field_groups = breakdowns[
            field_name
        ]

        for value, payload in (
            field_groups.items()
        ):
            print(
                f"{field_name}={value}",
                f"cases={payload['case_count']}",
                "SingleChunk=",
                _format_rate(
                    payload[
                        "single_chunk_fact_coverage"
                    ]
                ),
                "TwoChunk=",
                _format_rate(
                    payload[
                        "two_chunk_fact_coverage"
                    ]
                ),
                "Document=",
                _format_rate(
                    payload[
                        "document_fact_coverage"
                    ]
                ),
                f"Missing={payload['missing_fact_count']}",
            )

    missing_facts = tuple(
        (
            case_result,
            fact,
        )
        for case_result in result.case_results
        for fact in case_result.fact_coverages
        if fact.missing
    )

    print()
    print(
        "===== Missing Fact Examples ====="
    )

    if not missing_facts:
        print("None")
    else:
        for case_result, fact in (
            missing_facts[:10]
        ):
            preview = (
                fact.fact
                .replace("\r", " ")
                .replace("\n", " ")
            )

            print(
                f"[{case_result.case_id}]",
                case_result.source_type,
                preview[:160],
            )

    print()
    print("Saved:", result.output_path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "审计 Competition Frozen Dev "
            "Gold Evidence 在 Corpus 中的覆盖情况"
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
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    arguments = (
        build_argument_parser()
        .parse_args(argv)
    )

    result = (
        run_competition_evidence_coverage_dev(
            qa_file=arguments.qa,
            attachments_root=(
                arguments.attachments
            ),
            split_file=arguments.split,
            corpus_directory=(
                arguments.corpus
            ),
            output_path=arguments.output,
        )
    )

    print_audit_result(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())