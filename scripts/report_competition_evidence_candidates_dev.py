from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from statistics import mean, median

from app.schemas.competition_chunk import (
    CompetitionTextChunk,
)
from app.services.competition_corpus import (
    CompetitionCorpusBuildResult,
    load_competition_chunk_corpus,
)
from app.services.competition_evidence_candidates import (
    CompetitionEvidenceCandidate,
    rank_competition_evidence_candidates,
)


DEFAULT_AUDIT_FILE = Path(
    "data/competition/processed/eval/"
    "competition_evidence_coverage_dev_v1.json"
)

DEFAULT_OUTPUT_FILE = Path(
    "data/competition/processed/eval/"
    "competition_evidence_candidates_dev_v1.json"
)

DEFAULT_TOP_K = 5

EXPECTED_AUDIT_VERSION = (
    "competition_evidence_coverage_dev_v1"
)


@dataclass(frozen=True, slots=True)
class CompetitionMissingEvidenceFact:
    case_id: str
    source_type: str
    qa_type: str
    difficulty: str
    expected_source_id: str
    fact_index: int
    fact: str


@dataclass(frozen=True, slots=True)
class CompetitionEvidenceCandidateRecord:
    missing_fact: CompetitionMissingEvidenceFact
    candidates: tuple[
        CompetitionEvidenceCandidate,
        ...,
    ]


@dataclass(frozen=True, slots=True)
class CompetitionEvidenceCandidateReport:
    audit_sha256: str
    corpus: CompetitionCorpusBuildResult
    top_k: int
    records: tuple[
        CompetitionEvidenceCandidateRecord,
        ...,
    ]
    output_path: Path


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _require_string(
    payload: dict[str, object],
    field_name: str,
    *,
    context: str,
) -> str:
    value = payload.get(field_name)

    if (
        not isinstance(value, str)
        or not value
    ):
        raise RuntimeError(
            f"{context} 缺少有效的 "
            f"{field_name}"
        )

    return value


def load_missing_evidence_facts(
    audit_path: Path,
) -> tuple[
    str,
    tuple[
        CompetitionMissingEvidenceFact,
        ...,
    ],
]:
    try:
        payload = json.loads(
            audit_path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError(
            "无法读取 Evidence Coverage Audit："
            f"{audit_path}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            "Evidence Coverage Audit 根节点无效"
        )

    if (
        payload.get("audit_version")
        != EXPECTED_AUDIT_VERSION
    ):
        raise RuntimeError(
            "Evidence Coverage Audit "
            "版本不受支持"
        )

    corpus_id = _require_string(
        payload,
        "corpus_id",
        context="Evidence Coverage Audit",
    )

    cases = payload.get("cases")

    if not isinstance(cases, list):
        raise RuntimeError(
            "Evidence Coverage Audit "
            "缺少 cases"
        )

    records = []

    for case_payload in cases:
        if not isinstance(
            case_payload,
            dict,
        ):
            raise RuntimeError(
                "Evidence Coverage Audit "
                "包含无效 Case"
            )

        case_id = _require_string(
            case_payload,
            "case_id",
            context="Coverage Case",
        )

        source_type = _require_string(
            case_payload,
            "source_type",
            context=f"Coverage Case {case_id}",
        )

        qa_type = _require_string(
            case_payload,
            "qa_type",
            context=f"Coverage Case {case_id}",
        )

        difficulty = _require_string(
            case_payload,
            "difficulty",
            context=f"Coverage Case {case_id}",
        )

        expected_source_id = _require_string(
            case_payload,
            "expected_source_id",
            context=f"Coverage Case {case_id}",
        )

        fact_coverages = case_payload.get(
            "fact_coverages"
        )

        if not isinstance(
            fact_coverages,
            list,
        ):
            raise RuntimeError(
                f"Coverage Case {case_id} "
                "缺少 fact_coverages"
            )

        for fact_index, fact_payload in enumerate(
            fact_coverages
        ):
            if not isinstance(
                fact_payload,
                dict,
            ):
                raise RuntimeError(
                    f"Coverage Case {case_id} "
                    "包含无效 Fact"
                )

            if (
                fact_payload.get("missing")
                is not True
            ):
                continue

            fact = _require_string(
                fact_payload,
                "fact",
                context=(
                    f"Coverage Case {case_id} "
                    f"Fact {fact_index}"
                ),
            )

            records.append(
                CompetitionMissingEvidenceFact(
                    case_id=case_id,
                    source_type=source_type,
                    qa_type=qa_type,
                    difficulty=difficulty,
                    expected_source_id=(
                        expected_source_id
                    ),
                    fact_index=fact_index,
                    fact=fact,
                )
            )

    return (
        corpus_id,
        tuple(records),
    )


def _candidate_payload(
    candidate: CompetitionEvidenceCandidate,
) -> dict[str, object]:
    chunk = candidate.chunk

    return {
        "rank": candidate.rank,
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "doc_id": chunk.doc_id,
        "chunk_index": chunk.chunk_index,
        "chunk_type": chunk.chunk_type,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "paragraph_start_index": (
            chunk.paragraph_start_index
        ),
        "paragraph_end_index": (
            chunk.paragraph_end_index
        ),
        "table_index": chunk.table_index,
        "table_row_start": (
            chunk.table_row_start
        ),
        "table_row_end": (
            chunk.table_row_end
        ),
        "section_path": list(
            chunk.section_path
        ),
        "article": chunk.article,
        "table_title": chunk.table_title,
        "text": chunk.text,
        "normalized_fact_char_count": (
            candidate
            .normalized_fact_char_count
        ),
        "normalized_chunk_char_count": (
            candidate
            .normalized_chunk_char_count
        ),
        "unigram_overlap_count": (
            candidate.unigram_overlap_count
        ),
        "unigram_recall": (
            candidate.unigram_recall
        ),
        "bigram_overlap_count": (
            candidate.bigram_overlap_count
        ),
        "bigram_precision": (
            candidate.bigram_precision
        ),
        "bigram_recall": (
            candidate.bigram_recall
        ),
        "bigram_f1": candidate.bigram_f1,
        "sequence_ratio": (
            candidate.sequence_ratio
        ),
    }


def _record_payload(
    record: CompetitionEvidenceCandidateRecord,
) -> dict[str, object]:
    missing = record.missing_fact

    return {
        "case_id": missing.case_id,
        "source_type": missing.source_type,
        "qa_type": missing.qa_type,
        "difficulty": missing.difficulty,
        "expected_source_id": (
            missing.expected_source_id
        ),
        "fact_index": missing.fact_index,
        "fact": missing.fact,
        "candidates": [
            _candidate_payload(candidate)
            for candidate in record.candidates
        ],
    }


def _score_summary(
    records: tuple[
        CompetitionEvidenceCandidateRecord,
        ...,
    ],
) -> dict[str, object]:
    best_scores = [
        record.candidates[0].bigram_recall
        for record in records
        if record.candidates
    ]

    if not best_scores:
        return {
            "fact_count": len(records),
            "mean_best_bigram_recall": None,
            "median_best_bigram_recall": None,
            "min_best_bigram_recall": None,
            "max_best_bigram_recall": None,
            "best_bigram_recall_ge_0_8": 0,
            "best_bigram_recall_ge_0_5": 0,
            "best_bigram_recall_lt_0_3": 0,
        }

    return {
        "fact_count": len(records),
        "mean_best_bigram_recall": (
            mean(best_scores)
        ),
        "median_best_bigram_recall": (
            median(best_scores)
        ),
        "min_best_bigram_recall": (
            min(best_scores)
        ),
        "max_best_bigram_recall": (
            max(best_scores)
        ),
        "best_bigram_recall_ge_0_8": sum(
            score >= 0.8
            for score in best_scores
        ),
        "best_bigram_recall_ge_0_5": sum(
            score >= 0.5
            for score in best_scores
        ),
        "best_bigram_recall_lt_0_3": sum(
            score < 0.3
            for score in best_scores
        ),
    }


def _build_breakdowns(
    records: tuple[
        CompetitionEvidenceCandidateRecord,
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
                CompetitionEvidenceCandidateRecord
            ],
        ] = {}

        for record in records:
            value = str(
                getattr(
                    record.missing_fact,
                    field_name,
                )
            )

            groups.setdefault(
                value,
                [],
            ).append(record)

        breakdowns[field_name] = {
            value: _score_summary(
                tuple(group_records)
            )
            for value, group_records
            in sorted(groups.items())
        }

    return breakdowns


def _build_output_payload(
    result: CompetitionEvidenceCandidateReport,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "report_version": (
            "competition_evidence_candidates_dev_v1"
        ),
        "audit_sha256": result.audit_sha256,
        "corpus_id": (
            result.corpus.manifest.corpus_id
        ),
        "top_k": result.top_k,
        "missing_fact_count": (
            len(result.records)
        ),
        "summary": _score_summary(
            result.records
        ),
        "breakdowns": _build_breakdowns(
            result.records
        ),
        "records": [
            _record_payload(record)
            for record in result.records
        ],
    }


def run_competition_evidence_candidate_report(
    *,
    audit_path: Path,
    corpus_directory: Path,
    output_path: Path,
    top_k: int = DEFAULT_TOP_K,
) -> CompetitionEvidenceCandidateReport:
    if top_k < 1:
        raise ValueError(
            "top_k 必须大于等于 1"
        )

    (
        audit_corpus_id,
        missing_facts,
    ) = load_missing_evidence_facts(
        audit_path
    )

    corpus = load_competition_chunk_corpus(
        corpus_directory
    )

    if (
        audit_corpus_id
        != corpus.manifest.corpus_id
    ):
        raise RuntimeError(
            "Coverage Audit 与 Corpus "
            "corpus_id 不一致"
        )

    chunks_by_source: dict[
        str,
        list[
            CompetitionTextChunk
        ],
    ] = {}

    for chunk in corpus.chunks:
        chunks_by_source.setdefault(
            chunk.source_id,
            [],
        ).append(chunk)

    missing_source_ids = tuple(
        sorted(
            {
                record.expected_source_id
                for record in missing_facts
                if (
                    record.expected_source_id
                    not in chunks_by_source
                )
            }
        )
    )

    if missing_source_ids:
        raise RuntimeError(
            "Missing Fact 来源没有进入 Corpus："
            f"{missing_source_ids}"
        )

    records = tuple(
        CompetitionEvidenceCandidateRecord(
            missing_fact=missing_fact,
            candidates=(
                rank_competition_evidence_candidates(
                    fact=missing_fact.fact,
                    expected_source_id=(
                        missing_fact
                        .expected_source_id
                    ),
                    chunks=tuple(
                        chunks_by_source[
                            missing_fact
                            .expected_source_id
                        ]
                    ),
                    top_k=top_k,
                )
            ),
        )
        for missing_fact in missing_facts
    )

    result = CompetitionEvidenceCandidateReport(
        audit_sha256=_sha256_file(
            audit_path
        ),
        corpus=corpus,
        top_k=top_k,
        records=records,
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


def _format_score(
    value: object,
) -> str:
    if not isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        return "n/a"

    return f"{value:.4f}"


def _location_text(
    chunk: CompetitionTextChunk,
) -> str:
    if chunk.page_start is not None:
        return (
            f"page={chunk.page_start}"
            f"-{chunk.page_end}"
        )

    if chunk.table_index is not None:
        return (
            f"table={chunk.table_index} "
            f"rows={chunk.table_row_start}"
            f"-{chunk.table_row_end}"
        )

    return (
        "paragraph="
        f"{chunk.paragraph_start_index}"
        f"-{chunk.paragraph_end_index}"
    )


def print_candidate_report(
    result: CompetitionEvidenceCandidateReport,
) -> None:
    print(
        "=== Competition Frozen Dev "
        "Missing Evidence Candidate Report ==="
    )
    print(
        "Corpus ID:",
        result.corpus.manifest.corpus_id,
    )
    print(
        "Missing facts:",
        len(result.records),
    )
    print("Top K:", result.top_k)

    summary = _score_summary(
        result.records
    )

    print()
    print("===== Overall Best Candidate =====")
    print(
        "Mean Bigram Recall:",
        _format_score(
            summary[
                "mean_best_bigram_recall"
            ]
        ),
    )
    print(
        "Median Bigram Recall:",
        _format_score(
            summary[
                "median_best_bigram_recall"
            ]
        ),
    )
    print(
        "Range:",
        _format_score(
            summary[
                "min_best_bigram_recall"
            ]
        ),
        "-",
        _format_score(
            summary[
                "max_best_bigram_recall"
            ]
        ),
    )
    print(
        "Recall >= 0.8:",
        summary[
            "best_bigram_recall_ge_0_8"
        ],
    )
    print(
        "Recall >= 0.5:",
        summary[
            "best_bigram_recall_ge_0_5"
        ],
    )
    print(
        "Recall < 0.3:",
        summary[
            "best_bigram_recall_lt_0_3"
        ],
    )

    print()
    print("===== Breakdowns =====")

    breakdowns = _build_breakdowns(
        result.records
    )

    for field_name in (
        "source_type",
        "qa_type",
        "difficulty",
    ):
        for value, payload in (
            breakdowns[field_name].items()
        ):
            print(
                f"{field_name}={value}",
                f"facts={payload['fact_count']}",
                "Mean=",
                _format_score(
                    payload[
                        "mean_best_bigram_recall"
                    ]
                ),
                "Median=",
                _format_score(
                    payload[
                        "median_best_bigram_recall"
                    ]
                ),
                "<0.3=",
                payload[
                    "best_bigram_recall_lt_0_3"
                ],
            )

    print()
    print("===== First 10 Missing Facts =====")

    for record in result.records[:10]:
        missing = record.missing_fact
        best = record.candidates[0]

        print()
        print(
            f"[{missing.case_id}"
            f"#{missing.fact_index}]",
            missing.source_type,
            missing.fact,
        )
        print(
            "Best:",
            best.chunk.chunk_id,
            best.chunk.chunk_type,
            _location_text(best.chunk),
        )
        print(
            "Scores:",
            "BigramRecall=",
            _format_score(
                best.bigram_recall
            ),
            "BigramF1=",
            _format_score(
                best.bigram_f1
            ),
            "UnigramRecall=",
            _format_score(
                best.unigram_recall
            ),
            "Sequence=",
            _format_score(
                best.sequence_ratio
            ),
        )

        preview = (
            best.chunk.text
            .replace("\r", " ")
            .replace("\n", " ")
        )

        print(
            "Preview:",
            preview[:240],
        )

    print()
    print("Saved:", result.output_path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "为 Frozen Dev Missing Evidence "
            "生成正确来源内的词面候选报告"
        )
    )

    parser.add_argument(
        "--audit",
        type=Path,
        default=DEFAULT_AUDIT_FILE,
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

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    arguments = (
        build_argument_parser()
        .parse_args(argv)
    )

    result = (
        run_competition_evidence_candidate_report(
            audit_path=arguments.audit,
            corpus_directory=(
                arguments.corpus
            ),
            output_path=arguments.output,
            top_k=arguments.top_k,
        )
    )

    print_candidate_report(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())