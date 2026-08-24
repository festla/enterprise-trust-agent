from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from app.schemas.competition_gold import (
    CompetitionGoldChunkDataset,
    CompetitionGoldFactRecord,
)
from scripts.evaluate_competition_bm25_dev import (
    load_dev_case_ids,
)


DEFAULT_SPLIT_FILE = Path(
    "data/competition/processed/"
    "competition_eval_split_v1.json"
)

DEFAULT_COVERAGE_FILE = Path(
    "data/competition/processed/eval/"
    "competition_evidence_coverage_dev_v1.json"
)

DEFAULT_CANDIDATE_FILE = Path(
    "data/competition/processed/eval/"
    "competition_evidence_candidates_dev_v1.json"
)

DEFAULT_OUTPUT_FILE = Path(
    "data/competition/processed/eval/"
    "competition_dev_gold_chunks_v1.json"
)

DEFAULT_EXPECTED_TEXT_DEV_CASES = 69
DEFAULT_EXPECTED_FACTS = 97
DEFAULT_EXPECTED_MISSING_FACTS = 89

EXPECTED_COVERAGE_VERSION = (
    "competition_evidence_coverage_dev_v1"
)

EXPECTED_CANDIDATE_VERSION = (
    "competition_evidence_candidates_dev_v1"
)


@dataclass(frozen=True, slots=True)
class CompetitionGoldTemplateResult:
    dataset: CompetitionGoldChunkDataset
    dev_case_count: int
    fact_count: int
    missing_fact_count: int
    output_path: Path


def _sha256_file(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _load_json_object(
    path: Path,
    *,
    label: str,
) -> dict[str, object]:
    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError(
            f"无法读取 {label}：{path}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"{label} 根节点无效"
        )

    return payload


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


def _load_coverage_records(
    coverage_path: Path,
) -> tuple[
    str,
    tuple[str, ...],
    tuple[
        CompetitionGoldFactRecord,
        ...,
    ],
    dict[
        tuple[str, int],
        tuple[str, str],
    ],
]:
    payload = _load_json_object(
        coverage_path,
        label="Evidence Coverage Audit",
    )

    if (
        payload.get("audit_version")
        != EXPECTED_COVERAGE_VERSION
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

    if (
        not isinstance(cases, list)
        or not cases
    ):
        raise RuntimeError(
            "Evidence Coverage Audit "
            "缺少 cases"
        )

    case_ids: list[str] = []
    records: list[
        CompetitionGoldFactRecord
    ] = []

    missing_facts: dict[
        tuple[str, int],
        tuple[str, str],
    ] = {}

    seen_fact_keys: set[
        tuple[str, int]
    ] = set()

    for case_payload in cases:
        if not isinstance(
            case_payload,
            dict,
        ):
            raise RuntimeError(
                "Coverage Audit 包含无效 Case"
            )

        case_id = _require_string(
            case_payload,
            "case_id",
            context="Coverage Case",
        )

        expected_source_id = _require_string(
            case_payload,
            "expected_source_id",
            context=f"Coverage Case {case_id}",
        )

        fact_coverages = case_payload.get(
            "fact_coverages"
        )

        if (
            not isinstance(
                fact_coverages,
                list,
            )
            or not fact_coverages
        ):
            raise RuntimeError(
                f"Coverage Case {case_id} "
                "缺少 fact_coverages"
            )

        case_ids.append(case_id)

        for (
            fact_index,
            fact_payload,
        ) in enumerate(fact_coverages):
            if not isinstance(
                fact_payload,
                dict,
            ):
                raise RuntimeError(
                    f"Coverage Case {case_id} "
                    "包含无效 Fact"
                )

            fact = _require_string(
                fact_payload,
                "fact",
                context=(
                    f"Coverage Case {case_id} "
                    f"Fact {fact_index}"
                ),
            )

            missing = fact_payload.get(
                "missing"
            )

            if not isinstance(missing, bool):
                raise RuntimeError(
                    f"Coverage Case {case_id} "
                    f"Fact {fact_index} "
                    "缺少 missing"
                )

            fact_key = (
                case_id,
                fact_index,
            )

            if fact_key in seen_fact_keys:
                raise RuntimeError(
                    "Coverage Audit 包含重复 "
                    f"Fact Key：{fact_key}"
                )

            seen_fact_keys.add(fact_key)

            records.append(
                CompetitionGoldFactRecord(
                    case_id=case_id,
                    fact_index=fact_index,
                    fact=fact,
                    expected_source_id=(
                        expected_source_id
                    ),
                )
            )

            if missing:
                missing_facts[fact_key] = (
                    fact,
                    expected_source_id,
                )

    if (
        len(case_ids)
        != len(set(case_ids))
    ):
        raise RuntimeError(
            "Coverage Audit 包含重复 case_id"
        )

    summary = payload.get("summary")

    if not isinstance(summary, dict):
        raise RuntimeError(
            "Coverage Audit 缺少 summary"
        )

    if (
        summary.get("fact_count")
        != len(records)
    ):
        raise RuntimeError(
            "Coverage Audit fact_count "
            "与记录数量不一致"
        )

    if (
        summary.get("missing_fact_count")
        != len(missing_facts)
    ):
        raise RuntimeError(
            "Coverage Audit missing_fact_count "
            "与记录数量不一致"
        )

    return (
        corpus_id,
        tuple(sorted(case_ids)),
        tuple(
            sorted(
                records,
                key=lambda record: (
                    record.case_id,
                    record.fact_index,
                ),
            )
        ),
        missing_facts,
    )


def _load_candidate_records(
    candidate_path: Path,
    *,
    expected_corpus_id: str,
) -> dict[
    tuple[str, int],
    tuple[str, str],
]:
    payload = _load_json_object(
        candidate_path,
        label="Evidence Candidate Report",
    )

    if (
        payload.get("report_version")
        != EXPECTED_CANDIDATE_VERSION
    ):
        raise RuntimeError(
            "Evidence Candidate Report "
            "版本不受支持"
        )

    corpus_id = _require_string(
        payload,
        "corpus_id",
        context="Evidence Candidate Report",
    )

    if corpus_id != expected_corpus_id:
        raise RuntimeError(
            "Candidate Report 与 Coverage "
            "Audit 的 corpus_id 不一致"
        )

    raw_records = payload.get("records")

    if not isinstance(raw_records, list):
        raise RuntimeError(
            "Evidence Candidate Report "
            "缺少 records"
        )

    records: dict[
        tuple[str, int],
        tuple[str, str],
    ] = {}

    for raw_record in raw_records:
        if not isinstance(
            raw_record,
            dict,
        ):
            raise RuntimeError(
                "Candidate Report 包含"
                "无效记录"
            )

        case_id = _require_string(
            raw_record,
            "case_id",
            context="Candidate Record",
        )

        expected_source_id = _require_string(
            raw_record,
            "expected_source_id",
            context=(
                f"Candidate Record {case_id}"
            ),
        )

        fact = _require_string(
            raw_record,
            "fact",
            context=(
                f"Candidate Record {case_id}"
            ),
        )

        fact_index = raw_record.get(
            "fact_index"
        )

        if (
            not isinstance(fact_index, int)
            or isinstance(fact_index, bool)
            or fact_index < 0
        ):
            raise RuntimeError(
                f"Candidate Record {case_id} "
                "fact_index 无效"
            )

        candidates = raw_record.get(
            "candidates"
        )

        if (
            not isinstance(candidates, list)
            or not candidates
        ):
            raise RuntimeError(
                f"Candidate Record {case_id}"
                f"#{fact_index} 没有候选 Chunk"
            )

        for candidate in candidates:
            if not isinstance(
                candidate,
                dict,
            ):
                raise RuntimeError(
                    "Candidate Report 包含"
                    "无效 Chunk"
                )

            candidate_source_id = (
                _require_string(
                    candidate,
                    "source_id",
                    context=(
                        "Candidate Chunk "
                        f"{case_id}#{fact_index}"
                    ),
                )
            )

            _require_string(
                candidate,
                "chunk_id",
                context=(
                    "Candidate Chunk "
                    f"{case_id}#{fact_index}"
                ),
            )

            if (
                candidate_source_id
                != expected_source_id
            ):
                raise RuntimeError(
                    "Candidate Chunk 来源与 "
                    "expected_source_id 不一致"
                )

        fact_key = (
            case_id,
            fact_index,
        )

        if fact_key in records:
            raise RuntimeError(
                "Candidate Report 包含重复 "
                f"Fact Key：{fact_key}"
            )

        records[fact_key] = (
            fact,
            expected_source_id,
        )

    if (
        payload.get("missing_fact_count")
        != len(records)
    ):
        raise RuntimeError(
            "Candidate Report "
            "missing_fact_count 与记录数量不一致"
        )

    return records


def run_competition_gold_review_template(
    *,
    split_path: Path,
    coverage_path: Path,
    candidate_path: Path,
    output_path: Path,
    expected_text_dev_cases: (
        int | None
    ) = DEFAULT_EXPECTED_TEXT_DEV_CASES,
    expected_fact_count: (
        int | None
    ) = DEFAULT_EXPECTED_FACTS,
    expected_missing_fact_count: (
        int | None
    ) = DEFAULT_EXPECTED_MISSING_FACTS,
) -> CompetitionGoldTemplateResult:
    dev_case_ids = set(
        load_dev_case_ids(
            split_path
        )
    )

    (
        corpus_id,
        coverage_case_ids,
        records,
        missing_facts,
    ) = _load_coverage_records(
        coverage_path
    )

    leaked_case_ids = tuple(
        sorted(
            case_id
            for case_id
            in coverage_case_ids
            if case_id not in dev_case_ids
        )
    )

    if leaked_case_ids:
        raise RuntimeError(
            "Gold 模板包含非 Dev Case："
            f"{leaked_case_ids}"
        )

    if (
        expected_text_dev_cases is not None
        and len(coverage_case_ids)
        != expected_text_dev_cases
    ):
        raise RuntimeError(
            "Dev 文本 Case 数量改变："
            f"expected={expected_text_dev_cases}; "
            f"actual={len(coverage_case_ids)}"
        )

    if (
        expected_fact_count is not None
        and len(records)
        != expected_fact_count
    ):
        raise RuntimeError(
            "Dev Fact 数量改变："
            f"expected={expected_fact_count}; "
            f"actual={len(records)}"
        )

    if (
        expected_missing_fact_count is not None
        and len(missing_facts)
        != expected_missing_fact_count
    ):
        raise RuntimeError(
            "Missing Fact 数量改变："
            f"expected={expected_missing_fact_count}; "
            f"actual={len(missing_facts)}"
        )

    candidate_records = (
        _load_candidate_records(
            candidate_path,
            expected_corpus_id=corpus_id,
        )
    )

    missing_keys = set(
        missing_facts
    )

    candidate_keys = set(
        candidate_records
    )

    if candidate_keys != missing_keys:
        missing_candidates = tuple(
            sorted(
                missing_keys
                - candidate_keys
            )
        )

        unexpected_candidates = tuple(
            sorted(
                candidate_keys
                - missing_keys
            )
        )

        raise RuntimeError(
            "Missing Fact 与 Candidate "
            "记录不一致："
            f"missing={missing_candidates}; "
            f"unexpected={unexpected_candidates}"
        )

    for fact_key in sorted(
        missing_keys
    ):
        if (
            candidate_records[fact_key]
            != missing_facts[fact_key]
        ):
            raise RuntimeError(
                "Candidate Fact 文本或来源"
                "与 Coverage Audit 不一致："
                f"{fact_key}"
            )

    dataset = CompetitionGoldChunkDataset(
        corpus_id=corpus_id,
        candidate_report_sha256=(
            _sha256_file(
                candidate_path
            )
        ),
        records=records,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            dataset.model_dump(
                mode="json"
            ),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return CompetitionGoldTemplateResult(
        dataset=dataset,
        dev_case_count=(
            len(coverage_case_ids)
        ),
        fact_count=len(records),
        missing_fact_count=(
            len(missing_facts)
        ),
        output_path=output_path,
    )


def print_template_result(
    result: CompetitionGoldTemplateResult,
) -> None:
    print(
        "=== Competition Frozen Dev "
        "Gold Chunk Review Template ==="
    )
    print(
        "Corpus ID:",
        result.dataset.corpus_id,
    )
    print(
        "Dev text cases:",
        result.dev_case_count,
    )
    print(
        "Facts:",
        result.fact_count,
    )
    print(
        "Missing facts with candidates:",
        result.missing_fact_count,
    )
    print(
        "Pending reviews:",
        len(result.dataset.records),
    )
    print(
        "Confirmed:",
        sum(
            record.review_status
            == "confirmed"
            for record
            in result.dataset.records
        ),
    )
    print(
        "Unresolved:",
        sum(
            record.review_status
            == "unresolved"
            for record
            in result.dataset.records
        ),
    )
    print(
        "Saved:",
        result.output_path,
    )


def build_argument_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "生成 Frozen Dev Gold Chunk "
            "人工审核模板"
        )
    )

    parser.add_argument(
        "--split",
        type=Path,
        default=DEFAULT_SPLIT_FILE,
    )

    parser.add_argument(
        "--coverage",
        type=Path,
        default=DEFAULT_COVERAGE_FILE,
    )

    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_CANDIDATE_FILE,
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
        run_competition_gold_review_template(
            split_path=arguments.split,
            coverage_path=(
                arguments.coverage
            ),
            candidate_path=(
                arguments.candidates
            ),
            output_path=arguments.output,
        )
    )

    print_template_result(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())