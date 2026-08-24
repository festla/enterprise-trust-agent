from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Literal

from app.schemas.competition_chunk import (
    CompetitionTextChunk,
)
from app.schemas.competition_gold import (
    CompetitionGoldChunkDataset,
    CompetitionGoldChunkReference,
    CompetitionGoldFactRecord,
)
from app.services.competition_corpus import (
    CompetitionCorpusBuildResult,
    load_competition_chunk_corpus,
)


DEFAULT_COVERAGE_FILE = Path(
    "data/competition/processed/eval/"
    "competition_evidence_coverage_dev_v1.json"
)

DEFAULT_CANDIDATE_FILE = Path(
    "data/competition/processed/eval/"
    "competition_evidence_candidates_dev_v1.json"
)

DEFAULT_GOLD_FILE = Path(
    "data/competition/processed/eval/"
    "competition_dev_gold_chunks_v1.json"
)

EXPECTED_COVERAGE_VERSION = (
    "competition_evidence_coverage_dev_v1"
)

EXPECTED_CANDIDATE_VERSION = (
    "competition_evidence_candidates_dev_v1"
)

DEFAULT_EXPECTED_FACTS = 97

FactKey = tuple[str, int]

EvidenceMode = Literal[
    "single_chunk",
    "parent_child",
    "multi_chunk",
]

ChunkRole = Literal[
    "direct",
    "parent_context",
    "supporting",
]

InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class CoverageFact:
    case_id: str
    fact_index: int
    fact: str
    expected_source_id: str
    missing: bool
    single_chunk_ids: tuple[str, ...]
    two_chunk_window_ids: tuple[
        tuple[str, str],
        ...,
    ]

    @property
    def key(self) -> FactKey:
        return (
            self.case_id,
            self.fact_index,
        )


@dataclass(frozen=True, slots=True)
class CandidateSeed:
    chunk_id: str
    rank: int
    bigram_recall: float | None
    bigram_f1: float | None
    unigram_recall: float | None
    sequence_ratio: float | None


@dataclass(frozen=True, slots=True)
class CandidateFact:
    case_id: str
    fact_index: int
    fact: str
    expected_source_id: str
    candidates: tuple[
        CandidateSeed,
        ...,
    ]

    @property
    def key(self) -> FactKey:
        return (
            self.case_id,
            self.fact_index,
        )


@dataclass(frozen=True, slots=True)
class ReviewChunkOption:
    chunk: CompetitionTextChunk
    labels: tuple[str, ...]
    metrics: tuple[str, ...]


@dataclass(slots=True)
class CompetitionGoldReviewSession:
    dataset: CompetitionGoldChunkDataset
    corpus: CompetitionCorpusBuildResult

    coverage_by_key: dict[
        FactKey,
        CoverageFact,
    ]

    candidate_by_key: dict[
        FactKey,
        CandidateFact,
    ]

    options_by_key: dict[
        FactKey,
        tuple[
            ReviewChunkOption,
            ...,
        ],
    ]

    gold_path: Path

    @property
    def pending_record_indices(
        self,
    ) -> tuple[int, ...]:
        return tuple(
            index
            for index, record
            in enumerate(
                self.dataset.records
            )
            if (
                record.review_status
                == "pending"
            )
        )


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

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            f"{label} 根节点必须是 JSON Object"
        )

    return payload


def _require_string(
    payload: dict[str, object],
    field_name: str,
    *,
    context: str,
) -> str:
    value = payload.get(
        field_name
    )

    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise RuntimeError(
            f"{context} 缺少有效的 "
            f"{field_name}"
        )

    return value


def _require_non_negative_int(
    payload: dict[str, object],
    field_name: str,
    *,
    context: str,
) -> int:
    value = payload.get(
        field_name
    )

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise RuntimeError(
            f"{context} 的 "
            f"{field_name} 无效"
        )

    return value


def _optional_float(
    payload: dict[str, object],
    field_name: str,
) -> float | None:
    value = payload.get(
        field_name
    )

    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, float),
        )
    ):
        raise RuntimeError(
            "Candidate Chunk 的 "
            f"{field_name} 无效"
        )

    return float(value)


def _string_tuple(
    value: object,
    *,
    field_name: str,
    context: str,
) -> tuple[str, ...]:
    if not isinstance(
        value,
        list,
    ):
        raise RuntimeError(
            f"{context} 的 {field_name} "
            "必须是列表"
        )

    result: list[str] = []

    for item in value:
        if (
            not isinstance(item, str)
            or not item
        ):
            raise RuntimeError(
                f"{context} 的 {field_name} "
                "包含无效 Chunk ID"
            )

        result.append(item)

    if (
        len(result)
        != len(set(result))
    ):
        raise RuntimeError(
            f"{context} 的 {field_name} "
            "包含重复 Chunk ID"
        )

    return tuple(result)


def _window_tuple(
    value: object,
    *,
    context: str,
) -> tuple[
    tuple[str, str],
    ...,
]:
    if not isinstance(
        value,
        list,
    ):
        raise RuntimeError(
            f"{context} 的 "
            "two_chunk_window_ids "
            "必须是列表"
        )

    windows: list[
        tuple[str, str]
    ] = []

    for raw_window in value:
        if (
            not isinstance(
                raw_window,
                (list, tuple),
            )
            or len(raw_window) != 2
            or not all(
                isinstance(item, str)
                and item
                for item in raw_window
            )
        ):
            raise RuntimeError(
                f"{context} 包含无效的 "
                "two_chunk_window_ids"
            )

        windows.append(
            (
                raw_window[0],
                raw_window[1],
            )
        )

    if (
        len(windows)
        != len(set(windows))
    ):
        raise RuntimeError(
            f"{context} 包含重复的 "
            "two_chunk_window_ids"
        )

    return tuple(windows)


def load_coverage_facts(
    path: Path,
) -> tuple[
    str,
    dict[
        FactKey,
        CoverageFact,
    ],
]:
    payload = _load_json_object(
        path,
        label=(
            "Evidence Coverage Audit"
        ),
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
        context=(
            "Evidence Coverage Audit"
        ),
    )

    raw_cases = payload.get(
        "cases"
    )

    if (
        not isinstance(raw_cases, list)
        or not raw_cases
    ):
        raise RuntimeError(
            "Evidence Coverage Audit "
            "缺少 cases"
        )

    facts: dict[
        FactKey,
        CoverageFact,
    ] = {}

    for raw_case in raw_cases:
        if not isinstance(
            raw_case,
            dict,
        ):
            raise RuntimeError(
                "Evidence Coverage Audit "
                "包含无效 Case"
            )

        case_id = _require_string(
            raw_case,
            "case_id",
            context="Coverage Case",
        )

        source_id = _require_string(
            raw_case,
            "expected_source_id",
            context=(
                f"Coverage Case {case_id}"
            ),
        )

        raw_facts = raw_case.get(
            "fact_coverages"
        )

        if (
            not isinstance(
                raw_facts,
                list,
            )
            or not raw_facts
        ):
            raise RuntimeError(
                f"Coverage Case {case_id} "
                "缺少 fact_coverages"
            )

        for (
            fact_index,
            raw_fact,
        ) in enumerate(raw_facts):
            if not isinstance(
                raw_fact,
                dict,
            ):
                raise RuntimeError(
                    "Coverage Case "
                    f"{case_id} "
                    "包含无效 Fact"
                )

            context = (
                f"Coverage Case {case_id} "
                f"Fact {fact_index}"
            )

            fact = _require_string(
                raw_fact,
                "fact",
                context=context,
            )

            missing = raw_fact.get(
                "missing"
            )

            if not isinstance(
                missing,
                bool,
            ):
                raise RuntimeError(
                    f"{context} 缺少有效的 "
                    "missing"
                )

            record = CoverageFact(
                case_id=case_id,
                fact_index=fact_index,
                fact=fact,
                expected_source_id=(
                    source_id
                ),
                missing=missing,
                single_chunk_ids=(
                    _string_tuple(
                        raw_fact.get(
                            "single_chunk_ids",
                            [],
                        ),
                        field_name=(
                            "single_chunk_ids"
                        ),
                        context=context,
                    )
                ),
                two_chunk_window_ids=(
                    _window_tuple(
                        raw_fact.get(
                            "two_chunk_window_ids",
                            [],
                        ),
                        context=context,
                    )
                ),
            )

            if record.key in facts:
                raise RuntimeError(
                    "Coverage Audit "
                    "包含重复 Fact Key："
                    f"{record.key}"
                )

            facts[record.key] = record

    summary = payload.get(
        "summary"
    )

    if isinstance(
        summary,
        dict,
    ):
        fact_count = summary.get(
            "fact_count"
        )

        if (
            fact_count is not None
            and fact_count
            != len(facts)
        ):
            raise RuntimeError(
                "Coverage Audit "
                "fact_count 与事实数量不一致"
            )

    return (
        corpus_id,
        facts,
    )


def load_candidate_facts(
    path: Path,
) -> tuple[
    str,
    dict[
        FactKey,
        CandidateFact,
    ],
]:
    payload = _load_json_object(
        path,
        label=(
            "Evidence Candidate Report"
        ),
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
        context=(
            "Evidence Candidate Report"
        ),
    )

    raw_records = payload.get(
        "records"
    )

    if not isinstance(
        raw_records,
        list,
    ):
        raise RuntimeError(
            "Evidence Candidate Report "
            "缺少 records"
        )

    records: dict[
        FactKey,
        CandidateFact,
    ] = {}

    for raw_record in raw_records:
        if not isinstance(
            raw_record,
            dict,
        ):
            raise RuntimeError(
                "Evidence Candidate Report "
                "包含无效记录"
            )

        case_id = _require_string(
            raw_record,
            "case_id",
            context="Candidate Record",
        )

        fact_index = (
            _require_non_negative_int(
                raw_record,
                "fact_index",
                context=(
                    "Candidate Record "
                    f"{case_id}"
                ),
            )
        )

        context = (
            f"Candidate Record "
            f"{case_id}#{fact_index}"
        )

        expected_source_id = (
            _require_string(
                raw_record,
                "expected_source_id",
                context=context,
            )
        )

        raw_candidates = (
            raw_record.get(
                "candidates"
            )
        )

        if (
            not isinstance(
                raw_candidates,
                list,
            )
            or not raw_candidates
        ):
            raise RuntimeError(
                f"{context} "
                "没有候选 Chunk"
            )

        candidates: list[
            CandidateSeed
        ] = []

        seen_chunk_ids: set[str] = (
            set()
        )

        for raw_candidate in (
            raw_candidates
        ):
            if not isinstance(
                raw_candidate,
                dict,
            ):
                raise RuntimeError(
                    f"{context} "
                    "包含无效候选 Chunk"
                )

            chunk_id = _require_string(
                raw_candidate,
                "chunk_id",
                context=context,
            )

            candidate_source_id = (
                _require_string(
                    raw_candidate,
                    "source_id",
                    context=context,
                )
            )

            if (
                candidate_source_id
                != expected_source_id
            ):
                raise RuntimeError(
                    f"{context} 的候选 "
                    "Chunk 来源错误"
                )

            if (
                chunk_id
                in seen_chunk_ids
            ):
                raise RuntimeError(
                    f"{context} 包含重复"
                    f"候选 Chunk：{chunk_id}"
                )

            seen_chunk_ids.add(
                chunk_id
            )

            candidates.append(
                CandidateSeed(
                    chunk_id=chunk_id,
                    rank=(
                        _require_non_negative_int(
                            raw_candidate,
                            "rank",
                            context=context,
                        )
                    ),
                    bigram_recall=(
                        _optional_float(
                            raw_candidate,
                            "bigram_recall",
                        )
                    ),
                    bigram_f1=(
                        _optional_float(
                            raw_candidate,
                            "bigram_f1",
                        )
                    ),
                    unigram_recall=(
                        _optional_float(
                            raw_candidate,
                            "unigram_recall",
                        )
                    ),
                    sequence_ratio=(
                        _optional_float(
                            raw_candidate,
                            "sequence_ratio",
                        )
                    ),
                )
            )

        record = CandidateFact(
            case_id=case_id,
            fact_index=fact_index,
            fact=_require_string(
                raw_record,
                "fact",
                context=context,
            ),
            expected_source_id=(
                expected_source_id
            ),
            candidates=tuple(
                candidates
            ),
        )

        if record.key in records:
            raise RuntimeError(
                "Candidate Report "
                "包含重复 Fact Key："
                f"{record.key}"
            )

        records[record.key] = record

    declared_count = payload.get(
        "missing_fact_count"
    )

    if (
        declared_count is not None
        and declared_count
        != len(records)
    ):
        raise RuntimeError(
            "Candidate Report "
            "missing_fact_count "
            "与记录数量不一致"
        )

    return (
        corpus_id,
        records,
    )


def _metric_text(
    seed: CandidateSeed,
) -> str:
    values = (
        (
            "BigramRecall",
            seed.bigram_recall,
        ),
        (
            "BigramF1",
            seed.bigram_f1,
        ),
        (
            "UnigramRecall",
            seed.unigram_recall,
        ),
        (
            "Sequence",
            seed.sequence_ratio,
        ),
    )

    return " ".join(
        f"{name}={value:.4f}"
        for name, value in values
        if value is not None
    )


def _append_option_label(
    option_data: dict[
        str,
        dict[str, object],
    ],
    ordered_ids: list[str],
    *,
    chunk_id: str,
    label: str,
    metric: str | None = None,
) -> None:
    if (
        chunk_id
        not in option_data
    ):
        option_data[chunk_id] = {
            "labels": [],
            "metrics": [],
        }

        ordered_ids.append(
            chunk_id
        )

    labels = option_data[
        chunk_id
    ]["labels"]

    metrics = option_data[
        chunk_id
    ]["metrics"]

    assert isinstance(
        labels,
        list,
    )

    assert isinstance(
        metrics,
        list,
    )

    if label not in labels:
        labels.append(label)

    if (
        metric
        and metric not in metrics
    ):
        metrics.append(metric)


def _build_review_options(
    coverage: CoverageFact,
    candidate: CandidateFact | None,
    *,
    chunks_by_id: dict[
        str,
        CompetitionTextChunk,
    ],
    chunks_by_source_index: dict[
        tuple[str, int],
        CompetitionTextChunk,
    ],
) -> tuple[
    ReviewChunkOption,
    ...,
]:
    option_data: dict[
        str,
        dict[str, object],
    ] = {}

    ordered_ids: list[str] = []
    primary_ids: list[str] = []

    for chunk_id in (
        coverage.single_chunk_ids
    ):
        _append_option_label(
            option_data,
            ordered_ids,
            chunk_id=chunk_id,
            label=(
                "Coverage 精确单 Chunk"
            ),
        )

        primary_ids.append(
            chunk_id
        )

    for (
        window_index,
        window,
    ) in enumerate(
        coverage.two_chunk_window_ids,
        start=1,
    ):
        for chunk_id in window:
            _append_option_label(
                option_data,
                ordered_ids,
                chunk_id=chunk_id,
                label=(
                    "Coverage 双 Chunk "
                    f"窗口 {window_index}"
                ),
            )

            primary_ids.append(
                chunk_id
            )

    if candidate is not None:
        for seed in (
            candidate.candidates
        ):
            _append_option_label(
                option_data,
                ordered_ids,
                chunk_id=(
                    seed.chunk_id
                ),
                label=(
                    "词面候选 "
                    f"Top-{seed.rank}"
                ),
                metric=(
                    _metric_text(seed)
                ),
            )

            primary_ids.append(
                seed.chunk_id
            )

    unique_primary_ids = tuple(
        dict.fromkeys(
            primary_ids
        )
    )

    for primary_id in (
        unique_primary_ids
    ):
        primary = chunks_by_id.get(
            primary_id
        )

        if primary is None:
            raise RuntimeError(
                "候选引用了 Corpus 中"
                "不存在的 Chunk："
                f"{primary_id}"
            )

        if (
            primary.source_id
            != coverage.expected_source_id
        ):
            raise RuntimeError(
                "候选 Chunk 来源与"
                "事实来源不一致："
                f"{primary_id}"
            )

        for (
            delta,
            label,
        ) in (
            (
                -1,
                "前一相邻 Chunk",
            ),
            (
                1,
                "后一相邻 Chunk",
            ),
        ):
            neighbor = (
                chunks_by_source_index.get(
                    (
                        primary.source_id,
                        (
                            primary.chunk_index
                            + delta
                        ),
                    )
                )
            )

            if neighbor is None:
                continue

            _append_option_label(
                option_data,
                ordered_ids,
                chunk_id=(
                    neighbor.chunk_id
                ),
                label=(
                    f"{label}（相邻于 "
                    f"{primary_id}）"
                ),
            )

    options: list[
        ReviewChunkOption
    ] = []

    for chunk_id in ordered_ids:
        chunk = chunks_by_id.get(
            chunk_id
        )

        if chunk is None:
            raise RuntimeError(
                "审核选项引用了"
                "未知 Chunk："
                f"{chunk_id}"
            )

        raw = option_data[
            chunk_id
        ]

        raw_labels = raw[
            "labels"
        ]

        raw_metrics = raw[
            "metrics"
        ]

        assert isinstance(
            raw_labels,
            list,
        )

        assert isinstance(
            raw_metrics,
            list,
        )

        options.append(
            ReviewChunkOption(
                chunk=chunk,
                labels=tuple(
                    raw_labels
                ),
                metrics=tuple(
                    raw_metrics
                ),
            )
        )

    return tuple(options)


def load_review_session(
    *,
    coverage_path: Path,
    candidate_path: Path,
    gold_path: Path,
    corpus_directory: Path,
    expected_fact_count: (
        int | None
    ) = DEFAULT_EXPECTED_FACTS,
) -> CompetitionGoldReviewSession:
    try:
        dataset = (
            CompetitionGoldChunkDataset
            .model_validate_json(
                gold_path.read_bytes()
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"Gold Dataset 无效："
            f"{gold_path}"
        ) from exc

    candidate_sha256 = (
        _sha256_file(
            candidate_path
        )
    )

    if (
        dataset
        .candidate_report_sha256
        != candidate_sha256
    ):
        raise RuntimeError(
            "Gold Dataset 绑定的 "
            "Candidate SHA256 "
            "与当前文件不一致"
        )

    (
        coverage_corpus_id,
        coverage_by_key,
    ) = load_coverage_facts(
        coverage_path
    )

    (
        candidate_corpus_id,
        candidate_by_key,
    ) = load_candidate_facts(
        candidate_path
    )

    corpus = (
        load_competition_chunk_corpus(
            corpus_directory
        )
    )

    corpus_id = (
        corpus.manifest.corpus_id
    )

    if not (
        dataset.corpus_id
        == coverage_corpus_id
        == candidate_corpus_id
        == corpus_id
    ):
        raise RuntimeError(
            "Gold、Coverage、Candidate "
            "与 Corpus 的 corpus_id "
            "不一致"
        )

    if (
        expected_fact_count
        is not None
        and len(coverage_by_key)
        != expected_fact_count
    ):
        raise RuntimeError(
            "Fact 数量改变："
            f"expected="
            f"{expected_fact_count}; "
            f"actual="
            f"{len(coverage_by_key)}"
        )

    gold_by_key = {
        (
            record.case_id,
            record.fact_index,
        ): record
        for record
        in dataset.records
    }

    if (
        set(gold_by_key)
        != set(coverage_by_key)
    ):
        raise RuntimeError(
            "Gold Dataset 与 "
            "Coverage Audit 的 "
            "Fact Key 不一致"
        )

    missing_keys = {
        key
        for key, fact
        in coverage_by_key.items()
        if fact.missing
    }

    if (
        set(candidate_by_key)
        != missing_keys
    ):
        raise RuntimeError(
            "Candidate Fact Key "
            "必须与 Coverage "
            "Missing Fact 完全一致"
        )

    for (
        key,
        coverage,
    ) in coverage_by_key.items():
        gold = gold_by_key[key]

        if (
            gold.fact
            != coverage.fact
            or (
                gold.expected_source_id
                != coverage
                .expected_source_id
            )
        ):
            raise RuntimeError(
                "Gold 与 Coverage 的"
                "事实文本或来源不一致："
                f"{key}"
            )

        candidate = (
            candidate_by_key.get(
                key
            )
        )

        if (
            candidate is not None
            and (
                candidate.fact
                != coverage.fact
                or (
                    candidate
                    .expected_source_id
                    != coverage
                    .expected_source_id
                )
            )
        ):
            raise RuntimeError(
                "Candidate 与 Coverage 的"
                "事实文本或来源不一致："
                f"{key}"
            )

    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunk
        in corpus.chunks
    }

    if (
        len(chunks_by_id)
        != len(corpus.chunks)
    ):
        raise RuntimeError(
            "Corpus 包含重复 chunk_id"
        )

    chunks_by_source_index = {
        (
            chunk.source_id,
            chunk.chunk_index,
        ): chunk
        for chunk
        in corpus.chunks
    }

    for record in (
        dataset.records
    ):
        for reference in (
            record.gold_chunks
        ):
            chunk = chunks_by_id.get(
                reference.chunk_id
            )

            if (
                chunk is None
                or (
                    chunk.source_id
                    != record
                    .expected_source_id
                )
            ):
                raise RuntimeError(
                    "已有 Gold 引用了"
                    "不存在或来源错误的 "
                    "Chunk："
                    f"{reference.chunk_id}"
                )

    options_by_key = {
        key: _build_review_options(
            coverage,
            candidate_by_key.get(
                key
            ),
            chunks_by_id=(
                chunks_by_id
            ),
            chunks_by_source_index=(
                chunks_by_source_index
            ),
        )
        for key, coverage
        in coverage_by_key.items()
    }

    return (
        CompetitionGoldReviewSession(
            dataset=dataset,
            corpus=corpus,
            coverage_by_key=(
                coverage_by_key
            ),
            candidate_by_key=(
                candidate_by_key
            ),
            options_by_key=(
                options_by_key
            ),
            gold_path=gold_path,
        )
    )


def save_gold_dataset_atomic(
    dataset: CompetitionGoldChunkDataset,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    content = (
        json.dumps(
            dataset.model_dump(
                mode="json"
            ),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    (
        CompetitionGoldChunkDataset
        .model_validate_json(
            content
        )
    )

    temporary_path: (
        Path | None
    ) = None

    try:
        with (
            tempfile
            .NamedTemporaryFile(
                mode="wb",
                prefix=(
                    f".{output_path.name}."
                ),
                suffix=".tmp",
                dir=output_path.parent,
                delete=False,
            )
        ) as temporary_file:
            temporary_file.write(
                content
            )

            temporary_file.flush()

            os.fsync(
                temporary_file.fileno()
            )

            temporary_path = Path(
                temporary_file.name
            )

        backup_path = Path(
            f"{output_path}.bak"
        )

        if output_path.exists():
            shutil.copy2(
                output_path,
                backup_path,
            )

        os.replace(
            temporary_path,
            output_path,
        )

        return backup_path

    finally:
        if (
            temporary_path
            is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


def apply_review_decision(
    session: CompetitionGoldReviewSession,
    *,
    record_index: int,
    review_status: Literal[
        "confirmed",
        "unresolved",
    ],
    evidence_mode: (
        EvidenceMode | None
    ) = None,
    selections: tuple[
        tuple[
            str,
            ChunkRole,
        ],
        ...,
    ] = (),
    review_note: str | None = None,
) -> CompetitionGoldChunkDataset:
    if (
        record_index < 0
        or record_index
        >= len(
            session.dataset.records
        )
    ):
        raise IndexError(
            "record_index 超出范围"
        )

    old_record = (
        session.dataset.records[
            record_index
        ]
    )

    if (
        old_record.review_status
        != "pending"
    ):
        raise RuntimeError(
            "只能修改 pending 记录"
        )

    key = (
        old_record.case_id,
        old_record.fact_index,
    )

    allowed_ids = {
        option.chunk.chunk_id
        for option
        in session.options_by_key[
            key
        ]
    }

    selected_ids = [
        chunk_id
        for chunk_id, _
        in selections
    ]

    if any(
        chunk_id
        not in allowed_ids
        for chunk_id
        in selected_ids
    ):
        raise ValueError(
            "选择中包含当前事实"
            "未展示的 Chunk"
        )

    if (
        len(selected_ids)
        != len(set(selected_ids))
    ):
        raise ValueError(
            "不能重复选择同一个 Chunk"
        )

    payload = (
        old_record.model_dump(
            mode="json"
        )
    )

    payload.update(
        {
            "review_status": (
                review_status
            ),
            "evidence_mode": (
                evidence_mode
            ),
            "gold_chunks": [
                (
                    CompetitionGoldChunkReference(
                        chunk_id=(
                            chunk_id
                        ),
                        role=role,
                    )
                    .model_dump(
                        mode="json"
                    )
                )
                for chunk_id, role
                in selections
            ],
            "review_note": (
                review_note.strip()
                if review_note
                else None
            ),
        }
    )

    new_record = (
        CompetitionGoldFactRecord
        .model_validate(
            payload
        )
    )

    dataset_payload = (
        session.dataset.model_dump(
            mode="json"
        )
    )

    dataset_payload[
        "records"
    ][record_index] = (
        new_record.model_dump(
            mode="json"
        )
    )

    new_dataset = (
        CompetitionGoldChunkDataset
        .model_validate(
            dataset_payload
        )
    )

    save_gold_dataset_atomic(
        new_dataset,
        session.gold_path,
    )

    session.dataset = (
        new_dataset
    )

    return new_dataset


def _location_text(
    chunk: CompetitionTextChunk,
) -> str:
    parts = [
        f"type={chunk.chunk_type}",
        f"index={chunk.chunk_index}",
    ]

    if (
        chunk.page_start
        is not None
    ):
        page = str(
            chunk.page_start
        )

        if (
            chunk.page_end
            is not None
            and (
                chunk.page_end
                != chunk.page_start
            )
        ):
            page += (
                f"-{chunk.page_end}"
            )

        parts.append(
            f"page={page}"
        )

    if (
        chunk.paragraph_start_index
        is not None
    ):
        paragraph = str(
            chunk
            .paragraph_start_index
        )

        if (
            chunk.paragraph_end_index
            is not None
            and (
                chunk
                .paragraph_end_index
                != chunk
                .paragraph_start_index
            )
        ):
            paragraph += (
                "-"
                f"{chunk.paragraph_end_index}"
            )

        parts.append(
            f"paragraph={paragraph}"
        )

    if (
        chunk.table_index
        is not None
    ):
        parts.append(
            f"table={chunk.table_index}"
        )

    if chunk.section_path:
        parts.append(
            "section="
            + " / ".join(
                chunk.section_path
            )
        )

    if chunk.article:
        parts.append(
            f"article={chunk.article}"
        )

    if chunk.table_title:
        parts.append(
            "table_title="
            f"{chunk.table_title}"
        )

    return " | ".join(parts)


def _preview_text(
    text: str,
    max_chars: int,
) -> str:
    if (
        max_chars <= 0
        or len(text) <= max_chars
    ):
        return text

    return (
        text[:max_chars]
        + "\n... [截断 "
        + f"{len(text) - max_chars} "
        + "字符]"
    )


def _progress_text(
    dataset: CompetitionGoldChunkDataset,
) -> str:
    counts = {
        status: sum(
            record.review_status
            == status
            for record
            in dataset.records
        )
        for status in (
            "pending",
            "confirmed",
            "unresolved",
        )
    }

    return (
        f"总计={len(dataset.records)}"
        f" | pending={counts['pending']}"
        " | confirmed="
        f"{counts['confirmed']}"
        " | unresolved="
        f"{counts['unresolved']}"
    )


def print_review_record(
    session: CompetitionGoldReviewSession,
    *,
    record_index: int,
    pending_position: int,
    pending_total: int,
    output_fn: OutputFunction = print,
    preview_chars: int = 0,
) -> None:
    record = (
        session.dataset.records[
            record_index
        ]
    )

    key = (
        record.case_id,
        record.fact_index,
    )

    coverage = (
        session.coverage_by_key[
            key
        ]
    )

    options = (
        session.options_by_key[
            key
        ]
    )

    fact_class = (
        "missing + 词面 Top-K"
        if coverage.missing
        else "Coverage 精确覆盖"
    )

    output_fn("")

    output_fn(
        "=" * 88
    )

    output_fn(
        "审核进度："
        f"{pending_position}/"
        f"{pending_total}"
        " | "
        f"{_progress_text(session.dataset)}"
    )

    output_fn(
        f"Fact Key："
        f"{record.case_id}"
        f"#{record.fact_index}"
        " | 类别："
        f"{fact_class}"
    )

    output_fn(
        "Expected Source："
        f"{record.expected_source_id}"
    )

    output_fn(
        f"事实：{record.fact}"
    )

    output_fn(
        "-" * 88
    )

    if not options:
        output_fn(
            "当前事实没有可展示的 Chunk，"
            "只能 skip、unresolved 或 quit。"
        )

    for (
        option_number,
        option,
    ) in enumerate(
        options,
        start=1,
    ):
        output_fn("")

        output_fn(
            f"[{option_number}] "
            f"{option.chunk.chunk_id}"
            " | "
            + "；".join(
                option.labels
            )
        )

        output_fn(
            _location_text(
                option.chunk
            )
        )

        for metric in (
            option.metrics
        ):
            output_fn(metric)

        output_fn(
            _preview_text(
                option.chunk.text,
                preview_chars,
            )
        )

    output_fn("")

    output_fn("命令：")

    output_fn(
        "  c <编号>                 "
        "确认 single_chunk"
    )

    output_fn(
        "  p <父上下文编号> <直接证据编号>  "
        "确认 parent_child"
    )

    output_fn(
        "  m <编号1> <编号2> [...]   "
        "确认 multi_chunk；"
        "第一个为 direct"
    )

    output_fn(
        "  u <原因>                 "
        "标记 unresolved"
    )

    output_fn(
        "  s                        "
        "跳过，暂不修改"
    )

    output_fn(
        "  q                        "
        "安全退出，当前记录不修改"
    )


def _parse_option_numbers(
    raw: str,
    *,
    option_count: int,
) -> tuple[int, ...]:
    tokens = [
        token
        for token in re.split(
            r"[\s,，]+",
            raw.strip(),
        )
        if token
    ]

    if (
        not tokens
        or any(
            not token.isdigit()
            for token in tokens
        )
    ):
        raise ValueError(
            "请输入有效的候选编号"
        )

    numbers = tuple(
        int(token)
        for token in tokens
    )

    if any(
        number < 1
        or number > option_count
        for number in numbers
    ):
        raise ValueError(
            "候选编号必须在 "
            f"1 到 {option_count} 之间"
        )

    if (
        len(numbers)
        != len(set(numbers))
    ):
        raise ValueError(
            "候选编号不能重复"
        )

    return numbers


def _selection_for_numbers(
    options: tuple[
        ReviewChunkOption,
        ...,
    ],
    numbers: tuple[int, ...],
    roles: tuple[
        ChunkRole,
        ...,
    ],
) -> tuple[
    tuple[
        str,
        ChunkRole,
    ],
    ...,
]:
    if (
        len(numbers)
        != len(roles)
    ):
        raise ValueError(
            "候选数量与角色数量不一致"
        )

    return tuple(
        (
            options[
                number - 1
            ].chunk.chunk_id,
            role,
        )
        for number, role
        in zip(
            numbers,
            roles,
            strict=True,
        )
    )


def run_interactive_review(
    session: CompetitionGoldReviewSession,
    *,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
    preview_chars: int = 0,
) -> None:
    pending_indices = (
        session.pending_record_indices
    )

    if not pending_indices:
        output_fn(
            "没有 pending 记录，"
            "人工审核已经完成。"
        )

        output_fn(
            _progress_text(
                session.dataset
            )
        )

        return

    output_fn(
        "=== Competition Frozen Dev "
        "Gold Chunk Reviewer ==="
    )

    output_fn(
        "Corpus ID："
        f"{session.dataset.corpus_id}"
    )

    output_fn(
        f"Gold 文件："
        f"{session.gold_path}"
    )

    output_fn(
        _progress_text(
            session.dataset
        )
    )

    for (
        pending_position,
        record_index,
    ) in enumerate(
        pending_indices,
        start=1,
    ):
        while True:
            print_review_record(
                session,
                record_index=(
                    record_index
                ),
                pending_position=(
                    pending_position
                ),
                pending_total=len(
                    pending_indices
                ),
                output_fn=output_fn,
                preview_chars=(
                    preview_chars
                ),
            )

            record = (
                session.dataset.records[
                    record_index
                ]
            )

            key = (
                record.case_id,
                record.fact_index,
            )

            options = (
                session.options_by_key[
                    key
                ]
            )

            try:
                command_line = (
                    input_fn(
                        "review> "
                    )
                    .strip()
                )
            except (
                EOFError,
                KeyboardInterrupt,
            ):
                output_fn(
                    "\n安全退出；当前未确认"
                    "记录没有修改。"
                )

                output_fn(
                    _progress_text(
                        session.dataset
                    )
                )

                return

            if not command_line:
                output_fn(
                    "请输入命令。"
                )

                continue

            (
                command,
                _,
                arguments,
            ) = command_line.partition(
                " "
            )

            command = (
                command.lower()
            )

            arguments = (
                arguments.strip()
            )

            try:
                if command == "q":
                    output_fn(
                        "安全退出；当前未确认"
                        "记录没有修改。"
                    )

                    output_fn(
                        _progress_text(
                            session.dataset
                        )
                    )

                    return

                if command == "s":
                    output_fn(
                        "已跳过；当前记录"
                        "仍为 pending。"
                    )

                    break

                if command == "u":
                    note = arguments

                    if not note:
                        note = (
                            input_fn(
                                "unresolved 原因> "
                            )
                            .strip()
                        )

                    if not note:
                        raise ValueError(
                            "unresolved "
                            "必须填写原因"
                        )

                    apply_review_decision(
                        session,
                        record_index=(
                            record_index
                        ),
                        review_status=(
                            "unresolved"
                        ),
                        review_note=note,
                    )

                    output_fn(
                        "已原子保存 unresolved；"
                        "同时更新 .json.bak。"
                    )

                    break

                if command == "c":
                    numbers = (
                        _parse_option_numbers(
                            arguments,
                            option_count=(
                                len(options)
                            ),
                        )
                    )

                    if (
                        len(numbers)
                        != 1
                    ):
                        raise ValueError(
                            "single_chunk "
                            "必须选择且只能"
                            "选择一个编号"
                        )

                    selections = (
                        _selection_for_numbers(
                            options,
                            numbers,
                            ("direct",),
                        )
                    )

                    apply_review_decision(
                        session,
                        record_index=(
                            record_index
                        ),
                        review_status=(
                            "confirmed"
                        ),
                        evidence_mode=(
                            "single_chunk"
                        ),
                        selections=(
                            selections
                        ),
                    )

                    output_fn(
                        "已原子保存 "
                        "single_chunk；"
                        "同时更新 .json.bak。"
                    )

                    break

                if command == "p":
                    numbers = (
                        _parse_option_numbers(
                            arguments,
                            option_count=(
                                len(options)
                            ),
                        )
                    )

                    if (
                        len(numbers)
                        != 2
                    ):
                        raise ValueError(
                            "parent_child "
                            "必须依次选择"
                            "父上下文、直接证据"
                            "两个编号"
                        )

                    selections = (
                        _selection_for_numbers(
                            options,
                            numbers,
                            (
                                "parent_context",
                                "direct",
                            ),
                        )
                    )

                    apply_review_decision(
                        session,
                        record_index=(
                            record_index
                        ),
                        review_status=(
                            "confirmed"
                        ),
                        evidence_mode=(
                            "parent_child"
                        ),
                        selections=(
                            selections
                        ),
                    )

                    output_fn(
                        "已原子保存 "
                        "parent_child；"
                        "同时更新 .json.bak。"
                    )

                    break

                if command == "m":
                    numbers = (
                        _parse_option_numbers(
                            arguments,
                            option_count=(
                                len(options)
                            ),
                        )
                    )

                    if len(numbers) < 2:
                        raise ValueError(
                            "multi_chunk "
                            "至少选择两个编号"
                        )

                    roles: tuple[
                        ChunkRole,
                        ...,
                    ] = (
                        "direct",
                        *(
                            "supporting"
                            for _
                            in numbers[1:]
                        ),
                    )

                    selections = (
                        _selection_for_numbers(
                            options,
                            numbers,
                            roles,
                        )
                    )

                    apply_review_decision(
                        session,
                        record_index=(
                            record_index
                        ),
                        review_status=(
                            "confirmed"
                        ),
                        evidence_mode=(
                            "multi_chunk"
                        ),
                        selections=(
                            selections
                        ),
                    )

                    output_fn(
                        "已原子保存 "
                        "multi_chunk；"
                        "同时更新 .json.bak。"
                    )

                    break

                output_fn(
                    "未知命令，请使用 "
                    "c / p / m / u / s / q。"
                )

            except (
                ValueError,
                RuntimeError,
            ) as exc:
                output_fn(
                    f"输入无效：{exc}"
                )

    output_fn(
        "所有本轮 pending 记录"
        "均已处理或跳过。"
    )

    output_fn(
        _progress_text(
            session.dataset
        )
    )


def build_argument_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "交互式审核 Competition "
            "Frozen Dev Gold Chunks"
        )
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--coverage",
        type=Path,
        default=(
            DEFAULT_COVERAGE_FILE
        ),
    )

    parser.add_argument(
        "--candidates",
        type=Path,
        default=(
            DEFAULT_CANDIDATE_FILE
        ),
    )

    parser.add_argument(
        "--gold",
        type=Path,
        default=(
            DEFAULT_GOLD_FILE
        ),
    )

    parser.add_argument(
        "--preview-chars",
        type=int,
        default=0,
        help=(
            "每个 Chunk 最多显示字符数；"
            "0 表示完整显示"
        ),
    )

    parser.add_argument(
        "--expected-facts",
        type=int,
        default=(
            DEFAULT_EXPECTED_FACTS
        ),
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    arguments = (
        build_argument_parser()
        .parse_args(argv)
    )

    if (
        arguments.preview_chars
        < 0
    ):
        raise SystemExit(
            "--preview-chars "
            "不能小于 0"
        )

    if (
        arguments.expected_facts
        < 1
    ):
        raise SystemExit(
            "--expected-facts "
            "必须大于 0"
        )

    session = load_review_session(
        coverage_path=(
            arguments.coverage
        ),
        candidate_path=(
            arguments.candidates
        ),
        gold_path=(
            arguments.gold
        ),
        corpus_directory=(
            arguments.corpus
        ),
        expected_fact_count=(
            arguments
            .expected_facts
        ),
    )

    run_interactive_review(
        session,
        preview_chars=(
            arguments.preview_chars
        ),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())