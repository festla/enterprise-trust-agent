from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from app.schemas.competition_context_expansion import (
    CompetitionContextExpandedHit,
    CompetitionContextExpansionOrigin,
)
from app.services.competition_corpus import (
    load_competition_chunk_corpus,
)
from app.services.competition_evidence_assembler import (
    CompetitionEvidenceAssemblyConfig,
    assemble_competition_evidence_bundle,
)
from scripts.evaluate_competition_bm25_gold_dev import (
    load_competition_gold_dataset,
)


RETRIEVAL_RESULT_PATH = Path(
    "data/competition/processed/eval/"
    "competition_bm25_structured_context_"
    "source_filtered_dev_v1.json"
)

GOLD_PATH = Path(
    "data/competition/processed/eval/"
    "competition_dev_gold_chunks_v1.json"
)

CORPUS_PATH = Path(
    "data/competition/processed/corpora/"
    "competition_corpus_8cbec682dfce4962"
)

OUTPUT_PATH = Path(
    "data/competition/processed/eval/"
    "competition_evidence_assembly_dev_v1.json"
)


BUDGETS = (
    (16, 4000),
    (24, 4000),
    (32, 4000),
    (48, 4000),
)


@dataclass(
    frozen=True,
    slots=True,
)
class EvidenceAssemblyMetrics:
    max_evidences: int
    max_chars: int

    fact_count: int

    direct_hit_count: int
    complete_gold_count: int

    direct_hit_rate: float
    complete_gold_rate: float
    mean_gold_recall: float

    mean_evidences_per_case: float
    mean_chars_per_case: float

    total_evidences: int
    total_chars: int

    complete_by_mode: dict[str, int]


def _load_json(
    path: Path,
) -> dict[str, object]:
    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            f"JSON 根节点不是 Object：{path}"
        )

    return payload


def _rebuild_hits(
    *,
    case_payload: dict[str, object],
    chunks_by_id: dict[str, object],
) -> tuple[
    CompetitionContextExpandedHit,
    ...,
]:
    raw_hits = case_payload.get(
        "expanded_hits"
    )

    if not isinstance(
        raw_hits,
        list,
    ):
        raise RuntimeError(
            "Case 缺少 expanded_hits"
        )

    hits = []

    for raw_hit in raw_hits:
        if not isinstance(
            raw_hit,
            dict,
        ):
            raise RuntimeError(
                "expanded_hit 类型无效"
            )

        chunk_id = raw_hit.get(
            "chunk_id"
        )

        if not isinstance(
            chunk_id,
            str,
        ):
            raise RuntimeError(
                "expanded_hit 缺少 chunk_id"
            )

        chunk = chunks_by_id.get(
            chunk_id
        )

        if chunk is None:
            raise RuntimeError(
                "expanded_hit 引用了不存在的 "
                f"chunk：{chunk_id}"
            )

        raw_origins = raw_hit.get(
            "origins"
        )

        if not isinstance(
            raw_origins,
            list,
        ):
            raise RuntimeError(
                "expanded_hit 缺少 origins"
            )

        origins = tuple(
            CompetitionContextExpansionOrigin
            .model_validate(origin)
            for origin in raw_origins
        )

        rank = raw_hit.get(
            "rank"
        )

        effective_seed_rank = (
            raw_hit.get(
                "effective_seed_rank"
            )
        )

        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank < 1
        ):
            raise RuntimeError(
                "expanded_hit rank 无效"
            )

        if (
            not isinstance(
                effective_seed_rank,
                int,
            )
            or isinstance(
                effective_seed_rank,
                bool,
            )
            or effective_seed_rank < 1
        ):
            raise RuntimeError(
                "effective_seed_rank 无效"
            )

        hits.append(
            CompetitionContextExpandedHit(
                rank=rank,
                effective_seed_rank=(
                    effective_seed_rank
                ),
                origins=origins,
                chunk=chunk,
            )
        )

    hits.sort(
        key=lambda hit: hit.rank
    )

    return tuple(hits)


def _evaluate_budget(
    *,
    max_evidences: int,
    max_chars: int,
    case_hits: dict[
        str,
        tuple[
            CompetitionContextExpandedHit,
            ...,
        ],
    ],
    source_records: tuple,
    gold_records: tuple,
) -> EvidenceAssemblyMetrics:
    selected_by_case: dict[
        str,
        set[str],
    ] = {}

    total_evidences = 0
    total_chars = 0

    config = (
        CompetitionEvidenceAssemblyConfig(
            max_evidences=max_evidences,
            max_chars=max_chars,
        )
    )

    for (
        case_id,
        hits,
    ) in case_hits.items():
        bundle = (
            assemble_competition_evidence_bundle(
                hits=hits,
                source_records=source_records,
                config=config,
            )
        )

        evidence_ids = {
            evidence.evidence_id
            for evidence
            in bundle.evidences
        }

        selected_by_case[
            case_id
        ] = evidence_ids

        total_evidences += len(
            bundle.evidences
        )

        total_chars += sum(
            len(
                evidence.raw_content
            )
            for evidence
            in bundle.evidences
        )

    direct_hit_count = 0
    complete_gold_count = 0
    recall_sum = 0.0

    mode_fact_counts = Counter()
    mode_complete_counts = Counter()

    for record in gold_records:
        selected_ids = (
            selected_by_case.get(
                record.case_id
            )
        )

        if selected_ids is None:
            raise RuntimeError(
                "Evidence Assembly 缺少 "
                f"Case：{record.case_id}"
            )

        direct_ids = {
            reference.chunk_id
            for reference
            in record.gold_chunks
            if reference.role == "direct"
        }

        gold_ids = {
            reference.chunk_id
            for reference
            in record.gold_chunks
        }

        direct_hit = bool(
            direct_ids
            & selected_ids
        )

        complete = (
            gold_ids
            <= selected_ids
        )

        recovered = len(
            gold_ids
            & selected_ids
        )

        recall = (
            recovered
            / len(gold_ids)
        )

        direct_hit_count += int(
            direct_hit
        )

        complete_gold_count += int(
            complete
        )

        recall_sum += recall

        evidence_mode = (
            record.evidence_mode
        )

        if evidence_mode is None:
            raise RuntimeError(
                "Confirmed Gold 缺少 "
                "evidence_mode"
            )

        mode_fact_counts[
            evidence_mode
        ] += 1

        if complete:
            mode_complete_counts[
                evidence_mode
            ] += 1

    fact_count = len(
        gold_records
    )

    case_count = len(
        case_hits
    )

    complete_by_mode = {
        mode: (
            mode_complete_counts[
                mode
            ]
        )
        for mode
        in sorted(
            mode_fact_counts
        )
    }

    return EvidenceAssemblyMetrics(
        max_evidences=max_evidences,
        max_chars=max_chars,
        fact_count=fact_count,
        direct_hit_count=(
            direct_hit_count
        ),
        complete_gold_count=(
            complete_gold_count
        ),
        direct_hit_rate=(
            direct_hit_count
            / fact_count
        ),
        complete_gold_rate=(
            complete_gold_count
            / fact_count
        ),
        mean_gold_recall=(
            recall_sum
            / fact_count
        ),
        mean_evidences_per_case=(
            total_evidences
            / case_count
        ),
        mean_chars_per_case=(
            total_chars
            / case_count
        ),
        total_evidences=(
            total_evidences
        ),
        total_chars=total_chars,
        complete_by_mode=(
            complete_by_mode
        ),
    )


def _best_result(
    results: tuple[
        EvidenceAssemblyMetrics,
        ...,
    ],
) -> EvidenceAssemblyMetrics:
    return max(
        results,
        key=lambda result: (
            result.direct_hit_rate,
            result.complete_gold_rate,
            result.mean_gold_recall,
            -result.mean_chars_per_case,
            -result.mean_evidences_per_case,
        ),
    )


def main() -> None:
    retrieval = _load_json(
        RETRIEVAL_RESULT_PATH
    )

    corpus = (
        load_competition_chunk_corpus(
            CORPUS_PATH
        )
    )

    gold = (
        load_competition_gold_dataset(
            GOLD_PATH,
            expected_fact_count=97,
        )
    )

    if (
        retrieval.get("corpus_id")
        != corpus.manifest.corpus_id
        or gold.corpus_id
        != corpus.manifest.corpus_id
    ):
        raise RuntimeError(
            "Retrieval / Gold / Corpus "
            "身份不一致"
        )

    raw_cases = retrieval.get(
        "cases"
    )

    if not isinstance(
        raw_cases,
        list,
    ):
        raise RuntimeError(
            "Retrieval Result 缺少 cases"
        )

    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunk in corpus.chunks
    }

    case_hits: dict[
        str,
        tuple[
            CompetitionContextExpandedHit,
            ...,
        ],
    ] = {}

    for raw_case in raw_cases:
        if not isinstance(
            raw_case,
            dict,
        ):
            raise RuntimeError(
                "Retrieval Case 类型无效"
            )

        case_id = raw_case.get(
            "case_id"
        )

        if not isinstance(
            case_id,
            str,
        ):
            raise RuntimeError(
                "Retrieval Case 缺少 case_id"
            )

        case_hits[
            case_id
        ] = _rebuild_hits(
            case_payload=raw_case,
            chunks_by_id=chunks_by_id,
        )

    if len(case_hits) != 69:
        raise RuntimeError(
            "Frozen Dev Case 数量改变："
            f"{len(case_hits)}"
        )

    results = tuple(
        _evaluate_budget(
            max_evidences=max_evidences,
            max_chars=max_chars,
            case_hits=case_hits,
            source_records=(
                corpus.manifest
                .source_records
            ),
            gold_records=(
                gold.records
            ),
        )
        for (
            max_evidences,
            max_chars,
        ) in BUDGETS
    )

    best = _best_result(
        results
    )

    print()
    print(
        "===== Competition Evidence "
        "Assembly Dev ====="
    )

    print(
        "Cases:",
        len(case_hits),
    )

    print(
        "Gold facts:",
        len(gold.records),
    )

    print()

    for result in results:
        print(
            f"E={result.max_evidences:2d} "
            f"Chars={result.max_chars:4d} | "
            f"Direct="
            f"{result.direct_hit_count}/"
            f"{result.fact_count} "
            f"({result.direct_hit_rate:.4f}) | "
            f"Complete="
            f"{result.complete_gold_count}/"
            f"{result.fact_count} "
            f"({result.complete_gold_rate:.4f}) | "
            f"Recall="
            f"{result.mean_gold_recall:.4f} | "
            f"MeanE="
            f"{result.mean_evidences_per_case:.2f} | "
            f"MeanChars="
            f"{result.mean_chars_per_case:.1f}"
        )

        print(
            "    Complete by mode:",
            result.complete_by_mode,
        )

    print()
    print("BEST CANDIDATE")
    print(
        "  max_evidences:",
        best.max_evidences,
    )
    print(
        "  max_chars:",
        best.max_chars,
    )
    print(
        "  Direct:",
        f"{best.direct_hit_count}/"
        f"{best.fact_count}",
    )
    print(
        "  Complete:",
        f"{best.complete_gold_count}/"
        f"{best.fact_count}",
    )
    print(
        "  Recall:",
        f"{best.mean_gold_recall:.4f}",
    )
    print(
        "  Mean evidences:",
        f"{best.mean_evidences_per_case:.2f}",
    )
    print(
        "  Mean chars:",
        f"{best.mean_chars_per_case:.1f}",
    )

    payload = {
        "schema_version": 1,
        "evaluation_version": (
            "competition_evidence_"
            "assembly_dev_v1"
        ),
        "corpus_id": (
            corpus.manifest.corpus_id
        ),
        "fact_count": len(
            gold.records
        ),
        "case_count": len(
            case_hits
        ),
        "retrieval_result": str(
            RETRIEVAL_RESULT_PATH
        ),
        "results": [
            asdict(result)
            for result
            in results
        ],
        "best_candidate": (
            asdict(best)
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()