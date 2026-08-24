from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path

from app.schemas.competition_gold import (
    CompetitionGoldFactRecord,
)
from app.services.competition_gold_retrieval_eval import (
    CompetitionGoldChunkMatch,
    CompetitionGoldFactRetrievalResult,
    summarize_competition_gold_retrieval,
)
from scripts.evaluate_competition_bm25_gold_dev import (
    load_competition_gold_dataset,
)


DEFAULT_GOLD_FILE = Path(
    "data/competition/processed/eval/"
    "competition_dev_gold_chunks_v1.json"
)

DEFAULT_CONTEXT_FILE = Path(
    "data/competition/processed/eval/"
    "competition_bm25_context_propagation_depth2_dev_v1.json"
)

DEFAULT_OUTPUT_FILE = Path(
    "data/competition/processed/eval/"
    "competition_context_relation_ablation_dev_v1.json"
)

DEFAULT_CUTOFFS = (
    1,
    3,
    5,
    10,
)

EXPECTED_FACT_COUNT = 97


VARIANTS: dict[
    str,
    frozenset[str],
] = {
    #
    # 6B2a：
    # Seed + ±1
    #
    "adjacent_only": frozenset(
        {
            "seed",
            "previous_neighbor",
            "next_neighbor",
        }
    ),

    #
    # 我们现在最关心的：
    #
    # Seed + ±1 + Item Child
    #
    "item_child_only": frozenset(
        {
            "seed",
            "previous_neighbor",
            "next_neighbor",
            "item_child",
        }
    ),

    #
    # 用来判断 item_parent 本身贡献多少。
    #
    "item_parent_only": frozenset(
        {
            "seed",
            "previous_neighbor",
            "next_neighbor",
            "item_parent",
        }
    ),

    #
    # 当前完整 6B2e。
    #
    "full": frozenset(
        {
            "seed",
            "previous_neighbor",
            "next_neighbor",
            "item_parent",
            "item_child",
        }
    ),
}


def _load_json_object(
    path: Path,
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
            f"无法读取：{path}"
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "JSON 根节点必须为 Object"
        )

    return payload


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
    grouped: dict[
        str,
        list[
            CompetitionGoldFactRecord
        ],
    ] = defaultdict(
        list
    )

    for record in records:
        grouped[
            record.case_id
        ].append(
            record
        )

    return {
        case_id: tuple(
            case_records
        )
        for (
            case_id,
            case_records,
        ) in grouped.items()
    }


def _load_cases(
    payload: dict[str, object],
) -> dict[
    str,
    dict[str, object],
]:
    raw_cases = payload.get(
        "cases"
    )

    if not isinstance(
        raw_cases,
        list,
    ):
        raise RuntimeError(
            "Context Eval 缺少 cases"
        )

    result = {}

    for raw_case in raw_cases:
        if not isinstance(
            raw_case,
            dict,
        ):
            raise RuntimeError(
                "包含无效 Case"
            )

        case_id = raw_case.get(
            "case_id"
        )

        if (
            not isinstance(
                case_id,
                str,
            )
            or not case_id
        ):
            raise RuntimeError(
                "Case 缺少 case_id"
            )

        if case_id in result:
            raise RuntimeError(
                f"重复 Case：{case_id}"
            )

        result[
            case_id
        ] = raw_case

    return result


def _filtered_rank_map(
    raw_case: dict[str, object],
    *,
    allowed_relations: frozenset[str],
) -> tuple[
    dict[str, int],
    int,
]:
    """
    根据允许的 origin relation，
    重新计算每个 Chunk 的 effective_seed_rank。

    不能直接复用 JSON 里的 effective_seed_rank，
    因为过滤某些 origins 后，
    最小 seed_rank 可能发生变化。
    """

    raw_hits = raw_case.get(
        "expanded_hits"
    )

    if not isinstance(
        raw_hits,
        list,
    ):
        raise RuntimeError(
            "Case 缺少 expanded_hits"
        )

    rank_by_chunk_id: dict[
        str,
        int,
    ] = {}

    for raw_hit in raw_hits:
        if not isinstance(
            raw_hit,
            dict,
        ):
            raise RuntimeError(
                "包含无效 expanded hit"
            )

        chunk_id = raw_hit.get(
            "chunk_id"
        )

        if (
            not isinstance(
                chunk_id,
                str,
            )
            or not chunk_id
        ):
            raise RuntimeError(
                "Expanded Hit 缺少 chunk_id"
            )

        raw_origins = raw_hit.get(
            "origins"
        )

        if not isinstance(
            raw_origins,
            list,
        ):
            raise RuntimeError(
                "Expanded Hit 缺少 origins"
            )

        kept_ranks = []

        for raw_origin in raw_origins:
            if not isinstance(
                raw_origin,
                dict,
            ):
                raise RuntimeError(
                    "包含无效 Origin"
                )

            relation = raw_origin.get(
                "relation"
            )

            seed_rank = raw_origin.get(
                "seed_rank"
            )

            if (
                not isinstance(
                    relation,
                    str,
                )
                or isinstance(
                    seed_rank,
                    bool,
                )
                or not isinstance(
                    seed_rank,
                    int,
                )
                or seed_rank < 1
            ):
                raise RuntimeError(
                    "Origin relation/seed_rank 无效"
                )

            if (
                relation
                in allowed_relations
            ):
                kept_ranks.append(
                    seed_rank
                )

        if not kept_ranks:
            continue

        rank_by_chunk_id[
            chunk_id
        ] = min(
            kept_ranks
        )

    return (
        rank_by_chunk_id,
        len(
            rank_by_chunk_id
        ),
    )


def _evaluate_fact(
    *,
    record: CompetitionGoldFactRecord,
    rank_by_chunk_id: dict[
        str,
        int,
    ],
    retrieved_count: int,
) -> CompetitionGoldFactRetrievalResult:
    chunk_matches = tuple(
        CompetitionGoldChunkMatch(
            chunk_id=(
                reference.chunk_id
            ),
            role=(
                reference.role
            ),
            rank=(
                rank_by_chunk_id.get(
                    reference.chunk_id
                )
            ),
        )
        for reference
        in record.gold_chunks
    )

    return (
        CompetitionGoldFactRetrievalResult(
            case_id=record.case_id,
            fact_index=(
                record.fact_index
            ),
            fact=record.fact,
            expected_source_id=(
                record.expected_source_id
            ),
            evidence_mode=(
                record.evidence_mode
            ),
            retrieved_count=(
                retrieved_count
            ),
            chunk_matches=(
                chunk_matches
            ),
        )
    )


def _summary_payload(
    results: tuple[
        CompetitionGoldFactRetrievalResult,
        ...,
    ],
) -> dict[str, object]:
    summary = (
        summarize_competition_gold_retrieval(
            results,
            cutoffs=(
                DEFAULT_CUTOFFS
            ),
        )
    )

    return asdict(
        summary
    )


def _breakdown_payload(
    results: tuple[
        CompetitionGoldFactRetrievalResult,
        ...,
    ],
) -> dict[str, object]:
    result = {}

    for evidence_mode in (
        "single_chunk",
        "parent_child",
        "multi_chunk",
    ):
        group = tuple(
            item
            for item in results
            if (
                item.evidence_mode
                == evidence_mode
            )
        )

        summary = (
            summarize_competition_gold_retrieval(
                group,
                cutoffs=(
                    DEFAULT_CUTOFFS
                ),
            )
        )

        result[
            evidence_mode
        ] = asdict(
            summary
        )

    return result


def _metric_at(
    summary,
    cutoff: int,
):
    return summary.at(
        cutoff
    )


def _format_rate(
    value: float,
) -> str:
    return (
        f"{value:.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "对 Competition depth=2 "
            "Context Expansion 做 relation ablation"
        )
    )

    parser.add_argument(
        "--gold",
        type=Path,
        default=DEFAULT_GOLD_FILE,
    )

    parser.add_argument(
        "--context-eval",
        type=Path,
        default=DEFAULT_CONTEXT_FILE,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
    )

    args = (
        parser.parse_args()
    )

    gold = (
        load_competition_gold_dataset(
            args.gold,
            expected_fact_count=(
                EXPECTED_FACT_COUNT
            ),
        )
    )

    payload = (
        _load_json_object(
            args.context_eval
        )
    )

    cases = _load_cases(
        payload
    )

    gold_by_case = (
        _group_gold_records(
            gold.records
        )
    )

    if (
        set(cases)
        != set(gold_by_case)
    ):
        raise RuntimeError(
            "Context Eval Case 与 Gold Case 不一致"
        )

    raw_seed_count = 0

    for raw_case in cases.values():
        raw_seed_hits = raw_case.get(
            "seed_hits"
        )

        if not isinstance(
            raw_seed_hits,
            list,
        ):
            raise RuntimeError(
                "Case 缺少 seed_hits"
            )

        raw_seed_count += len(
            raw_seed_hits
        )

    variant_payloads = {}

    print(
        "=== Competition Context "
        "Relation Ablation ==="
    )

    print(
        "Source:",
        args.context_eval,
    )

    print(
        "Gold facts:",
        len(
            gold.records
        ),
    )

    print(
        "Seed chunks:",
        raw_seed_count,
    )

    print()

    for (
        variant_name,
        allowed_relations,
    ) in VARIANTS.items():
        all_results = []

        total_context_chunks = 0

        for case_id in sorted(
            cases
        ):
            (
                rank_by_chunk_id,
                retrieved_count,
            ) = _filtered_rank_map(
                cases[
                    case_id
                ],
                allowed_relations=(
                    allowed_relations
                ),
            )

            total_context_chunks += (
                retrieved_count
            )

            for record in (
                gold_by_case[
                    case_id
                ]
            ):
                all_results.append(
                    _evaluate_fact(
                        record=record,
                        rank_by_chunk_id=(
                            rank_by_chunk_id
                        ),
                        retrieved_count=(
                            retrieved_count
                        ),
                    )
                )

        result_tuple = tuple(
            all_results
        )

        summary = (
            summarize_competition_gold_retrieval(
                result_tuple,
                cutoffs=(
                    DEFAULT_CUTOFFS
                ),
            )
        )

        metric_10 = (
            _metric_at(
                summary,
                10,
            )
        )

        breakdowns = {}

        for evidence_mode in (
            "single_chunk",
            "parent_child",
            "multi_chunk",
        ):
            mode_results = tuple(
                result
                for result
                in result_tuple
                if (
                    result.evidence_mode
                    == evidence_mode
                )
            )

            mode_summary = (
                summarize_competition_gold_retrieval(
                    mode_results,
                    cutoffs=(
                        DEFAULT_CUTOFFS
                    ),
                )
            )

            mode_metric_10 = (
                _metric_at(
                    mode_summary,
                    10,
                )
            )

            breakdowns[
                evidence_mode
            ] = {
                "fact_count": (
                    len(
                        mode_results
                    )
                ),
                "direct_hit_at_10": (
                    mode_metric_10
                    .fact_direct_hit_rate
                ),
                "complete_gold_at_10": (
                    mode_metric_10
                    .fact_complete_gold_hit_rate
                ),
                "gold_recall_at_10": (
                    mode_metric_10
                    .mean_gold_chunk_recall
                ),
            }

        context_ratio = (
            total_context_chunks
            / raw_seed_count
        )

        variant_payloads[
            variant_name
        ] = {
            "allowed_relations": (
                sorted(
                    allowed_relations
                )
            ),
            "context_chunk_count": (
                total_context_chunks
            ),
            "context_ratio": (
                context_ratio
            ),
            "summary": (
                _summary_payload(
                    result_tuple
                )
            ),
            "breakdowns": (
                breakdowns
            ),
        }

        print(
            variant_name
        )

        print(
            "  Context:",
            total_context_chunks,
            "ratio=",
            _format_rate(
                context_ratio
            ),
        )

        print(
            "  Overall @10:"
        )

        print(
            "    DirectHit:",
            _format_rate(
                metric_10
                .fact_direct_hit_rate
            ),
        )

        print(
            "    CompleteGold:",
            _format_rate(
                metric_10
                .fact_complete_gold_hit_rate
            ),
        )

        print(
            "    GoldRecall:",
            _format_rate(
                metric_10
                .mean_gold_chunk_recall
            ),
        )

        for evidence_mode in (
            "parent_child",
            "multi_chunk",
        ):
            mode = (
                breakdowns[
                    evidence_mode
                ]
            )

            print(
                f"  {evidence_mode} @10:"
            )

            print(
                "    DirectHit:",
                _format_rate(
                    float(
                        mode[
                            "direct_hit_at_10"
                        ]
                    )
                ),
            )

            print(
                "    CompleteGold:",
                _format_rate(
                    float(
                        mode[
                            "complete_gold_at_10"
                        ]
                    )
                ),
            )

            print(
                "    GoldRecall:",
                _format_rate(
                    float(
                        mode[
                            "gold_recall_at_10"
                        ]
                    )
                ),
            )

        print()

    output_payload = {
        "schema_version": 1,
        "ablation_version": (
            "competition_context_"
            "relation_ablation_dev_v1"
        ),
        "source_context_eval": (
            str(
                args.context_eval
            )
        ),
        "seed_chunk_count": (
            raw_seed_count
        ),
        "variants": (
            variant_payloads
        ),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            output_payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "Saved:",
        args.output,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )