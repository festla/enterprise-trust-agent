from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from app.schemas.competition_chunk import (
    CompetitionTextChunk,
)
from app.schemas.competition_gold import (
    CompetitionGoldFactRecord,
)
from app.services.competition_corpus import (
    load_competition_chunk_corpus,
)
from scripts.evaluate_competition_bm25_gold_dev import (
    load_competition_gold_dataset,
)


DEFAULT_GOLD_FILE = Path(
    "data/competition/processed/eval/"
    "competition_dev_gold_chunks_v1.json"
)

DEFAULT_CONTEXT_EVAL_FILE = Path(
    "data/competition/processed/eval/"
    "competition_bm25_context_propagation_dev_v1.json"
)

DEFAULT_OUTPUT_FILE = Path(
    "data/competition/processed/eval/"
    "competition_parent_child_failure_diagnosis_dev_v1.json"
)

EXPECTED_CONTEXT_EVAL_VERSION = (
    "competition_bm25_context_expansion_dev_v1"
)

DEFAULT_EXPECTED_FACTS = 97


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


def _load_eval_cases(
    payload: dict[str, object],
) -> dict[
    str,
    dict[str, object],
]:
    if (
        payload.get(
            "evaluation_version"
        )
        != EXPECTED_CONTEXT_EVAL_VERSION
    ):
        raise RuntimeError(
            "Context Expansion Eval "
            "版本不受支持"
        )

    raw_cases = payload.get(
        "cases"
    )

    if not isinstance(
        raw_cases,
        list,
    ):
        raise RuntimeError(
            "Context Expansion Eval 缺少 cases"
        )

    result: dict[
        str,
        dict[str, object],
    ] = {}

    for raw_case in raw_cases:
        if not isinstance(
            raw_case,
            dict,
        ):
            raise RuntimeError(
                "Context Expansion Eval "
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
                "Context Case 缺少 case_id"
            )

        if case_id in result:
            raise RuntimeError(
                "Context Eval 包含重复 Case："
                f"{case_id}"
            )

        result[
            case_id
        ] = raw_case

    return result


def _load_hit_map(
    raw_case: dict[str, object],
    *,
    field_name: str,
) -> dict[
    str,
    dict[str, object],
]:
    raw_hits = raw_case.get(
        field_name
    )

    if not isinstance(
        raw_hits,
        list,
    ):
        raise RuntimeError(
            f"Case 缺少 {field_name}"
        )

    result: dict[
        str,
        dict[str, object],
    ] = {}

    for raw_hit in raw_hits:
        if not isinstance(
            raw_hit,
            dict,
        ):
            raise RuntimeError(
                f"{field_name} 包含无效 Hit"
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
                f"{field_name} Hit 缺少 chunk_id"
            )

        if chunk_id in result:
            raise RuntimeError(
                f"{field_name} 包含重复 chunk_id："
                f"{chunk_id}"
            )

        result[
            chunk_id
        ] = raw_hit

    return result


def _strict_item_prefix(
    prefix: tuple[str, ...],
    value: tuple[str, ...],
) -> bool:
    if not prefix:
        return False

    if (
        len(prefix)
        >= len(value)
    ):
        return False

    return (
        value[:len(prefix)]
        == prefix
    )


def _item_depth_relation(
    left: CompetitionTextChunk,
    right: CompetitionTextChunk,
) -> int | None:
    """
    如果两个 Chunk 存在 Item ancestor/descendant
    关系，返回 item_path 的层级深度差。

    否则返回 None。
    """

    if (
        left.source_id
        != right.source_id
    ):
        return None

    if _strict_item_prefix(
        left.item_path,
        right.item_path,
    ):
        return (
            len(right.item_path)
            - len(left.item_path)
        )

    if _strict_item_prefix(
        right.item_path,
        left.item_path,
    ):
        return (
            len(left.item_path)
            - len(right.item_path)
        )

    return None


def _minimum_seed_distance(
    *,
    target: CompetitionTextChunk,
    seed_chunks: tuple[
        CompetitionTextChunk,
        ...,
    ],
) -> int | None:
    distances = tuple(
        abs(
            target.chunk_index
            - seed.chunk_index
        )
        for seed in seed_chunks
        if (
            seed.source_id
            == target.source_id
        )
    )

    if not distances:
        return None

    return min(
        distances
    )


def _origin_relations(
    raw_hit: dict[str, object],
) -> tuple[str, ...]:
    raw_origins = raw_hit.get(
        "origins"
    )

    if not isinstance(
        raw_origins,
        list,
    ):
        return ()

    relations = []

    for raw_origin in raw_origins:
        if not isinstance(
            raw_origin,
            dict,
        ):
            continue

        relation = raw_origin.get(
            "relation"
        )

        if isinstance(
            relation,
            str,
        ):
            relations.append(
                relation
            )

    return tuple(
        relations
    )


def _origin_classes(
    raw_hit: dict[str, object],
) -> tuple[str, ...]:
    relations = set(
        _origin_relations(
            raw_hit
        )
    )

    classes = []

    if "seed" in relations:
        classes.append(
            "seed"
        )

    if relations & {
        "previous_neighbor",
        "next_neighbor",
    }:
        classes.append(
            "adjacent"
        )

    if relations & {
        "item_parent",
        "item_child",
    }:
        classes.append(
            "item_hierarchy"
        )

    return tuple(
        classes
    )


def _adjacent_context_ids(
    expanded_hits: dict[
        str,
        dict[str, object],
    ],
) -> set[str]:
    result: set[str] = set()

    for (
        chunk_id,
        raw_hit,
    ) in expanded_hits.items():
        relations = set(
            _origin_relations(
                raw_hit
            )
        )

        if relations & {
            "previous_neighbor",
            "next_neighbor",
        }:
            result.add(
                chunk_id
            )

    return result


def _can_reach_by_item_depth(
    *,
    target: CompetitionTextChunk,
    triggers: tuple[
        CompetitionTextChunk,
        ...,
    ],
    max_depth_delta: int,
) -> bool:
    for trigger in triggers:
        depth = _item_depth_relation(
            trigger,
            target,
        )

        if (
            depth is not None
            and depth
            <= max_depth_delta
        ):
            return True

    return False


def _can_reach_by_one_more_item_hop(
    *,
    target: CompetitionTextChunk,
    expanded_chunks: tuple[
        CompetitionTextChunk,
        ...,
    ],
) -> bool:
    """
    判断：

        当前 Expanded Context
            ↓
        再允许一轮 depth=1 item hierarchy

    是否能触达 target。

    如果为 True，说明第二跳 hierarchy
    理论上可能解决该 Missing Gold Chunk。
    """

    return _can_reach_by_item_depth(
        target=target,
        triggers=expanded_chunks,
        max_depth_delta=1,
    )


def _fact_payload(
    *,
    record: CompetitionGoldFactRecord,
    seed_hits: dict[
        str,
        dict[str, object],
    ],
    expanded_hits: dict[
        str,
        dict[str, object],
    ],
    chunks_by_id: dict[
        str,
        CompetitionTextChunk,
    ],
) -> dict[str, object]:
    seed_ids = set(
        seed_hits
    )

    expanded_ids = set(
        expanded_hits
    )

    adjacent_ids = (
        _adjacent_context_ids(
            expanded_hits
        )
    )

    trigger_ids = (
        seed_ids
        | adjacent_ids
    )

    seed_chunks = tuple(
        chunks_by_id[
            chunk_id
        ]
        for chunk_id in seed_ids
    )

    current_trigger_chunks = tuple(
        chunks_by_id[
            chunk_id
        ]
        for chunk_id in trigger_ids
    )

    expanded_chunks = tuple(
        chunks_by_id[
            chunk_id
        ]
        for chunk_id in expanded_ids
    )

    gold_ids = {
        reference.chunk_id
        for reference
        in record.gold_chunks
    }

    retrieved_ids = (
        gold_ids
        & expanded_ids
    )

    missing_ids = (
        gold_ids
        - expanded_ids
    )

    gold_details = []

    missing_details = []

    for reference in record.gold_chunks:
        chunk = chunks_by_id.get(
            reference.chunk_id
        )

        if chunk is None:
            raise RuntimeError(
                "Gold 引用了 Corpus 中不存在的 "
                "Chunk："
                f"{reference.chunk_id}"
            )

        retrieved = (
            reference.chunk_id
            in expanded_ids
        )

        raw_hit = (
            expanded_hits.get(
                reference.chunk_id
            )
        )

        if raw_hit is None:
            origin_classes: tuple[
                str,
                ...
            ] = ()

            origin_relations: tuple[
                str,
                ...
            ] = ()
        else:
            origin_classes = (
                _origin_classes(
                    raw_hit
                )
            )

            origin_relations = (
                _origin_relations(
                    raw_hit
                )
            )

        detail = {
            "chunk_id": (
                reference.chunk_id
            ),
            "role": (
                reference.role
            ),
            "retrieved": (
                retrieved
            ),
            "chunk_index": (
                chunk.chunk_index
            ),
            "item_path": list(
                chunk.item_path
            ),
            "section_path": list(
                chunk.section_path
            ),
            "article": (
                chunk.article
            ),
            "origin_classes": list(
                origin_classes
            ),
            "origin_relations": list(
                origin_relations
            ),
        }

        gold_details.append(
            detail
        )

        if retrieved:
            continue

        min_seed_distance = (
            _minimum_seed_distance(
                target=chunk,
                seed_chunks=seed_chunks,
            )
        )

        one_more_item_hop = (
            _can_reach_by_one_more_item_hop(
                target=chunk,
                expanded_chunks=(
                    expanded_chunks
                ),
            )
        )

        depth_two_from_current_trigger = (
            _can_reach_by_item_depth(
                target=chunk,
                triggers=(
                    current_trigger_chunks
                ),
                max_depth_delta=2,
            )
        )

        depth_three_from_current_trigger = (
            _can_reach_by_item_depth(
                target=chunk,
                triggers=(
                    current_trigger_chunks
                ),
                max_depth_delta=3,
            )
        )

        missing_details.append(
            {
                **detail,
                "min_seed_chunk_distance": (
                    min_seed_distance
                ),
                "within_seed_window_2": (
                    min_seed_distance
                    is not None
                    and min_seed_distance <= 2
                ),
                "within_seed_window_3": (
                    min_seed_distance
                    is not None
                    and min_seed_distance <= 3
                ),
                "reachable_by_one_more_item_hop": (
                    one_more_item_hop
                ),
                "reachable_by_depth_2_from_seed_or_adjacent": (
                    depth_two_from_current_trigger
                ),
                "reachable_by_depth_3_from_seed_or_adjacent": (
                    depth_three_from_current_trigger
                ),
            }
        )

    recoverable_one_more = (
        bool(missing_details)
        and all(
            bool(
                detail[
                    "reachable_by_one_more_item_hop"
                ]
            )
            for detail
            in missing_details
        )
    )

    recoverable_depth_two = (
        bool(missing_details)
        and all(
            bool(
                detail[
                    "reachable_by_depth_2_from_seed_or_adjacent"
                ]
            )
            for detail
            in missing_details
        )
    )

    recoverable_depth_three = (
        bool(missing_details)
        and all(
            bool(
                detail[
                    "reachable_by_depth_3_from_seed_or_adjacent"
                ]
            )
            for detail
            in missing_details
        )
    )

    recoverable_window_two = (
        bool(missing_details)
        and all(
            bool(
                detail[
                    "within_seed_window_2"
                ]
            )
            for detail
            in missing_details
        )
    )

    recoverable_window_three = (
        bool(missing_details)
        and all(
            bool(
                detail[
                    "within_seed_window_3"
                ]
            )
            for detail
            in missing_details
        )
    )

    return {
        "case_id": (
            record.case_id
        ),
        "fact_index": (
            record.fact_index
        ),
        "fact": (
            record.fact
        ),
        "gold_chunk_count": (
            len(record.gold_chunks)
        ),
        "retrieved_gold_count": (
            len(retrieved_ids)
        ),
        "missing_gold_count": (
            len(missing_ids)
        ),
        "gold_chunks": (
            gold_details
        ),
        "missing_chunks": (
            missing_details
        ),
        "recoverable_by_one_more_item_hop": (
            recoverable_one_more
        ),
        "recoverable_by_depth_2_from_seed_or_adjacent": (
            recoverable_depth_two
        ),
        "recoverable_by_depth_3_from_seed_or_adjacent": (
            recoverable_depth_three
        ),
        "recoverable_by_seed_window_2": (
            recoverable_window_two
        ),
        "recoverable_by_seed_window_3": (
            recoverable_window_three
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "诊断 Competition parent-child "
            "Context Expansion 失败原因"
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
        default=DEFAULT_CONTEXT_EVAL_FILE,
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

    arguments = (
        parser.parse_args()
    )

    gold_dataset = (
        load_competition_gold_dataset(
            arguments.gold,
            expected_fact_count=(
                DEFAULT_EXPECTED_FACTS
            ),
        )
    )

    corpus = (
        load_competition_chunk_corpus(
            arguments.corpus
        )
    )

    context_payload = (
        _load_json_object(
            arguments.context_eval,
            label=(
                "Context Expansion Eval"
            ),
        )
    )

    cases = _load_eval_cases(
        context_payload
    )

    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunk in corpus.chunks
    }

    parent_child_records = tuple(
        record
        for record
        in gold_dataset.records
        if (
            record.review_status
            == "confirmed"
            and record.evidence_mode
            == "parent_child"
        )
    )

    incomplete_payloads = []

    retrieved_role_counts = Counter()
    missing_role_counts = Counter()

    retrieved_origin_class_counts = Counter()

    complete_count = 0

    incomplete_with_some_gold = 0
    incomplete_with_no_gold = 0

    recoverable_one_more_count = 0
    recoverable_depth_two_count = 0
    recoverable_depth_three_count = 0
    recoverable_window_two_count = 0
    recoverable_window_three_count = 0

    for record in parent_child_records:
        raw_case = cases.get(
            record.case_id
        )

        if raw_case is None:
            raise RuntimeError(
                "Context Eval 缺少 Gold Case："
                f"{record.case_id}"
            )

        seed_hits = _load_hit_map(
            raw_case,
            field_name="seed_hits",
        )

        expanded_hits = _load_hit_map(
            raw_case,
            field_name="expanded_hits",
        )

        gold_ids = {
            reference.chunk_id
            for reference
            in record.gold_chunks
        }

        expanded_ids = set(
            expanded_hits
        )

        if (
            gold_ids
            <= expanded_ids
        ):
            complete_count += 1
            continue

        payload = _fact_payload(
            record=record,
            seed_hits=seed_hits,
            expanded_hits=(
                expanded_hits
            ),
            chunks_by_id=(
                chunks_by_id
            ),
        )

        incomplete_payloads.append(
            payload
        )

        retrieved_count = int(
            payload[
                "retrieved_gold_count"
            ]
        )

        if retrieved_count > 0:
            incomplete_with_some_gold += 1
        else:
            incomplete_with_no_gold += 1

        for detail in payload[
            "gold_chunks"
        ]:
            assert isinstance(
                detail,
                dict,
            )

            role = detail[
                "role"
            ]

            retrieved = detail[
                "retrieved"
            ]

            assert isinstance(
                role,
                str,
            )

            if retrieved:
                retrieved_role_counts[
                    role
                ] += 1

                raw_classes = detail.get(
                    "origin_classes",
                    [],
                )

                if isinstance(
                    raw_classes,
                    list,
                ):
                    for origin_class in (
                        raw_classes
                    ):
                        if isinstance(
                            origin_class,
                            str,
                        ):
                            (
                                retrieved_origin_class_counts[
                                    origin_class
                                ]
                            ) += 1

            else:
                missing_role_counts[
                    role
                ] += 1

        if bool(
            payload[
                "recoverable_by_one_more_item_hop"
            ]
        ):
            recoverable_one_more_count += 1

        if bool(
            payload[
                "recoverable_by_depth_2_from_seed_or_adjacent"
            ]
        ):
            recoverable_depth_two_count += 1

        if bool(
            payload[
                "recoverable_by_depth_3_from_seed_or_adjacent"
            ]
        ):
            recoverable_depth_three_count += 1

        if bool(
            payload[
                "recoverable_by_seed_window_2"
            ]
        ):
            recoverable_window_two_count += 1

        if bool(
            payload[
                "recoverable_by_seed_window_3"
            ]
        ):
            recoverable_window_three_count += 1

    incomplete_count = (
        len(incomplete_payloads)
    )

    unresolved_by_tested_extensions = sum(
        not any(
            (
                bool(
                    payload[
                        "recoverable_by_one_more_item_hop"
                    ]
                ),
                bool(
                    payload[
                        "recoverable_by_depth_2_from_seed_or_adjacent"
                    ]
                ),
                bool(
                    payload[
                        "recoverable_by_depth_3_from_seed_or_adjacent"
                    ]
                ),
                bool(
                    payload[
                        "recoverable_by_seed_window_2"
                    ]
                ),
                bool(
                    payload[
                        "recoverable_by_seed_window_3"
                    ]
                ),
            )
        )
        for payload
        in incomplete_payloads
    )

    result_payload = {
        "schema_version": 1,
        "diagnosis_version": (
            "competition_parent_child_"
            "failure_diagnosis_dev_v1"
        ),
        "corpus_id": (
            corpus.manifest.corpus_id
        ),
        "source_context_eval": (
            str(
                arguments.context_eval
            )
        ),
        "summary": {
            "parent_child_fact_count": (
                len(
                    parent_child_records
                )
            ),
            "complete_fact_count": (
                complete_count
            ),
            "incomplete_fact_count": (
                incomplete_count
            ),
            "incomplete_with_some_gold": (
                incomplete_with_some_gold
            ),
            "incomplete_with_no_gold": (
                incomplete_with_no_gold
            ),
            "retrieved_gold_role_counts": dict(
                sorted(
                    retrieved_role_counts.items()
                )
            ),
            "missing_gold_role_counts": dict(
                sorted(
                    missing_role_counts.items()
                )
            ),
            "retrieved_gold_origin_class_counts": dict(
                sorted(
                    retrieved_origin_class_counts.items()
                )
            ),
            "recoverable_by_one_more_item_hop": (
                recoverable_one_more_count
            ),
            "recoverable_by_depth_2_from_seed_or_adjacent": (
                recoverable_depth_two_count
            ),
            "recoverable_by_depth_3_from_seed_or_adjacent": (
                recoverable_depth_three_count
            ),
            "recoverable_by_seed_window_2": (
                recoverable_window_two_count
            ),
            "recoverable_by_seed_window_3": (
                recoverable_window_three_count
            ),
            "unresolved_by_tested_extensions": (
                unresolved_by_tested_extensions
            ),
        },
        "incomplete_facts": (
            incomplete_payloads
        ),
    }

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    arguments.output.write_text(
        json.dumps(
            result_payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "=== Competition Parent-Child "
        "Failure Diagnosis ==="
    )

    print(
        "Corpus ID:",
        corpus.manifest.corpus_id,
    )

    print()

    print(
        "Parent-child facts:",
        len(
            parent_child_records
        ),
    )

    print(
        "Complete:",
        complete_count,
    )

    print(
        "Incomplete:",
        incomplete_count,
    )

    print(
        "Incomplete with some Gold:",
        incomplete_with_some_gold,
    )

    print(
        "Incomplete with no Gold:",
        incomplete_with_no_gold,
    )

    print()

    print(
        "Retrieved Gold roles:",
        dict(
            sorted(
                retrieved_role_counts.items()
            )
        ),
    )

    print(
        "Missing Gold roles:",
        dict(
            sorted(
                missing_role_counts.items()
            )
        ),
    )

    print()

    print(
        "Retrieved Gold origin classes:",
        dict(
            sorted(
                retrieved_origin_class_counts.items()
            )
        ),
    )

    print()

    print(
        "Fact fully recoverable by "
        "ONE MORE item hop:",
        recoverable_one_more_count,
    )

    print(
        "Fact fully recoverable by "
        "depth<=2 from seed/adjacent:",
        recoverable_depth_two_count,
    )

    print(
        "Fact fully recoverable by "
        "depth<=3 from seed/adjacent:",
        recoverable_depth_three_count,
    )

    print(
        "Fact fully recoverable by "
        "seed window ±2:",
        recoverable_window_two_count,
    )

    print(
        "Fact fully recoverable by "
        "seed window ±3:",
        recoverable_window_three_count,
    )

    print()

    print(
        "Still unresolved by tested "
        "extensions:",
        unresolved_by_tested_extensions,
    )

    print()

    print(
        "Saved:",
        arguments.output,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )