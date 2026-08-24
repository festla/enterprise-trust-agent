from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

from app.schemas.competition_chunk import (
    CompetitionTextChunk,
)
from app.schemas.competition_gold import (
    CompetitionGoldChunkReference,
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
    "competition_bm25_context_expansion_dev_v1.json"
)

DEFAULT_OUTPUT_FILE = Path(
    "data/competition/processed/eval/"
    "competition_context_structure_diagnosis_dev_v1.json"
)


@dataclass(frozen=True, slots=True)
class StructuralRelation:
    same_source: bool
    chunk_distance: int | None
    adjacent_1: bool
    adjacent_2: bool
    adjacent_3: bool

    same_section_path: bool

    same_article: bool

    same_item_path: bool
    left_item_prefix_of_right: bool
    right_item_prefix_of_left: bool
    item_hierarchy_related: bool

    same_table: bool


def _is_prefix(
    prefix: tuple[str, ...],
    value: tuple[str, ...],
) -> bool:
    if not prefix:
        return False

    if len(prefix) >= len(value):
        return False

    return (
        value[:len(prefix)]
        == prefix
    )


def describe_relation(
    left: CompetitionTextChunk,
    right: CompetitionTextChunk,
) -> StructuralRelation:
    same_source = (
        left.source_id
        == right.source_id
    )

    if same_source:
        chunk_distance = abs(
            left.chunk_index
            - right.chunk_index
        )
    else:
        chunk_distance = None

    same_section_path = (
        same_source
        and bool(left.section_path)
        and left.section_path
        == right.section_path
    )

    same_article = (
        same_source
        and left.article is not None
        and right.article is not None
        and left.article
        == right.article
    )

    same_item_path = (
        same_source
        and bool(left.item_path)
        and left.item_path
        == right.item_path
    )

    left_prefix = (
        same_source
        and _is_prefix(
            left.item_path,
            right.item_path,
        )
    )

    right_prefix = (
        same_source
        and _is_prefix(
            right.item_path,
            left.item_path,
        )
    )

    same_table = (
        same_source
        and left.chunk_type == "table"
        and right.chunk_type == "table"
        and left.table_index is not None
        and right.table_index is not None
        and left.table_index
        == right.table_index
    )

    return StructuralRelation(
        same_source=same_source,
        chunk_distance=chunk_distance,
        adjacent_1=(
            chunk_distance is not None
            and chunk_distance <= 1
        ),
        adjacent_2=(
            chunk_distance is not None
            and chunk_distance <= 2
        ),
        adjacent_3=(
            chunk_distance is not None
            and chunk_distance <= 3
        ),
        same_section_path=(
            same_section_path
        ),
        same_article=same_article,
        same_item_path=same_item_path,
        left_item_prefix_of_right=(
            left_prefix
        ),
        right_item_prefix_of_left=(
            right_prefix
        ),
        item_hierarchy_related=(
            left_prefix
            or right_prefix
        ),
        same_table=same_table,
    )


def _load_context_eval(
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
            "无法读取 Context Expansion "
            f"评测文件：{path}"
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "Context Expansion "
            "评测根节点必须为 JSON Object"
        )

    if (
        payload.get(
            "evaluation_version"
        )
        != (
            "competition_bm25_"
            "context_expansion_dev_v1"
        )
    ):
        raise RuntimeError(
            "Context Expansion "
            "评测版本不受支持"
        )

    return payload


def _load_expanded_ids_by_case(
    payload: dict[str, object],
) -> dict[
    str,
    set[str],
]:
    raw_cases = payload.get(
        "cases"
    )

    if not isinstance(
        raw_cases,
        list,
    ):
        raise RuntimeError(
            "Context Expansion "
            "评测缺少 cases"
        )

    result: dict[
        str,
        set[str],
    ] = {}

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

        raw_hits = raw_case.get(
            "expanded_hits"
        )

        if not isinstance(
            raw_hits,
            list,
        ):
            raise RuntimeError(
                f"{case_id} 缺少 expanded_hits"
            )

        ids: set[str] = set()

        for raw_hit in raw_hits:
            if not isinstance(
                raw_hit,
                dict,
            ):
                raise RuntimeError(
                    f"{case_id} 包含无效 expanded hit"
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
                    f"{case_id} Hit 缺少 chunk_id"
                )

            ids.add(
                chunk_id
            )

        result[
            case_id
        ] = ids

    return result


def _references_by_role(
    record: CompetitionGoldFactRecord,
    *,
    role: str,
) -> tuple[
    CompetitionGoldChunkReference,
    ...,
]:
    return tuple(
        reference
        for reference in record.gold_chunks
        if reference.role == role
    )


def _relation_flags(
    relation: StructuralRelation,
) -> tuple[str, ...]:
    flags = []

    if relation.adjacent_1:
        flags.append(
            "distance_le_1"
        )

    if relation.adjacent_2:
        flags.append(
            "distance_le_2"
        )

    if relation.adjacent_3:
        flags.append(
            "distance_le_3"
        )

    if relation.same_section_path:
        flags.append(
            "same_section_path"
        )

    if relation.same_article:
        flags.append(
            "same_article"
        )

    if relation.same_item_path:
        flags.append(
            "same_item_path"
        )

    if relation.item_hierarchy_related:
        flags.append(
            "item_hierarchy"
        )

    if relation.same_table:
        flags.append(
            "same_table"
        )

    return tuple(flags)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "诊断 Competition Gold "
            "Parent-Child / Multi-Chunk "
            "结构关系"
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

    args = parser.parse_args()

    gold = (
        load_competition_gold_dataset(
            args.gold,
            expected_fact_count=97,
        )
    )

    corpus = (
        load_competition_chunk_corpus(
            args.corpus
        )
    )

    context_payload = (
        _load_context_eval(
            args.context_eval
        )
    )

    expanded_ids_by_case = (
        _load_expanded_ids_by_case(
            context_payload
        )
    )

    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunk in corpus.chunks
    }

    relation_counter = Counter()
    incomplete_relation_counter = Counter()

    parent_child_total = 0
    parent_child_complete = 0

    incomplete_records = []

    for record in gold.records:
        if (
            record.review_status
            != "confirmed"
        ):
            continue

        if (
            record.evidence_mode
            != "parent_child"
        ):
            continue

        parent_child_total += 1

        expanded_ids = (
            expanded_ids_by_case.get(
                record.case_id,
                set(),
            )
        )

        direct_refs = (
            _references_by_role(
                record,
                role="direct",
            )
        )

        parent_refs = (
            _references_by_role(
                record,
                role="parent_context",
            )
        )

        all_gold_ids = {
            reference.chunk_id
            for reference
            in record.gold_chunks
        }

        complete = (
            all_gold_ids
            <= expanded_ids
        )

        if complete:
            parent_child_complete += 1

        record_relations = Counter()

        pair_payloads = []

        for parent_ref in parent_refs:
            parent_chunk = (
                chunks_by_id.get(
                    parent_ref.chunk_id
                )
            )

            if parent_chunk is None:
                raise RuntimeError(
                    "Gold 引用了未知 parent Chunk："
                    f"{parent_ref.chunk_id}"
                )

            for direct_ref in direct_refs:
                direct_chunk = (
                    chunks_by_id.get(
                        direct_ref.chunk_id
                    )
                )

                if direct_chunk is None:
                    raise RuntimeError(
                        "Gold 引用了未知 direct Chunk："
                        f"{direct_ref.chunk_id}"
                    )

                relation = (
                    describe_relation(
                        parent_chunk,
                        direct_chunk,
                    )
                )

                flags = (
                    _relation_flags(
                        relation
                    )
                )

                for flag in flags:
                    relation_counter[
                        flag
                    ] += 1

                    record_relations[
                        flag
                    ] += 1

                pair_payloads.append(
                    {
                        "parent_chunk_id": (
                            parent_chunk.chunk_id
                        ),
                        "direct_chunk_id": (
                            direct_chunk.chunk_id
                        ),
                        "parent_chunk_index": (
                            parent_chunk.chunk_index
                        ),
                        "direct_chunk_index": (
                            direct_chunk.chunk_index
                        ),
                        "parent_section_path": list(
                            parent_chunk.section_path
                        ),
                        "direct_section_path": list(
                            direct_chunk.section_path
                        ),
                        "parent_article": (
                            parent_chunk.article
                        ),
                        "direct_article": (
                            direct_chunk.article
                        ),
                        "parent_item_path": list(
                            parent_chunk.item_path
                        ),
                        "direct_item_path": list(
                            direct_chunk.item_path
                        ),
                        "parent_table_index": (
                            parent_chunk.table_index
                        ),
                        "direct_table_index": (
                            direct_chunk.table_index
                        ),
                        "relation": {
                            "chunk_distance": (
                                relation.chunk_distance
                            ),
                            "flags": list(
                                flags
                            ),
                        },
                    }
                )

        if not complete:
            for flag in record_relations:
                incomplete_relation_counter[
                    flag
                ] += 1

            missing_gold_ids = sorted(
                all_gold_ids
                - expanded_ids
            )

            retrieved_gold_ids = sorted(
                all_gold_ids
                & expanded_ids
            )

            missing_direct_ids = sorted(
                reference.chunk_id
                for reference in direct_refs
                if (
                    reference.chunk_id
                    not in expanded_ids
                )
            )

            retrieved_parent_ids = sorted(
                reference.chunk_id
                for reference in parent_refs
                if (
                    reference.chunk_id
                    in expanded_ids
                )
            )

            incomplete_records.append(
                {
                    "case_id": (
                        record.case_id
                    ),
                    "fact_index": (
                        record.fact_index
                    ),
                    "fact": record.fact,
                    "missing_gold_ids": (
                        missing_gold_ids
                    ),
                    "retrieved_gold_ids": (
                        retrieved_gold_ids
                    ),
                    "missing_direct_ids": (
                        missing_direct_ids
                    ),
                    "retrieved_parent_ids": (
                        retrieved_parent_ids
                    ),
                    "relations": (
                        pair_payloads
                    ),
                }
            )

    #
    # Multi-chunk 结构统计
    #
    multi_counter = Counter()
    multi_total = 0

    for record in gold.records:
        if (
            record.review_status
            != "confirmed"
            or record.evidence_mode
            != "multi_chunk"
        ):
            continue

        multi_total += 1

        chunks = tuple(
            chunks_by_id[
                reference.chunk_id
            ]
            for reference
            in record.gold_chunks
        )

        for left_index in range(
            len(chunks)
        ):
            for right_index in range(
                left_index + 1,
                len(chunks),
            ):
                relation = (
                    describe_relation(
                        chunks[left_index],
                        chunks[right_index],
                    )
                )

                for flag in (
                    _relation_flags(
                        relation
                    )
                ):
                    multi_counter[
                        flag
                    ] += 1

    payload = {
        "schema_version": 1,
        "diagnosis_version": (
            "competition_context_"
            "structure_diagnosis_dev_v1"
        ),
        "corpus_id": (
            corpus.manifest.corpus_id
        ),
        "parent_child": {
            "fact_count": (
                parent_child_total
            ),
            "complete_after_adjacent_expansion": (
                parent_child_complete
            ),
            "incomplete_fact_count": (
                parent_child_total
                - parent_child_complete
            ),
            "all_gold_pair_relation_counts": dict(
                sorted(
                    relation_counter.items()
                )
            ),
            "incomplete_fact_relation_counts": dict(
                sorted(
                    incomplete_relation_counter.items()
                )
            ),
            "incomplete_records": (
                incomplete_records
            ),
        },
        "multi_chunk": {
            "fact_count": (
                multi_total
            ),
            "gold_pair_relation_counts": dict(
                sorted(
                    multi_counter.items()
                )
            ),
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
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

    print(
        "=== Competition Context "
        "Structure Diagnosis ==="
    )

    print(
        "Corpus ID:",
        corpus.manifest.corpus_id,
    )

    print()

    print(
        "Parent-child facts:",
        parent_child_total,
    )

    print(
        "Complete after ±1:",
        parent_child_complete,
    )

    print(
        "Incomplete after ±1:",
        (
            parent_child_total
            - parent_child_complete
        ),
    )

    print()

    print(
        "Parent-child Gold pair relations:"
    )

    for (
        relation_name,
        count,
    ) in sorted(
        relation_counter.items()
    ):
        print(
            f"  {relation_name}:",
            count,
        )

    print()

    print(
        "Relations among still-incomplete "
        "parent-child facts:"
    )

    for (
        relation_name,
        count,
    ) in sorted(
        incomplete_relation_counter.items()
    ):
        print(
            f"  {relation_name}:",
            count,
        )

    print()

    print(
        "Multi-chunk Gold pair relations:"
    )

    for (
        relation_name,
        count,
    ) in sorted(
        multi_counter.items()
    ):
        print(
            f"  {relation_name}:",
            count,
        )

    print()

    print(
        "Saved:",
        args.output,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )