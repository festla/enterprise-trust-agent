from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


DEFAULT_SPARSE_FILE = Path(
    "data/competition/processed/eval/"
    "competition_bm25_structured_context_dev_v1.json"
)

DEFAULT_DENSE_FILE = Path(
    "data/competition/processed/eval/"
    "competition_dense_gold_dev_v1.json"
)

DEFAULT_OUTPUT_FILE = Path(
    "data/competition/processed/eval/"
    "competition_dense_sparse_complementarity_dev_v1.json"
)

DEFAULT_TOP_K = 10

DENSE_MODES = (
    "question_only",
    "question_with_options",
)


@dataclass(
    frozen=True,
    slots=True,
)
class GoldMatchView:
    chunk_id: str
    role: str
    rank: int | None

    def hit_at(
        self,
        cutoff: int,
    ) -> bool:
        return (
            self.rank is not None
            and self.rank <= cutoff
        )


@dataclass(
    frozen=True,
    slots=True,
)
class FactView:
    case_id: str
    fact_index: int
    fact: str
    evidence_mode: str
    expected_source_id: str
    matches: tuple[
        GoldMatchView,
        ...,
    ]

    @property
    def key(
        self,
    ) -> tuple[str, int]:
        return (
            self.case_id,
            self.fact_index,
        )

    def direct_hit_at(
        self,
        cutoff: int,
    ) -> bool:
        return any(
            match.role == "direct"
            and match.hit_at(cutoff)
            for match in self.matches
        )

    def complete_hit_at(
        self,
        cutoff: int,
    ) -> bool:
        return all(
            match.hit_at(cutoff)
            for match in self.matches
        )

    def recall_at(
        self,
        cutoff: int,
    ) -> float:
        return (
            sum(
                match.hit_at(cutoff)
                for match in self.matches
            )
            / len(self.matches)
        )


def _load_json(
    path: Path,
) -> dict[str, Any]:
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
            f"无法读取 JSON：{path}"
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            f"JSON 根节点必须是 object：{path}"
        )

    return payload


def _require_list(
    value: object,
    *,
    label: str,
) -> list[Any]:
    if not isinstance(
        value,
        list,
    ):
        raise RuntimeError(
            f"{label} 必须是 list"
        )

    return value


def _parse_fact(
    *,
    case_id: str,
    payload: dict[str, Any],
) -> FactView:
    try:
        fact_index = payload[
            "fact_index"
        ]
        fact = payload[
            "fact"
        ]
        evidence_mode = payload[
            "evidence_mode"
        ]
        expected_source_id = payload[
            "expected_source_id"
        ]
        chunk_matches = payload[
            "chunk_matches"
        ]
    except KeyError as exc:
        raise RuntimeError(
            "Fact payload 缺少字段："
            f"{case_id}"
        ) from exc

    if (
        not isinstance(
            fact_index,
            int,
        )
        or not isinstance(
            fact,
            str,
        )
        or not isinstance(
            evidence_mode,
            str,
        )
        or not isinstance(
            expected_source_id,
            str,
        )
    ):
        raise RuntimeError(
            f"Fact 字段类型错误：{case_id}"
        )

    raw_matches = _require_list(
        chunk_matches,
        label=(
            f"{case_id}#"
            f"{fact_index}.chunk_matches"
        ),
    )

    matches = []

    for raw_match in raw_matches:
        if not isinstance(
            raw_match,
            dict,
        ):
            raise RuntimeError(
                "chunk_match 必须是 object"
            )

        chunk_id = raw_match.get(
            "chunk_id"
        )
        role = raw_match.get(
            "role"
        )
        rank = raw_match.get(
            "rank"
        )

        if (
            not isinstance(
                chunk_id,
                str,
            )
            or not isinstance(
                role,
                str,
            )
            or (
                rank is not None
                and not isinstance(
                    rank,
                    int,
                )
            )
        ):
            raise RuntimeError(
                "Gold Chunk Match 字段无效："
                f"{case_id}#{fact_index}"
            )

        matches.append(
            GoldMatchView(
                chunk_id=chunk_id,
                role=role,
                rank=rank,
            )
        )

    if not matches:
        raise RuntimeError(
            "Fact 不应没有 Gold Chunk："
            f"{case_id}#{fact_index}"
        )

    return FactView(
        case_id=case_id,
        fact_index=fact_index,
        fact=fact,
        evidence_mode=(
            evidence_mode
        ),
        expected_source_id=(
            expected_source_id
        ),
        matches=tuple(matches),
    )


def _build_fact_map(
    *,
    cases: list[Any],
    facts_field: str,
) -> dict[
    tuple[str, int],
    FactView,
]:
    result: dict[
        tuple[str, int],
        FactView,
    ] = {}

    for raw_case in cases:
        if not isinstance(
            raw_case,
            dict,
        ):
            raise RuntimeError(
                "Case 必须是 object"
            )

        case_id = raw_case.get(
            "case_id"
        )

        if not isinstance(
            case_id,
            str,
        ):
            raise RuntimeError(
                "Case 缺少 case_id"
            )

        raw_facts = _require_list(
            raw_case.get(
                facts_field
            ),
            label=(
                f"{case_id}."
                f"{facts_field}"
            ),
        )

        for raw_fact in raw_facts:
            if not isinstance(
                raw_fact,
                dict,
            ):
                raise RuntimeError(
                    "Fact 必须是 object"
                )

            fact = _parse_fact(
                case_id=case_id,
                payload=raw_fact,
            )

            if fact.key in result:
                raise RuntimeError(
                    "重复 Fact Key："
                    f"{fact.key}"
                )

            result[
                fact.key
            ] = fact

    return result


def _load_sparse_facts(
    payload: dict[str, Any],
) -> dict[
    tuple[str, int],
    FactView,
]:
    cases = _require_list(
        payload.get("cases"),
        label="sparse.cases",
    )

    return _build_fact_map(
        cases=cases,
        facts_field=(
            "expanded_facts"
        ),
    )


def _load_dense_mode_facts(
    payload: dict[str, Any],
    *,
    mode: str,
) -> dict[
    tuple[str, int],
    FactView,
]:
    modes = payload.get(
        "modes"
    )

    if not isinstance(
        modes,
        dict,
    ):
        raise RuntimeError(
            "dense.modes 无效"
        )

    mode_payload = modes.get(
        mode
    )

    if not isinstance(
        mode_payload,
        dict,
    ):
        raise RuntimeError(
            "Dense 缺少 Mode："
            f"{mode}"
        )

    cases = _require_list(
        mode_payload.get(
            "cases"
        ),
        label=(
            f"dense.{mode}.cases"
        ),
    )

    return _build_fact_map(
        cases=cases,
        facts_field="facts",
    )


def _validate_same_fact(
    left: FactView,
    right: FactView,
) -> None:
    if (
        left.key != right.key
        or left.fact != right.fact
        or (
            left.evidence_mode
            != right.evidence_mode
        )
        or (
            left.expected_source_id
            != right.expected_source_id
        )
    ):
        raise RuntimeError(
            "Sparse / Dense Fact "
            "身份不一致："
            f"{left.key}"
        )

    left_gold = {
        (
            match.chunk_id,
            match.role,
        )
        for match in left.matches
    }

    right_gold = {
        (
            match.chunk_id,
            match.role,
        )
        for match in right.matches
    }

    if left_gold != right_gold:
        raise RuntimeError(
            "Sparse / Dense Gold Chunk "
            "不一致："
            f"{left.key}"
        )


def _merge_fact_views(
    *views: FactView,
) -> FactView:
    if not views:
        raise ValueError(
            "至少需要一个 FactView"
        )

    first = views[0]

    for other in views[1:]:
        _validate_same_fact(
            first,
            other,
        )

    ranks_by_chunk: dict[
        str,
        list[int],
    ] = {
        match.chunk_id: []
        for match in first.matches
    }

    roles_by_chunk = {
        match.chunk_id: (
            match.role
        )
        for match in first.matches
    }

    for view in views:
        for match in view.matches:
            if match.rank is not None:
                ranks_by_chunk[
                    match.chunk_id
                ].append(
                    match.rank
                )

    merged_matches = tuple(
        GoldMatchView(
            chunk_id=(
                match.chunk_id
            ),
            role=roles_by_chunk[
                match.chunk_id
            ],
            rank=(
                min(
                    ranks_by_chunk[
                        match.chunk_id
                    ]
                )
                if ranks_by_chunk[
                    match.chunk_id
                ]
                else None
            ),
        )
        for match in first.matches
    )

    return FactView(
        case_id=first.case_id,
        fact_index=(
            first.fact_index
        ),
        fact=first.fact,
        evidence_mode=(
            first.evidence_mode
        ),
        expected_source_id=(
            first.expected_source_id
        ),
        matches=merged_matches,
    )


def _summary(
    facts: dict[
        tuple[str, int],
        FactView,
    ],
    *,
    cutoff: int,
) -> dict[str, object]:
    values = tuple(
        facts.values()
    )

    return {
        "fact_count": len(values),
        "direct_hit_count": sum(
            fact.direct_hit_at(
                cutoff
            )
            for fact in values
        ),
        "complete_gold_count": sum(
            fact.complete_hit_at(
                cutoff
            )
            for fact in values
        ),
        "mean_gold_recall": (
            sum(
                fact.recall_at(
                    cutoff
                )
                for fact in values
            )
            / len(values)
        ),
    }


def _diagnose_mode(
    *,
    sparse: dict[
        tuple[str, int],
        FactView,
    ],
    dense: dict[
        tuple[str, int],
        FactView,
    ],
    cutoff: int,
) -> dict[str, object]:
    if set(sparse) != set(dense):
        raise RuntimeError(
            "Sparse / Dense Fact Key "
            "集合不一致"
        )

    union: dict[
        tuple[str, int],
        FactView,
    ] = {}

    rescued_direct_facts = []
    repaired_complete_facts = []
    recovered_missing_chunks = []

    sparse_direct_misses = 0
    sparse_incomplete = 0
    sparse_missing_chunk_count = 0

    multi_chunk_sparse_incomplete = 0
    multi_chunk_repaired = 0

    for key in sorted(sparse):
        sparse_fact = sparse[
            key
        ]
        dense_fact = dense[
            key
        ]

        _validate_same_fact(
            sparse_fact,
            dense_fact,
        )

        union_fact = (
            _merge_fact_views(
                sparse_fact,
                dense_fact,
            )
        )

        union[key] = union_fact

        sparse_direct = (
            sparse_fact
            .direct_hit_at(
                cutoff
            )
        )

        dense_direct = (
            dense_fact
            .direct_hit_at(
                cutoff
            )
        )

        sparse_complete = (
            sparse_fact
            .complete_hit_at(
                cutoff
            )
        )

        union_complete = (
            union_fact
            .complete_hit_at(
                cutoff
            )
        )

        if not sparse_direct:
            sparse_direct_misses += 1

            if dense_direct:
                direct_matches = [
                    match
                    for match
                    in dense_fact.matches
                    if (
                        match.role
                        == "direct"
                        and match.hit_at(
                            cutoff
                        )
                    )
                ]

                rescued_direct_facts.append(
                    {
                        "case_id": (
                            sparse_fact.case_id
                        ),
                        "fact_index": (
                            sparse_fact.fact_index
                        ),
                        "fact": (
                            sparse_fact.fact
                        ),
                        "evidence_mode": (
                            sparse_fact
                            .evidence_mode
                        ),
                        "dense_direct_hits": [
                            {
                                "chunk_id": (
                                    match.chunk_id
                                ),
                                "rank": (
                                    match.rank
                                ),
                            }
                            for match
                            in direct_matches
                        ],
                    }
                )

        if not sparse_complete:
            sparse_incomplete += 1

            if (
                sparse_fact.evidence_mode
                == "multi_chunk"
            ):
                multi_chunk_sparse_incomplete += 1

            if union_complete:
                repaired_complete_facts.append(
                    {
                        "case_id": (
                            sparse_fact.case_id
                        ),
                        "fact_index": (
                            sparse_fact.fact_index
                        ),
                        "fact": (
                            sparse_fact.fact
                        ),
                        "evidence_mode": (
                            sparse_fact
                            .evidence_mode
                        ),
                    }
                )

                if (
                    sparse_fact.evidence_mode
                    == "multi_chunk"
                ):
                    multi_chunk_repaired += 1

        sparse_matches = {
            match.chunk_id: match
            for match
            in sparse_fact.matches
        }

        dense_matches = {
            match.chunk_id: match
            for match
            in dense_fact.matches
        }

        for chunk_id, sparse_match in (
            sparse_matches.items()
        ):
            if sparse_match.hit_at(
                cutoff
            ):
                continue

            sparse_missing_chunk_count += 1

            dense_match = (
                dense_matches[
                    chunk_id
                ]
            )

            if dense_match.hit_at(
                cutoff
            ):
                recovered_missing_chunks.append(
                    {
                        "case_id": (
                            sparse_fact.case_id
                        ),
                        "fact_index": (
                            sparse_fact.fact_index
                        ),
                        "evidence_mode": (
                            sparse_fact
                            .evidence_mode
                        ),
                        "chunk_id": (
                            chunk_id
                        ),
                        "role": (
                            sparse_match.role
                        ),
                        "dense_rank": (
                            dense_match.rank
                        ),
                    }
                )

    return {
        "dense_only_summary": (
            _summary(
                dense,
                cutoff=cutoff,
            )
        ),
        "sparse_union_dense_summary": (
            _summary(
                union,
                cutoff=cutoff,
            )
        ),
        "sparse_direct_misses": (
            sparse_direct_misses
        ),
        "direct_misses_rescued": len(
            rescued_direct_facts
        ),
        "sparse_incomplete_facts": (
            sparse_incomplete
        ),
        "incomplete_facts_repaired": len(
            repaired_complete_facts
        ),
        "sparse_missing_gold_chunks": (
            sparse_missing_chunk_count
        ),
        "missing_gold_chunks_recovered": len(
            recovered_missing_chunks
        ),
        "multi_chunk_sparse_incomplete": (
            multi_chunk_sparse_incomplete
        ),
        "multi_chunk_facts_repaired": (
            multi_chunk_repaired
        ),
        "rescued_direct_facts": (
            rescued_direct_facts
        ),
        "repaired_complete_facts": (
            repaired_complete_facts
        ),
        "recovered_missing_chunks": (
            recovered_missing_chunks
        ),
    }


def _build_dense_any_oracle(
    *,
    dense_modes: dict[
        str,
        dict[
            tuple[str, int],
            FactView,
        ],
    ],
) -> dict[
    tuple[str, int],
    FactView,
]:
    first_mode = DENSE_MODES[0]

    keys = set(
        dense_modes[
            first_mode
        ]
    )

    for mode in DENSE_MODES[1:]:
        if (
            set(
                dense_modes[mode]
            )
            != keys
        ):
            raise RuntimeError(
                "Dense 两种 Query Mode "
                "Fact Key 不一致"
            )

    return {
        key: _merge_fact_views(
            *(
                dense_modes[
                    mode
                ][key]
                for mode
                in DENSE_MODES
            )
        )
        for key in sorted(keys)
    }


def _format_summary(
    summary: dict[str, object],
) -> str:
    fact_count = int(
        summary["fact_count"]
    )

    direct = int(
        summary[
            "direct_hit_count"
        ]
    )

    complete = int(
        summary[
            "complete_gold_count"
        ]
    )

    recall = float(
        summary[
            "mean_gold_recall"
        ]
    )

    return (
        f"Direct={direct}/{fact_count} "
        f"({direct / fact_count:.4f}) "
        f"Complete={complete}/{fact_count} "
        f"({complete / fact_count:.4f}) "
        f"Recall={recall:.4f}"
    )


def print_result(
    *,
    sparse_summary: dict[
        str,
        object,
    ],
    analyses: dict[
        str,
        dict[str, object],
    ],
    cutoff: int,
) -> None:
    print(
        "=== Competition Dense / Sparse "
        "Complementarity Analysis ==="
    )

    print(
        "Cutoff:",
        cutoff,
    )

    print()

    print(
        "Sparse Final:"
    )

    print(
        " ",
        _format_summary(
            sparse_summary
        ),
    )

    for mode, analysis in (
        analyses.items()
    ):
        print()
        print(
            f"===== {mode} ====="
        )

        print(
            "Dense only:"
        )

        print(
            " ",
            _format_summary(
                analysis[
                    "dense_only_summary"
                ]
            ),
        )

        print(
            "Sparse U Dense:"
        )

        print(
            " ",
            _format_summary(
                analysis[
                    "sparse_union_dense_summary"
                ]
            ),
        )

        print(
            "Direct rescue:",
            (
                analysis[
                    "direct_misses_rescued"
                ]
            ),
            "/",
            (
                analysis[
                    "sparse_direct_misses"
                ]
            ),
        )

        print(
            "Complete repair:",
            (
                analysis[
                    "incomplete_facts_repaired"
                ]
            ),
            "/",
            (
                analysis[
                    "sparse_incomplete_facts"
                ]
            ),
        )

        print(
            "Missing Gold chunks recovered:",
            (
                analysis[
                    "missing_gold_chunks_recovered"
                ]
            ),
            "/",
            (
                analysis[
                    "sparse_missing_gold_chunks"
                ]
            ),
        )

        print(
            "Multi-chunk repaired:",
            (
                analysis[
                    "multi_chunk_facts_repaired"
                ]
            ),
            "/",
            (
                analysis[
                    "multi_chunk_sparse_incomplete"
                ]
            ),
        )

        rescued = analysis[
            "rescued_direct_facts"
        ]

        if rescued:
            print(
                "Rescued Direct facts:"
            )

            for item in rescued:
                print(
                    " ",
                    f"{item['case_id']}#"
                    f"{item['fact_index']}",
                    item[
                        "evidence_mode"
                    ],
                )


def run(
    *,
    sparse_path: Path,
    dense_path: Path,
    output_path: Path,
    cutoff: int,
) -> dict[str, object]:
    if cutoff < 1:
        raise ValueError(
            "cutoff 必须 >= 1"
        )

    sparse_payload = (
        _load_json(
            sparse_path
        )
    )

    dense_payload = (
        _load_json(
            dense_path
        )
    )

    if (
        sparse_payload.get(
            "corpus_id"
        )
        != dense_payload.get(
            "corpus_id"
        )
    ):
        raise RuntimeError(
            "Sparse / Dense corpus_id "
            "不一致"
        )

    if (
        sparse_payload.get(
            "gold_sha256"
        )
        != dense_payload.get(
            "gold_sha256"
        )
    ):
        raise RuntimeError(
            "Sparse / Dense gold_sha256 "
            "不一致"
        )

    sparse_facts = (
        _load_sparse_facts(
            sparse_payload
        )
    )

    dense_modes = {
        mode: (
            _load_dense_mode_facts(
                dense_payload,
                mode=mode,
            )
        )
        for mode in DENSE_MODES
    }

    if len(sparse_facts) != 97:
        raise RuntimeError(
            "Frozen Gold Fact 数量改变："
            f"{len(sparse_facts)}"
        )

    sparse_summary = (
        _summary(
            sparse_facts,
            cutoff=cutoff,
        )
    )

    # Frozen 6B2 Final Gate：
    # 防止误拿旧版 Context Expansion 文件。
    if cutoff == 10:
        if (
            sparse_summary[
                "direct_hit_count"
            ]
            != 95
            or sparse_summary[
                "complete_gold_count"
            ]
            != 91
        ):
            raise RuntimeError(
                "Sparse Final 指标不是 "
                "6B2 Frozen Baseline："
                f"{sparse_summary}"
            )

    analyses: dict[
        str,
        dict[str, object],
    ] = {}

    for mode in DENSE_MODES:
        analyses[
            mode
        ] = _diagnose_mode(
            sparse=sparse_facts,
            dense=dense_modes[
                mode
            ],
            cutoff=cutoff,
        )

    # Diagnostic Oracle：
    # 表示两种 Dense Query Top-K 的并集。
    #
    # 注意它最多相当于 20 个 Dense 候选，
    # 因此只用于判断“Dense 有没有互补信息”，
    # 不能直接当最终 Hybrid 指标。
    dense_any_oracle = (
        _build_dense_any_oracle(
            dense_modes=(
                dense_modes
            )
        )
    )

    analyses[
        "dense_any_query_oracle"
    ] = _diagnose_mode(
        sparse=sparse_facts,
        dense=dense_any_oracle,
        cutoff=cutoff,
    )

    output = {
        "schema_version": 1,
        "analysis_version": (
            "competition_dense_sparse_"
            "complementarity_dev_v1"
        ),
        "corpus_id": (
            sparse_payload[
                "corpus_id"
            ]
        ),
        "gold_sha256": (
            sparse_payload[
                "gold_sha256"
            ]
        ),
        "cutoff": cutoff,
        "sparse_file": str(
            sparse_path
        ),
        "dense_file": str(
            dense_path
        ),
        "sparse_summary": (
            sparse_summary
        ),
        "analyses": analyses,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print_result(
        sparse_summary=(
            sparse_summary
        ),
        analyses=analyses,
        cutoff=cutoff,
    )

    print()
    print(
        "Saved:",
        output_path,
    )

    return output


def build_argument_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "分析 Dense 对最终 Sparse "
            "Hard Cases 的互补价值"
        )
    )

    parser.add_argument(
        "--sparse",
        type=Path,
        default=(
            DEFAULT_SPARSE_FILE
        ),
    )

    parser.add_argument(
        "--dense",
        type=Path,
        default=(
            DEFAULT_DENSE_FILE
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_OUTPUT_FILE
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    return parser


def main(
) -> int:
    args = (
        build_argument_parser()
        .parse_args()
    )

    run(
        sparse_path=args.sparse,
        dense_path=args.dense,
        output_path=args.output,
        cutoff=args.top_k,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )