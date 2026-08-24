from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SPARSE_FINAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "competition"
    / "processed"
    / "eval"
    / "competition_bm25_structured_context_dev_v1.json"
)

DENSE_COMPLEMENTARITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "competition"
    / "processed"
    / "eval"
    / (
        "competition_dense_sparse_"
        "source_filtered_"
        "complementarity_dev_v1.json"
    )
)

EXPECTED_CORPUS_ID = (
    "competition_corpus_8cbec682dfce4962"
)

EXPECTED_FACT_COUNT = 97
EXPECTED_TOP_K = 10

EXPECTED_EVIDENCE_MODE_COUNTS = {
    "single_chunk": 54,
    "parent_child": 35,
    "multi_chunk": 8,
}

EXPECTED_COMPLETE_COUNTS = {
    "single_chunk": 54,
    "parent_child": 33,
    "multi_chunk": 4,
}

EXPECTED_DIRECT_COUNT = 95
EXPECTED_COMPLETE_COUNT = 91

# Frozen baseline:
# reported GoldRecall@10 = 0.9464
MIN_GOLD_RECALL = 0.9463


@dataclass(
    frozen=True,
    slots=True,
)
class AcceptanceCheck:
    name: str
    passed: bool
    detail: str


def _load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            f"文件不存在：{path}"
        )

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"JSON 无效：{path}"
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            f"JSON 根节点必须是 object：{path}"
        )

    return payload


def _fact_hit_at(
    fact: dict[str, Any],
    *,
    cutoff: int,
) -> tuple[
    bool,
    bool,
    float,
]:
    matches = fact.get(
        "chunk_matches"
    )

    if (
        not isinstance(matches, list)
        or not matches
    ):
        raise RuntimeError(
            "Fact 缺少 chunk_matches"
        )

    hit_count = 0
    direct_hit = False

    for match in matches:
        if not isinstance(
            match,
            dict,
        ):
            raise RuntimeError(
                "chunk_match 必须是 object"
            )

        role = match.get("role")
        rank = match.get("rank")

        hit = (
            isinstance(rank, int)
            and rank <= cutoff
        )

        if hit:
            hit_count += 1

        if (
            role == "direct"
            and hit
        ):
            direct_hit = True

    complete = (
        hit_count == len(matches)
    )

    recall = (
        hit_count
        / len(matches)
    )

    return (
        direct_hit,
        complete,
        recall,
    )


def _collect_sparse_facts(
    sparse: dict[str, Any],
) -> list[
    dict[str, Any]
]:
    cases = sparse.get(
        "cases"
    )

    if not isinstance(
        cases,
        list,
    ):
        raise RuntimeError(
            "Sparse Final 缺少 cases"
        )

    facts = []

    for case in cases:
        if not isinstance(
            case,
            dict,
        ):
            raise RuntimeError(
                "Sparse case 必须是 object"
            )

        case_facts = case.get(
            "expanded_facts"
        )

        if not isinstance(
            case_facts,
            list,
        ):
            raise RuntimeError(
                "Sparse case 缺少 "
                "expanded_facts"
            )

        facts.extend(
            case_facts
        )

    return facts


def _calculate_sparse_metrics(
    sparse: dict[str, Any],
) -> dict[str, Any]:
    facts = (
        _collect_sparse_facts(
            sparse
        )
    )

    direct_count = 0
    complete_count = 0
    recall_sum = 0.0

    evidence_mode_counts: Counter[
        str
    ] = Counter()

    complete_mode_counts: Counter[
        str
    ] = Counter()

    for fact in facts:
        if not isinstance(
            fact,
            dict,
        ):
            raise RuntimeError(
                "Fact 必须是 object"
            )

        evidence_mode = fact.get(
            "evidence_mode"
        )

        if not isinstance(
            evidence_mode,
            str,
        ):
            raise RuntimeError(
                "Fact 缺少 evidence_mode"
            )

        (
            direct,
            complete,
            recall,
        ) = _fact_hit_at(
            fact,
            cutoff=EXPECTED_TOP_K,
        )

        evidence_mode_counts[
            evidence_mode
        ] += 1

        direct_count += int(
            direct
        )

        complete_count += int(
            complete
        )

        recall_sum += recall

        if complete:
            complete_mode_counts[
                evidence_mode
            ] += 1

    fact_count = len(facts)

    if fact_count == 0:
        raise RuntimeError(
            "Sparse Final 没有 Fact"
        )

    return {
        "fact_count": fact_count,
        "direct_count": (
            direct_count
        ),
        "complete_count": (
            complete_count
        ),
        "gold_recall": (
            recall_sum
            / fact_count
        ),
        "evidence_mode_counts": (
            dict(
                evidence_mode_counts
            )
        ),
        "complete_mode_counts": (
            dict(
                complete_mode_counts
            )
        ),
    }


def _check_artifact_identity(
    sparse: dict[str, Any],
    dense: dict[str, Any],
) -> AcceptanceCheck:
    sparse_corpus = sparse.get(
        "corpus_id"
    )

    dense_corpus = dense.get(
        "corpus_id"
    )

    sparse_gold = sparse.get(
        "gold_sha256"
    )

    dense_gold = dense.get(
        "gold_sha256"
    )

    passed = (
        sparse_corpus
        == EXPECTED_CORPUS_ID
        and dense_corpus
        == EXPECTED_CORPUS_ID
        and isinstance(
            sparse_gold,
            str,
        )
        and sparse_gold
        and sparse_gold
        == dense_gold
    )

    return AcceptanceCheck(
        name="frozen_artifact_identity",
        passed=passed,
        detail=(
            f"corpus_id={sparse_corpus}; "
            "Sparse/Dense Gold binding="
            f"{sparse_gold == dense_gold}"
        ),
    )


def _check_sparse_config(
    sparse: dict[str, Any],
) -> AcceptanceCheck:
    config = sparse.get(
        "expansion_config"
    )

    expected = {
        "seed_candidate_count": 10,
        "previous_window": 1,
        "next_window": 1,
        "enable_item_hierarchy": True,
        "enable_item_parent": False,
        "enable_item_child": True,
        "max_item_depth_delta": 2,
        "item_hierarchy_trigger_mode": (
            "seed_and_adjacent"
        ),
    }

    mismatches = []

    if not isinstance(
        config,
        dict,
    ):
        mismatches.append(
            "expansion_config missing"
        )
    else:
        for key, expected_value in (
            expected.items()
        ):
            actual = config.get(
                key
            )

            if actual != expected_value:
                mismatches.append(
                    f"{key}: "
                    f"actual={actual!r}, "
                    f"expected="
                    f"{expected_value!r}"
                )

    return AcceptanceCheck(
        name="final_sparse_strategy",
        passed=not mismatches,
        detail=(
            "Option-Anchored Top10 "
            "+ ±1 "
            "+ child-only depth<=2 "
            "+ seed_and_adjacent"
            if not mismatches
            else "; ".join(
                mismatches
            )
        ),
    )


def _check_child_only(
    sparse: dict[str, Any],
) -> AcceptanceCheck:
    relation_counts = sparse.get(
        "origin_relation_counts"
    )

    if not isinstance(
        relation_counts,
        dict,
    ):
        return AcceptanceCheck(
            name="child_only_gate",
            passed=False,
            detail=(
                "origin_relation_counts "
                "missing"
            ),
        )

    parent_count = int(
        relation_counts.get(
            "item_parent",
            0,
        )
    )

    child_count = int(
        relation_counts.get(
            "item_child",
            0,
        )
    )

    passed = (
        parent_count == 0
        and child_count > 0
    )

    return AcceptanceCheck(
        name="child_only_gate",
        passed=passed,
        detail=(
            f"item_parent={parent_count}; "
            f"item_child={child_count}"
        ),
    )


def _check_sparse_metrics(
    metrics: dict[str, Any],
) -> AcceptanceCheck:
    fact_count = int(
        metrics["fact_count"]
    )

    direct_count = int(
        metrics["direct_count"]
    )

    complete_count = int(
        metrics["complete_count"]
    )

    recall = float(
        metrics["gold_recall"]
    )

    passed = (
        fact_count
        == EXPECTED_FACT_COUNT
        and direct_count
        == EXPECTED_DIRECT_COUNT
        and complete_count
        == EXPECTED_COMPLETE_COUNT
        and recall
        >= MIN_GOLD_RECALL
    )

    return AcceptanceCheck(
        name="frozen_sparse_metrics",
        passed=passed,
        detail=(
            f"facts={fact_count}; "
            f"Direct="
            f"{direct_count}/"
            f"{fact_count}; "
            f"Complete="
            f"{complete_count}/"
            f"{fact_count}; "
            f"Recall={recall:.4f}"
        ),
    )


def _check_evidence_modes(
    metrics: dict[str, Any],
) -> AcceptanceCheck:
    evidence_counts = (
        metrics[
            "evidence_mode_counts"
        ]
    )

    complete_counts = (
        metrics[
            "complete_mode_counts"
        ]
    )

    passed = (
        evidence_counts
        == EXPECTED_EVIDENCE_MODE_COUNTS
        and complete_counts
        == EXPECTED_COMPLETE_COUNTS
    )

    return AcceptanceCheck(
        name="evidence_mode_gate",
        passed=passed,
        detail=(
            f"facts={evidence_counts}; "
            f"complete={complete_counts}"
        ),
    )


def _check_dense_no_go(
    dense: dict[str, Any],
) -> AcceptanceCheck:
    analyses = dense.get(
        "analyses"
    )

    if not isinstance(
        analyses,
        dict,
    ):
        return AcceptanceCheck(
            name="dense_hybrid_no_go",
            passed=False,
            detail="analyses missing",
        )

    oracle = analyses.get(
        "dense_any_query_oracle"
    )

    if not isinstance(
        oracle,
        dict,
    ):
        return AcceptanceCheck(
            name="dense_hybrid_no_go",
            passed=False,
            detail=(
                "dense_any_query_oracle "
                "missing"
            ),
        )

    direct_rescue = int(
        oracle.get(
            "direct_misses_rescued",
            -1,
        )
    )

    complete_repair = int(
        oracle.get(
            "incomplete_facts_repaired",
            -1,
        )
    )

    multi_repair = int(
        oracle.get(
            "multi_chunk_facts_repaired",
            -1,
        )
    )

    recovered_chunks = int(
        oracle.get(
            "missing_gold_chunks_recovered",
            -1,
        )
    )

    passed = (
        direct_rescue == 0
        and complete_repair == 0
        and multi_repair == 0
    )

    return AcceptanceCheck(
        name="dense_hybrid_no_go",
        passed=passed,
        detail=(
            f"direct_rescue="
            f"{direct_rescue}; "
            f"complete_repair="
            f"{complete_repair}; "
            f"multi_chunk_repair="
            f"{multi_repair}; "
            f"chunk_recovery="
            f"{recovered_chunks}"
        ),
    )


def _check_sparse_monotonicity(
    sparse: dict[str, Any],
) -> AcceptanceCheck:
    seed_summary = sparse.get(
        "seed_summary"
    )

    expanded_summary = sparse.get(
        "expanded_summary"
    )

    if (
        not isinstance(
            seed_summary,
            dict,
        )
        or not isinstance(
            expanded_summary,
            dict,
        )
    ):
        return AcceptanceCheck(
            name="context_monotonicity",
            passed=False,
            detail=(
                "seed/expanded summary "
                "missing"
            ),
        )

    seed_metrics = seed_summary.get(
        "metrics"
    )

    expanded_metrics = (
        expanded_summary.get(
            "metrics"
        )
    )

    if (
        not isinstance(
            seed_metrics,
            list,
        )
        or not isinstance(
            expanded_metrics,
            list,
        )
    ):
        return AcceptanceCheck(
            name="context_monotonicity",
            passed=False,
            detail="metrics missing",
        )

    expanded_by_cutoff = {
        metric.get(
            "cutoff"
        ): metric
        for metric in expanded_metrics
        if isinstance(
            metric,
            dict,
        )
    }

    violations = []

    fields = (
        "fact_direct_hit_rate",
        "fact_any_gold_hit_rate",
        "fact_complete_gold_hit_rate",
        "mean_gold_chunk_recall",
    )

    for seed_metric in seed_metrics:
        if not isinstance(
            seed_metric,
            dict,
        ):
            continue

        cutoff = seed_metric.get(
            "cutoff"
        )

        expanded_metric = (
            expanded_by_cutoff.get(
                cutoff
            )
        )

        if expanded_metric is None:
            violations.append(
                f"missing cutoff={cutoff}"
            )
            continue

        for field in fields:
            before = float(
                seed_metric[
                    field
                ]
            )

            after = float(
                expanded_metric[
                    field
                ]
            )

            if (
                after
                + 1e-12
                < before
            ):
                violations.append(
                    f"{field}@{cutoff}: "
                    f"{before:.6f}"
                    " -> "
                    f"{after:.6f}"
                )

    return AcceptanceCheck(
        name="context_monotonicity",
        passed=not violations,
        detail=(
            "all Gold coverage metrics "
            "non-decreasing"
            if not violations
            else "; ".join(
                violations
            )
        ),
    )


def main() -> None:
    sparse = _load_json(
        SPARSE_FINAL_PATH
    )

    dense = _load_json(
        DENSE_COMPLEMENTARITY_PATH
    )

    sparse_metrics = (
        _calculate_sparse_metrics(
            sparse
        )
    )

    checks = (
        _check_artifact_identity(
            sparse,
            dense,
        ),
        _check_sparse_config(
            sparse
        ),
        _check_child_only(
            sparse
        ),
        _check_sparse_metrics(
            sparse_metrics
        ),
        _check_evidence_modes(
            sparse_metrics
        ),
        _check_sparse_monotonicity(
            sparse
        ),
        _check_dense_no_go(
            dense
        ),
    )

    print(
        "\n===== Week 8 Retrieval "
        "Acceptance ====="
    )

    for check in checks:
        marker = (
            "PASS"
            if check.passed
            else "FAIL"
        )

        print(
            f"[{marker}] "
            f"{check.name}: "
            f"{check.detail}"
        )

    passed_count = sum(
        check.passed
        for check in checks
    )

    print(
        "\nAcceptance: "
        f"{passed_count}/"
        f"{len(checks)}"
    )

    if (
        passed_count
        != len(checks)
    ):
        raise SystemExit(1)

    print()
    print(
        "WEEK 8 RETRIEVAL ACCEPTED"
    )

    print()
    print(
        "Frozen strategy:"
    )

    print(
        "  Option-Anchored Top10"
    )

    print(
        "  + Adjacent ±1"
    )

    print(
        "  + Item Child depth<=2"
    )

    print(
        "  + seed_and_adjacent trigger"
    )

    print()
    print(
        "Hybrid decision:"
    )

    print(
        "  NO-GO: Dense adds no "
        "fact-level recovery"
    )


if __name__ == "__main__":
    main()