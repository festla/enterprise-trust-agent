from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from app.schemas.competition_context_expansion import (
    CompetitionContextExpansionConfig,
)
from scripts.evaluate_competition_bm25_context_expansion_dev import (
    DEFAULT_BASELINE_FILE,
    DEFAULT_GOLD_FILE,
    DEFAULT_TOP_K,
    print_context_expansion_result,
    run_competition_bm25_context_expansion_dev,
)


DEFAULT_OUTPUT_FILE = Path(
    "data/competition/processed/eval/"
    "competition_bm25_item_hierarchy_dev_v1.json"
)


def build_argument_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "评测 Competition BM25 "
            "Option-Anchored + ±1 + "
            "Item Hierarchy Expansion"
        )
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_FILE,
    )

    parser.add_argument(
        "--gold",
        type=Path,
        default=DEFAULT_GOLD_FILE,
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

    parser.add_argument(
        "--previous-window",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--next-window",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-item-depth-delta",
        type=int,
        default=1,
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    arguments = (
        build_argument_parser()
        .parse_args(argv)
    )

    expansion_config = (
        CompetitionContextExpansionConfig(
            seed_candidate_count=(
                arguments.top_k
            ),
            previous_window=(
                arguments.previous_window
            ),
            next_window=(
                arguments.next_window
            ),
            enable_item_hierarchy=True,
            max_item_depth_delta=(
                arguments
                .max_item_depth_delta
            ),
        )
    )

    result = (
        run_competition_bm25_context_expansion_dev(
            baseline_path=(
                arguments.baseline
            ),
            gold_path=(
                arguments.gold
            ),
            corpus_directory=(
                arguments.corpus
            ),
            output_path=(
                arguments.output
            ),
            expansion_config=(
                expansion_config
            ),
            top_k=(
                arguments.top_k
            ),
        )
    )

    print_context_expansion_result(
        result
    )

    relation_counts = Counter(
        origin.relation
        for case
        in result.case_results
        for hit
        in case.expanded_hits
        for origin
        in hit.origins
    )

    print()

    print(
        "Origin relations:"
    )

    for (
        relation,
        count,
    ) in sorted(
        relation_counts.items()
    ):
        print(
            f"  {relation}:",
            count,
        )

    hierarchy_count = (
        relation_counts[
            "item_parent"
        ]
        + relation_counts[
            "item_child"
        ]
    )

    print()

    print(
        "Item hierarchy origins:",
        hierarchy_count,
    )

    print(
        "Max item depth delta:",
        arguments.max_item_depth_delta,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )