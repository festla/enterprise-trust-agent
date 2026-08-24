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
    "competition_bm25_structured_context_dev_v1.json"
)


def build_argument_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Competition 最终 Sparse Retrieval："
            "Option-Anchored + ±1 + "
            "Item Child depth<=2"
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

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    arguments = (
        build_argument_parser()
        .parse_args(argv)
    )

    config = (
        CompetitionContextExpansionConfig(
            seed_candidate_count=(
                arguments.top_k
            ),
            previous_window=1,
            next_window=1,
            enable_item_hierarchy=True,

            #
            # 6B2f Ablation 最终结论：
            #
            enable_item_parent=False,
            enable_item_child=True,

            max_item_depth_delta=2,
            item_hierarchy_trigger_mode=(
                "seed_and_adjacent"
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
            expansion_config=config,
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
        "Final sparse strategy:"
    )

    print(
        "  Option-Anchored Top-K"
    )

    print(
        "  Adjacent window: ±1"
    )

    print(
        "  Item parent: DISABLED"
    )

    print(
        "  Item child: ENABLED"
    )

    print(
        "  Item child depth: 2"
    )

    print(
        "  Hierarchy trigger:",
        (
            config
            .item_hierarchy_trigger_mode
        ),
    )

    print()

    print(
        "Origin relations:",
        dict(
            sorted(
                relation_counts.items()
            )
        ),
    )

    if (
        relation_counts[
            "item_parent"
        ]
        != 0
    ):
        raise RuntimeError(
            "最终 Sparse Strategy "
            "不应产生 item_parent origin"
        )

    print()

    print(
        "Child-only gate: PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )