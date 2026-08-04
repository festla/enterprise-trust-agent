from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class DenseBaselineSummaryError(ValueError):
    """Dense Baseline 汇总异常。"""


REPORT_IDS = (
    "midea_group_2024",
    "hisense_home_2024",
)

STRATEGIES = (
    "fixed_length",
    "paragraph",
    "section_paragraph",
)


def _load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise DenseBaselineSummaryError(
            f"文件不存在：{path}"
        )

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, dict):
        raise DenseBaselineSummaryError(
            f"JSON 顶层必须是对象：{path}"
        )

    return payload


def _load_jsonl(
    path: Path,
) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        raise DenseBaselineSummaryError(
            f"文件不存在：{path}"
        )

    rows: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        payload = json.loads(line)

        if not isinstance(payload, dict):
            raise DenseBaselineSummaryError(
                "JSONL 记录必须是对象："
                f"path={path}，"
                f"line={line_number}"
            )

        rows.append(payload)

    return tuple(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "汇总 Week 4 六组 Dense "
            "Retrieval Baseline"
        )
    )

    parser.add_argument(
        "--evaluation-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    experiment_rows: list[
        dict[str, Any]
    ] = []

    failed_rows: list[
        dict[str, Any]
    ] = []

    totals = {
        strategy: {
            "case_count": 0,
            "hit_at_1_count": 0,
            "hit_at_3_count": 0,
            "hit_at_5_count": 0,
        }
        for strategy in STRATEGIES
    }

    for report_id in REPORT_IDS:
        for strategy in STRATEGIES:
            directory = (
                args.evaluation_root
                / report_id
                / strategy
            )

            summary = _load_json(
                directory / "summary.json"
            )

            results = _load_jsonl(
                directory / "results.jsonl"
            )

            row = {
                "report_id": report_id,
                "strategy": strategy,
                "case_count": (
                    summary["case_count"]
                ),
                "hit_at_1_count": (
                    summary[
                        "hit_at_1_count"
                    ]
                ),
                "hit_at_3_count": (
                    summary[
                        "hit_at_3_count"
                    ]
                ),
                "hit_at_5_count": (
                    summary[
                        "hit_at_5_count"
                    ]
                ),
                "recall_at_1": (
                    summary["recall_at_1"]
                ),
                "recall_at_3": (
                    summary["recall_at_3"]
                ),
                "recall_at_5": (
                    summary["recall_at_5"]
                ),
            }

            experiment_rows.append(row)

            total = totals[strategy]

            for field in (
                "case_count",
                "hit_at_1_count",
                "hit_at_3_count",
                "hit_at_5_count",
            ):
                total[field] += row[field]

            for result in results:
                first_rank = result.get(
                    "first_relevant_rank"
                )

                if (
                    first_rank is None
                    or first_rank > 5
                ):
                    failed_rows.append(
                        {
                            "report_id": (
                                report_id
                            ),
                            "strategy": (
                                strategy
                            ),
                            "case_id": (
                                result["case_id"]
                            ),
                            "metric_name": (
                                result.get(
                                    "metric_name"
                                )
                            ),
                            "first_relevant_rank": (
                                first_rank
                            ),
                            "top_pdf_pages": [
                                hit["pdf_page"]
                                for hit in result.get(
                                    "top_hits",
                                    [],
                                )
                            ],
                        }
                    )

    overall_rows: list[
        dict[str, Any]
    ] = []

    for strategy in STRATEGIES:
        total = totals[strategy]
        case_count = total["case_count"]

        overall_rows.append(
            {
                "strategy": strategy,
                **total,
                "recall_at_1": (
                    total["hit_at_1_count"]
                    / case_count
                ),
                "recall_at_3": (
                    total["hit_at_3_count"]
                    / case_count
                ),
                "recall_at_5": (
                    total["hit_at_5_count"]
                    / case_count
                ),
            }
        )

    output = {
        "schema_version": 1,
        "experiment_count": len(
            experiment_rows
        ),
        "report_results": experiment_rows,
        "overall_by_strategy": (
            overall_rows
        ),
        "top5_failures": failed_rows,
    }

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output_path.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "REPORT | STRATEGY | CASES | "
        "HIT@1 | HIT@3 | HIT@5 | "
        "R@1 | R@3 | R@5"
    )

    for row in experiment_rows:
        print(
            f"{row['report_id']} | "
            f"{row['strategy']} | "
            f"{row['case_count']} | "
            f"{row['hit_at_1_count']} | "
            f"{row['hit_at_3_count']} | "
            f"{row['hit_at_5_count']} | "
            f"{row['recall_at_1']:.3f} | "
            f"{row['recall_at_3']:.3f} | "
            f"{row['recall_at_5']:.3f}"
        )

    print()
    print(
        "OVERALL 16 CASES"
    )

    for row in overall_rows:
        print(
            f"{row['strategy']} | "
            f"R@1={row['recall_at_1']:.3f} | "
            f"R@3={row['recall_at_3']:.3f} | "
            f"R@5={row['recall_at_5']:.3f}"
        )

    print()
    print(
        f"top5_failure_count="
        f"{len(failed_rows)}"
    )

    for row in failed_rows:
        print(
            f"- {row['report_id']} | "
            f"{row['strategy']} | "
            f"{row['case_id']} | "
            f"rank={row['first_relevant_rank']} | "
            f"top_pages={row['top_pdf_pages']}"
        )

    print(
        f"output_path={args.output_path}"
    )


if __name__ == "__main__":
    main()