from __future__ import annotations

import random
from collections import Counter, defaultdict

from app.schemas.competition import (
    CompetitionQaCase,
    CompetitionSourceResolution,
)


class CompetitionSplitError(
    RuntimeError
):
    pass


def _build_source_groups(
    *,
    cases: tuple[
        CompetitionQaCase,
        ...,
    ],
    resolutions: tuple[
        CompetitionSourceResolution,
        ...,
    ],
) -> dict[
    str,
    list[CompetitionQaCase],
]:
    resolution_by_case = {
        resolution.case_id: resolution
        for resolution in resolutions
    }

    if len(resolution_by_case) != len(cases):
        raise CompetitionSplitError(
            "Case 与 Resolution 数量不一致"
        )

    groups: dict[
        str,
        list[CompetitionQaCase],
    ] = defaultdict(list)

    for case in cases:
        try:
            resolution = (
                resolution_by_case[
                    case.case_id
                ]
            )
        except KeyError as exc:
            raise CompetitionSplitError(
                "缺少 Source Resolution: "
                f"{case.case_id}"
            ) from exc

        groups[
            resolution.source_id
        ].append(case)

    return dict(groups)


def _counter_for_cases(
    cases: list[
        CompetitionQaCase
    ],
    *,
    field: str,
) -> Counter[str]:
    return Counter(
        str(getattr(case, field))
        for case in cases
    )


def _candidate_score(
    *,
    all_cases: list[
        CompetitionQaCase
    ],
    dev_cases: list[
        CompetitionQaCase
    ],
    dev_ratio: float,
) -> float:
    """
    越小越好。

    目标：
    1. Dev 总题量接近目标比例；
    2. source_type 分布接近总体；
    3. qa_type 分布接近总体；
    4. difficulty 分布接近总体。

    source_type / qa_type 比 difficulty 权重略高。
    """

    total_count = len(all_cases)

    target_dev_count = (
        total_count * dev_ratio
    )

    dev_count = len(dev_cases)

    count_error = (
        abs(
            dev_count
            - target_dev_count
        )
        / total_count
    )

    def distribution_error(
        field: str,
    ) -> float:
        overall = _counter_for_cases(
            all_cases,
            field=field,
        )

        actual = _counter_for_cases(
            dev_cases,
            field=field,
        )

        error = 0.0

        for key, overall_count in (
            overall.items()
        ):
            target = (
                overall_count
                * dev_ratio
            )

            error += abs(
                actual.get(key, 0)
                - target
            )

        return (
            error
            / max(
                1.0,
                target_dev_count,
            )
        )

    source_type_error = (
        distribution_error(
            "source_type"
        )
    )

    qa_type_error = (
        distribution_error(
            "qa_type"
        )
    )

    difficulty_error = (
        distribution_error(
            "difficulty"
        )
    )

    return (
        3.0 * count_error
        + 2.0 * source_type_error
        + 2.0 * qa_type_error
        + 1.0 * difficulty_error
    )


def _contains_required_categories(
    cases: list[
        CompetitionQaCase
    ],
) -> bool:
    if not cases:
        return False

    source_types = {
        case.source_type
        for case in cases
    }

    qa_types = {
        case.qa_type
        for case in cases
    }

    difficulties = {
        case.difficulty
        for case in cases
    }

    return (
        source_types
        == {
            "excel",
            "word",
            "pdf",
        }
        and qa_types
        == {
            "表格取数",
            "表格比较",
            "表格计算",
            "单事实检索",
            "多事实检索",
        }
        and difficulties
        == {
            "easy",
            "medium",
            "hard",
        }
    )


def build_grouped_balanced_dev_test_split(
    *,
    cases: tuple[
        CompetitionQaCase,
        ...,
    ],
    resolutions: tuple[
        CompetitionSourceResolution,
        ...,
    ],
    dev_ratio: float = 1.0 / 3.0,
    seed: int = 2026,
    search_iterations: int = 25_000,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
]:
    if not 0.0 < dev_ratio < 1.0:
        raise CompetitionSplitError(
            "dev_ratio 必须位于 (0, 1)"
        )

    if search_iterations < 1:
        raise CompetitionSplitError(
            "search_iterations 必须 >= 1"
        )

    groups = _build_source_groups(
        cases=cases,
        resolutions=resolutions,
    )

    group_items = list(
        groups.items()
    )

    if len(group_items) < 2:
        raise CompetitionSplitError(
            "至少需要两个 Source Group"
        )

    all_cases = list(cases)

    target_dev_count = round(
        len(all_cases)
        * dev_ratio
    )

    # 允许实际 Dev 数量围绕目标轻微波动。
    #
    # 当前 300 QA：
    # target ≈ 100
    #
    # Group 最大可能一次带入十几道题，
    # 因此不能要求恰好 100。
    min_dev_count = max(
        1,
        target_dev_count - 15,
    )

    max_dev_count = min(
        len(all_cases) - 1,
        target_dev_count + 15,
    )

    rng = random.Random(seed)

    best_score: float | None = None

    best_dev_source_ids: set[
        str
    ] | None = None

    for _ in range(
        search_iterations
    ):
        shuffled = list(
            group_items
        )

        rng.shuffle(shuffled)

        dev_source_ids: set[
            str
        ] = set()

        dev_cases: list[
            CompetitionQaCase
        ] = []

        # 每次随机排列后，
        # 尝试不同 prefix。
        #
        # 一个 prefix 就是一组完整 Source，
        # 所以永远不会拆开同一附件。
        for (
            source_id,
            source_cases,
        ) in shuffled:
            dev_source_ids.add(
                source_id
            )

            dev_cases.extend(
                source_cases
            )

            dev_count = len(
                dev_cases
            )

            if (
                dev_count
                < min_dev_count
            ):
                continue

            if (
                dev_count
                > max_dev_count
            ):
                break

            test_cases = [
                case
                for (
                    other_source_id,
                    other_source_cases,
                ) in group_items
                if (
                    other_source_id
                    not in dev_source_ids
                )
                for case
                in other_source_cases
            ]

            # 两边都必须覆盖：
            #
            # excel / word / pdf
            # 五种 QA
            # easy / medium / hard
            if not (
                _contains_required_categories(
                    dev_cases
                )
                and
                _contains_required_categories(
                    test_cases
                )
            ):
                continue

            score = (
                _candidate_score(
                    all_cases=all_cases,
                    dev_cases=dev_cases,
                    dev_ratio=dev_ratio,
                )
            )

            if (
                best_score is None
                or score < best_score
            ):
                best_score = score

                best_dev_source_ids = (
                    set(
                        dev_source_ids
                    )
                )

    if (
        best_dev_source_ids
        is None
    ):
        raise CompetitionSplitError(
            "没有搜索到满足约束的 "
            "Dev/Test Split"
        )

    dev_ids: list[str] = []

    test_ids: list[str] = []

    for (
        source_id,
        source_cases,
    ) in groups.items():
        target = (
            dev_ids
            if (
                source_id
                in best_dev_source_ids
            )
            else test_ids
        )

        target.extend(
            case.case_id
            for case
            in source_cases
        )

    dev_id_set = set(
        dev_ids
    )

    test_id_set = set(
        test_ids
    )

    if dev_id_set & test_id_set:
        raise CompetitionSplitError(
            "Dev/Test 存在 Case 泄漏"
        )

    if (
        len(dev_id_set)
        + len(test_id_set)
        != len(cases)
    ):
        raise CompetitionSplitError(
            "Dev/Test 未覆盖全部 Case"
        )

    # ========================================================
    # 最重要的 Source Leakage Check
    # ========================================================

    resolution_by_case = {
        resolution.case_id:
        resolution
        for resolution
        in resolutions
    }

    dev_source_ids = {
        resolution_by_case[
            case_id
        ].source_id
        for case_id
        in dev_ids
    }

    test_source_ids = {
        resolution_by_case[
            case_id
        ].source_id
        for case_id
        in test_ids
    }

    if (
        dev_source_ids
        & test_source_ids
    ):
        raise CompetitionSplitError(
            "发现 Source Leakage："
            "同一附件同时进入 Dev/Test"
        )

    return (
        tuple(
            sorted(dev_ids)
        ),
        tuple(
            sorted(test_ids)
        ),
    )