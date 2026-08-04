from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.schemas.complex_plan_eval import (
    ComplexFinancialEvalCase,
)
from app.schemas.complex_plan_eval_result import (
    ComplexPlanRunResult,
)
from app.services.complex_plan_oracle import (
    GoldOracleAnswerGenerator,
    GoldOracleCalculator,
    GoldOracleRetriever,
    execute_gold_oracle_case,
)


_RUN_ID_PREFIX_PATTERN = re.compile(
    r"^complex_run_[a-z0-9_]+$"
)


class ComplexPlanBatchRunnerError(
    ValueError
):
    """复杂问题批量运行参数无效。"""


class ComplexPlanBatchWriteError(
    OSError
):
    """复杂问题批量结果写入失败。"""


@dataclass(frozen=True, slots=True)
class ComplexPlanBatchRun:
    """一次复杂问题批量运行的内存结果。"""

    results: tuple[
        ComplexPlanRunResult,
        ...,
    ]

    def __post_init__(self) -> None:
        if not self.results:
            raise ComplexPlanBatchRunnerError(
                "批量运行结果不能为空"
            )

        case_ids = [
            result.case_id
            for result in self.results
        ]

        if len(case_ids) != len(
            set(case_ids)
        ):
            raise ComplexPlanBatchRunnerError(
                "批量结果包含重复 case_id"
            )

        run_ids = [
            result.run_id
            for result in self.results
        ]

        if len(run_ids) != len(
            set(run_ids)
        ):
            raise ComplexPlanBatchRunnerError(
                "批量结果包含重复 run_id"
            )

    @property
    def case_count(self) -> int:
        return len(self.results)

    @property
    def completed_count(self) -> int:
        return sum(
            result.status == "completed"
            for result in self.results
        )

    @property
    def failed_count(self) -> int:
        return sum(
            result.status == "failed"
            for result in self.results
        )

    @property
    def refused_count(self) -> int:
        return sum(
            result.status == "refused"
            for result in self.results
        )

    @property
    def all_completed(self) -> bool:
        return (
            self.completed_count
            == self.case_count
        )


def run_complex_plan_batch(
    *,
    cases: Sequence[
        ComplexFinancialEvalCase
    ],
    run_id_prefix: str,
    retriever: GoldOracleRetriever,
    generator: GoldOracleAnswerGenerator,
    calculator: (
        GoldOracleCalculator | None
    ) = None,
    top_k: int = 5,
) -> ComplexPlanBatchRun:
    """依次执行全部 Case，单题失败不终止批次。"""

    case_tuple = tuple(cases)

    if not case_tuple:
        raise ComplexPlanBatchRunnerError(
            "cases 不能为空"
        )

    case_ids = [
        case.case_id
        for case in case_tuple
    ]

    if len(case_ids) != len(
        set(case_ids)
    ):
        raise ComplexPlanBatchRunnerError(
            "cases 包含重复 case_id"
        )

    if (
        run_id_prefix
        != run_id_prefix.strip()
    ):
        raise ComplexPlanBatchRunnerError(
            "run_id_prefix 不能包含首尾空白"
        )

    if (
        _RUN_ID_PREFIX_PATTERN.fullmatch(
            run_id_prefix
        )
        is None
    ):
        raise ComplexPlanBatchRunnerError(
            "run_id_prefix 必须以 "
            "complex_run_ 开头，且只能包含"
            "小写字母、数字和下划线"
        )

    if top_k <= 0:
        raise ComplexPlanBatchRunnerError(
            "top_k 必须大于 0"
        )

    results: list[
        ComplexPlanRunResult
    ] = []

    for case in case_tuple:
        run_id = (
            f"{run_id_prefix}_"
            f"{case.case_id}"
        )

        result = execute_gold_oracle_case(
            run_id=run_id,
            case=case,
            retriever=retriever,
            calculator=calculator,
            generator=generator,
            top_k=top_k,
        )

        results.append(result)

    return ComplexPlanBatchRun(
        results=tuple(results)
    )


def write_complex_plan_batch_results(
    *,
    batch: ComplexPlanBatchRun,
    output_path: Path,
) -> Path:
    """把批量运行结果写成 UTF-8 JSONL。

    为避免误覆盖历史评测结果，目标文件已经存在时拒绝写入。
    """

    if output_path.suffix.lower() != ".jsonl":
        raise ComplexPlanBatchWriteError(
            "批量结果文件必须使用 .jsonl 后缀"
        )

    if output_path.exists():
        raise ComplexPlanBatchWriteError(
            "批量结果文件已经存在，拒绝覆盖："
            f"{output_path}"
        )

    content = "".join(
        result.model_dump_json()
        + "\n"
        for result in batch.results
    ).encode("utf-8")

    try:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open("xb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())

    except FileExistsError as exc:
        raise ComplexPlanBatchWriteError(
            "批量结果文件已经存在，拒绝覆盖："
            f"{output_path}"
        ) from exc

    except OSError as exc:
        raise ComplexPlanBatchWriteError(
            "无法写入批量运行结果："
            f"{output_path}"
        ) from exc

    return output_path