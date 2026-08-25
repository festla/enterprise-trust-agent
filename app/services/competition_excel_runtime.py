from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from pathlib import Path

from app.schemas.competition import (
    CompetitionQuestion,
    CompetitionSourceRecord,
    CompetitionSourceResolution,
)
from app.schemas.competition_excel import (
    CompetitionExcelWorkbook,
)
from app.schemas.competition_excel_solver import (
    CompetitionExcelCalculationResult,
    CompetitionExcelCompareResult,
    CompetitionExcelLookupResult,
)
from app.services.competition_excel_calculation import (
    solve_excel_table_calculation,
)
from app.services.competition_excel_compare import (
    solve_excel_table_compare,
)
from app.services.competition_excel_lookup import (
    solve_excel_table_lookup,
)
from app.services.competition_excel_parser import (
    parse_competition_excel,
)
from app.services.competition_source_resolver import (
    CompetitionSourceResolver,
    build_competition_source_manifest,
)


CompetitionExcelSolverResult = (
    CompetitionExcelLookupResult
    | CompetitionExcelCompareResult
    | CompetitionExcelCalculationResult
)


@dataclass(
    frozen=True,
    slots=True,
)
class CompetitionExcelRuntimeResult:
    """
    一次 Excel Solver Runtime 的结果。

    不保存整个 Workbook，
    避免以后把大型 Workbook 塞进 Agent State。
    """

    resolution: CompetitionSourceResolution

    excel_format: str

    solver_type: str

    prediction: CompetitionExcelSolverResult


@dataclass(
    frozen=True,
    slots=True,
)
class CompetitionExcelRuntimeResources:
    """
    Excel 的进程级资源。

    Workbook cache 在多个 Question 之间复用，
    避免同一附件反复解析。
    """

    attachments_root: Path

    source_resolver: CompetitionSourceResolver

    source_by_id: dict[
        str,
        CompetitionSourceRecord,
    ]

    workbook_cache: dict[
        str,
        CompetitionExcelWorkbook,
    ] = field(
        default_factory=dict
    )


def load_competition_excel_runtime_resources(
    *,
    attachments_root: Path,
) -> CompetitionExcelRuntimeResources:
    manifest = (
        build_competition_source_manifest(
            attachments_root
        )
    )

    resolver = (
        CompetitionSourceResolver(
            manifest
        )
    )

    source_by_id = {
        source.source_id: source
        for source in manifest
    }

    return CompetitionExcelRuntimeResources(
        attachments_root=(
            attachments_root
        ),
        source_resolver=resolver,
        source_by_id=source_by_id,
    )


def _load_workbook(
    *,
    resources: (
        CompetitionExcelRuntimeResources
    ),
    source_id: str,
) -> CompetitionExcelWorkbook:
    cached = (
        resources
        .workbook_cache
        .get(source_id)
    )

    if cached is not None:
        return cached

    source = (
        resources
        .source_by_id
        .get(source_id)
    )

    if source is None:
        raise RuntimeError(
            "Source Resolution 返回了"
            "不存在于 Manifest 的 source_id："
            f"{source_id}"
        )

    workbook = (
        parse_competition_excel(
            attachments_root=(
                resources
                .attachments_root
            ),
            source=source,
        )
    )

    resources.workbook_cache[
        source_id
    ] = workbook

    return workbook


def solve_competition_excel_question(
    *,
    question: CompetitionQuestion,
    resources: (
        CompetitionExcelRuntimeResources
    ),
    resolution: (
        CompetitionSourceResolution
        | None
    ) = None,
) -> CompetitionExcelRuntimeResult:
    """
    Competition Excel Runtime V1。

    输入严格为 CompetitionQuestion，
    不接受 CompetitionQaCase，
    因此无 Gold 泄漏。
    """

    if question.source_type != "excel":
        raise ValueError(
            "Excel Runtime "
            "只接受 source_type=excel"
        )

    active_resolution = (
        resolution
        if resolution is not None
        else (
            resources
            .source_resolver
            .resolve(question)
        )
    )

    if (
        active_resolution.case_id
        != question.case_id
    ):
        raise RuntimeError(
            "Question 与 Excel "
            "Source Resolution "
            "case_id 不一致"
        )

    if (
        active_resolution.source_type
        != "excel"
    ):
        raise RuntimeError(
            "Excel Runtime 收到"
            "非 Excel Source Resolution"
        )

    workbook = _load_workbook(
        resources=resources,
        source_id=(
            active_resolution.source_id
        ),
    )

    if question.qa_type == "表格取数":
        prediction = (
            solve_excel_table_lookup(
                question=question,
                workbook=workbook,
            )
        )

        solver_type = "lookup"

    elif question.qa_type == "表格比较":
        prediction = (
            solve_excel_table_compare(
                question=question,
                workbook=workbook,
            )
        )

        solver_type = "compare"

    elif question.qa_type == "表格计算":
        prediction = (
            solve_excel_table_calculation(
                question=question,
                workbook=workbook,
            )
        )

        solver_type = "calculation"

    else:
        raise ValueError(
            "当前 Excel Runtime "
            "不支持 qa_type："
            f"{question.qa_type}"
        )

    return CompetitionExcelRuntimeResult(
        resolution=active_resolution,
        excel_format=(
            workbook.excel_format
        ),
        solver_type=solver_type,
        prediction=prediction,
    )