from __future__ import annotations

from collections.abc import (
    Callable,
)

from app.schemas.competition_agent_execution import (
    CompetitionAgentTraceEvent,
)
from app.services.competition_excel_evidence import (
    build_competition_excel_evidence_bundle,
    build_competition_excel_final_answer,
)
from app.services.competition_excel_runtime import (
    CompetitionExcelRuntimeResources,
    solve_competition_excel_question,
)
from app.workflow.competition_state import (
    CompetitionAgentState,
)


CompetitionAgentNode = Callable[
    [CompetitionAgentState],
    dict[str, object],
]


def build_excel_solver_node(
    resources: (
        CompetitionExcelRuntimeResources
    ),
) -> CompetitionAgentNode:
    """
    Excel Tool Node。

    已解析的 Source
        ↓
    Excel Workbook
        ↓
    qa_type dispatch
        ↓
    Lookup / Compare / Calculation
        ↓
    CompetitionEvidenceBundle
        ↓
    CompetitionFinalAnswer

    注意：
    这一条路径不调用 LLM。
    """

    def excel_solver_node(
        state: CompetitionAgentState,
    ) -> dict[str, object]:
        question = state.get(
            "question"
        )

        if question is None:
            raise RuntimeError(
                "Excel Solver Node "
                "缺少 question"
            )

        if (
            question.source_type
            != "excel"
        ):
            raise RuntimeError(
                "Excel Solver Node "
                "收到非 Excel Question"
            )

        resolution = state.get(
            "source_resolution"
        )

        if resolution is None:
            raise RuntimeError(
                "Excel Solver Node "
                "缺少 source_resolution"
            )

        # -----------------------------------------
        # 1. 调用统一 Excel Runtime
        # -----------------------------------------

        runtime_result = (
            solve_competition_excel_question(
                question=question,
                resources=resources,
                resolution=resolution,
            )
        )

        # -----------------------------------------
        # 2. 找到真正 Source Record
        # -----------------------------------------

        source = (
            resources
            .source_by_id
            .get(
                resolution.source_id
            )
        )

        if source is None:
            raise RuntimeError(
                "Excel Runtime 中不存在 "
                "resolved source_id："
                f"{resolution.source_id}"
            )

        # -----------------------------------------
        # 3. Solver Result
        #    → Unified Evidence
        # -----------------------------------------

        evidence_bundle = (
            build_competition_excel_evidence_bundle(
                question=question,
                source=source,
                result=(
                    runtime_result
                    .prediction
                ),
                attachments_root=(
                    resources
                    .attachments_root
                ),
            )
        )

        # -----------------------------------------
        # 4. Deterministic FinalAnswer
        # -----------------------------------------

        answer = (
            build_competition_excel_final_answer(
                question=question,
                result=(
                    runtime_result
                    .prediction
                ),
                evidence_bundle=(
                    evidence_bundle
                ),
                solver_type=(
                    runtime_result
                    .solver_type
                ),
            )
        )

        return {
            "excel_runtime": (
                runtime_result
            ),
            "evidence_bundle": (
                evidence_bundle
            ),
            "answer": answer,
            "trace": (
                CompetitionAgentTraceEvent(
                    stage="excel_solver",
                    outcome="success",
                    message=(
                        "Excel structured "
                        "solver completed"
                    ),
                    details={
                        "solver_type": (
                            runtime_result
                            .solver_type
                        ),
                        "excel_format": (
                            runtime_result
                            .excel_format
                        ),
                        "evidence_count": len(
                            evidence_bundle
                            .evidences
                        ),
                        "answer": (
                            answer.answer
                        ),
                    },
                ),
            ),
        }

    return excel_solver_node