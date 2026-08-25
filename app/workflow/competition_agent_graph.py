from __future__ import annotations

from typing import Literal

from langgraph.graph import (
    END,
    START,
    StateGraph,
)
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
)

from app.services.competition_excel_runtime import (
    CompetitionExcelRuntimeResources,
)
from app.workflow.competition_excel_nodes import (
    build_excel_solver_node,
)
from app.workflow.competition_decision_nodes import (
    build_answer_node,
    build_sufficiency_node,
    citation_node,
    conflict_refuse_node,
    deterministic_refuse_node,
    readiness_node,
    semantic_refuse_node,
    failed_result_node,
)
from app.workflow.competition_runtime import (
    CompetitionAgentModelResources,
    CompetitionAgentRuntimeResources,
)
from app.workflow.competition_state import (
    CompetitionAgentState,
)
from app.workflow.competition_text_nodes import (
    build_evidence_assembly_node,
    build_retrieval_node,
    build_source_resolution_node,
)


# ============================================================
# Routers
# ============================================================


def route_after_source_resolution(
    state: CompetitionAgentState,
) -> Literal[
    "retrieval",
    "excel_solver",
    "readiness",
]:
    resolution = state.get(
        "source_resolution"
    )

    # Source 无法安全定位
    if resolution is None:
        return "readiness"

    question = state[
        "question"
    ]

    if (
        question.source_type
        == "excel"
    ):
        return "excel_solver"

    if (
        question.source_type
        in {
            "word",
            "pdf",
        }
    ):
        return "retrieval"

    raise RuntimeError(
        "未知 Competition source_type："
        f"{question.source_type}"
    )


def route_after_readiness(
    state: CompetitionAgentState,
) -> Literal[
    "refuse",
    "sufficiency",
]:
    decision = state[
        "readiness"
    ]

    if decision.status == "refused":
        return "refuse"

    return "sufficiency"


def route_after_sufficiency(
    state: CompetitionAgentState,
) -> Literal[
    "failed",
    "refuse",
    "answer",
]:
    if (
        state.get("failure")
        is not None
    ):
        return "failed"

    assessment = state.get(
        "sufficiency"
    )

    if assessment is None:
        raise RuntimeError(
            "Sufficiency Router "
            "缺少 sufficiency"
        )

    if (
        assessment.status
        == "insufficient"
    ):
        return "refuse"

    return "answer"


def route_after_answer(
    state: CompetitionAgentState,
) -> Literal[
    "failed",
    "conflict",
    "citation",
]:
    if (
        state.get("failure")
        is not None
    ):
        return "failed"

    sufficiency = state[
        "sufficiency"
    ]

    answer = state.get(
        "answer"
    )

    if answer is None:
        raise RuntimeError(
            "Answer Router "
            "缺少 answer"
        )

    if (
        sufficiency.supported_answer
        is None
    ):
        raise RuntimeError(
            "sufficient 状态缺少 "
            "supported_answer"
        )

    if (
        sufficiency.supported_answer
        != answer.answer
    ):
        return "conflict"

    return "citation"

def route_after_citation(
    state: CompetitionAgentState,
) -> Literal[
    "failed",
    "complete",
]:
    if (
        state.get("failure")
        is not None
    ):
        return "failed"

    if (
        state.get("result")
        is None
    ):
        raise RuntimeError(
            "Citation 成功后缺少 result"
        )

    return "complete"

# ============================================================
# Graph
# ============================================================


def build_competition_agent_graph(
    *,
    runtime_resources: (
        CompetitionAgentRuntimeResources
    ),
    excel_resources: (
        CompetitionExcelRuntimeResources
    ),
    model_resources: (
        CompetitionAgentModelResources
    ),
    checkpointer: (
        BaseCheckpointSaver
        | None
    ) = None,
):
    """
    Competition Trusted Document Agent V1。
    """

    builder = StateGraph(
        CompetitionAgentState
    )

    # -------------------------
    # Deterministic Retrieval
    # -------------------------

    builder.add_node(
        "source_resolution",
        build_source_resolution_node(
            runtime_resources
        ),
    )

    builder.add_node(
        "retrieval",
        build_retrieval_node(
            runtime_resources
        ),
    )

    builder.add_node(
        "evidence_assembly",
        build_evidence_assembly_node(
            runtime_resources
        ),
    )

    # -------------------------
    # Trusted Decision
    # -------------------------

    builder.add_node(
        "readiness",
        readiness_node,
    )

    builder.add_node(
        "deterministic_refuse",
        deterministic_refuse_node,
    )

    builder.add_node(
        "sufficiency",
        build_sufficiency_node(
            model_resources
        ),
    )

    builder.add_node(
        "semantic_refuse",
        semantic_refuse_node,
    )

    builder.add_node(
        "answer",
        build_answer_node(
            model_resources
        ),
    )

    builder.add_node(
        "conflict_refuse",
        conflict_refuse_node,
    )

    builder.add_node(
        "citation",
        citation_node,
    )

    # ========================================================
    # failed_node
    # ========================================================

    builder.add_node(
        "failed",
        failed_result_node,
    )

    # ========================================================
    # excel solver node
    # ========================================================

    builder.add_node(
        "excel_solver",
        build_excel_solver_node(
            excel_resources
        ),
    )

    # ========================================================
    # Graph Flow
    # ========================================================

    builder.add_edge(
        START,
        "source_resolution",
    )

    # Source routing
    builder.add_conditional_edges(
        "source_resolution",
        route_after_source_resolution,
        {
            "retrieval": (
                "retrieval"
            ),
            "excel_solver": (
                "excel_solver"
            ),
            "readiness": (
                "readiness"
            ),
        },
    )

    # Retrieval path
    builder.add_edge(
        "retrieval",
        "evidence_assembly",
    )

    builder.add_edge(
        "evidence_assembly",
        "readiness",
    )

    # Deterministic gate
    builder.add_conditional_edges(
        "readiness",
        route_after_readiness,
        {
            "refuse": (
                "deterministic_refuse"
            ),
            "sufficiency": (
                "sufficiency"
            ),
        },
    )

    builder.add_edge(
        "deterministic_refuse",
        END,
    )

    # Semantic gate
    builder.add_conditional_edges(
        "sufficiency",
        route_after_sufficiency,
        {
            "failed": "failed",
            "refuse": (
                "semantic_refuse"
            ),
            "answer": (
                "answer"
            ),
        },
    )

    builder.add_edge(
        "semantic_refuse",
        END,
    )

    # Answer audit
    builder.add_conditional_edges(
        "answer",
        route_after_answer,
        {
            "failed": "failed",
            "conflict": (
                "conflict_refuse"
            ),
            "citation": (
                "citation"
            ),
        },
    )

    builder.add_edge(
        "excel_solver",
        "citation",
    )

    builder.add_edge(
        "conflict_refuse",
        END,
    )

    builder.add_conditional_edges(
        "citation",
        route_after_citation,
        {
            "failed": "failed",
            "complete": END,
        },
    )

    builder.add_edge(
        "failed",
        END,
    )

    return builder.compile(
        checkpointer=checkpointer,
    )