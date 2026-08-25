from __future__ import annotations

from typing import Literal

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from app.workflow.competition_decision_nodes import (
    build_answer_node,
    build_sufficiency_node,
    citation_node,
    conflict_refuse_node,
    deterministic_refuse_node,
    readiness_node,
    semantic_refuse_node,
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
    "readiness",
]:
    """
    Source 找不到时：

    不进入 Retrieval，
    直接交给 Readiness，
    最终生成 source_unresolved refusal。
    """

    if (
        state.get(
            "source_resolution"
        )
        is None
    ):
        return "readiness"

    return "retrieval"


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
    "refuse",
    "answer",
]:
    assessment = state[
        "sufficiency"
    ]

    if (
        assessment.status
        == "insufficient"
    ):
        return "refuse"

    return "answer"


def route_after_answer(
    state: CompetitionAgentState,
) -> Literal[
    "conflict",
    "citation",
]:
    sufficiency = state[
        "sufficiency"
    ]

    answer = state[
        "answer"
    ]

    if (
        sufficiency.supported_answer
        is None
    ):
        raise RuntimeError(
            "sufficient 状态"
            "缺少 supported_answer"
        )

    if (
        sufficiency.supported_answer
        != answer.answer
    ):
        return "conflict"

    return "citation"


# ============================================================
# Graph
# ============================================================


def build_competition_agent_graph(
    *,
    runtime_resources: (
        CompetitionAgentRuntimeResources
    ),
    model_resources: (
        CompetitionAgentModelResources
    ),
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
            "conflict": (
                "conflict_refuse"
            ),
            "citation": (
                "citation"
            ),
        },
    )

    builder.add_edge(
        "conflict_refuse",
        END,
    )

    builder.add_edge(
        "citation",
        END,
    )

    return builder.compile()