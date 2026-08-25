from __future__ import annotations

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from app.workflow.competition_runtime import (
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


def build_competition_text_preparation_graph(
    resources: (
        CompetitionAgentRuntimeResources
    ),
):
    """
    Competition Text Agent Graph V1。

    当前阶段：

        START
          ↓
        Source Resolution
          ↓
        Retrieval
          ↓
        Evidence Assembly
          ↓
        END

    8A4 会继续接：
        Evidence Readiness
        Sufficiency
        Answer / Refuse
        Citation
    """

    builder = StateGraph(
        CompetitionAgentState
    )

    builder.add_node(
        "source_resolution",
        build_source_resolution_node(
            resources
        ),
    )

    builder.add_node(
        "retrieval",
        build_retrieval_node(
            resources
        ),
    )

    builder.add_node(
        "evidence_assembly",
        build_evidence_assembly_node(
            resources
        ),
    )

    builder.add_edge(
        START,
        "source_resolution",
    )

    builder.add_edge(
        "source_resolution",
        "retrieval",
    )

    builder.add_edge(
        "retrieval",
        "evidence_assembly",
    )

    builder.add_edge(
        "evidence_assembly",
        END,
    )

    return builder.compile()