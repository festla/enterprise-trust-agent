from __future__ import annotations

from app.services.competition_abstention import (
    assess_competition_evidence_readiness,
)
from app.workflow.competition_agent_state import (
    CompetitionAgentState,
)
from app.schemas.competition_runtime_result import (
    CompetitionRefusedResult,
)

def readiness_node(
    state: CompetitionAgentState,
) -> dict[str, object]:
    """
    确定性 Evidence Readiness Node。

    读取：
        question
        resolved_source_id
        retrieval_hit_count
        evidence_bundle

    写入：
        readiness
    """

    decision = (
        assess_competition_evidence_readiness(
            question=(
                state["question"]
            ),
            resolved_source_id=(
                state[
                    "resolved_source_id"
                ]
            ),
            retrieval_hit_count=(
                state[
                    "retrieval_hit_count"
                ]
            ),
            evidence_bundle=(
                state[
                    "evidence_bundle"
                ]
            ),
        )
    )

    return {
        "readiness": decision,
    }

def deterministic_refuse_node(
    state: CompetitionAgentState,
) -> dict[str, object]:
    """
    确定性拒答 Node。

    只有 readiness 已明确判定 refused
    时才应该进入这里。

    读取：
        question
        readiness

    写入：
        result
    """

    decision = state["readiness"]

    if decision.status != "refused":
        raise RuntimeError(
            "deterministic_refuse_node "
            "只能处理 refused 状态"
        )

    if decision.refusal_code is None:
        raise RuntimeError(
            "refused 状态缺少 refusal_code"
        )

    result = CompetitionRefusedResult(
        case_id=(
            state["question"].case_id
        ),
        answer_text=(
            "当前证据不足，"
            "无法可靠作答。"
        ),
        refusal_code=(
            decision.refusal_code
        ),
        refusal_reason=(
            decision.reason
        ),
    )

    return {
        "result": result,
    }