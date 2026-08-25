from __future__ import annotations

from collections.abc import (
    Callable,
)

from app.rag.competition_answer_generation import (
    generate_competition_answer,
)
from app.rag.competition_sufficiency import (
    assess_competition_semantic_sufficiency,
)
from app.schemas.competition_runtime_result import (
    CompetitionAnsweredResult,
    CompetitionRefusedResult,
)
from app.services.competition_abstention import (
    assess_competition_evidence_readiness,
)
from app.services.competition_citation import (
    build_competition_citation_bundle,
)
from app.workflow.competition_runtime import (
    CompetitionAgentModelResources,
)
from app.workflow.competition_state import (
    CompetitionAgentState,
)


CompetitionAgentNode = Callable[
    [CompetitionAgentState],
    dict[str, object],
]


# ============================================================
# Readiness
# ============================================================


def readiness_node(
    state: CompetitionAgentState,
) -> dict[str, object]:
    """
    第一层确定性 Evidence Gate。

    可以处理：
        source unresolved
        no retrieval hits
        no evidence
        source mismatch
    """

    resolution = state.get(
        "source_resolution"
    )

    retrieval = state.get(
        "retrieval"
    )

    evidence_bundle = state.get(
        "evidence_bundle"
    )

    resolved_source_id = (
        resolution.source_id
        if resolution is not None
        else None
    )

    retrieval_hit_count = (
        retrieval.retrieval_hit_count
        if retrieval is not None
        else 0
    )

    decision = (
        assess_competition_evidence_readiness(
            question=(
                state["question"]
            ),
            resolved_source_id=(
                resolved_source_id
            ),
            retrieval_hit_count=(
                retrieval_hit_count
            ),
            evidence_bundle=(
                evidence_bundle
            ),
        )
    )

    return {
        "readiness": decision,
    }


# ============================================================
# Deterministic Refusal
# ============================================================


def deterministic_refuse_node(
    state: CompetitionAgentState,
) -> dict[str, object]:
    decision = state[
        "readiness"
    ]

    if decision.status != "refused":
        raise RuntimeError(
            "Deterministic Refuse Node "
            "收到非 refused 状态"
        )

    if decision.refusal_code is None:
        raise RuntimeError(
            "refused 状态缺少 "
            "refusal_code"
        )

    result = (
        CompetitionRefusedResult(
            case_id=(
                state[
                    "question"
                ].case_id
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
    )

    return {
        "result": result,
    }


# ============================================================
# Semantic Sufficiency
# ============================================================


def build_sufficiency_node(
    model_resources: (
        CompetitionAgentModelResources
    ),
) -> CompetitionAgentNode:

    def sufficiency_node(
        state: CompetitionAgentState,
    ) -> dict[str, object]:
        evidence_bundle = state.get(
            "evidence_bundle"
        )

        if evidence_bundle is None:
            raise RuntimeError(
                "Sufficiency Node "
                "缺少 EvidenceBundle"
            )

        assessment = (
            assess_competition_semantic_sufficiency(
                question=(
                    state["question"]
                ),
                evidence_bundle=(
                    evidence_bundle
                ),
                provider=(
                    model_resources
                    .sufficiency_provider
                ),
            )
        )

        return {
            "sufficiency": (
                assessment
            ),
        }

    return sufficiency_node


# ============================================================
# Semantic Refusal
# ============================================================


def semantic_refuse_node(
    state: CompetitionAgentState,
) -> dict[str, object]:
    assessment = state[
        "sufficiency"
    ]

    result = (
        CompetitionRefusedResult(
            case_id=(
                state[
                    "question"
                ].case_id
            ),
            answer_text=(
                "当前证据不足以"
                "唯一确定答案，"
                "拒绝猜测。"
            ),
            refusal_code=(
                "semantic_evidence_insufficient"
            ),
            refusal_reason=(
                assessment.reason
            ),
        )
    )

    return {
        "result": result,
    }


# ============================================================
# Answer
# ============================================================


def build_answer_node(
    model_resources: (
        CompetitionAgentModelResources
    ),
) -> CompetitionAgentNode:

    def answer_node(
        state: CompetitionAgentState,
    ) -> dict[str, object]:
        evidence_bundle = state.get(
            "evidence_bundle"
        )

        if evidence_bundle is None:
            raise RuntimeError(
                "Answer Node "
                "缺少 EvidenceBundle"
            )

        answer = (
            generate_competition_answer(
                question=(
                    state["question"]
                ),
                evidence_bundle=(
                    evidence_bundle
                ),
                provider=(
                    model_resources
                    .answer_provider
                ),
            )
        )

        return {
            "answer": answer,
        }

    return answer_node


# ============================================================
# Judge / Generator Conflict
# ============================================================


def conflict_refuse_node(
    state: CompetitionAgentState,
) -> dict[str, object]:
    sufficiency = state[
        "sufficiency"
    ]

    answer = state[
        "answer"
    ]

    result = (
        CompetitionRefusedResult(
            case_id=(
                state[
                    "question"
                ].case_id
            ),
            answer_text=(
                "证据审计结果与"
                "回答生成结果不一致，"
                "拒绝返回不确定答案。"
            ),
            refusal_code=(
                "answer_sufficiency_conflict"
            ),
            refusal_reason=(
                "Evidence Sufficiency Judge "
                f"支持选项 "
                f"{sufficiency.supported_answer}，"
                "但 Answer Generator "
                f"生成选项 {answer.answer}。"
            ),
        )
    )

    return {
        "result": result,
    }


# ============================================================
# Citation
# ============================================================


def citation_node(
    state: CompetitionAgentState,
) -> dict[str, object]:
    evidence_bundle = state.get(
        "evidence_bundle"
    )

    if evidence_bundle is None:
        raise RuntimeError(
            "Citation Node "
            "缺少 EvidenceBundle"
        )

    answer = state[
        "answer"
    ]

    citations = (
        build_competition_citation_bundle(
            answer=answer,
            evidence_bundle=(
                evidence_bundle
            ),
        )
    )

    result = (
        CompetitionAnsweredResult(
            case_id=(
                state[
                    "question"
                ].case_id
            ),
            answer=answer,
            citations=citations,
        )
    )

    return {
        "citations": citations,
        "result": result,
    }