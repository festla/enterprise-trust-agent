from __future__ import annotations

from collections.abc import (
    Callable,
)

from app.rag.competition_answer_generation import (
    CompetitionAnswerGenerationError,
    generate_competition_answer,
)
from app.rag.competition_sufficiency import (
    CompetitionEvidenceSufficiencyError,
    assess_competition_semantic_sufficiency,
)
from app.rag.bailian_answer_provider import (
    BailianAnswerProviderRequestError,
    BailianAnswerProviderResponseError,
)
from app.schemas.competition_agent_execution import (
    CompetitionAgentFailure,
    CompetitionAgentTraceEvent,
)
from app.schemas.competition_runtime_result import (
    CompetitionAnsweredResult,
    CompetitionFailedResult,
    CompetitionRefusedResult,
)
from app.services.competition_abstention import (
    assess_competition_evidence_readiness,
)
from app.services.competition_citation import (
    CompetitionCitationError,
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
# Failure Helper
# ============================================================

def _build_failure_update(
    *,
    stage: str,
    failure_code: str,
    retryable: bool,
    exc: Exception,
) -> dict[str, object]:
    failure = CompetitionAgentFailure(
        stage=stage,
        failure_code=failure_code,
        retryable=retryable,
        error_type=(
            type(exc).__name__
        ),
        message=str(exc),
    )

    trace_event = (
        CompetitionAgentTraceEvent(
            stage=stage,
            outcome="system_failed",
            message=(
                f"{stage} failed"
            ),
            details={
                "failure_code": (
                    failure_code
                ),
                "retryable": (
                    retryable
                ),
                "error_type": (
                    type(exc).__name__
                ),
            },
        )
    )

    return {
        "failure": failure,
        "trace": (
            trace_event,
        ),
    }

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

    outcome = (
        "success"
        if (
            decision.status
            == "ready_for_generation"
        )
        else "business_blocked"
    )

    return {
        "readiness": decision,
        "trace": (
            CompetitionAgentTraceEvent(
                stage="readiness",
                outcome=outcome,
                message=(
                    "Evidence readiness evaluated"
                ),
                details={
                    "status": (
                        decision.status
                    ),
                    "evidence_count": (
                        decision.evidence_count
                    ),
                    "refusal_code": (
                        decision.refusal_code
                    ),
                },
            ),
        ),
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

        try:
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

        except BailianAnswerProviderRequestError as exc:
            return _build_failure_update(
                stage="sufficiency",
                failure_code=(
                    "provider_request_error"
                ),
                retryable=True,
                exc=exc,
            )

        except BailianAnswerProviderResponseError as exc:
            return _build_failure_update(
                stage="sufficiency",
                failure_code=(
                    "provider_response_error"
                ),
                retryable=False,
                exc=exc,
            )

        except CompetitionEvidenceSufficiencyError as exc:
            return _build_failure_update(
                stage="sufficiency",
                failure_code=(
                    "validation_error"
                ),
                retryable=False,
                exc=exc,
            )


        outcome = (
            "success"
            if assessment.status
            == "sufficient"
            else "business_blocked"
        )

        return {
            "sufficiency": (
                assessment
            ),
            "trace": (
                CompetitionAgentTraceEvent(
                    stage="sufficiency",
                    outcome=outcome,
                    message=(
                        "Semantic evidence "
                        "sufficiency evaluated"
                    ),
                    details={
                        "status": (
                            assessment.status
                        ),
                        "supported_answer": (
                            assessment
                            .supported_answer
                        ),
                        "citation_count": len(
                            assessment
                            .citation_ids
                        ),
                    },
                ),
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

        try:
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

        except BailianAnswerProviderRequestError as exc:
            return _build_failure_update(
                stage="answer",
                failure_code=(
                    "provider_request_error"
                ),
                retryable=True,
                exc=exc,
            )

        except BailianAnswerProviderResponseError as exc:
            return _build_failure_update(
                stage="answer",
                failure_code=(
                    "provider_response_error"
                ),
                retryable=False,
                exc=exc,
            )

        except CompetitionAnswerGenerationError as exc:
            return _build_failure_update(
                stage="answer",
                failure_code=(
                    "validation_error"
                ),
                retryable=False,
                exc=exc,
            )


        return {
            "answer": answer,
            "trace": (
                CompetitionAgentTraceEvent(
                    stage="answer",
                    outcome="success",
                    message=(
                        "Answer generated successfully"
                    ),
                    details={
                        "answer": (
                            answer.answer
                        ),
                        "citation_count": len(
                            answer.citation_ids
                        ),
                        "generator_id": (
                            answer.generator_id
                        ),
                    },
                ),
            ),
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

    try:
        citations = (
            build_competition_citation_bundle(
                answer=answer,
                evidence_bundle=(
                    evidence_bundle
                ),
            )
        )

    except CompetitionCitationError as exc:
        return _build_failure_update(
            stage="citation",
            failure_code=(
                "citation_binding_error"
            ),
            retryable=False,
            exc=exc,
        )


    result = (
        CompetitionAnsweredResult(
            case_id=(
                state["question"].case_id
            ),
            answer=answer,
            citations=citations,
        )
    )

    return {
        "citations": citations,
        "result": result,
        "trace": (
            CompetitionAgentTraceEvent(
                stage="citation",
                outcome="success",
                message=(
                    "Citation binding completed"
                ),
                details={
                    "citation_count": len(
                        citations.citations
                    ),
                },
            ),
        ),
    }

# ============================================================
# Failed Node
# ============================================================

def failed_result_node(
    state: CompetitionAgentState,
) -> dict[str, object]:
    failure = state.get(
        "failure"
    )

    if failure is None:
        raise RuntimeError(
            "Failed Node 缺少 failure"
        )

    result = CompetitionFailedResult(
        case_id=(
            state["question"].case_id
        ),
        failure_stage=(
            failure.stage
        ),
        failure_code=(
            failure.failure_code
        ),
        retryable=(
            failure.retryable
        ),
        error_type=(
            failure.error_type
        ),
        error_message=(
            failure.message
        ),
    )

    return {
        "result": result,
    }