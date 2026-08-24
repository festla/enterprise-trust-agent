from __future__ import annotations

from app.rag.competition_answer_generation import (
    CompetitionAnswerProvider,
    generate_competition_answer,
)
from app.rag.competition_sufficiency import (
    CompetitionEvidenceSufficiencyProvider,
    assess_competition_semantic_sufficiency,
)
from app.schemas.competition import (
    CompetitionQuestion,
)
from app.schemas.competition_evidence import (
    CompetitionEvidenceBundle,
)
from app.schemas.competition_runtime_result import (
    CompetitionAnsweredResult,
    CompetitionEvidenceReadinessDecision,
    CompetitionExecutionResult,
    CompetitionRefusedResult,
)
from app.services.competition_citation import (
    build_competition_citation_bundle,
)


def assess_competition_evidence_readiness(
    *,
    question: CompetitionQuestion,
    resolved_source_id: str | None,
    retrieval_hit_count: int,
    evidence_bundle: (
        CompetitionEvidenceBundle
        | None
    ),
) -> CompetitionEvidenceReadinessDecision:
    """
    确定性 Evidence Gate。

    当前只处理能够明确判断的失败：

    - Source Resolver 没有定位来源；
    - Retriever 没有结果；
    - Retriever 有结果但没有形成 Evidence；
    - Evidence 来源与 resolved source 不一致。

    不在这里使用 BM25 score 阈值，
    避免拍脑袋设置“低于多少就拒答”。
    """

    if not resolved_source_id:
        return (
            CompetitionEvidenceReadinessDecision(
                status="refused",
                evidence_count=0,
                refusal_code=(
                    "source_unresolved"
                ),
                reason=(
                    "无法确定问题对应的"
                    "知识来源，拒绝猜测。"
                ),
            )
        )

    if retrieval_hit_count <= 0:
        return (
            CompetitionEvidenceReadinessDecision(
                status="refused",
                evidence_count=0,
                refusal_code=(
                    "no_retrieval_hits"
                ),
                reason=(
                    "指定知识来源中"
                    "未检索到可用内容。"
                ),
            )
        )

    if evidence_bundle is None:
        return (
            CompetitionEvidenceReadinessDecision(
                status="refused",
                evidence_count=0,
                refusal_code=(
                    "insufficient_evidence"
                ),
                reason=(
                    "检索结果未能形成"
                    "可用于回答的 Evidence。"
                ),
            )
        )

    evidence_count = len(
        evidence_bundle.evidences
    )

    evidence_source_ids = {
        evidence.source.source_id
        for evidence
        in evidence_bundle.evidences
    }

    if (
        resolved_source_id
        not in evidence_source_ids
    ):
        return (
            CompetitionEvidenceReadinessDecision(
                status="refused",
                evidence_count=(
                    evidence_count
                ),
                refusal_code=(
                    "evidence_source_mismatch"
                ),
                reason=(
                    "Evidence 与问题解析出的"
                    "知识来源不一致。"
                ),
            )
        )

    return (
        CompetitionEvidenceReadinessDecision(
            status="ready_for_generation",
            evidence_count=evidence_count,
            reason=(
                "Evidence 已通过"
                "确定性准备检查。"
            ),
        )
    )


def run_competition_answer_with_abstention(
    *,
    question: CompetitionQuestion,
    resolved_source_id: str | None,
    retrieval_hit_count: int,
    evidence_bundle: (
        CompetitionEvidenceBundle
        | None
    ),
    sufficiency_provider: (
        CompetitionEvidenceSufficiencyProvider
    ),
    answer_provider: (
        CompetitionAnswerProvider
    ),
) -> CompetitionExecutionResult:
    """
    Competition 可信回答执行链。

    1. Deterministic Evidence Gate
    2. Semantic Evidence Sufficiency
    3. Answer Generation
    4. Judge / Answer Consistency
    5. Citation Binding

    Evidence 不足时：
        不生成答案。

    Judge 与 Answer 冲突时：
        拒绝返回不确定答案。
    """

    decision = (
        assess_competition_evidence_readiness(
            question=question,
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

    if decision.status == "refused":
        assert (
            decision.refusal_code
            is not None
        )

        return CompetitionRefusedResult(
            case_id=question.case_id,
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

    if evidence_bundle is None:
        raise RuntimeError(
            "ready_for_generation "
            "不应缺少 evidence_bundle"
        )

    # --------------------------------
    # Semantic Sufficiency Gate
    # --------------------------------

    sufficiency = (
        assess_competition_semantic_sufficiency(
            question=question,
            evidence_bundle=(
                evidence_bundle
            ),
            provider=(
                sufficiency_provider
            ),
        )
    )

    if (
        sufficiency.status
        == "insufficient"
    ):
        return CompetitionRefusedResult(
            case_id=question.case_id,
            answer_text=(
                "当前证据不足以"
                "唯一确定答案，拒绝猜测。"
            ),
            refusal_code=(
                "semantic_evidence_insufficient"
            ),
            refusal_reason=(
                sufficiency.reason
            ),
        )

    if (
        sufficiency.supported_answer
        is None
    ):
        raise RuntimeError(
            "sufficient 状态必须"
            "包含 supported_answer"
        )

    # --------------------------------
    # Answer Generation
    # --------------------------------

    answer = (
        generate_competition_answer(
            question=question,
            evidence_bundle=(
                evidence_bundle
            ),
            provider=(
                answer_provider
            ),
        )
    )

    # --------------------------------
    # Judge / Generator Consistency
    # --------------------------------

    if (
        answer.answer
        != sufficiency.supported_answer
    ):
        return CompetitionRefusedResult(
            case_id=question.case_id,
            answer_text=(
                "证据审计结果与回答生成结果"
                "不一致，拒绝返回不确定答案。"
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

    # --------------------------------
    # Citation Binding
    # --------------------------------

    citations = (
        build_competition_citation_bundle(
            answer=answer,
            evidence_bundle=(
                evidence_bundle
            ),
        )
    )

    return CompetitionAnsweredResult(
        case_id=question.case_id,
        answer=answer,
        citations=citations,
    )