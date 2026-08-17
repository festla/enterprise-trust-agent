from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)

import pytest

from app.schemas.agent_runtime import (
    AgentState,
)

from app.schemas.trust import (
    AnswerDraft,
    Claim,
    ClaimSupport,
    VerificationIssue,
    VerificationReport,
)

from app.services.registry import (
    RegistryBundle,
)

from app.services.runtime_completion import (
    RuntimeAnswerGenerationError,
    RuntimeAnswerGenerator,
)


# ============================================================
# Step 3.3
#
# Generator Hard Trust Gate
#
# 这些测试故意不通过 AgentRuntime.run()。
#
# 因为我们要验证的正是：
#
# 即使调用方绕过 Runtime / LangGraph，
# 直接调用 RuntimeAnswerGenerator.generate()，
# Generator 自己也不会生成未经验证的答案。
# ============================================================


def _now(
) -> datetime:
    return datetime(
        2026,
        8,
        15,
        14,
        0,
        tzinfo=timezone.utc,
    )


def _build_answer_draft(
) -> AnswerDraft:
    return AnswerDraft(
        draft_id=(
            "draft_trust_gate"
        ),

        draft_type="financial",

        claims=(
            Claim(
                claim_id=(
                    "claim_trust_gate"
                ),

                claim_type=(
                    "financial_fact"
                ),

                claim_text=(
                    "美的集团2024年"
                    "营业收入为"
                    "407149600000元"
                ),

                support=(
                    ClaimSupport(
                        fact_ids=(
                            "fact_trust_gate",
                        ),

                        citation_ids=(
                            "citation_trust_gate",
                        ),
                    )
                ),

                confidence=1.0,
            ),
        ),
    )


def _build_state(
    *,
    verification_report=(
        None
    ),
) -> AgentState:
    now = _now()

    return AgentState(
        request_id=(
            "request_trust_gate"
        ),

        trace_id=(
            "trace_trust_gate"
        ),

        run_id=(
            "run_trust_gate"
        ),

        thread_id=(
            "thread_trust_gate"
        ),

        query=(
            "美的集团2024年"
            "营业收入是多少？"
        ),

        intent=(
            "financial_fact"
        ),

        answer_draft=(
            _build_answer_draft()
        ),

        verification_report=(
            verification_report
        ),

        status="verifying",

        current_node=(
            "verify_answer"
        ),

        next_node=(
            "generate_answer"
        ),

        started_at=now,

        updated_at=now,
    )


def _build_generator(
) -> RuntimeAnswerGenerator:
    # 这两个 Hard Gate 测试都会在真正读取
    # RegistryBundle 之前抛出异常，
    # 因此空 Registry 足够。
    return RuntimeAnswerGenerator(
        registry_bundle=(
            RegistryBundle()
        )
    )


# ============================================================
# Case 1
#
# AnswerDraft 已存在，
# 但完全没有 VerificationReport。
#
# 旧逻辑：
#
#   AnswerDraft
#       ↓
#   Generator
#
# 可能继续执行。
#
# Step3.3：
#
#   AnswerDraft
#       ↓
#   VerificationReport missing
#       ↓
#      BLOCK
# ============================================================


def test_generate_rejects_missing_verification_report(
) -> None:
    generator = (
        _build_generator()
    )

    state = _build_state(
        verification_report=None
    )

    with pytest.raises(
        RuntimeAnswerGenerationError,
        match=(
            "verification_report"
        ),
    ):
        generator.generate(
            state
        )


# ============================================================
# Case 2
#
# VerificationReport 虽然存在，
# 但 passed=False。
#
# 这种状态同样不能发布最终答案。
# ============================================================


def test_generate_rejects_failed_verification_report(
) -> None:
    report = VerificationReport(
        passed=False,

        numeric_verified=False,

        evidence_verified=True,

        citation_verified=True,

        evidence_sufficient=False,

        issues=(
            VerificationIssue(
                issue_type=(
                    "unsupported_claim"
                ),

                severity="error",

                message=(
                    "回答中的 Claim "
                    "无法被可信证据支持"
                ),

                claim_id=(
                    "claim_trust_gate"
                ),
            ),
        ),
    )

    generator = (
        _build_generator()
    )

    state = _build_state(
        verification_report=report
    )

    with pytest.raises(
        RuntimeAnswerGenerationError,
        match=(
            "VerificationReport 已通过"
        ),
    ):
        generator.generate(
            state
        )