from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Literal,
    Protocol,
)
from uuid import uuid4

from app.schemas.trust import (
    HumanReviewRequest,
    PolicyDecision,
    RiskLevel,
    VerificationReport,
)


# ============================================================
# Week7 - Step 6.1
#
# Deterministic Risk Policy Core
#
# 这一层只负责：
#
# RiskLevel
#      +
# VerificationReport
#      ↓
# PolicyDecision
#
# 当前策略：
#
# verification failed
#     → refuse
#
# low + verified
#     → allow
#
# medium + verified
#     → allow
#
# high + verified
#     → require_human
#
# 暂时不负责：
#
# - AgentState
# - Runtime Graph
# - Checkpoint
# - Human Decision
# - Resume
#
# 这些在 Step6.2 / Step6.3 接入。
# ============================================================


ReviewerRole = Literal[
    "reviewer",
    "admin",
]


class RuntimeRiskPolicyError(
    ValueError
):
    """Risk Policy 基础异常。"""


class PolicyIdFactory(
    Protocol
):
    def new_id(
        self,
        prefix: str,
    ) -> str:
        """生成 Policy / HITL 所需 ID。"""


@dataclass(
    frozen=True,
    slots=True,
)
class UUIDPolicyIdFactory:
    """生产环境默认 ID Factory。"""

    def new_id(
        self,
        prefix: str,
    ) -> str:
        return (
            f"{prefix}_{uuid4().hex}"
        )


@dataclass(
    frozen=True,
    slots=True,
)
class RuntimeRiskPolicy:
    """可信校验完成后的确定性风险策略。"""

    id_factory: PolicyIdFactory = field(
        default_factory=(
            UUIDPolicyIdFactory
        )
    )

    high_risk_reviewer_role: (
        ReviewerRole
    ) = "reviewer"

    def evaluate(
        self,
        *,
        risk_level: RiskLevel,
        verification_report: (
            VerificationReport
        ),
        claim_ids: tuple[
            str,
            ...
        ] = (),
    ) -> PolicyDecision:
        """根据风险等级和可信校验结果生成最终策略决策。"""

        self._validate_risk_level(
            risk_level
        )

        # ====================================================
        # Rule 1
        #
        # Trust Verification 是 Hard Gate。
        #
        # 不管 Risk Level 多低，
        # 可信校验没有通过都不能放行。
        # ====================================================

        if not (
            verification_report.passed
        ):
            return PolicyDecision(
                action="refuse",
                risk_level=risk_level,
                reason=(
                    "可信校验未通过，"
                    "Risk Policy 拒绝放行"
                ),
                verification_passed=False,
                human_review=None,
            )

        # ====================================================
        # Rule 2
        #
        # Low Risk：
        # 已通过所有可信校验，可以自动执行。
        # ====================================================

        if risk_level == "low":
            return PolicyDecision(
                action="allow",
                risk_level="low",
                reason=(
                    "低风险请求已通过可信校验，"
                    "允许自动执行"
                ),
                verification_passed=True,
                human_review=None,
            )

        # ====================================================
        # Rule 3
        #
        # Medium Risk：
        #
        # 第一版策略中，
        # 只有 Verification 全部通过才允许自动执行。
        #
        # VerificationReport 的 Schema 已保证：
        #
        # passed=True
        #     ↓
        # numeric_verified
        # evidence_verified
        # citation_verified
        # evidence_sufficient
        #
        # 全部为 True。
        # ====================================================

        if risk_level == "medium":
            return PolicyDecision(
                action="allow",
                risk_level="medium",
                reason=(
                    "中风险请求已通过全部可信校验，"
                    "允许自动执行"
                ),
                verification_passed=True,
                human_review=None,
            )

        # ====================================================
        # Rule 4
        #
        # High Risk：
        #
        # 即使 Answer Trust Verification 已通过，
        # 仍不能直接生成最终答案。
        #
        # 必须：
        #
        # require_human
        #      ↓
        # HumanReviewRequest
        # ====================================================

        review_request = (
            HumanReviewRequest(
                review_id=(
                    self.id_factory.new_id(
                        "review"
                    )
                ),
                risk_level="high",
                reason=(
                    "高风险请求即使已通过可信校验，"
                    "仍需要人工复核后才能继续"
                ),
                claim_ids=claim_ids,
                required_reviewer_role=(
                    self
                    .high_risk_reviewer_role
                ),
            )
        )

        return PolicyDecision(
            action="require_human",
            risk_level="high",
            reason=(
                "高风险请求需要人工复核"
            ),
            verification_passed=True,
            human_review=(
                review_request
            ),
        )

    @staticmethod
    def _validate_risk_level(
        risk_level: str,
    ) -> None:
        if risk_level not in {
            "low",
            "medium",
            "high",
        }:
            raise RuntimeRiskPolicyError(
                "未知 RiskLevel："
                f"{risk_level}"
            )