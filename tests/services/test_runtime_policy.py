from __future__ import annotations

import pytest

from app.schemas.trust import (
    VerificationIssue,
    VerificationReport,
)
from app.services.runtime_policy import (
    RuntimeRiskPolicy,
    RuntimeRiskPolicyError,
)


# ============================================================
# Week7 - Step6.1
#
# Deterministic Risk Policy Tests
# ============================================================


class SequentialPolicyIdFactory:
    def __init__(
        self,
    ) -> None:
        self.counter = 0

    def new_id(
        self,
        prefix: str,
    ) -> str:
        self.counter += 1

        return (
            f"{prefix}_{self.counter}"
        )


def _build_policy(
) -> RuntimeRiskPolicy:
    return RuntimeRiskPolicy(
        id_factory=(
            SequentialPolicyIdFactory()
        )
    )


def _passed_verification(
) -> VerificationReport:
    return VerificationReport(
        passed=True,
        numeric_verified=True,
        evidence_verified=True,
        citation_verified=True,
        evidence_sufficient=True,
        issues=(),
    )


def _failed_verification(
) -> VerificationReport:
    return VerificationReport(
        passed=False,
        numeric_verified=False,
        evidence_verified=False,
        citation_verified=False,
        evidence_sufficient=False,
        issues=(
            VerificationIssue(
                issue_type=(
                    "missing_evidence"
                ),
                severity="error",
                message=(
                    "测试：缺少可信证据"
                ),
            ),
        ),
    )


# ============================================================
# Low Risk
# ============================================================


def test_low_risk_verified_request_is_allowed(
) -> None:
    policy = _build_policy()

    decision = policy.evaluate(
        risk_level="low",
        verification_report=(
            _passed_verification()
        ),
    )

    assert (
        decision.action
        == "allow"
    )

    assert (
        decision.risk_level
        == "low"
    )

    assert (
        decision.verification_passed
        is True
    )

    assert (
        decision.human_review
        is None
    )


# ============================================================
# Medium Risk
# ============================================================


def test_medium_risk_verified_request_is_allowed(
) -> None:
    policy = _build_policy()

    decision = policy.evaluate(
        risk_level="medium",
        verification_report=(
            _passed_verification()
        ),
    )

    assert (
        decision.action
        == "allow"
    )

    assert (
        decision.risk_level
        == "medium"
    )

    assert (
        decision.verification_passed
        is True
    )

    assert (
        decision.human_review
        is None
    )


# ============================================================
# High Risk
# ============================================================


def test_high_risk_verified_request_requires_human(
) -> None:
    policy = _build_policy()

    decision = policy.evaluate(
        risk_level="high",
        verification_report=(
            _passed_verification()
        ),
    )

    assert (
        decision.action
        == "require_human"
    )

    assert (
        decision.risk_level
        == "high"
    )

    assert (
        decision.verification_passed
        is True
    )

    assert (
        decision.human_review
        is not None
    )

    assert (
        decision
        .human_review
        .review_id
        == "review_1"
    )

    assert (
        decision
        .human_review
        .required_reviewer_role
        == "reviewer"
    )


def test_high_risk_review_preserves_claim_ids(
) -> None:
    policy = _build_policy()

    decision = policy.evaluate(
        risk_level="high",
        verification_report=(
            _passed_verification()
        ),
        claim_ids=(
            "claim_revenue",
            "claim_margin",
        ),
    )

    assert (
        decision.human_review
        is not None
    )

    assert (
        decision
        .human_review
        .claim_ids
        == (
            "claim_revenue",
            "claim_margin",
        )
    )


def test_high_risk_can_require_admin_review(
) -> None:
    policy = RuntimeRiskPolicy(
        id_factory=(
            SequentialPolicyIdFactory()
        ),
        high_risk_reviewer_role=(
            "admin"
        ),
    )

    decision = policy.evaluate(
        risk_level="high",
        verification_report=(
            _passed_verification()
        ),
    )

    assert (
        decision.human_review
        is not None
    )

    assert (
        decision
        .human_review
        .required_reviewer_role
        == "admin"
    )


# ============================================================
# Verification Hard Gate
#
# Verification FAIL：
#
# low / medium / high
# 全部不得放行。
# ============================================================


@pytest.mark.parametrize(
    "risk_level",
    [
        "low",
        "medium",
        "high",
    ],
)
def test_failed_verification_is_always_refused(
    risk_level,
) -> None:
    policy = _build_policy()

    decision = policy.evaluate(
        risk_level=risk_level,
        verification_report=(
            _failed_verification()
        ),
    )

    assert (
        decision.action
        == "refuse"
    )

    assert (
        decision.risk_level
        == risk_level
    )

    assert (
        decision.verification_passed
        is False
    )

    assert (
        decision.human_review
        is None
    )


# ============================================================
# Invalid Risk
# ============================================================


def test_unknown_risk_level_is_rejected(
) -> None:
    policy = _build_policy()

    with pytest.raises(
        RuntimeRiskPolicyError,
        match="未知 RiskLevel",
    ):
        policy.evaluate(
            risk_level=(
                "extreme"  # type: ignore[arg-type]
            ),
            verification_report=(
                _passed_verification()
            ),
        )