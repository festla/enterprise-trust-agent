import pytest

from pydantic import ValidationError

from app.schemas.trust import (
    AnswerDraft,
    Claim,
    ClaimSupport,
    HumanReviewRequest,
    PolicyDecision,
    VerificationIssue,
    VerificationReport,
)

def test_claim_support_accepts_fact() -> None:
    support = ClaimSupport(
        fact_ids=(
            "fact_midea_2024_revenue",
        ),
    )

    assert support.fact_ids == (
        "fact_midea_2024_revenue",
    )


def test_claim_support_accepts_calculation_and_citation() -> None:
    support = ClaimSupport(
        calculation_ids=(
            "calculation_gross_margin_2024",
        ),
        citation_ids=(
            "citation_1",
        ),
    )

    assert len(
        support.calculation_ids
    ) == 1

    assert len(
        support.citation_ids
    ) == 1


def test_claim_support_rejects_empty_support() -> None:
    with pytest.raises(
        ValidationError
    ):
        ClaimSupport()


def test_claim_support_rejects_duplicate_fact_ids() -> None:
    with pytest.raises(
        ValidationError
    ):
        ClaimSupport(
            fact_ids=(
                "fact_midea_2024_revenue",
                "fact_midea_2024_revenue",
            ),
        )


def test_claim_can_be_created() -> None:
    claim = Claim(
        claim_id="claim_revenue_2024",
        claim_type="financial_fact",
        claim_text=(
            "美的集团2024年营业收入为"
            "407149600000元。"
        ),
        support=ClaimSupport(
            fact_ids=(
                "fact_midea_2024_revenue",
            ),
            citation_ids=(
                "citation_1",
            ),
        ),
        confidence=1.0,
    )

    assert (
        claim.claim_type
        == "financial_fact"
    )

    assert claim.confidence == 1.0


def test_claim_rejects_invalid_confidence() -> None:
    with pytest.raises(
        ValidationError
    ):
        Claim(
            claim_id="claim_revenue_2024",
            claim_type="financial_fact",
            claim_text="测试结论",
            support=ClaimSupport(
                fact_ids=(
                    "fact_midea_2024_revenue",
                ),
            ),
            confidence=1.5,
        )

def test_verification_issue_can_be_created() -> None:
    issue = VerificationIssue(
        issue_type="year_mismatch",
        severity="error",
        message="回答年份与事实年份不一致",
        claim_id="claim_revenue_2024",
        expected_value="2024",
        actual_value="2025",
    )

    assert (
        issue.issue_type
        == "year_mismatch"
    )

    assert issue.severity == "error"


def test_verification_report_accepts_passed_result() -> None:
    report = VerificationReport(
        passed=True,
        numeric_verified=True,
        evidence_verified=True,
        citation_verified=True,
        evidence_sufficient=True,
    )

    assert report.passed is True
    assert report.issues == ()


def test_verification_report_accepts_failed_result() -> None:
    issue = VerificationIssue(
        issue_type="unit_mismatch",
        severity="error",
        message="回答单位与事实单位不一致",
        claim_id="claim_revenue_2024",
        expected_value="CNY",
        actual_value="CNY_100M",
    )

    report = VerificationReport(
        passed=False,
        numeric_verified=False,
        evidence_verified=True,
        citation_verified=True,
        evidence_sufficient=True,
        issues=(issue,),
    )

    assert report.passed is False
    assert len(report.issues) == 1


def test_verification_report_rejects_passed_with_issue() -> None:
    issue = VerificationIssue(
        issue_type="citation_mismatch",
        severity="error",
        message="引用无法支持结论",
    )

    with pytest.raises(
        ValidationError
    ):
        VerificationReport(
            passed=True,
            numeric_verified=True,
            evidence_verified=True,
            citation_verified=True,
            evidence_sufficient=True,
            issues=(issue,),
        )


def test_verification_report_rejects_failed_without_issue() -> None:
    with pytest.raises(
        ValidationError
    ):
        VerificationReport(
            passed=False,
        )


def test_verification_report_rejects_partial_pass() -> None:
    with pytest.raises(
        ValidationError
    ):
        VerificationReport(
            passed=True,
            numeric_verified=True,
            evidence_verified=True,
            citation_verified=False,
            evidence_sufficient=True,
        )


def test_human_review_request_can_be_created() -> None:
    request = HumanReviewRequest(
        review_id="review_major_risk_001",
        risk_level="high",
        reason="需要人工确认重大风险结论",
        claim_ids=(
            "claim_major_financial_risk",
        ),
    )

    assert request.risk_level == "high"
    assert (
        request.required_reviewer_role
        == "reviewer"
    )


def test_human_review_rejects_low_risk() -> None:
    with pytest.raises(
        ValidationError
    ):
        HumanReviewRequest(
            review_id="review_low_001",
            risk_level="low",
            reason="低风险请求",
        )


def test_policy_allows_verified_result() -> None:
    decision = PolicyDecision(
        action="allow",
        risk_level="low",
        reason="可信校验全部通过",
        verification_passed=True,
    )

    assert decision.action == "allow"


def test_policy_rejects_allow_when_verification_failed() -> None:
    with pytest.raises(
        ValidationError
    ):
        PolicyDecision(
            action="allow",
            risk_level="low",
            reason="测试",
            verification_passed=False,
        )


def test_policy_require_human_needs_review_request() -> None:
    with pytest.raises(
        ValidationError
    ):
        PolicyDecision(
            action="require_human",
            risk_level="high",
            reason="高风险任务",
            verification_passed=True,
        )


def test_policy_accepts_human_review() -> None:
    review = HumanReviewRequest(
        review_id="review_high_001",
        risk_level="high",
        reason="重大风险判断需要审核",
    )

    decision = PolicyDecision(
        action="require_human",
        risk_level="high",
        reason="进入人工审核",
        verification_passed=True,
        human_review=review,
    )

    assert (
        decision.action
        == "require_human"
    )


def test_policy_rejects_mismatched_review_risk() -> None:
    review = HumanReviewRequest(
        review_id="review_medium_001",
        risk_level="medium",
        reason="需要审核",
    )

    with pytest.raises(
        ValidationError
    ):
        PolicyDecision(
            action="require_human",
            risk_level="high",
            reason="进入人工审核",
            verification_passed=True,
            human_review=review,
        )


def test_answer_draft_can_be_created() -> None:
    claim = Claim(
        claim_id="claim_revenue_2024",
        claim_type="financial_fact",
        claim_text=(
            "美的集团2024年营业收入为"
            "407149600000元。"
        ),
        support=ClaimSupport(
            fact_ids=(
                "fact_midea_2024_revenue",
            ),
            citation_ids=(
                "citation_1",
            ),
        ),
    )

    draft = AnswerDraft(
        draft_id="draft_revenue_2024",
        draft_type="financial",
        claims=(claim,),
    )

    assert draft.draft_type == "financial"
    assert len(draft.claims) == 1


def test_answer_draft_rejects_empty_claims() -> None:
    with pytest.raises(
        ValidationError
    ):
        AnswerDraft(
            draft_id="draft_empty",
            draft_type="financial",
            claims=(),
        )


def test_answer_draft_rejects_duplicate_claim_ids() -> None:
    claim = Claim(
        claim_id="claim_revenue_2024",
        claim_type="financial_fact",
        claim_text="测试结论",
        support=ClaimSupport(
            fact_ids=(
                "fact_midea_2024_revenue",
            ),
        ),
    )

    with pytest.raises(
        ValidationError
    ):
        AnswerDraft(
            draft_id="draft_duplicate",
            draft_type="financial",
            claims=(
                claim,
                claim,
            ),
        )


def test_answer_draft_accepts_uncertainty_and_limitation() -> None:
    claim = Claim(
        claim_id="claim_document_risk",
        claim_type="document_analysis",
        claim_text="年报披露了相关经营风险。",
        support=ClaimSupport(
            citation_ids=(
                "citation_1",
            ),
        ),
    )

    draft = AnswerDraft(
        draft_id="draft_document_risk",
        draft_type="document",
        claims=(claim,),
        uncertainty_notes=(
            "证据仅来自公开年报披露。",
        ),
        limitation_notes=(
            "不构成投资建议。",
        ),
    )

    assert len(
        draft.uncertainty_notes
    ) == 1

    assert len(
        draft.limitation_notes
    ) == 1