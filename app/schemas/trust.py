from __future__ import annotations

from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


ClaimType = Literal[
    "financial_fact",
    "financial_calculation",
    "document_analysis",
    "limitation",
]


_CLAIM_ID_PATTERN = (
    r"^claim_[a-z0-9_]+$"
)

_FACT_ID_PATTERN = (
    r"^fact_[a-z0-9_]+$"
)

_CALCULATION_ID_PATTERN = (
    r"^calculation_[a-z0-9_]+$"
)

_CITATION_ID_PATTERN = (
    r"^citation_[a-z0-9_]+$"
)


def _validate_unique_values(
    values: tuple[str, ...],
    *,
    field_name: str,
) -> tuple[str, ...]:
    if len(values) != len(set(values)):
        raise ValueError(
            f"{field_name} 不能包含重复值"
        )

    return values


class ClaimSupport(BaseModel):
    """一条 Claim 所依赖的可验证支持对象。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    fact_ids: tuple[
        str,
        ...
    ] = ()

    calculation_ids: tuple[
        str,
        ...
    ] = ()

    citation_ids: tuple[
        str,
        ...
    ] = ()

    @field_validator(
        "fact_ids",
        "calculation_ids",
        "citation_ids",
    )
    @classmethod
    def validate_unique_ids(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        return _validate_unique_values(
            values,
            field_name="support ids",
        )

    @field_validator("fact_ids")
    @classmethod
    def validate_fact_ids(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        import re

        for value in values:
            if re.fullmatch(
                _FACT_ID_PATTERN,
                value,
            ) is None:
                raise ValueError(
                    f"非法 fact_id：{value}"
                )

        return values

    @field_validator("calculation_ids")
    @classmethod
    def validate_calculation_ids(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        import re

        for value in values:
            if re.fullmatch(
                _CALCULATION_ID_PATTERN,
                value,
            ) is None:
                raise ValueError(
                    "非法 calculation_id："
                    f"{value}"
                )

        return values

    @field_validator("citation_ids")
    @classmethod
    def validate_citation_ids(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        import re

        for value in values:
            if re.fullmatch(
                _CITATION_ID_PATTERN,
                value,
            ) is None:
                raise ValueError(
                    f"非法 citation_id：{value}"
                )

        return values

    @model_validator(mode="after")
    def validate_has_support(
        self,
    ) -> Self:
        if not (
            self.fact_ids
            or self.calculation_ids
            or self.citation_ids
        ):
            raise ValueError(
                "ClaimSupport 至少需要一个支持对象"
            )

        return self


class Claim(BaseModel):
    """最终回答中的一条可独立验证结论。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    claim_id: str = Field(
        pattern=_CLAIM_ID_PATTERN,
    )

    claim_type: ClaimType

    claim_text: str = Field(
        min_length=1,
        max_length=4000,
    )

    support: ClaimSupport

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )

VerificationIssueType = Literal[
    "missing_evidence",
    "unsupported_claim",
    "year_mismatch",
    "unit_mismatch",
    "statement_scope_mismatch",
    "calculation_input_mismatch",
    "citation_mismatch",
    "evidence_conflict",
    "permission_denied",
    "prompt_injection_detected",
]

VerificationSeverity = Literal[
    "warning",
    "error",
    "critical",
]


class VerificationIssue(BaseModel):
    """一次可信校验过程中发现的结构化问题。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    issue_type: VerificationIssueType

    severity: VerificationSeverity

    message: str = Field(
        min_length=1,
        max_length=2000,
    )

    claim_id: str | None = Field(
        default=None,
        pattern=_CLAIM_ID_PATTERN,
    )

    expected_value: str | None = Field(
        default=None,
        max_length=1000,
    )

    actual_value: str | None = Field(
        default=None,
        max_length=1000,
    )

class VerificationReport(BaseModel):
    """一次回答可信校验的汇总结果。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    passed: bool

    numeric_verified: bool = False

    evidence_verified: bool = False

    citation_verified: bool = False

    evidence_sufficient: bool = False

    issues: tuple[
        VerificationIssue,
        ...
    ] = ()

    @model_validator(mode="after")
    def validate_report_contract(
        self,
    ) -> Self:
        if self.passed and self.issues:
            raise ValueError(
                "passed=True 时不能包含 VerificationIssue"
            )

        if (
            self.passed
            and not (
                self.numeric_verified
                and self.evidence_verified
                and self.citation_verified
                and self.evidence_sufficient
            )
        ):
            raise ValueError(
                "passed=True 时所有可信校验必须通过"
            )

        if (
            not self.passed
            and not self.issues
        ):
            raise ValueError(
                "passed=False 时必须说明失败原因"
            )

        return self


RiskLevel = Literal[
    "low",
    "medium",
    "high",
]

UserRole = Literal[
    "viewer",
    "reviewer",
    "admin",
]

PolicyAction = Literal[
    "allow",
    "refuse",
    "require_human",
]

_REVIEW_ID_PATTERN = (
    r"^review_[a-z0-9_]+$"
)


class HumanReviewRequest(BaseModel):
    """需要人工介入时保存的结构化审核请求。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    review_id: str = Field(
        pattern=_REVIEW_ID_PATTERN,
    )

    risk_level: RiskLevel

    reason: str = Field(
        min_length=1,
        max_length=2000,
    )

    claim_ids: tuple[
        str,
        ...
    ] = ()

    required_reviewer_role: Literal[
        "reviewer",
        "admin",
    ] = "reviewer"

    @field_validator("claim_ids")
    @classmethod
    def validate_claim_ids(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        import re

        _validate_unique_values(
            values,
            field_name="claim_ids",
        )

        for value in values:
            if re.fullmatch(
                _CLAIM_ID_PATTERN,
                value,
            ) is None:
                raise ValueError(
                    f"非法 claim_id：{value}"
                )

        return values

    @model_validator(mode="after")
    def validate_review_risk(
        self,
    ) -> Self:
        if self.risk_level == "low":
            raise ValueError(
                "low risk 不应该创建人工复核请求"
            )

        return self


class PolicyDecision(BaseModel):
    """可信校验完成后的系统放行决策。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    action: PolicyAction

    risk_level: RiskLevel

    reason: str = Field(
        min_length=1,
        max_length=2000,
    )

    verification_passed: bool

    human_review: (
        HumanReviewRequest | None
    ) = None

    @model_validator(mode="after")
    def validate_policy_contract(
        self,
    ) -> Self:
        if (
            self.action == "allow"
            and not self.verification_passed
        ):
            raise ValueError(
                "可信校验未通过时不能 allow"
            )

        if (
            self.action == "require_human"
            and self.human_review is None
        ):
            raise ValueError(
                "require_human 必须包含人工复核请求"
            )

        if (
            self.action != "require_human"
            and self.human_review is not None
        ):
            raise ValueError(
                "只有 require_human "
                "才能携带 human_review"
            )

        if (
            self.human_review is not None
            and self.human_review.risk_level
            != self.risk_level
        ):
            raise ValueError(
                "PolicyDecision 与 HumanReviewRequest "
                "风险等级必须一致"
            )

        return self