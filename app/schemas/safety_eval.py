from __future__ import annotations

from typing import (
    Literal,
    Self,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.schemas.agent_runtime import (
    StopReason,
)
from app.schemas.trust import (
    PolicyAction,
    UserRole,
    VerificationIssueType,
)


SafetyEvalCategory = Literal[
    "evidence_citation",
    "numeric_scope",
    "rbac",
    "prompt_injection",
    "unsupported_boundary",
    "risk_hitl",
    "normal_safe",
]


SafetyControl = Literal[
    "trust_verification",
    "rbac",
    "prompt_injection",
    "runtime_boundary",
    "risk_policy",
    "hitl",
]


SafetyExpectedOutcome = Literal[
    "allow",
    "deny",
    "detect",
    "refuse",
    "require_human",
]


SafetyScenario = Literal[
    # Evidence / Citation
    "missing_evidence",
    "unsupported_claim",
    "citation_mismatch",
    "evidence_conflict",
    "missing_citation",
    "document_evidence_insufficient",

    # Numeric / Scope
    "year_mismatch",
    "unit_mismatch",
    "statement_scope_mismatch",
    "calculation_input_mismatch",
    "numeric_value_tamper",
    "calculation_result_tamper",

    # RBAC
    "viewer_calculation_denied",
    "viewer_financial_read_allowed",
    "viewer_document_read_allowed",
    "reviewer_calculation_allowed",
    "permission_snapshot_tamper_denied",

    # Prompt Injection
    "prompt_instruction_override_en",
    "prompt_instruction_override_zh",
    "prompt_system_prompt_extraction",
    "prompt_authority_hijack",
    "prompt_tool_manipulation",
    "prompt_security_bypass",

    # Unsupported / Boundary
    "unsupported_stock_prediction",
    "unsupported_investment_advice",
    "unsupported_external_fact",
    "unsupported_write_operation",
    "unsupported_out_of_scope",

    # Risk / HITL
    "policy_high_requires_human",
    "hitl_reviewer_approve",
    "hitl_reviewer_reject",
    "hitl_viewer_denied",
    "hitl_admin_required_reviewer_denied",
    "hitl_admin_approve",

    # Normal Safe
    "safe_financial_fact",
    "safe_calculation",
    "safe_comparison",
    "safe_document",
    "safe_viewer_read",
    "safe_prompt_like_document",
]


class SafetyEvalCase(
    BaseModel
):
    """Week7 单条安全评测 Case。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    case_id: str = Field(
        pattern=r"^safety_[0-9]{3}$",
    )

    name: str = Field(
        min_length=1,
        max_length=256,
    )

    category: SafetyEvalCategory

    scenario: SafetyScenario

    expected_control: SafetyControl

    expected_outcome: (
        SafetyExpectedOutcome
    )

    adversarial: bool

    question: str = Field(
        min_length=1,
        max_length=4000,
    )

    user_role: UserRole = (
        "reviewer"
    )

    document_text: str | None = Field(
        default=None,
        max_length=20_000,
    )

    expected_stop_reason: (
        StopReason | None
    ) = None

    expected_issue_type: (
        VerificationIssueType | None
    ) = None

    expected_rule_ids: tuple[
        str,
        ...
    ] = ()

    expected_policy_action: (
        PolicyAction | None
    ) = None

    reviewer_role: (
        UserRole | None
    ) = None

    human_approved: (
        bool | None
    ) = None

    @field_validator(
        "expected_rule_ids"
    )
    @classmethod
    def validate_rule_ids(
        cls,
        value: tuple[
            str,
            ...
        ],
    ) -> tuple[str, ...]:
        if (
            len(value)
            != len(set(value))
        ):
            raise ValueError(
                "expected_rule_ids "
                "不能包含重复值"
            )

        return value

    @model_validator(
        mode="after"
    )
    def validate_case_contract(
        self,
    ) -> Self:
        if (
            self.category
            == "prompt_injection"
            and not self.document_text
        ):
            raise ValueError(
                "Prompt Injection Case "
                "必须提供 document_text"
            )

        if (
            self.expected_rule_ids
            and self.category
            != "prompt_injection"
        ):
            raise ValueError(
                "只有 Prompt Injection Case "
                "才能填写 expected_rule_ids"
            )

        if (
            self.category
            == "normal_safe"
        ):
            if self.adversarial:
                raise ValueError(
                    "normal_safe "
                    "不能是 adversarial"
                )

            if (
                self.expected_outcome
                != "allow"
            ):
                raise ValueError(
                    "normal_safe "
                    "必须 expected_outcome=allow"
                )

        if (
            self.expected_outcome
            == "require_human"
            and self.expected_stop_reason
            != "human_review_required"
        ):
            raise ValueError(
                "require_human Case "
                "必须使用 "
                "human_review_required"
            )

        return self

SafetyActualOutcome = Literal[
    "allow",
    "deny",
    "detect",
    "refuse",
    "require_human",
    "error",
]

class SafetyEvalCaseResult(
    BaseModel
):
    """一条 Week7 Safety Eval 的实际执行结果。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    case_id: str = Field(
        pattern=r"^safety_[0-9]{3}$",
    )

    category: SafetyEvalCategory

    scenario: SafetyScenario

    adversarial: bool

    expected_control: SafetyControl

    expected_outcome: (
        SafetyExpectedOutcome
    )

    actual_outcome: (
        SafetyActualOutcome
    )

    expected_stop_reason: (
        StopReason | None
    ) = None

    actual_stop_reason: (
        StopReason | None
    ) = None

    expected_issue_type: (
        VerificationIssueType | None
    ) = None

    actual_issue_types: tuple[
        VerificationIssueType,
        ...
    ] = ()

    expected_rule_ids: tuple[
        str,
        ...
    ] = ()

    actual_rule_ids: tuple[
        str,
        ...
    ] = ()

    expected_policy_action: (
        PolicyAction | None
    ) = None

    actual_policy_action: (
        PolicyAction | None
    ) = None

    answer_released: bool = False

    outcome_ok: bool

    stop_reason_ok: bool

    issue_ok: bool

    rule_ids_ok: bool

    policy_ok: bool

    unsafe_answer_released: bool

    case_pass: bool

    error_message: str | None = Field(
        default=None,
        max_length=2000,
    )

class SafetyEvalSummary(
    BaseModel
):
    """Week7 40-case Safety Eval 指标汇总。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    eval_name: str = (
        "week7_safety_eval_v1"
    )

    case_count: int = Field(
        ge=1,
    )

    passed_count: int = Field(
        ge=0,
    )

    adversarial_count: int = Field(
        ge=0,
    )

    normal_safe_count: int = Field(
        ge=0,
    )

    # ========================================================
    # Unsafe Answer Release Rate
    # ========================================================

    unsafe_release_applicable_count: int = Field(
        ge=0,
    )

    unsafe_answer_release_count: int = Field(
        ge=0,
    )

    unsafe_answer_release_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    # ========================================================
    # False Refusal Rate
    # ========================================================

    false_refusal_applicable_count: int = Field(
        ge=0,
    )

    false_refusal_count: int = Field(
        ge=0,
    )

    false_refusal_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    # ========================================================
    # Trust Violation Detection Rate
    # ========================================================

    trust_violation_applicable_count: int = Field(
        ge=0,
    )

    trust_violation_detected_count: int = Field(
        ge=0,
    )

    trust_violation_detection_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    # ========================================================
    # Prompt Injection Detection Rate
    # ========================================================

    prompt_injection_applicable_count: int = Field(
        ge=0,
    )

    prompt_injection_detected_count: int = Field(
        ge=0,
    )

    prompt_injection_detection_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    # ========================================================
    # Permission Denial Accuracy
    # ========================================================

    permission_denial_applicable_count: int = Field(
        ge=0,
    )

    permission_denial_correct_count: int = Field(
        ge=0,
    )

    permission_denial_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    # ========================================================
    # HITL Routing Accuracy
    # ========================================================

    hitl_applicable_count: int = Field(
        ge=0,
    )

    hitl_routing_correct_count: int = Field(
        ge=0,
    )

    hitl_routing_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    # ========================================================
    # Overall
    # ========================================================

    overall_safety_success_rate: float = Field(
        ge=0.0,
        le=1.0,
    )