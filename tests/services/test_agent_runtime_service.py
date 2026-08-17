from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)

from app.schemas.company import (
    Company,
)
from app.schemas.enums import (
    MetricOrigin,
    ReportType,
    StatementScope,
    StatementType,
)
from app.schemas.metric import (
    FinancialMetric,
)
from app.schemas.report import (
    Report,
)
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
from app.services.agent_runtime import (
    AgentRuntime,
    AgentRuntimeError,
)
from app.services.registry import (
    RegistryBundle,
)
from app.services.runtime_intent_router import (
    RuntimeIntentRouter,
)
from app.services.runtime_planner import (
    RuntimePlanner,
)
from app.services.runtime_answer_draft import (
    RuntimeAnswerDraftBuilder,
)
from app.services.runtime_query_parser import (
    RuntimeQueryParser,
)
from app.services.runtime_trust_verifier import (
    RuntimeTrustVerifier,
)
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

# ============================================================
# 固定 Clock：
#
# Runtime 测试不应该因为真实时间变化而不稳定。
# ============================================================


class FixedClock:
    def now(
        self,
    ) -> datetime:
        return datetime(
            2026,
            8,
            11,
            3,
            0,
            tzinfo=timezone.utc,
        )


# ============================================================
# Production：
#
# request_41f81...
#
# Test：
#
# request_1
# request_2
#
# 更容易断言。
# ============================================================


class SequentialIdFactory:
    def __init__(
        self,
    ) -> None:
        self._counters: dict[
            str,
            int,
        ] = {}

    def new_id(
        self,
        prefix: str,
    ) -> str:
        next_value = (
            self._counters.get(
                prefix,
                0,
            )
            + 1
        )

        self._counters[
            prefix
        ] = next_value

        return (
            f"{prefix}_{next_value}"
        )


# ============================================================
# 只构造 8A 集成测试需要的 Registry。
# ============================================================


def _build_bundle(
) -> RegistryBundle:
    bundle = RegistryBundle()

    bundle.companies.add(
        Company.model_construct(
            company_id=(
                "midea_group"
            ),
            legal_name_cn=(
                "美的集团股份有限公司"
            ),
            short_name_cn=(
                "美的集团"
            ),
            stock_code="000333",
        )
    )

    bundle.reports.add(
        Report.model_construct(
            report_id=(
                "midea_group_2024"
            ),
            company_id=(
                "midea_group"
            ),
            fiscal_year=2024,
            report_type=(
                ReportType.ANNUAL_REPORT
            ),
        )
    )

    bundle.metrics.add(
        FinancialMetric.model_construct(
            metric_id="revenue",
            display_name_cn=(
                "营业收入"
            ),
            display_name_en=(
                "Revenue"
            ),
            metric_origin=(
                MetricOrigin.REPORTED
            ),
            statement_type=(
                StatementType
                .INCOME_STATEMENT
            ),
            allowed_scopes=[
                StatementScope
                .CONSOLIDATED,
                StatementScope
                .PARENT_COMPANY,
            ],
            formula_id=None,
        )
    )

    return bundle


def _build_runtime(
) -> AgentRuntime:
    bundle = _build_bundle()

    return AgentRuntime(
        query_parser=(
            RuntimeQueryParser(
                registry_bundle=bundle
            )
        ),
        intent_router=(
            RuntimeIntentRouter()
        ),
        planner=(
            RuntimePlanner(
                registry_bundle=bundle
            )
        ),
        clock=FixedClock(),
        id_factory=(
            SequentialIdFactory()
        ),
    )

def _build_test_answer_draft(
) -> AnswerDraft:
    return AnswerDraft(
        draft_id="draft_run_1",
        draft_type="financial",
        claims=(
            Claim(
                claim_id=(
                    "claim_test_revenue"
                ),
                claim_type=(
                    "financial_fact"
                ),
                claim_text=(
                    "美的集团2024年"
                    "营业收入为测试值。"
                ),
                support=ClaimSupport(
                    fact_ids=(
                        "fact_test_revenue",
                    ),
                    citation_ids=(
                        "citation_1",
                    ),
                ),
                confidence=1.0,
            ),
        ),
    )

def _build_verifying_state(
) -> AgentState:
    now = FixedClock().now()

    return AgentState(
        request_id="request_1",
        trace_id="trace_1",
        run_id="run_1",
        thread_id="thread_1",

        query=(
            "美的集团2024年"
            "营业收入是多少？"
        ),

        intent="financial_fact",

        status="verifying",

        current_node="verify_evidence",
        next_node="prepare_answer",

        started_at=now,
        updated_at=now,
    )
# ============================================================
# 初始状态：
#
# 还没有做任何业务逻辑。
# ============================================================


def test_create_state_starts_created(
) -> None:
    runtime = _build_runtime()

    state = runtime.create_state(
        query=(
            "美的集团2024年营业收入是多少？"
        )
    )

    assert (
        state.status
        == "created"
    )

    assert (
        state.request_id
        == "request_1"
    )

    assert (
        state.trace_id
        == "trace_1"
    )

    assert (
        state.run_id
        == "run_1"
    )

    assert (
        state.thread_id
        == "thread_1"
    )

    assert (
        state.parsed_query
        is None
    )

    assert state.intent is None

    assert (
        state.runtime_plan
        is None
    )

    assert (
        state.current_node
        == "parse_query"
    )

    assert (
        state.next_node
        == "parse_query"
    )

    assert (
        state.user_role
        == "reviewer"
    )

    assert (
        state.granted_permissions
        == (
            "execute_calculation",
            "read_documents",
            "read_financial_data",
        )
    )

# week7 Step4test
def test_create_state_resolves_viewer_permissions(
) -> None:
    runtime = _build_runtime()

    state = runtime.create_state(
        query=(
            "美的集团2024年"
            "营业收入是多少？"
        ),
        user_role="viewer",
    )

    assert (
        state.user_role
        == "viewer"
    )

    assert (
        state.granted_permissions
        == (
            "read_documents",
            "read_financial_data",
        )
    )

def test_prepare_preserves_access_context(
) -> None:
    runtime = _build_runtime()

    state = runtime.prepare(
        query=(
            "美的集团2024年"
            "营业收入是多少？"
        ),
        user_role="viewer",
    )

    assert (
        state.user_role
        == "viewer"
    )

    assert (
        state.granted_permissions
        == (
            "read_documents",
            "read_financial_data",
        )
    )

    assert (
        state.status
        == "planned"
    )

def test_trajectory_preserves_access_context(
) -> None:
    runtime = _build_runtime()

    state = runtime.create_state(
        query=(
            "一个不支持的问题"
        ),
        user_role="viewer",
    )

    now = FixedClock().now()

    terminal_state = (
        state.model_copy(
            update={
                "status": (
                    "refused"
                ),
                "stop_reason": (
                    "unsupported"
                ),
                "current_node": (
                    "finish"
                ),
                "next_node": (
                    "finish"
                ),
                "completed_at": now,
                "updated_at": now,
            }
        )
    )

    trajectory = (
        runtime._build_trajectory(
            terminal_state
        )
    )

    assert (
        trajectory.user_role
        == "viewer"
    )

    assert (
        trajectory
        .granted_permissions
        == (
            "read_documents",
            "read_financial_data",
        )
    )


# ============================================================
# 8A 最核心测试：
#
# Question
# ↓
# Parser
# ↓
# Router
# ↓
# Planner
# ↓
# planned
# ============================================================


def test_prepare_financial_fact_reaches_planned(
) -> None:
    runtime = _build_runtime()

    state = runtime.prepare(
        query=(
            "美的集团2024年营业收入是多少？"
        )
    )

    assert (
        state.status
        == "planned"
    )

    assert (
        state.intent
        == "financial_fact"
    )

    assert (
        state.runtime_plan
        is not None
    )

    assert (
        state.runtime_plan.intent
        == "financial_fact"
    )

    assert (
        state.current_node
        == "create_plan"
    )

    assert (
        state.next_node
        == "execute_plan"
    )

    # 8A 还没有执行 Plan。
    assert state.current_step == 0

    assert (
        state.completed_step_ids
        == ()
    )

    assert (
        state.tool_results
        == ()
    )


# ============================================================
# AgentState 是一次运行的“唯一事实源”。
#
# Parser 结果需要同步到 State。
# ============================================================


def test_prepare_copies_parsed_identity_to_state(
) -> None:
    runtime = _build_runtime()

    state = runtime.prepare(
        query=(
            "美的集团2024年营业收入是多少？"
        )
    )

    assert (
        state.company_ids
        == (
            "midea_group",
        )
    )

    assert (
        state.report_ids
        == (
            "midea_group_2024",
        )
    )

    assert (
        state.years
        == (
            2024,
        )
    )

    assert (
        state.metric_ids
        == (
            "revenue",
        )
    )

    assert (
        state.confidence
        == 1.0
    )


# ============================================================
# 8A 已经开始可观测：
#
# parse_query
# route_intent
# create_plan
#
# 每一个 Node 都有 NodeSpan。
# ============================================================


def test_prepare_records_three_node_spans(
) -> None:
    runtime = _build_runtime()

    state = runtime.prepare(
        query=(
            "美的集团2024年营业收入是多少？"
        )
    )

    assert tuple(
        span.node_name
        for span
        in state.node_spans
    ) == (
        "parse_query",
        "route_intent",
        "create_plan",
    )

    assert all(
        span.status
        == "completed"
        for span
        in state.node_spans
    )

    assert (
        state.step_count
        == 3
    )

    # Checkpoint 尚未接入。
    assert all(
        span.checkpoint_revision
        == 0
        for span
        in state.node_spans
    )


# ============================================================
# Runtime 并不只支持 financial_fact。
#
# Parser / Router / Planner 的具体 Intent
# 对 Runtime 来说只是“注入的业务逻辑”。
# ============================================================


def test_prepare_document_query_reaches_planned(
) -> None:
    runtime = _build_runtime()

    state = runtime.prepare(
        query=(
            "美的集团2024年"
            "主要经营风险有哪些？"
        )
    )

    assert (
        state.status
        == "planned"
    )

    assert (
        state.intent
        == "document_evidence"
    )

    assert (
        state.runtime_plan
        is not None
    )

    assert (
        state.runtime_plan
        .tool_by_step_id
        == {
            "s1": (
                "retrieve_documents"
            )
        }
    )


# ============================================================
# Unsupported：
#
# 不是 Exception，
# 也不是 Runtime Crash。
#
# 它是一个正常的终止状态。
# ============================================================


def test_unsupported_query_is_refused(
) -> None:
    runtime = _build_runtime()

    state = runtime.prepare(
        query="帮我写一首诗"
    )

    assert (
        state.status
        == "refused"
    )

    assert (
        state.intent
        == "unsupported"
    )

    assert (
        state.stop_reason
        == "unsupported"
    )

    assert (
        state.completed_at
        is not None
    )

    assert (
        state.runtime_plan
        is None
    )

    assert (
        state.current_node
        == "finish"
    )

    assert (
        state.next_node
        == "finish"
    )

    assert tuple(
        span.node_name
        for span
        in state.node_spans
    ) == (
        "parse_query",
        "route_intent",
    )


# ============================================================
# Missing != Unsupported
#
#
# “营业收入是多少？”
#
# 系统会做，
# 但不知道：
#
# 哪家公司？
# 哪一年？
#
# 所以等待澄清。
# ============================================================


def test_missing_identity_waits_for_human(
) -> None:
    runtime = _build_runtime()

    state = runtime.prepare(
        query="营业收入是多少？"
    )

    assert (
        state.intent
        == "financial_fact"
    )

    assert (
        state.status
        == "awaiting_human"
    )

    assert (
        state.stop_reason
        == "human_review_required"
    )

    assert (
        state.pending_human_review
        is True
    )

    assert (
        state.human_review_reason
        is not None
    )

    assert (
        "company_ids"
        in state.human_review_reason
    )

    assert (
        "years"
        in state.human_review_reason
    )

    assert (
        state.runtime_plan
        is None
    )

    assert (
        state.current_node
        == "await_human"
    )


# ============================================================
# thread_id 后面会用于：
#
# Checkpoint
# LangGraph thread
# 多轮恢复
#
# 所以调用方传入时不能被 Runtime 自己覆盖。
# ============================================================


def test_prepare_preserves_caller_thread_id(
) -> None:
    runtime = _build_runtime()

    state = runtime.prepare(
        query=(
            "美的集团2024年营业收入是多少？"
        ),
        thread_id="thread_demo_001",
    )

    assert (
        state.thread_id
        == "thread_demo_001"
    )


def test_prepare_answer_node_stores_draft(
) -> None:
    builder = MagicMock(
        spec=RuntimeAnswerDraftBuilder
    )

    builder.build.return_value = (
        _build_test_answer_draft()
    )

    runtime = replace(
        _build_runtime(),
        answer_draft_builder=builder,
    )

    state = (
        _build_verifying_state()
    )

    updated_state = (
        runtime
        ._run_prepare_answer_node(
            state
        )
    )

    builder.build.assert_called_once_with(
        state
    )

    assert (
        updated_state.answer_draft
        is not None
    )

    assert (
        updated_state
        .answer_draft
        .draft_id
        == "draft_run_1"
    )

    assert (
        updated_state.current_node
        == "prepare_answer"
    )

    assert (
        updated_state.next_node
        == "verify_answer"
    )

    assert (
        updated_state.step_count
        == state.step_count + 1
    )

    assert (
        updated_state
        .node_spans[-1]
        .node_name
        == "prepare_answer"
    )

def test_prepare_answer_node_requires_builder(
) -> None:
    runtime = _build_runtime()

    state = (
        _build_verifying_state()
    )

    with pytest.raises(
        AgentRuntimeError,
        match="answer_draft_builder",
    ):
        runtime._run_prepare_answer_node(
            state
        )

def _build_state_ready_for_verify_answer(
) -> AgentState:
    """构造已经完成 prepare_answer 的 State。"""

    state = (
        _build_verifying_state()
    )

    return state.model_copy(
        update={
            "answer_draft": (
                _build_test_answer_draft()
            ),
            "current_node": (
                "prepare_answer"
            ),
            "next_node": (
                "verify_answer"
            ),
        }
    )


def test_verify_answer_node_passes_to_generator(
) -> None:
    """Trust Verification 通过后才能进入 Generator。"""

    report = VerificationReport(
        passed=True,
        numeric_verified=True,
        evidence_verified=True,
        citation_verified=True,
        evidence_sufficient=True,
        issues=(),
    )

    verifier = MagicMock(
        spec=RuntimeTrustVerifier
    )

    verifier.verify.return_value = (
        report
    )

    runtime = replace(
        _build_runtime(),
        trust_verifier=verifier,
    )

    state = (
        _build_state_ready_for_verify_answer()
    )

    updated_state = (
        runtime
        ._run_verify_answer_node(
            state
        )
    )

    verifier.verify.assert_called_once_with(
        state
    )

    assert (
        updated_state
        .verification_report
        == report
    )

    assert (
        updated_state.status
        == "verifying"
    )

    assert (
        updated_state.stop_reason
        is None
    )

    assert (
        updated_state.current_node
        == "verify_answer"
    )

    assert (
        updated_state.next_node
        == "generate_answer"
    )

    assert (
        updated_state.completed_at
        is None
    )

    assert (
        updated_state.step_count
        == state.step_count + 1
    )

    assert (
        updated_state
        .node_spans[-1]
        .node_name
        == "verify_answer"
    )

    assert (
        updated_state
        .node_spans[-1]
        .output_summary[
            "passed"
        ]
        is True
    )


def test_verify_answer_node_refuses_failed_report(
) -> None:
    """VerificationReport FAIL 时必须拒答。"""

    issue = VerificationIssue(
        issue_type=(
            "unsupported_claim"
        ),
        severity="error",
        message=(
            "Claim 与确定性事实不一致"
        ),
        claim_id=(
            "claim_test_revenue"
        ),
        expected_value=(
            "407149600000"
        ),
        actual_value="1",
    )

    report = VerificationReport(
        passed=False,
        numeric_verified=False,
        evidence_verified=True,
        citation_verified=True,
        evidence_sufficient=False,
        issues=(
            issue,
        ),
    )

    verifier = MagicMock(
        spec=RuntimeTrustVerifier
    )

    verifier.verify.return_value = (
        report
    )

    runtime = replace(
        _build_runtime(),
        trust_verifier=verifier,
    )

    state = (
        _build_state_ready_for_verify_answer()
    )

    updated_state = (
        runtime
        ._run_verify_answer_node(
            state
        )
    )

    verifier.verify.assert_called_once_with(
        state
    )

    assert (
        updated_state
        .verification_report
        == report
    )

    assert (
        updated_state.status
        == "refused"
    )

    assert (
        updated_state.stop_reason
        == "insufficient_evidence"
    )

    assert (
        updated_state.answer
        is None
    )

    assert (
        updated_state.current_node
        == "verify_answer"
    )

    assert (
        updated_state.next_node
        == "finish"
    )

    assert (
        updated_state.completed_at
        is not None
    )

    assert (
        updated_state
        .node_spans[-1]
        .node_name
        == "verify_answer"
    )

    assert (
        updated_state
        .node_spans[-1]
        .output_summary[
            "passed"
        ]
        is False
    )

    assert (
        updated_state
        .node_spans[-1]
        .output_summary[
            "issue_count"
        ]
        == 1
    )


def test_verify_answer_node_requires_trust_verifier(
) -> None:
    """没有 TrustVerifier 时不能进入 verify_answer。"""

    runtime = _build_runtime()

    state = (
        _build_state_ready_for_verify_answer()
    )

    with pytest.raises(
        AgentRuntimeError,
        match="trust_verifier",
    ):
        runtime._run_verify_answer_node(
            state
        )