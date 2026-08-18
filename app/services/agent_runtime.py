from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from datetime import (
    datetime,
    timezone,
)
from time import perf_counter
from typing import Protocol
from uuid import uuid4

from app.schemas.agent_runtime import (
    AgentErrorRecord,
    AgentIntent,
    AgentState,
    AgentTrajectory,
    HumanReviewDecision,
    NodeSpan,
    ParsedFinancialQuery,
    PromptInjectionFinding,
    RuntimePlan,
)
from app.schemas.trust import (
    PolicyDecision,
    RiskLevel,
    UserRole,
)
from app.services.checkpoint_store import (
    CheckpointStore,
)
from app.services.runtime_completion import (
    RuntimeAnswerGenerationError,
    RuntimeAnswerGenerator,
    RuntimeEvidenceError,
    RuntimeEvidenceVerifier,
)
from app.services.runtime_plan_executor import (
    RuntimePermissionSnapshotMismatchError,
    RuntimePlanExecutor,
    RuntimePlanExecutorError,
    RuntimePromptInjectionDetectedError,
)
from app.services.tool_registry import (
    ToolExecutionFailedError,
    ToolRegistryError,
)
from app.services.trajectory_store import (
    TrajectoryAlreadyExistsError,
    TrajectoryStore,
)
from app.services.runtime_answer_draft import (
    RuntimeAnswerDraftBuilder,
    RuntimeAnswerDraftError,
)
from app.services.runtime_trust_verifier import (
    RuntimeTrustVerificationError,
    RuntimeTrustVerifier,
)
from app.services.runtime_access_control import (
    RuntimeAccessController,
)
from app.services.runtime_policy import (
    RuntimeRiskPolicy,
    reviewer_role_satisfies,
)


class AgentRuntimeError(
    ValueError
):
    """Framework-independent Agent Runtime 基础异常。"""


class QueryParserProvider(
    Protocol
):
    parser_version: str

    def parse(
        self,
        question: str,
    ) -> ParsedFinancialQuery:
        """解析用户问题。"""


class IntentRouterProvider(
    Protocol
):
    router_version: str

    def route(
        self,
        parsed_query: ParsedFinancialQuery,
    ) -> AgentIntent:
        """判断任务 Intent。"""


class RuntimePlannerProvider(
    Protocol
):
    planner_version: str

    def create_plan(
        self,
        *,
        parsed_query: ParsedFinancialQuery,
        intent: AgentIntent,
    ) -> RuntimePlan:
        """生成 RuntimePlan。"""


class RuntimeClock(
    Protocol
):
    def now(
        self,
    ) -> datetime:
        """返回当前时间。"""


class RuntimeIdFactory(
    Protocol
):
    def new_id(
        self,
        prefix: str,
    ) -> str:
        """生成 Runtime ID。"""


@dataclass(
    frozen=True,
    slots=True,
)
class UTCClock:
    def now(
        self,
    ) -> datetime:
        return datetime.now(
            timezone.utc
        )


@dataclass(
    frozen=True,
    slots=True,
)
class UUIDRuntimeIdFactory:
    def new_id(
        self,
        prefix: str,
    ) -> str:
        return (
            f"{prefix}_{uuid4().hex}"
        )


def _unique_in_order(
    values: tuple[
        str,
        ...
    ],
) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        result.append(value)

    return tuple(result)


_TERMINAL_STATUSES = {
    "completed",
    "refused",
    "failed",
}


@dataclass(
    frozen=True,
    slots=True,
)
class AgentRuntime:
    query_parser: QueryParserProvider

    intent_router: IntentRouterProvider

    planner: RuntimePlannerProvider

    access_controller: (
        RuntimeAccessController
    ) = field(
        default_factory=(
            RuntimeAccessController
        )
    )

    plan_executor: (
        RuntimePlanExecutor | None
    ) = None

    verifier: (
        RuntimeEvidenceVerifier | None
    ) = None

    answer_draft_builder: (
        RuntimeAnswerDraftBuilder | None
    ) = None

    trust_verifier: (
        RuntimeTrustVerifier | None
    ) = None

    risk_policy: (
        RuntimeRiskPolicy | None
    ) = None

    answer_generator: (
        RuntimeAnswerGenerator | None
    ) = None

    checkpoint_store: (
        CheckpointStore | None
    ) = None

    trajectory_store: (
        TrajectoryStore | None
    ) = None

    clock: RuntimeClock = field(
        default_factory=UTCClock
    )

    id_factory: RuntimeIdFactory = field(
        default_factory=(
            UUIDRuntimeIdFactory
        )
    )

    def create_state(
        self,
        *,
        query: str,
        request_id: str | None = None,
        trace_id: str | None = None,
        run_id: str | None = None,
        thread_id: str | None = None,
        user_role: UserRole = (
            "reviewer"
        ),
        max_steps: int = 32,
    ) -> AgentState:
        """创建带有 RBAC Access Context 的初始状态。"""

        now = self.clock.now()

        # ========================================================
        # Step4.2
        #
        # Role
        #   ↓
        # AccessController
        #   ↓
        # Effective Permissions
        #
        # frozenset 转换成排序 tuple，
        # 使 Checkpoint / Trajectory 序列化结果稳定。
        # ========================================================

        granted_permissions = tuple(
            sorted(
                self
                .access_controller
                .permissions_for_role(
                    user_role
                )
            )
        )

        return AgentState(
            request_id=(
                request_id
                or self.id_factory.new_id(
                    "request"
                )
            ),

            trace_id=(
                trace_id
                or self.id_factory.new_id(
                    "trace"
                )
            ),

            run_id=(
                run_id
                or self.id_factory.new_id(
                    "run"
                )
            ),

            thread_id=(
                thread_id
                or self.id_factory.new_id(
                    "thread"
                )
            ),

            query=query,

            user_role=(
                user_role
            ),

            granted_permissions=(
                granted_permissions
            ),

            status="created",

            max_steps=max_steps,

            current_node=(
                "parse_query"
            ),

            next_node=(
                "parse_query"
            ),

            started_at=now,

            updated_at=now,
        )

    def prepare(
        self,
        *,
        query: str,
        request_id: str | None = None,
        trace_id: str | None = None,
        run_id: str | None = None,
        thread_id: str | None = None,
        user_role: UserRole = (
            "reviewer"
        ),
        max_steps: int = 32,
    ) -> AgentState:
        state = self.create_state(
            query=query,
            request_id=request_id,
            trace_id=trace_id,
            run_id=run_id,
            thread_id=thread_id,
            user_role=user_role,
            max_steps=max_steps,
        )

        state = self._persist_checkpoint(
            state
        )

        state = self._run_parse_node(
            state
        )

        state = self._persist_checkpoint(
            state
        )

        state = self._run_route_node(
            state
        )

        state = self._persist_checkpoint(
            state
        )

        if state.intent == "unsupported":
            state = self._mark_refused(
                state
            )

            return self._persist_checkpoint(
                state
            )

        parsed_query = (
            self._require_parsed_query(
                state
            )
        )

        missing_fields = (
            self._blocking_missing_fields(
                parsed_query
            )
        )

        if missing_fields:
            state = (
                self
                ._mark_awaiting_human(
                    state,
                    blocking_missing_fields=(
                        missing_fields
                    ),
                )
            )

            return self._persist_checkpoint(
                state
            )

        state = self._run_plan_node(
            state
        )

        return self._persist_checkpoint(
            state
        )

    def run(
        self,
        *,
        query: str,
        request_id: str | None = None,
        trace_id: str | None = None,
        run_id: str | None = None,
        thread_id: str | None = None,
        user_role: UserRole = (
            "reviewer"
        ),
        max_steps: int = 32,
    ) -> AgentState:
        """启动一次新的 Agent 运行。"""

        state = self.prepare(
            query=query,
            request_id=request_id,
            trace_id=trace_id,
            run_id=run_id,
            thread_id=thread_id,
            user_role=user_role,
            max_steps=max_steps,
        )

        return self._continue_from_state(
            state
        )

    def resume(
        self,
        *,
        run_id: str,
        thread_id: str,
    ) -> AgentState:
        """从最新领域 Checkpoint 恢复已有运行。"""

        if self.checkpoint_store is None:
            raise AgentRuntimeError(
                "Runtime resume 需要配置 "
                "checkpoint_store"
            )

        record = (
            self.checkpoint_store
            .load_latest(
                run_id=run_id,
                thread_id=thread_id,
            )
        )

        state = record.state

        return self._continue_from_state(
            state
        )

    def submit_policy_review(
        self,
        *,
        run_id: str,
        thread_id: str,
        approved: bool,
        reviewer_id: str,
        reviewer_role: UserRole,
        reason: str,
    ) -> AgentState:
        """提交 Risk Policy 触发的人工审核决定并继续 Runtime。"""

        if self.checkpoint_store is None:
            raise AgentRuntimeError(
                "Policy Review 需要配置 "
                "checkpoint_store"
            )

        record = (
            self.checkpoint_store
            .load_latest(
                run_id=run_id,
                thread_id=thread_id,
            )
        )

        state = record.state

        # ========================================================
        # Gate 1
        #
        # 只能审核真正暂停中的 Runtime。
        # ========================================================

        if (
            state.status
            != "awaiting_human"
            or not state.pending_human_review
        ):
            raise AgentRuntimeError(
                "当前 Runtime 不处于"
                "人工审核状态"
            )

        # ========================================================
        # Gate 2
        #
        # 这里只处理 Risk Policy 触发的 HITL，
        # 不处理“缺字段澄清”。
        # ========================================================

        policy_decision = (
            state.policy_decision
        )

        if (
            policy_decision is None
            or policy_decision.action
            != "require_human"
        ):
            raise AgentRuntimeError(
                "当前人工等待不是 "
                "Risk Policy Review"
            )

        human_review = (
            policy_decision.human_review
        )

        if human_review is None:
            raise AgentRuntimeError(
                "PolicyDecision 缺少 "
                "HumanReviewRequest"
            )

        # ========================================================
        # Gate 3
        #
        # HITL 不能绕过 Trust Verification。
        # ========================================================

        if (
            state.verification_report
            is None
            or not state
            .verification_report
            .passed
            or not policy_decision
            .verification_passed
        ):
            raise AgentRuntimeError(
                "可信校验未通过，"
                "人工审核不能覆盖 Trust Gate"
            )

        # ========================================================
        # Gate 4
        #
        # 人工审核本身也有授权边界。
        # ========================================================

        if not reviewer_role_satisfies(
            reviewer_role=(
                reviewer_role
            ),
            required_role=(
                human_review
                .required_reviewer_role
            ),
        ):
            raise AgentRuntimeError(
                "人工审核角色权限不足："
                f"required="
                f"{human_review.required_reviewer_role}; "
                f"actual={reviewer_role}"
            )

        now = self.clock.now()

        human_decision = (
            HumanReviewDecision(
                approved=approved,
                corrected_query=None,
                reason=reason,
                reviewer_id=reviewer_id,
                reviewer_role=(
                    reviewer_role
                ),
                decided_at=now,
            )
        )

        # ========================================================
        # Audit Span
        #
        # 不保存敏感业务正文，
        # 只保存 Review ID、权限和决定。
        # ========================================================

        span = (
            self._build_completed_span(
                node_name="await_human",
                input_summary={
                    "review_id": (
                        human_review.review_id
                    ),
                    "required_reviewer_role": (
                        human_review
                        .required_reviewer_role
                    ),
                },
                output_summary={
                    "approved": approved,
                    "reviewer_role": (
                        reviewer_role
                    ),
                },
                started_at=now,
                completed_at=now,
                timer_start=(
                    perf_counter()
                ),
                checkpoint_revision=(
                    state
                    .checkpoint_revision
                ),
            )
        )

        # ========================================================
        # REJECT
        #
        # 人工拒绝属于业务安全决策，
        # 不是 Runtime Failure。
        # ========================================================

        if not approved:
            state = self._replace_state(
                state,
                human_decision=(
                    human_decision
                ),
                answer=None,
                status="refused",
                stop_reason=(
                    "human_rejected"
                ),
                pending_human_review=False,
                human_review_reason=None,
                current_node=(
                    "await_human"
                ),
                next_node="finish",
                node_spans=(
                    state.node_spans
                    + (
                        span,
                    )
                ),
                step_count=(
                    state.step_count
                    + 1
                ),
                completed_at=now,
                updated_at=now,
            )

            state = (
                self._persist_checkpoint(
                    state
                )
            )

            return self._continue_from_state(
                state
            )

        # ========================================================
        # APPROVE
        #
        # 注意：
        #
        # 不重新执行 Tool
        # 不重新做 Verification
        # 不重新执行 Policy
        #
        # 因为等待人工时，这些步骤都已经完成。
        #
        # 直接：
        #
        # awaiting_human
        #     ↓
        # generate_answer
        # ========================================================

        state = self._replace_state(
            state,
            human_decision=(
                human_decision
            ),
            status="verifying",
            stop_reason=None,
            pending_human_review=False,
            human_review_reason=None,
            current_node=(
                "await_human"
            ),
            next_node=(
                "generate_answer"
            ),
            node_spans=(
                state.node_spans
                + (
                    span,
                )
            ),
            step_count=(
                state.step_count
                + 1
            ),
            updated_at=now,
        )

        state = (
            self._persist_checkpoint(
                state
            )
        )

        return self._continue_from_state(
            state
        )

    def _continue_from_state(
        self,
        state: AgentState,
    ) -> AgentState:
        """根据 AgentState.next_node 从断点继续运行。"""

        while True:
            if (
                state.status
                in _TERMINAL_STATUSES
            ):
                self._save_trajectory(
                    state
                )

                return state

            if (
                state.status
                == "awaiting_human"
            ):
                return state

            try:
                if (
                    state.step_count
                    >= state.max_steps
                ):
                    raise AgentRuntimeError(
                        "Runtime 已达到 max_steps"
                    )

                # route_intent 的结果可能已经保存成
                # 一个独立 Checkpoint。
                #
                # 因此恢复时需要再次执行 route 后的
                # 状态判断，而不是直接进入 Planner。
                if state.status == "routed":
                    if (
                        state.intent
                        == "unsupported"
                    ):
                        state = (
                            self._mark_refused(
                                state
                            )
                        )

                        state = (
                            self._persist_checkpoint(
                                state
                            )
                        )

                        continue

                    parsed_query = (
                        self._require_parsed_query(
                            state
                        )
                    )

                    missing_fields = (
                        self
                        ._blocking_missing_fields(
                            parsed_query
                        )
                    )

                    if missing_fields:
                        state = (
                            self
                            ._mark_awaiting_human(
                                state,
                                blocking_missing_fields=(
                                    missing_fields
                                ),
                            )
                        )

                        state = (
                            self
                            ._persist_checkpoint(
                                state
                            )
                        )

                        return state

                next_node = (
                    state.next_node
                )

                if next_node == "parse_query":
                    state = (
                        self._run_parse_node(
                            state
                        )
                    )

                    state = (
                        self._persist_checkpoint(
                            state
                        )
                    )

                    continue

                if next_node == "route_intent":
                    state = (
                        self._run_route_node(
                            state
                        )
                    )

                    state = (
                        self._persist_checkpoint(
                            state
                        )
                    )

                    continue

                if next_node == "create_plan":
                    state = (
                        self._run_plan_node(
                            state
                        )
                    )

                    state = (
                        self._persist_checkpoint(
                            state
                        )
                    )

                    continue

                if next_node == "execute_plan":
                    self._require_execution_services()

                    assert (
                        self.plan_executor
                        is not None
                    )

                    state = (
                        self.plan_executor
                        .execute_next_step(
                            state
                        )
                    )

                    state = (
                        self._persist_checkpoint(
                            state
                        )
                    )

                    continue

                if (
                    next_node
                    == "verify_evidence"
                ):
                    self._require_execution_services()

                    assert (
                        self.verifier
                        is not None
                    )

                    state = (
                        self.verifier.verify(
                            state
                        )
                    )

                    state = (
                        self._persist_checkpoint(
                            state
                        )
                    )

                    continue

                if (
                    next_node
                    == "prepare_answer"
                ):
                    state = (
                        self._run_prepare_answer_node(
                            state
                        )
                    )

                    state = (
                        self._persist_checkpoint(
                            state
                        )
                    )

                    continue
                if (
                    next_node
                    == "verify_answer"
                ):
                    state = (
                        self._run_verify_answer_node(
                            state
                        )
                    )

                    state = (
                        self._persist_checkpoint(
                            state
                        )
                    )

                    continue

                if (
                    next_node
                    == "evaluate_policy"
                ):
                    state = (
                        self._run_policy_node(
                            state
                        )
                    )

                    state = (
                        self._persist_checkpoint(
                            state
                        )
                    )

                    continue

                if (
                    next_node
                    == "generate_answer"
                ):
                    self._require_execution_services()

                    assert (
                        self.answer_generator
                        is not None
                    )

                    state = (
                        self.answer_generator
                        .generate(
                            state
                        )
                    )

                    state = (
                        self._persist_checkpoint(
                            state
                        )
                    )

                    continue

                if (
                    next_node
                    == "await_human"
                ):
                    if (
                        state.status
                        != "awaiting_human"
                    ):
                        raise AgentRuntimeError(
                            "next_node=await_human "
                            "但 State 不是 "
                            "awaiting_human"
                        )

                    return state

                if next_node == "finish":
                    if (
                        state.status
                        not in _TERMINAL_STATUSES
                    ):
                        raise AgentRuntimeError(
                            "非终止状态不能直接进入 "
                            "finish"
                        )

                    self._save_trajectory(
                        state
                    )

                    return state

                if (
                    next_node
                    == "handle_failure"
                ):
                    raise AgentRuntimeError(
                        "无法从未完成的 "
                        "handle_failure 节点恢复"
                    )

                raise AgentRuntimeError(
                    "Runtime 遇到未知 next_node："
                    f"{next_node}"
                )

            except Exception as exc:
                state = (
                    self._handle_failure(
                        state=state,
                        error=exc,
                    )
                )

                state = (
                    self._persist_checkpoint(
                        state
                    )
                )

    def _run_parse_node(
        self,
        state: AgentState,
    ) -> AgentState:
        started_at = self.clock.now()
        timer_start = perf_counter()

        parsed_query = (
            self.query_parser.parse(
                state.query
            )
        )

        completed_at = self.clock.now()

        target_metric_ids = (
            _unique_in_order(
                parsed_query.metric_ids
                + parsed_query
                .calculation_metric_ids
            )
        )

        span = (
            self._build_completed_span(
                node_name="parse_query",
                input_summary={
                    "query_length": len(
                        state.query
                    ),
                },
                output_summary={
                    "company_count": len(
                        parsed_query.company_ids
                    ),
                    "report_count": len(
                        parsed_query.report_ids
                    ),
                    "metric_count": len(
                        target_metric_ids
                    ),
                    "parser_version": (
                        self.query_parser
                        .parser_version
                    ),
                },
                started_at=started_at,
                completed_at=completed_at,
                timer_start=timer_start,
                checkpoint_revision=(
                    state.checkpoint_revision
                ),
            )
        )

        return self._replace_state(
            state,
            parsed_query=parsed_query,
            company_ids=(
                parsed_query.company_ids
            ),
            report_ids=(
                parsed_query.report_ids
            ),
            years=(
                parsed_query.years
            ),
            metric_ids=(
                target_metric_ids
            ),
            confidence=(
                parsed_query.confidence
            ),
            status="parsing",
            current_node="parse_query",
            next_node="route_intent",
            node_spans=(
                state.node_spans
                + (
                    span,
                )
            ),
            step_count=(
                state.step_count + 1
            ),
            updated_at=completed_at,
        )

    def _run_route_node(
        self,
        state: AgentState,
    ) -> AgentState:
        parsed_query = (
            self._require_parsed_query(
                state
            )
        )

        started_at = self.clock.now()
        timer_start = perf_counter()

        intent = (
            self.intent_router.route(
                parsed_query
            )
        )

        completed_at = self.clock.now()

        span = (
            self._build_completed_span(
                node_name="route_intent",
                input_summary={
                    "comparison_requested": (
                        parsed_query
                        .comparison_requested
                    ),
                    "ranking_requested": (
                        parsed_query
                        .ranking_requested
                    ),
                    "explanation_requested": (
                        parsed_query
                        .explanation_requested
                    ),
                },
                output_summary={
                    "intent": intent,
                    "router_version": (
                        self.intent_router
                        .router_version
                    ),
                },
                started_at=started_at,
                completed_at=completed_at,
                timer_start=timer_start,
                checkpoint_revision=(
                    state.checkpoint_revision
                ),
            )
        )

        return self._replace_state(
            state,
            intent=intent,
            status="routed",
            current_node="route_intent",
            next_node=(
                "finish"
                if intent == "unsupported"
                else "create_plan"
            ),
            node_spans=(
                state.node_spans
                + (
                    span,
                )
            ),
            step_count=(
                state.step_count + 1
            ),
            updated_at=completed_at,
        )

    def _run_plan_node(
        self,
        state: AgentState,
    ) -> AgentState:
        parsed_query = (
            self._require_parsed_query(
                state
            )
        )

        intent = (
            self._require_supported_intent(
                state
            )
        )

        started_at = self.clock.now()
        timer_start = perf_counter()

        runtime_plan = (
            self.planner.create_plan(
                parsed_query=parsed_query,
                intent=intent,
            )
        )

        completed_at = self.clock.now()

        span = (
            self._build_completed_span(
                node_name="create_plan",
                input_summary={
                    "intent": intent,
                },
                output_summary={
                    "planner_version": (
                        runtime_plan
                        .planner_version
                    ),
                    "step_count": len(
                        runtime_plan
                        .plan.steps
                    ),
                    "financial_query_count": len(
                        runtime_plan
                        .financial_queries
                    ),
                    "document_query_count": len(
                        runtime_plan
                        .document_queries
                    ),
                },
                started_at=started_at,
                completed_at=completed_at,
                timer_start=timer_start,
                checkpoint_revision=(
                    state.checkpoint_revision
                ),
            )
        )

        return self._replace_state(
            state,
            runtime_plan=runtime_plan,
            planner_version=(
                runtime_plan
                .planner_version
            ),
            status="planned",
            current_node="create_plan",
            next_node="execute_plan",
            node_spans=(
                state.node_spans
                + (
                    span,
                )
            ),
            step_count=(
                state.step_count + 1
            ),
            updated_at=completed_at,
        )

    def _run_prepare_answer_node(
        self,
        state: AgentState,
    ) -> AgentState:
        if (
            self.answer_draft_builder
            is None
        ):
            raise AgentRuntimeError(
                "prepare_answer 需要配置 "
                "answer_draft_builder"
            )

        if (
            state.status
            != "verifying"
        ):
            raise AgentRuntimeError(
                "只有 Verifying 状态才能进入 "
                "prepare_answer"
            )

        if (
            state.step_count
            >= state.max_steps
        ):
            raise AgentRuntimeError(
                "Runtime 已达到 max_steps"
            )

        started_at = (
            self.clock.now()
        )

        timer_start = (
            perf_counter()
        )

        answer_draft = (
            self.answer_draft_builder
            .build(
                state
            )
        )

        completed_at = (
            self.clock.now()
        )

        span = (
            self._build_completed_span(
                node_name=(
                    "prepare_answer"
                ),
                input_summary={
                    "intent": (
                        state.intent
                    ),
                    "citation_count": len(
                        state.citations
                    ),
                    "fact_count": len(
                        state.resolved_fact_ids
                    ),
                    "calculation_count": len(
                        state.calculation_ids
                    ),
                },
                output_summary={
                    "draft_id": (
                        answer_draft.draft_id
                    ),
                    "draft_type": (
                        answer_draft.draft_type
                    ),
                    "claim_count": len(
                        answer_draft.claims
                    ),
                },
                started_at=(
                    started_at
                ),
                completed_at=(
                    completed_at
                ),
                timer_start=(
                    timer_start
                ),
                checkpoint_revision=(
                    state
                    .checkpoint_revision
                ),
            )
        )

        return self._replace_state(
            state,
            answer_draft=(
                answer_draft
            ),

            # 新 Draft 尚未经过可信验证。
            verification_report=None,

            status="verifying",

            current_node=(
                "prepare_answer"
            ),

            # Step3.2 的核心变化：
            # 不允许直接进入 Generator。
            next_node=(
                "verify_answer"
            ),

            node_spans=(
                state.node_spans
                + (
                    span,
                )
            ),

            step_count=(
                state.step_count
                + 1
            ),

            updated_at=(
                completed_at
            ),
        )

    def _run_verify_answer_node(
        self,
        state: AgentState,
    ) -> AgentState:
        """对 AnswerDraft 执行回答级可信校验。"""

        if (
            self.trust_verifier
            is None
        ):
            raise AgentRuntimeError(
                "verify_answer 需要配置 "
                "trust_verifier"
            )

        if (
            state.status
            != "verifying"
        ):
            raise AgentRuntimeError(
                "只有 Verifying 状态才能进入 "
                "verify_answer"
            )

        if (
            state.answer_draft
            is None
        ):
            raise AgentRuntimeError(
                "verify_answer 缺少 "
                "answer_draft"
            )

        if (
            state.step_count
            >= state.max_steps
        ):
            raise AgentRuntimeError(
                "Runtime 已达到 max_steps"
            )

        started_at = (
            self.clock.now()
        )

        timer_start = (
            perf_counter()
        )

        # ========================================================
        # Trust Verification
        #
        # RuntimeTrustVerifier 不修改 State。
        # 它只返回结构化 VerificationReport。
        # ========================================================

        verification_report = (
            self.trust_verifier
            .verify(
                state
            )
        )

        completed_at = (
            self.clock.now()
        )

        issue_types = tuple(
            issue.issue_type
            for issue
            in verification_report.issues
        )

        span = (
            self._build_completed_span(
                node_name=(
                    "verify_answer"
                ),
                input_summary={
                    "draft_id": (
                        state
                        .answer_draft
                        .draft_id
                    ),
                    "claim_count": len(
                        state
                        .answer_draft
                        .claims
                    ),
                },
                output_summary={
                    "passed": (
                        verification_report
                        .passed
                    ),
                    "numeric_verified": (
                        verification_report
                        .numeric_verified
                    ),
                    "evidence_verified": (
                        verification_report
                        .evidence_verified
                    ),
                    "citation_verified": (
                        verification_report
                        .citation_verified
                    ),
                    "evidence_sufficient": (
                        verification_report
                        .evidence_sufficient
                    ),
                    "issue_count": len(
                        verification_report
                        .issues
                    ),
                    "issue_types": (
                        issue_types
                    ),
                },
                started_at=(
                    started_at
                ),
                completed_at=(
                    completed_at
                ),
                timer_start=(
                    timer_start
                ),
                checkpoint_revision=(
                    state
                    .checkpoint_revision
                ),
            )
        )

        # ========================================================
        # PASS
        #
        # 校验成功：
        #
        # AnswerDraft
        #     ↓
        # VerificationReport(PASS)
        #     ↓
        # evaluate_policy
        # ========================================================

        if (
            verification_report.passed
        ):
            return self._replace_state(
                state,
                verification_report=(
                    verification_report
                ),
                status="verifying",
                current_node=(
                    "verify_answer"
                ),
                next_node=(
                    "evaluate_policy"
                ),
                node_spans=(
                    state.node_spans
                    + (
                        span,
                    )
                ),
                step_count=(
                    state.step_count
                    + 1
                ),
                updated_at=(
                    completed_at
                ),
            )

        # ========================================================
        # FAIL
        #
        # Trust Verification 本身执行成功，
        # 但发现回答不可信。
        #
        # 这不是 Runtime Crash，
        # 而是 Controlled Refusal。
        # ========================================================

        return self._replace_state(
            state,
            verification_report=(
                verification_report
            ),

            answer=None,

            status="refused",

            stop_reason=(
                "insufficient_evidence"
            ),

            current_node=(
                "verify_answer"
            ),

            next_node="finish",

            node_spans=(
                state.node_spans
                + (
                    span,
                )
            ),

            step_count=(
                state.step_count
                + 1
            ),

            completed_at=(
                completed_at
            ),

            updated_at=(
                completed_at
            ),
        )

    @staticmethod
    def _risk_level_for_state(
        state: AgentState,
    ) -> RiskLevel:
        """根据当前受支持 Intent 做第一版确定性风险分级。"""

        intent = state.intent

        if intent == "financial_fact":
            return "low"

        if intent in {
            "financial_calculation",
            "financial_comparison",
            "document_evidence",
        }:
            return "medium"

        raise AgentRuntimeError(
            "无法为当前 Intent 分配 RiskLevel："
            f"{intent}"
        )

    def _run_policy_node(
        self,
        state: AgentState,
    ) -> AgentState:
        """执行可信校验后的确定性 Risk Policy。"""

        if self.risk_policy is None:
            raise AgentRuntimeError(
                "evaluate_policy 需要配置 "
                "risk_policy"
            )

        if (
            state.status
            != "verifying"
        ):
            raise AgentRuntimeError(
                "只有 Verifying 状态才能进入 "
                "evaluate_policy"
            )

        if (
            state.verification_report
            is None
        ):
            raise AgentRuntimeError(
                "evaluate_policy 缺少 "
                "verification_report"
            )

        if (
            state.answer_draft
            is None
        ):
            raise AgentRuntimeError(
                "evaluate_policy 缺少 "
                "answer_draft"
            )

        if (
            state.step_count
            >= state.max_steps
        ):
            raise AgentRuntimeError(
                "Runtime 已达到 max_steps"
            )

        started_at = (
            self.clock.now()
        )

        timer_start = (
            perf_counter()
        )

        risk_level = (
            self._risk_level_for_state(
                state
            )
        )

        claim_ids = tuple(
            claim.claim_id
            for claim
            in state.answer_draft.claims
        )

        decision = (
            self.risk_policy.evaluate(
                risk_level=risk_level,
                verification_report=(
                    state
                    .verification_report
                ),
                claim_ids=claim_ids,
            )
        )

        completed_at = (
            self.clock.now()
        )

        span = (
            self._build_completed_span(
                node_name=(
                    "evaluate_policy"
                ),
                input_summary={
                    "intent": (
                        state.intent
                    ),
                    "risk_level": (
                        risk_level
                    ),
                    "verification_passed": (
                        state
                        .verification_report
                        .passed
                    ),
                    "claim_count": len(
                        claim_ids
                    ),
                },
                output_summary={
                    "action": (
                        decision.action
                    ),
                    "risk_level": (
                        decision.risk_level
                    ),
                    "requires_human": (
                        decision.action
                        == "require_human"
                    ),
                },
                started_at=started_at,
                completed_at=completed_at,
                timer_start=timer_start,
                checkpoint_revision=(
                    state
                    .checkpoint_revision
                ),
            )
        )

        # ========================================================
        # ALLOW
        # ========================================================

        if decision.action == "allow":
            return self._replace_state(
                state,
                risk_level=(
                    risk_level
                ),
                policy_decision=(
                    decision
                ),
                status="verifying",
                current_node=(
                    "evaluate_policy"
                ),
                next_node=(
                    "generate_answer"
                ),
                node_spans=(
                    state.node_spans
                    + (
                        span,
                    )
                ),
                step_count=(
                    state.step_count
                    + 1
                ),
                updated_at=(
                    completed_at
                ),
            )

        # ========================================================
        # REFUSE
        # ========================================================

        if decision.action == "refuse":
            return self._replace_state(
                state,
                risk_level=(
                    risk_level
                ),
                policy_decision=(
                    decision
                ),
                answer=None,
                status="refused",
                stop_reason=(
                    "policy_refused"
                ),
                pending_human_review=False,
                human_review_reason=None,
                current_node=(
                    "evaluate_policy"
                ),
                next_node="finish",
                node_spans=(
                    state.node_spans
                    + (
                        span,
                    )
                ),
                step_count=(
                    state.step_count
                    + 1
                ),
                completed_at=(
                    completed_at
                ),
                updated_at=(
                    completed_at
                ),
            )

        # ========================================================
        # REQUIRE HUMAN
        #
        # Step6.2 先完成 Routing。
        # Step6.3 再处理 approve / reject / resume。
        # ========================================================

        human_review = (
            decision.human_review
        )

        if human_review is None:
            raise AgentRuntimeError(
                "require_human 缺少 "
                "HumanReviewRequest"
            )

        return self._replace_state(
            state,
            risk_level=(
                risk_level
            ),
            policy_decision=(
                decision
            ),
            status="awaiting_human",
            stop_reason=(
                "human_review_required"
            ),
            pending_human_review=True,
            human_review_reason=(
                human_review.reason
            ),
            current_node=(
                "await_human"
            ),
            next_node=(
                "await_human"
            ),
            node_spans=(
                state.node_spans
                + (
                    span,
                )
            ),
            step_count=(
                state.step_count
                + 1
            ),
            updated_at=(
                completed_at
            ),
        )

    def _mark_refused(
        self,
        state: AgentState,
    ) -> AgentState:
        now = self.clock.now()

        return self._replace_state(
            state,
            status="refused",
            stop_reason="unsupported",
            current_node="finish",
            next_node="finish",
            completed_at=now,
            updated_at=now,
        )

    def _mark_awaiting_human(
        self,
        state: AgentState,
        *,
        blocking_missing_fields: tuple[
            str,
            ...
        ],
    ) -> AgentState:
        now = self.clock.now()

        return self._replace_state(
            state,
            status="awaiting_human",
            stop_reason=(
                "human_review_required"
            ),
            pending_human_review=True,
            human_review_reason=(
                "生成执行计划所需字段缺失："
                + ", ".join(
                    blocking_missing_fields
                )
            ),
            current_node="await_human",
            next_node="await_human",
            updated_at=now,
        )

    def _handle_failure(
        self,
        *,
        state: AgentState,
        error: Exception,
    ) -> AgentState:
        now = self.clock.now()

        stop_reason, final_status = (
            self._classify_failure(
                error
            )
        )

        stage = self._failure_stage(
            error
        )

        extra_tool_traces = ()

        if isinstance(
            error,
            ToolExecutionFailedError,
        ):
            extra_tool_traces = (
                error.traces
            )

        elif isinstance(
            error,
            RuntimePlanExecutorError,
        ):
            if (
                error.execution_result
                is not None
            ):
                extra_tool_traces = (
                    error.execution_result
                    .traces
                )

        tool_call_traces = (
            self._merge_tool_traces(
                state.tool_call_traces,
                extra_tool_traces,
            )
        )

        prompt_injection_findings = (
            state.prompt_injection_findings
        )

        if isinstance(
            error,
            RuntimePromptInjectionDetectedError,
        ):
            detection_result = (
                error.detection_result
            )

            finding = (
                PromptInjectionFinding(
                    chunk_id=(
                        error.chunk_id
                    ),
                    document_id=(
                        error.document_id
                    ),
                    severity=(
                        detection_result
                        .severity
                    ),
                    matched_rule_ids=(
                        detection_result
                        .matched_rule_ids
                    ),
                    reason=(
                        detection_result
                        .reason
                    ),
                )
            )

            if (
                finding
                not in
                prompt_injection_findings
            ):
                prompt_injection_findings += (
                    finding,
                )

        calculation_traces = (
            state.calculation_traces
        )

        if (
            isinstance(
                error,
                RuntimePlanExecutorError,
            )
            and error.calculation_trace
            is not None
        ):
            existing_ids = {
                trace.calculation_id
                for trace
                in calculation_traces
            }

            if (
                error.calculation_trace
                .calculation_id
                not in existing_ids
            ):
                calculation_traces += (
                    error.calculation_trace,
                )

        error_message = str(error)

        if not error_message:
            error_message = (
                error.__class__.__name__
            )

        error_message = (
            error_message[:2000]
        )

        error_record = AgentErrorRecord(
            stage=stage,
            error_type=(
                error.__class__.__name__
            ),
            message=error_message,
            retryable=False,
            occurred_at=now,
        )

        span = NodeSpan(
            span_id=(
                self.id_factory.new_id(
                    "span"
                )
            ),
            node_name="handle_failure",
            attempt=1,
            status="completed",
            input_summary={
                "error_type": (
                    error
                    .__class__.__name__
                ),
                "stage": stage,
            },
            output_summary={
                "final_status": (
                    final_status
                ),
                "stop_reason": (
                    stop_reason
                ),
            },
            started_at=now,
            completed_at=now,
            latency_ms=0.0,
            checkpoint_revision=(
                state.checkpoint_revision
            ),
            error_type=None,
            error_message=None,
        )

        retry_increment = max(
            len(extra_tool_traces) - 1,
            0,
        )

        return self._replace_state(
            state,
            status=final_status,
            stop_reason=stop_reason,
            answer=None,
            prompt_injection_findings=(
                prompt_injection_findings
            ),
            pending_human_review=False,
            human_review_reason=None,
            tool_call_traces=(
                tool_call_traces
            ),
            calculation_traces=(
                calculation_traces
            ),
            retry_count=(
                state.retry_count
                + retry_increment
            ),
            errors=(
                state.errors
                + (
                    error_record,
                )
            ),
            node_spans=(
                state.node_spans
                + (
                    span,
                )
            ),
            step_count=min(
                state.step_count + 1,
                state.max_steps,
            ),
            current_node=(
                "handle_failure"
            ),
            next_node="finish",
            completed_at=now,
            updated_at=now,
        )

    @staticmethod
    def _classify_failure(
        error: Exception,
    ) -> tuple[
        str,
        str,
    ]:
        message = str(error)

        if "max_steps" in message:
            return (
                "max_steps_exceeded",
                "failed",
            )

        if isinstance(
            error,
            RuntimeEvidenceError,
        ):
            return (
                "insufficient_evidence",
                "refused",
            )

        if isinstance(
            error,
            RuntimeAnswerDraftError,
        ):
            return (
                "internal_error",
                "failed",
            )

        if isinstance(
            error,
            RuntimePromptInjectionDetectedError,
        ):
            # ========================================================
            # Week7 Step5.3
            #
            # Prompt Injection 被安全机制主动拦截，
            # 不属于 Runtime Crash。
            #
            # 因此：
            #
            # failed / internal_error
            #     ❌
            #
            # refused / prompt_injection_detected
            #     ✅
            # ========================================================

            return (
                "prompt_injection_detected",
                "refused",
            )

        if isinstance(
            error,
            RuntimePermissionSnapshotMismatchError,
        ):
            return (
                "permission_denied",
                "refused",
            )

        if (
            isinstance(
                error,
                RuntimePlanExecutorError,
            )
            and (
                "没有返回可用"
                in message
            )
        ):
            return (
                "insufficient_evidence",
                "refused",
            )

        if (
            isinstance(
                error,
                RuntimePlanExecutorError,
            )
            and (
                error.calculation_trace
                is not None
                or (
                    "Calculation 领域执行失败"
                    in message
                )
            )
        ):
            return (
                "calculation_failed",
                "failed",
            )

        if isinstance(
            error,
            ToolExecutionFailedError,
        ):
            # ========================================================
            # Week7 Step4.3
            #
            # Permission Denied 不是 Runtime Crash。
            #
            # 它表示系统成功执行了安全策略，
            # 并主动阻止了一次未授权 Tool Call。
            #
            # 因此：
            #
            # failed
            #     ❌
            #
            # refused / permission_denied
            #     ✅
            # ========================================================

            if any(
                trace.error_type
                == "ToolPermissionDeniedError"
                for trace
                in error.traces
            ):
                return (
                    "permission_denied",
                    "refused",
                )

            if any(
                trace.status
                == "timed_out"
                for trace
                in error.traces
            ):
                return (
                    "tool_timeout",
                    "failed",
                )

            return (
                "tool_failure",
                "failed",
            )

        if isinstance(
            error,
            ToolRegistryError,
        ):
            return (
                "tool_failure",
                "failed",
            )

        if isinstance(
            error,
            RuntimeAnswerGenerationError,
        ):
            return (
                "internal_error",
                "failed",
            )

        return (
            "internal_error",
            "failed",
        )

    @staticmethod
    def _failure_stage(
        error: Exception,
    ) -> str:
        if isinstance(
            error,
            RuntimeEvidenceError,
        ):
            return "verify_evidence"

        if isinstance(
            error,
            RuntimeAnswerDraftError,
        ):
            return "prepare_answer"

        if isinstance(
            error,
            RuntimeTrustVerificationError,
        ):
            return "verify_answer"

        if isinstance(
            error,
            RuntimeAnswerGenerationError,
        ):
            return "generate_answer"

        return "execute_plan"

    def _persist_checkpoint(
        self,
        state: AgentState,
    ) -> AgentState:
        if self.checkpoint_store is None:
            return state

        record = (
            self.checkpoint_store.save(
                state,
                expected_revision=(
                    state.checkpoint_revision
                ),
            )
        )

        return record.state

    def _save_trajectory(
        self,
        state: AgentState,
    ) -> None:
        """幂等完成一次 Runtime Trajectory finalization。"""

        if self.trajectory_store is None:
            return

        trajectory = (
            self._build_trajectory(
                state
            )
        )

        try:
            self.trajectory_store.save(
                trajectory
            )

        except (
            TrajectoryAlreadyExistsError
        ):
            # Store 本身仍然禁止覆盖。
            # Runtime finalize 则允许重复调用。
            return

    @staticmethod
    def _build_trajectory(
        state: AgentState,
    ) -> AgentTrajectory:
        if (
            state.status
            not in _TERMINAL_STATUSES
        ):
            raise AgentRuntimeError(
                "只有终止状态才能保存 Trajectory"
            )

        if state.completed_at is None:
            raise AgentRuntimeError(
                "终止状态缺少 completed_at"
            )

        latency_ms = max(
            (
                state.completed_at
                - state.started_at
            )
            .total_seconds()
            * 1000.0,
            0.0,
        )

        return AgentTrajectory(
            request_id=state.request_id,
            trace_id=state.trace_id,
            run_id=state.run_id,
            thread_id=state.thread_id,
            query=state.query,
            user_role=(
                state.user_role
            ),
            granted_permissions=(
                state.granted_permissions
            ),
            intent=state.intent,
            planner_version=(
                state.planner_version
            ),
            retriever_version=(
                state.retriever_version
            ),
            calculator_version=(
                state.calculator_version
            ),
            generator_version=(
                state.generator_version
            ),
            prompt_version=(
                state.prompt_version
            ),
            prompt_sha256=(
                state.prompt_sha256
            ),
            model_name=state.model_name,
            parsed_query=(
                state.parsed_query
            ),
            runtime_plan=(
                state.runtime_plan
            ),
            node_spans=(
                state.node_spans
            ),
            tool_call_traces=(
                state.tool_call_traces
            ),
            retrieval_traces=(
                state.retrieval_traces
            ),
            calculation_traces=(
                state.calculation_traces
            ),
            retrieved_documents=(
                state.retrieved_documents
            ),
            prompt_injection_findings=(
                state.prompt_injection_findings
            ),  
            resolved_fact_ids=(
                state.resolved_fact_ids
            ),
            evidence_ids=(
                state.evidence_ids
            ),
            calculation_ids=(
                state.calculation_ids
            ),
            citations=state.citations,
            answer_draft=(
                state.answer_draft
            ),
            verification_report=(
                state.verification_report
            ),
            risk_level=(
                state.risk_level
            ),
            policy_decision=(
                state.policy_decision
            ),
            human_decision=(
                state.human_decision
            ),
            errors=state.errors,
            answer=state.answer,
            input_tokens=(
                state.input_tokens
            ),
            output_tokens=(
                state.output_tokens
            ),
            estimated_cost=(
                state.estimated_cost
            ),
            started_at=(
                state.started_at
            ),
            completed_at=(
                state.completed_at
            ),
            latency_ms=latency_ms,
            final_status=state.status,
            stop_reason=(
                state.stop_reason
            ),
        )

    @staticmethod
    def _merge_tool_traces(
        existing: tuple,
        incoming: tuple,
    ) -> tuple:
        result = list(existing)

        known_ids = {
            trace.tool_call_id
            for trace in existing
        }

        for trace in incoming:
            if (
                trace.tool_call_id
                in known_ids
            ):
                continue

            result.append(trace)

            known_ids.add(
                trace.tool_call_id
            )

        return tuple(result)

    def _require_execution_services(
        self,
    ) -> None:
        missing: list[str] = []

        if self.plan_executor is None:
            missing.append(
                "plan_executor"
            )

        if self.verifier is None:
            missing.append(
                "verifier"
            )

        if self.risk_policy is None:
            missing.append(
                "risk_policy"
            )

        if self.answer_generator is None:
            missing.append(
                "answer_generator"
            )

        if missing:
            raise AgentRuntimeError(
                "AgentRuntime 缺少完整执行组件："
                f"{missing}"
            )

    def _build_completed_span(
        self,
        *,
        node_name: str,
        input_summary: dict[
            str,
            object,
        ],
        output_summary: dict[
            str,
            object,
        ],
        started_at: datetime,
        completed_at: datetime,
        timer_start: float,
        checkpoint_revision: int,
    ) -> NodeSpan:
        return NodeSpan(
            span_id=(
                self.id_factory.new_id(
                    "span"
                )
            ),
            node_name=node_name,
            attempt=1,
            status="completed",
            input_summary=(
                input_summary
            ),
            output_summary=(
                output_summary
            ),
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=max(
                (
                    perf_counter()
                    - timer_start
                )
                * 1000.0,
                0.0,
            ),
            checkpoint_revision=(
                checkpoint_revision
            ),
            error_type=None,
            error_message=None,
        )

    @staticmethod
    def _require_parsed_query(
        state: AgentState,
    ) -> ParsedFinancialQuery:
        if state.parsed_query is None:
            raise AgentRuntimeError(
                "Runtime 尚未生成 parsed_query"
            )

        return state.parsed_query

    @staticmethod
    def _require_supported_intent(
        state: AgentState,
    ) -> AgentIntent:
        if state.intent is None:
            raise AgentRuntimeError(
                "Runtime 尚未生成 intent"
            )

        if state.intent == "unsupported":
            raise AgentRuntimeError(
                "unsupported intent "
                "不能进入 Planner"
            )

        return state.intent

    @staticmethod
    def _blocking_missing_fields(
        parsed_query: ParsedFinancialQuery,
    ) -> tuple[str, ...]:
        required = {
            "company_ids",
            "years",
            "report_ids",
        }

        return tuple(
            field_name
            for field_name
            in parsed_query.missing_fields
            if field_name in required
        )

    @staticmethod
    def _replace_state(
        state: AgentState,
        **updates: object,
    ) -> AgentState:
        payload = state.model_dump(
            mode="python"
        )

        payload.update(
            updates
        )

        return AgentState.model_validate(
            payload
        )