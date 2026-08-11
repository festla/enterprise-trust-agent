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
    AgentIntent,
    AgentState,
    NodeSpan,
    ParsedFinancialQuery,
    RuntimePlan,
)


class AgentRuntimeError(
    ValueError
):
    """Framework-independent Agent Runtime 基础异常。"""


# ============================================================
# AgentRuntime 不应该强绑定：
#
# RuntimeQueryParser
# RuntimeIntentRouter
# RuntimePlanner
#
# 它只要求这些对象实现对应接口。
#
#
# AgentRuntime
#      ↓
# Protocol
#      ↓
# Deterministic Parser / Router / Planner
#
# 后面即使换成：
#
# LLM Structured Output Parser
# LLM Planner
#
# Runtime 主循环也不需要重写。
# ============================================================


class QueryParserProvider(
    Protocol
):
    parser_version: str

    def parse(
        self,
        question: str,
    ) -> ParsedFinancialQuery:
        """把用户问题解析为结构化 Query。"""


class IntentRouterProvider(
    Protocol
):
    router_version: str

    def route(
        self,
        parsed_query: ParsedFinancialQuery,
    ) -> AgentIntent:
        """根据 Parsed Query 判断任务 Intent。"""


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
        """生成可执行 RuntimePlan。"""


# ============================================================
# 时间和 ID 也通过接口注入。
#
# 原因：
#
# Production：
#
#     datetime.now(...)
#     uuid4()
#
# Test：
#
#     固定时间
#     request_1 / run_1 ...
#
# 这样测试不会依赖随机值。
# ============================================================


class RuntimeClock(
    Protocol
):
    def now(
        self,
    ) -> datetime:
        """返回带时区的当前时间。"""


class RuntimeIdFactory(
    Protocol
):
    def new_id(
        self,
        prefix: str,
    ) -> str:
        """生成一个 Runtime ID。"""


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
    values: tuple[str, ...],
) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        result.append(value)

    return tuple(result)


# ============================================================
# 这就是 Week 6 Step 8 的核心类。
#
#
#                    AgentRuntime
#
#                         │
#            ┌────────────┼────────────┐
#            ↓            ↓            ↓
#          Parser       Router       Planner
#
#            │            │            │
#            └────────────┴────────────┘
#                         ↓
#
#                    AgentState
#
#
# 8A 只执行到：
#
# status = "planned"
#
# 暂时不执行任何 Tool。
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class AgentRuntime:
    query_parser: QueryParserProvider

    intent_router: IntentRouterProvider

    planner: RuntimePlannerProvider

    clock: RuntimeClock = field(
        default_factory=UTCClock
    )

    id_factory: RuntimeIdFactory = field(
        default_factory=(
            UUIDRuntimeIdFactory
        )
    )

    # ========================================================
    # create_state 只负责：
    #
    # “一次 Agent Run 的初始化”
    #
    # 此时：
    #
    # parsed_query = None
    # intent = None
    # runtime_plan = None
    #
    # status = created
    # ========================================================

    def create_state(
        self,
        *,
        query: str,
        request_id: str | None = None,
        trace_id: str | None = None,
        run_id: str | None = None,
        thread_id: str | None = None,
        max_steps: int = 32,
    ) -> AgentState:
        now = self.clock.now()

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

    # ========================================================
    # prepare() 是 8A 的主入口。
    #
    #
    # create
    #   ↓
    # parse
    #   ↓
    # route
    #   ↓
    # plan
    #   ↓
    # planned
    #
    #
    # 8B 后面会变成：
    #
    # prepare
    #   ↓
    # execute_plan
    # ========================================================

    def prepare(
        self,
        *,
        query: str,
        request_id: str | None = None,
        trace_id: str | None = None,
        run_id: str | None = None,
        thread_id: str | None = None,
        max_steps: int = 32,
    ) -> AgentState:
        state = self.create_state(
            query=query,
            request_id=request_id,
            trace_id=trace_id,
            run_id=run_id,
            thread_id=thread_id,
            max_steps=max_steps,
        )

        state = self._run_parse_node(
            state
        )

        state = self._run_route_node(
            state
        )

        # ====================================================
        # unsupported 不应该继续交给 Planner。
        #
        # 这是正常业务拒答，
        # 不是系统崩溃。
        # ====================================================

        if state.intent == "unsupported":
            return self._mark_refused(
                state
            )

        parsed_query = (
            self._require_parsed_query(
                state
            )
        )

        blocking_missing_fields = (
            self
            ._blocking_missing_fields(
                parsed_query
            )
        )

        # ====================================================
        # Missing != Unsupported
        #
        #
        # “营业收入是多少？”
        #
        # Router：
        #     financial_fact
        #
        # 但是：
        #     company/year 不完整
        #
        # 所以不能：
        #
        #     refused
        #
        # 而应该：
        #
        #     awaiting_human
        #
        # 后面 Step 9 LangGraph Interrupt
        # 就会真正接住这个状态。
        # ====================================================

        if blocking_missing_fields:
            return (
                self
                ._mark_awaiting_human(
                    state,
                    blocking_missing_fields=(
                        blocking_missing_fields
                    ),
                )
            )

        state = self._run_plan_node(
            state
        )

        return state

    # ========================================================
    # Parse Node
    # ========================================================

    def _run_parse_node(
        self,
        state: AgentState,
    ) -> AgentState:
        node_started_at = (
            self.clock.now()
        )

        timer_start = (
            perf_counter()
        )

        state = self._replace_state(
            state,
            status="parsing",
            current_node=(
                "parse_query"
            ),
            next_node=(
                "route_intent"
            ),
            updated_at=(
                node_started_at
            ),
        )

        parsed_query = (
            self.query_parser.parse(
                state.query
            )
        )

        node_completed_at = (
            self.clock.now()
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
                    "year_count": len(
                        parsed_query.years
                    ),
                    "reported_metric_count": len(
                        parsed_query.metric_ids
                    ),
                    "derived_metric_count": len(
                        parsed_query
                        .calculation_metric_ids
                    ),
                    "confidence": (
                        parsed_query.confidence
                    ),
                    "parser_version": (
                        self.query_parser
                        .parser_version
                    ),
                },
                started_at=(
                    node_started_at
                ),
                completed_at=(
                    node_completed_at
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

        # ====================================================
        # AgentState.metric_ids 是 Runtime 层的
        # “目标指标摘要”。
        #
        # 因此这里把：
        #
        # reported metric
        # +
        # derived metric
        #
        # 都保存进去。
        #
        # 详细区分仍然保留在 parsed_query 内。
        # ====================================================

        target_metric_ids = (
            _unique_in_order(
                parsed_query.metric_ids
                + parsed_query
                .calculation_metric_ids
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
            node_spans=(
                state.node_spans
                + (
                    span,
                )
            ),
            step_count=(
                state.step_count + 1
            ),
            updated_at=(
                node_completed_at
            ),
        )

    # ========================================================
    # Route Node
    # ========================================================

    def _run_route_node(
        self,
        state: AgentState,
    ) -> AgentState:
        parsed_query = (
            self._require_parsed_query(
                state
            )
        )

        node_started_at = (
            self.clock.now()
        )

        timer_start = (
            perf_counter()
        )

        intent = (
            self.intent_router.route(
                parsed_query
            )
        )

        node_completed_at = (
            self.clock.now()
        )

        next_node = (
            "finish"
            if intent == "unsupported"
            else "create_plan"
        )

        span = (
            self._build_completed_span(
                node_name="route_intent",
                input_summary={
                    "has_reported_metrics": (
                        bool(
                            parsed_query
                            .metric_ids
                        )
                    ),
                    "has_derived_metrics": (
                        bool(
                            parsed_query
                            .calculation_metric_ids
                        )
                    ),
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
                started_at=(
                    node_started_at
                ),
                completed_at=(
                    node_completed_at
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
            intent=intent,
            status="routed",
            current_node=(
                "route_intent"
            ),
            next_node=next_node,
            node_spans=(
                state.node_spans
                + (
                    span,
                )
            ),
            step_count=(
                state.step_count + 1
            ),
            updated_at=(
                node_completed_at
            ),
        )

    # ========================================================
    # Plan Node
    # ========================================================

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

        node_started_at = (
            self.clock.now()
        )

        timer_start = (
            perf_counter()
        )

        runtime_plan = (
            self.planner.create_plan(
                parsed_query=(
                    parsed_query
                ),
                intent=intent,
            )
        )

        node_completed_at = (
            self.clock.now()
        )

        span = (
            self._build_completed_span(
                node_name="create_plan",
                input_summary={
                    "intent": intent,
                    "report_count": len(
                        parsed_query.report_ids
                    ),
                    "target_metric_count": (
                        len(
                            parsed_query
                            .metric_ids
                        )
                        + len(
                            parsed_query
                            .calculation_metric_ids
                        )
                    ),
                },
                output_summary={
                    "planner_version": (
                        runtime_plan
                        .planner_version
                    ),
                    "plan_step_count": len(
                        runtime_plan
                        .plan
                        .steps
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
                started_at=(
                    node_started_at
                ),
                completed_at=(
                    node_completed_at
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
            runtime_plan=(
                runtime_plan
            ),
            planner_version=(
                runtime_plan
                .planner_version
            ),
            status="planned",
            current_node=(
                "create_plan"
            ),
            next_node=(
                "execute_plan"
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
            updated_at=(
                node_completed_at
            ),
        )

    # ========================================================
    # Expected early exits
    # ========================================================

    def _mark_refused(
        self,
        state: AgentState,
    ) -> AgentState:
        now = self.clock.now()

        return self._replace_state(
            state,
            status="refused",
            stop_reason="unsupported",
            completed_at=now,
            current_node="finish",
            next_node="finish",
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

        reason = (
            "生成执行计划所需字段缺失："
            + ", ".join(
                blocking_missing_fields
            )
        )

        return self._replace_state(
            state,
            status="awaiting_human",
            stop_reason=(
                "human_review_required"
            ),
            pending_human_review=True,
            human_review_reason=reason,
            current_node="await_human",
            next_node="await_human",
            updated_at=now,
        )

    # ========================================================
    # 为什么不用：
    #
    #     state.status = "refused"
    #     state.stop_reason = ...
    #     state.completed_at = ...
    #
    #
    # 因为 AgentState 开启了：
    #
    #     validate_assignment=True
    #
    # 并且有跨字段 Model Validator。
    #
    # 如果先设置：
    #
    #     status = refused
    #
    # 此时：
    #
    #     completed_at 还是 None
    #     stop_reason 还是 None
    #
    # 模型会立刻判定为非法状态。
    #
    #
    # 所以 Runtime State Transition
    # 应该“原子更新”：
    #
    # old state
    #    ↓
    # dump
    #    ↓
    # 同时 update 多个字段
    #    ↓
    # 重新完整校验
    #    ↓
    # new state
    #
    # 这是 8A 最值得理解的地方之一。
    # ========================================================

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

    # ========================================================
    # Node Span
    # ========================================================

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
        latency_ms = max(
            (
                perf_counter()
                - timer_start
            )
            * 1000,
            0.0,
        )

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
            started_at=(
                started_at
            ),
            completed_at=(
                completed_at
            ),
            latency_ms=(
                latency_ms
            ),

            # =================================================
            # 【重点理解】
            #
            # 8A 还没有接 CheckpointStore，
            # 所以这里仍然是初始 revision。
            #
            # 8D 会正式更新它。
            # =================================================

            checkpoint_revision=(
                checkpoint_revision
            ),
            error_type=None,
            error_message=None,
        )

    # ========================================================
    # Guards
    # ========================================================

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
        blocking_names = {
            "company_ids",
            "years",
            "report_ids",
        }

        return tuple(
            field_name
            for field_name
            in parsed_query.missing_fields
            if field_name
            in blocking_names
        )