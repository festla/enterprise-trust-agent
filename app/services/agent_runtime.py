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
    NodeSpan,
    ParsedFinancialQuery,
    RuntimePlan,
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
    RuntimePlanExecutor,
    RuntimePlanExecutorError,
)
from app.services.tool_registry import (
    ToolExecutionFailedError,
    ToolRegistryError,
)
from app.services.trajectory_store import (
    TrajectoryStore,
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


@dataclass(
    frozen=True,
    slots=True,
)
class AgentRuntime:
    query_parser: QueryParserProvider

    intent_router: IntentRouterProvider

    planner: RuntimePlannerProvider

    plan_executor: (
        RuntimePlanExecutor | None
    ) = None

    verifier: (
        RuntimeEvidenceVerifier | None
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

    id_factory: RuntimeIdFactory = (
        field(
            default_factory=(
                UUIDRuntimeIdFactory
            )
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
            current_node="parse_query",
            next_node="parse_query",
            started_at=now,
            updated_at=now,
        )

    # prepare 保持 8A 语义：
    # 只执行 parse -> route -> plan。
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

    # 完整 Framework-independent Runtime 入口。
    def run(
        self,
        *,
        query: str,
        request_id: str | None = None,
        trace_id: str | None = None,
        run_id: str | None = None,
        thread_id: str | None = None,
        max_steps: int = 32,
    ) -> AgentState:
        state = self.prepare(
            query=query,
            request_id=request_id,
            trace_id=trace_id,
            run_id=run_id,
            thread_id=thread_id,
            max_steps=max_steps,
        )

        if state.status in {
            "completed",
            "refused",
            "failed",
        }:
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
            self._require_execution_services()

            assert (
                self.plan_executor
                is not None
            )

            assert self.verifier is not None

            assert (
                self.answer_generator
                is not None
            )

            while (
                state.runtime_plan
                is not None
                and state.current_step
                < len(
                    state.runtime_plan
                    .plan.steps
                )
            ):
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

            state = self.verifier.verify(
                state
            )

            state = self._persist_checkpoint(
                state
            )

            state = (
                self.answer_generator
                .generate(
                    state
                )
            )

            state = self._persist_checkpoint(
                state
            )

        except Exception as exc:
            state = self._handle_failure(
                state=state,
                error=exc,
            )

            state = self._persist_checkpoint(
                state
            )

        if state.status in {
            "completed",
            "refused",
            "failed",
        }:
            self._save_trajectory(
                state
            )

        return state

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
                    state
                    .checkpoint_revision
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
                    state
                    .checkpoint_revision
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
                parsed_query=(
                    parsed_query
                ),
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
                    state
                    .checkpoint_revision
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

    # 将执行异常转换为可审计的终止状态。
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
                or "Calculation 领域执行失败"
                in message
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
            if any(
                trace.status == "timed_out"
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
                    state
                    .checkpoint_revision
                ),
            )
        )

        return record.state

    def _save_trajectory(
        self,
        state: AgentState,
    ) -> None:
        if self.trajectory_store is None:
            return

        trajectory = (
            self._build_trajectory(
                state
            )
        )

        self.trajectory_store.save(
            trajectory
        )

    @staticmethod
    def _build_trajectory(
        state: AgentState,
    ) -> AgentTrajectory:
        if (
            state.status
            not in {
                "completed",
                "refused",
                "failed",
            }
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

        payload.update(updates)

        return AgentState.model_validate(
            payload
        )