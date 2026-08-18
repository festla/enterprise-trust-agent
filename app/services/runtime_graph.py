from __future__ import annotations

from typing import (
    Any,
    TypedDict,
)

from langgraph.checkpoint.memory import (
    InMemorySaver,
)
from langgraph.graph import (
    END,
    START,
    StateGraph,
)
from langgraph.types import (
    interrupt,
)

from app.schemas.agent_runtime import (
    AgentState,
    HumanReviewDecision,
    NodeSpan
)
from app.services.agent_runtime import (
    AgentRuntime,
    AgentRuntimeError,
)
from app.services.trajectory_store import (
    TrajectoryAlreadyExistsError,
)

"""
                    用户问题
                       │
                       ▼
           create_runtime_graph_input()
                       │
                       ▼
                AgentState
                       │
                _dump_state()
                       │
                       ▼
              RuntimeGraphState
                       │
                       ▼
                LangGraph 开始
                       │
              START → parse_query
                       │
                       ▼
                  route_intent
                  /     |      \
                 /      |       \
        unsupported   缺字段     正常
            │           │         │
            ▼           ▼         ▼
          finish   await_human  create_plan
                        │           │
                  人工补充问题       ▼
                        │       execute_plan
                        └──→parse     │
                                  可能循环
                                     │
                                     ▼
                              verify_evidence
                                     │
                                     ▼
                              generate_answer
                                     │
                                     ▼
                                   finish
                                     │
                                     ▼
                                    END
"""

class RuntimeGraphState(
    TypedDict
):
    """LangGraph 中传递的最小状态包装。"""

    agent_state: dict[
        str,
        Any,
    ]

_TERMINAL_STATUSES = {
    "completed",
    "refused",
    "failed",
}

def create_runtime_graph_input(
    runtime: AgentRuntime,
    *,
    query: str,
    request_id: str | None = None,
    trace_id: str | None = None,
    run_id: str | None = None,
    thread_id: str | None = None,
    max_steps: int = 32,
) -> RuntimeGraphState:
    """创建一次 LangGraph Runtime 的初始输入。"""

    state = runtime.create_state(
        query=query,
        request_id=request_id,
        trace_id=trace_id,
        run_id=run_id,
        thread_id=thread_id,
        max_steps=max_steps,
    )

    # 如果 Runtime 配置了自己的 CheckpointStore，
    # 同时保存领域 AgentState。
    state = runtime._persist_checkpoint(
        state
    )

    return _dump_state(
        state
    )


def runtime_graph_config(
    thread_id: str,
) -> dict[
    str,
    dict[str, str],
]:
    """生成 LangGraph Checkpointer 所需配置。"""

    return {
        "configurable": {
            "thread_id": (
                thread_id
            ),
        }
    }

def extract_agent_state(
    graph_result: dict[
        str,
        Any,
    ],
) -> AgentState:
    """从 LangGraph 输出恢复严格 AgentState。"""

    raw_state = graph_result.get(
        "agent_state"
    )

    if not isinstance(
        raw_state,
        dict,
    ):
        raise AgentRuntimeError(
            "LangGraph 输出缺少 "
            "agent_state"
        )

    return AgentState.model_validate(
        raw_state
    )

def _load_state(
    graph_state: RuntimeGraphState,
) -> AgentState:
    return AgentState.model_validate(
        graph_state[
            "agent_state"
        ]
    )

def _dump_state(
    state: AgentState,
) -> RuntimeGraphState:
    return {
        "agent_state": (
            state.model_dump(
                mode="json"
            )
        )
    }

def _persist_state(
    runtime: AgentRuntime,
    state: AgentState,
) -> AgentState:
    return runtime._persist_checkpoint(
        state
    )

def _handle_node_error(
    *,
    runtime: AgentRuntime,
    state: AgentState,
    error: Exception,
) -> AgentState:
    """把 Graph Node 异常交回 Runtime 统一处理。"""

    failed_state = (
        runtime._handle_failure(
            state=state,
            error=error,
        )
    )

    return _persist_state(
        runtime,
        failed_state,
    )

def _is_terminal(
    state: AgentState,
) -> bool:
    return (
        state.status
        in _TERMINAL_STATUSES
    )

def _parse_human_decision(
    *,
    value: object,
    runtime: AgentRuntime,
) -> HumanReviewDecision:
    """把 interrupt 恢复值转换为严格人工决策。"""

    if not isinstance(
        value,
        dict,
    ):
        raise AgentRuntimeError(
            "Human Resume 必须是对象"
        )

    approved = value.get(
        "approved"
    )

    if not isinstance(
        approved,
        bool,
    ):
        raise AgentRuntimeError(
            "Human Resume 缺少 "
            "approved: bool"
        )

    corrected_query = value.get(
        "corrected_query"
    )

    if (
        corrected_query is not None
        and not isinstance(
            corrected_query,
            str,
        )
    ):
        raise AgentRuntimeError(
            "corrected_query 必须是字符串"
        )

    if (
        approved
        and (
            corrected_query is None
            or not corrected_query.strip()
        )
    ):
        raise AgentRuntimeError(
            "批准继续执行时必须提供 "
            "corrected_query"
        )

    reason = value.get(
        "reason",
        (
            "用户补充了 Runtime "
            "所需信息"
            if approved
            else "用户拒绝继续执行"
        ),
    )

    if (
        not isinstance(reason, str)
        or not reason.strip()
    ):
        raise AgentRuntimeError(
            "Human Resume reason "
            "必须是非空字符串"
        )

    reviewer_id = value.get(
        "reviewer_id",
        "human",
    )

    if (
        not isinstance(
            reviewer_id,
            str,
        )
        or not reviewer_id.strip()
    ):
        raise AgentRuntimeError(
            "reviewer_id 必须是"
            "非空字符串"
        )

    return HumanReviewDecision(
        approved=approved,
        corrected_query=(
            corrected_query
        ),
        reason=reason,
        reviewer_id=(
            reviewer_id
        ),
        decided_at=(
            runtime.clock.now()
        ),
    )


def _apply_human_decision(
    *,
    runtime: AgentRuntime,
    state: AgentState,
    decision: HumanReviewDecision,
) -> AgentState:
    """根据人工决策拒绝运行或重新进入 Parser。"""

    now = runtime.clock.now()

    if (
        state.step_count
        >= state.max_steps
    ):
        raise AgentRuntimeError(
            "Runtime 已达到 max_steps"
        )

    span = NodeSpan(
        span_id=(
            runtime.id_factory.new_id(
                "span"
            )
        ),
        node_name="await_human",
        attempt=1,
        status="completed",
        input_summary={
            "reason": (
                state.human_review_reason
            ),
        },
        output_summary={
            "approved": (
                decision.approved
            ),
            "has_corrected_query": (
                decision.corrected_query
                is not None
            ),
            "reviewer_id": (
                decision.reviewer_id
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

    payload = state.model_dump(
        mode="python"
    )

    if not decision.approved:
        payload.update(
            {
                "status": "refused",
                "stop_reason": (
                    "human_rejected"
                ),
                "pending_human_review": (
                    False
                ),
                "human_review_reason": None,
                "human_decision": (
                    decision
                ),
                "current_node": (
                    "await_human"
                ),
                "next_node": "finish",
                "node_spans": (
                    state.node_spans
                    + (
                        span,
                    )
                ),
                "step_count": (
                    state.step_count
                    + 1
                ),
                "completed_at": now,
                "updated_at": now,
            }
        )

        return AgentState.model_validate(
            payload
        )

    corrected_query = (
        decision.corrected_query
    )

    if corrected_query is None:
        raise AgentRuntimeError(
            "批准继续执行时缺少 "
            "corrected_query"
        )

    # 人工补充信息后重新进入 Parser。
    # 运行 ID、thread_id、历史 NodeSpan 和
    # checkpoint revision 均保持不变。
    payload.update(
        {
            "query": corrected_query,
            "parsed_query": None,
            "intent": None,
            "company_ids": (),
            "report_ids": (),
            "years": (),
            "metric_ids": (),
            "runtime_plan": None,
            "current_step": 0,
            "runtime_refs": {},
            "completed_step_ids": (),
            "tool_results": (),
            "tool_call_traces": (),
            "retrieval_traces": (),
            "calculation_traces": (),
            "retrieved_documents": (),
            "resolved_fact_ids": (),
            "evidence_ids": (),
            "calculation_ids": (),
            "answer_draft": None,
            "verification_report": None,
            "risk_level": None,
            "policy_decision": None,   
            "answer": None,
            "citations": (),
            "status": "created",
            "stop_reason": None,
            "pending_human_review": False,
            "human_review_reason": None,
            "human_decision": decision,
            "current_node": (
                "parse_query"
            ),
            "next_node": (
                "parse_query"
            ),
            "planner_version": None,
            "retriever_version": None,
            "calculator_version": None,
            "generator_version": None,
            "prompt_version": None,
            "prompt_sha256": None,
            "model_name": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost": 0.0,
            "node_spans": (
                state.node_spans
                + (
                    span,
                )
            ),
            "step_count": (
                state.step_count
                + 1
            ),
            "completed_at": None,
            "updated_at": now,
        }
    )

    return AgentState.model_validate(
        payload
    )


def build_agent_runtime_graph(
    runtime: AgentRuntime,
    *,
    checkpointer: Any | None = None,
):
    """把 framework-independent Runtime 包装成 LangGraph。"""

    if checkpointer is None:
        # 默认 Checkpointer 适用于本地学习、
        # 单测与 Demo。
        checkpointer = (
            InMemorySaver()
        )

    builder = StateGraph(
        RuntimeGraphState
    )

    def parse_query_node(
        graph_state: RuntimeGraphState,
    ) -> RuntimeGraphState:
        state = _load_state(
            graph_state
        )

        try:
            state = (
                runtime._run_parse_node(
                    state
                )
            )

            state = _persist_state(
                runtime,
                state,
            )

        except Exception as exc:
            state = (
                _handle_node_error(
                    runtime=runtime,
                    state=state,
                    error=exc,
                )
            )

        return _dump_state(
            state
        )

    def route_intent_node(
        graph_state: RuntimeGraphState,
    ) -> RuntimeGraphState:
        state = _load_state(
            graph_state
        )

        try:
            state = (
                runtime._run_route_node(
                    state
                )
            )

            if (
                state.intent
                == "unsupported"
            ):
                state = (
                    runtime._mark_refused(
                        state
                    )
                )

            else:
                parsed_query = (
                    runtime
                    ._require_parsed_query(
                        state
                    )
                )

                missing_fields = (
                    runtime
                    ._blocking_missing_fields(
                        parsed_query
                    )
                )

                if missing_fields:
                    state = (
                        runtime
                        ._mark_awaiting_human(
                            state,
                            blocking_missing_fields=(
                                missing_fields
                            ),
                        )
                    )

            state = _persist_state(
                runtime,
                state,
            )

        except Exception as exc:
            state = (
                _handle_node_error(
                    runtime=runtime,
                    state=state,
                    error=exc,
                )
            )

        return _dump_state(
            state
        )

    def create_plan_node(
        graph_state: RuntimeGraphState,
    ) -> RuntimeGraphState:
        state = _load_state(
            graph_state
        )

        try:
            state = (
                runtime._run_plan_node(
                    state
                )
            )

            state = _persist_state(
                runtime,
                state,
            )

        except Exception as exc:
            state = (
                _handle_node_error(
                    runtime=runtime,
                    state=state,
                    error=exc,
                )
            )

        return _dump_state(
            state
        )

    def execute_plan_node(
        graph_state: RuntimeGraphState,
    ) -> RuntimeGraphState:
        state = _load_state(
            graph_state
        )

        try:
            if (
                runtime.plan_executor
                is None
            ):
                raise AgentRuntimeError(
                    "AgentRuntime 缺少 "
                    "plan_executor"
                )

            state = (
                runtime.plan_executor
                .execute_next_step(
                    state
                )
            )

            state = _persist_state(
                runtime,
                state,
            )

        except Exception as exc:
            state = (
                _handle_node_error(
                    runtime=runtime,
                    state=state,
                    error=exc,
                )
            )

        return _dump_state(
            state
        )

    def verify_evidence_node(
        graph_state: RuntimeGraphState,
    ) -> RuntimeGraphState:
        state = _load_state(
            graph_state
        )

        try:
            if runtime.verifier is None:
                raise AgentRuntimeError(
                    "AgentRuntime 缺少 verifier"
                )

            state = (
                runtime.verifier.verify(
                    state
                )
            )

            state = _persist_state(
                runtime,
                state,
            )

        except Exception as exc:
            state = (
                _handle_node_error(
                    runtime=runtime,
                    state=state,
                    error=exc,
                )
            )

        return _dump_state(
            state
        )

    def prepare_answer_node(
        graph_state: RuntimeGraphState,
    ) -> RuntimeGraphState:
        """把 Runtime 产物转换成可验证的 AnswerDraft。"""

        state = _load_state(
            graph_state
        )

        try:
            state = (
                runtime
                ._run_prepare_answer_node(
                    state
                )
            )

            state = _persist_state(
                runtime,
                state,
            )

        except Exception as exc:
            state = (
                _handle_node_error(
                    runtime=runtime,
                    state=state,
                    error=exc,
                )
            )

        return _dump_state(
            state
        )

    def verify_answer_node(
        graph_state: RuntimeGraphState,
    ) -> RuntimeGraphState:
        """执行 AnswerDraft 的可信校验。"""

        state = _load_state(
            graph_state
        )

        try:
            state = (
                runtime
                ._run_verify_answer_node(
                    state
                )
            )

            state = _persist_state(
                runtime,
                state,
            )

        except Exception as exc:
            state = (
                _handle_node_error(
                    runtime=runtime,
                    state=state,
                    error=exc,
                )
            )

        return _dump_state(
            state
        )

    def evaluate_policy_node(
        graph_state: RuntimeGraphState,
    ) -> RuntimeGraphState:
        """执行 Answer Trust Verification 后的 Risk Policy。"""

        state = _load_state(
            graph_state
        )

        try:
            state = (
                runtime
                ._run_policy_node(
                    state
                )
            )

            state = _persist_state(
                runtime,
                state,
            )

        except Exception as exc:
            state = (
                _handle_node_error(
                    runtime=runtime,
                    state=state,
                    error=exc,
                )
            )

        return _dump_state(
            state
        )

    def generate_answer_node(
        graph_state: RuntimeGraphState,
    ) -> RuntimeGraphState:
        state = _load_state(
            graph_state
        )

        try:
            if (
                runtime.answer_generator
                is None
            ):
                raise AgentRuntimeError(
                    "AgentRuntime 缺少 "
                    "answer_generator"
                )

            state = (
                runtime
                .answer_generator
                .generate(
                    state
                )
            )

            state = _persist_state(
                runtime,
                state,
            )

        except Exception as exc:
            state = (
                _handle_node_error(
                    runtime=runtime,
                    state=state,
                    error=exc,
                )
            )

        return _dump_state(
            state
        )

    def await_human_node(
        graph_state: RuntimeGraphState,
    ) -> RuntimeGraphState:
        state = _load_state(
            graph_state
        )

        # interrupt 必须在副作用之前调用。
        # 恢复时该 Node 会从头重新执行。
        resume_value = interrupt(
            {
                "type": (
                    "missing_required_fields"
                ),
                "query": state.query,
                "reason": (
                    state.human_review_reason
                ),
                "instruction": (
                    "请补充完整问题后继续，"
                    "或拒绝本次执行。"
                ),
            }
        )

        try:
            decision = (
                _parse_human_decision(
                    value=resume_value,
                    runtime=runtime,
                )
            )

            state = (
                _apply_human_decision(
                    runtime=runtime,
                    state=state,
                    decision=decision,
                )
            )

            state = _persist_state(
                runtime,
                state,
            )

        except Exception as exc:
            state = (
                _handle_node_error(
                    runtime=runtime,
                    state=state,
                    error=exc,
                )
            )

        return _dump_state(
            state
        )

    def finish_node(
        graph_state: RuntimeGraphState,
    ) -> RuntimeGraphState:
        state = _load_state(
            graph_state
        )

        if (
            state.status
            in _TERMINAL_STATUSES
        ):
            try:
                runtime._save_trajectory(
                    state
                )

            # finish Node 必须保持幂等。
            except (
                TrajectoryAlreadyExistsError
            ):
                pass

        return _dump_state(
            state
        )

    def after_parse(
        graph_state: RuntimeGraphState,
    ) -> str:
        state = _load_state(
            graph_state
        )

        if _is_terminal(state):
            return "finish"

        return "route_intent"

    def after_route(
        graph_state: RuntimeGraphState,
    ) -> str:
        state = _load_state(
            graph_state
        )

        if _is_terminal(state):
            return "finish"

        if (
            state.status
            == "awaiting_human"
        ):
            return "await_human"

        return "create_plan"

    def after_plan(
        graph_state: RuntimeGraphState,
    ) -> str:
        state = _load_state(
            graph_state
        )

        if _is_terminal(state):
            return "finish"

        return "execute_plan"

    def after_execute(
        graph_state: RuntimeGraphState,
    ) -> str:
        state = _load_state(
            graph_state
        )

        if _is_terminal(state):
            return "finish"

        if (
            state.next_node
            == "verify_evidence"
        ):
            return "verify_evidence"

        return "execute_plan"

    def after_verify(
        graph_state: RuntimeGraphState,
    ) -> str:
        state = _load_state(
            graph_state
        )

        if _is_terminal(state):
            return "finish"

        return "prepare_answer"

    def after_prepare_answer(
        graph_state: RuntimeGraphState,
    ) -> str:
        state = _load_state(
            graph_state
        )

        if _is_terminal(
            state
        ):
            return "finish"

        return "verify_answer"

    def after_verify_answer(
        graph_state: RuntimeGraphState,
    ) -> str:
        state = _load_state(
            graph_state
        )

        # Verification 不通过时，
        # _run_verify_answer_node 已经把
        # State 变成 refused。
        if _is_terminal(
            state
        ):
            return "finish"

        return "evaluate_policy"

    def after_policy(
        graph_state: RuntimeGraphState,
    ) -> str:
        state = _load_state(
            graph_state
        )

        if _is_terminal(
            state
        ):
            return "finish"

        if (
            state.status
            == "awaiting_human"
        ):
            return "await_human"

        return "generate_answer"

    def after_generate(
        graph_state: RuntimeGraphState,
    ) -> str:
        return "finish"

    def after_human(
        graph_state: RuntimeGraphState,
    ) -> str:
        state = _load_state(
            graph_state
        )

        if _is_terminal(state):
            return "finish"

        return "parse_query"

    builder.add_node(
        "parse_query",
        parse_query_node,
    )

    builder.add_node(
        "route_intent",
        route_intent_node,
    )

    builder.add_node(
        "create_plan",
        create_plan_node,
    )

    builder.add_node(
        "execute_plan",
        execute_plan_node,
    )

    builder.add_node(
        "verify_evidence",
        verify_evidence_node,
    )

    builder.add_node(
        "prepare_answer",
        prepare_answer_node,
    )

    builder.add_node(
        "verify_answer",
        verify_answer_node,
    )

    builder.add_node(
        "evaluate_policy",
        evaluate_policy_node,
    )

    builder.add_node(
        "generate_answer",
        generate_answer_node,
    )

    builder.add_node(
        "await_human",
        await_human_node,
    )

    builder.add_node(
        "finish",
        finish_node,
    )

    builder.add_edge(
        START,
        "parse_query",
    )

    builder.add_conditional_edges(
        "parse_query",
        after_parse,
        {
            "route_intent": (
                "route_intent"
            ),
            "finish": "finish",
        },
    )

    builder.add_conditional_edges(
        "route_intent",
        after_route,
        {
            "create_plan": (
                "create_plan"
            ),
            "await_human": (
                "await_human"
            ),
            "finish": "finish",
        },
    )

    builder.add_conditional_edges(
        "create_plan",
        after_plan,
        {
            "execute_plan": (
                "execute_plan"
            ),
            "finish": "finish",
        },
    )

    builder.add_conditional_edges(
        "execute_plan",
        after_execute,
        {
            "execute_plan": (
                "execute_plan"
            ),
            "verify_evidence": (
                "verify_evidence"
            ),
            "finish": "finish",
        },
    )

    builder.add_conditional_edges(
        "verify_evidence",
        after_verify,
        {
            "prepare_answer": (
                "prepare_answer"
            ),
            "finish": "finish",
        },
    )

    builder.add_conditional_edges(
        "prepare_answer",
        after_prepare_answer,
        {
            "verify_answer": (
                "verify_answer"
            ),
            "finish": "finish",
        },
    )

    builder.add_conditional_edges(
        "verify_answer",
        after_verify_answer,
        {
            "evaluate_policy": (
                "evaluate_policy"
            ),
            "finish": "finish",
        },
    )

    builder.add_conditional_edges(
        "evaluate_policy",
        after_policy,
        {
            "generate_answer": (
                "generate_answer"
            ),
            "await_human": (
                "await_human"
            ),
            "finish": "finish",
        },
    )

    builder.add_conditional_edges(
        "generate_answer",
        after_generate,
        {
            "finish": "finish",
        },
    )

    builder.add_conditional_edges(
        "await_human",
        after_human,
        {
            "parse_query": (
                "parse_query"
            ),
            "finish": "finish",
        },
    )

    builder.add_edge(
        "finish",
        END,
    )

    return builder.compile(
        checkpointer=checkpointer,
        name=(
            "enterprise_trust_agent_runtime"
        ),
    )