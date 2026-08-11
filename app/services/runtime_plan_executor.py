from __future__ import annotations

from collections.abc import (
    Collection,
    Mapping,
)
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
    AgentState,
    NodeSpan,
    RuntimePlan,
)
from app.schemas.complex_plan_eval_result import (
    ComplexCalculationTrace,
    ComplexPlanStepOutput,
    ComplexRetrievalQueryOutput,
    ComplexRetrievalTrace,
)
from app.schemas.tool_registry import (
    DocumentEvidenceQuery,
    ExecuteCalculationInput,
    ExecuteCalculationOutput,
    QueryFinancialDataInput,
    QueryFinancialDataOutput,
    RetrievedDocument,
    RetrieveDocumentsInput,
    RetrieveDocumentsOutput,
    ToolExecutionResult,
    ToolPermission,
)


class RuntimePlanExecutorError(
    ValueError
):
    """RuntimePlan 执行阶段基础异常。"""


# ============================================================
# RuntimePlanExecutor 不直接依赖具体 ToolExecutor。
#
# 它只要求：
#
#     execute(...)
#
# Production:
#     ToolExecutor
#
# Test:
#     Fake / Stub Executor
#
# ============================================================


class ToolExecutorProvider(
    Protocol
):
    def execute(
        self,
        *,
        tool_name: str,
        arguments: Mapping[
            str,
            object,
        ],
        request_id: str,
        run_id: str,
        step_id: str,
        granted_permissions: Collection[
            ToolPermission
        ],
    ) -> ToolExecutionResult:
        """执行一次受控 Tool Call。"""


class PlanExecutorClock(
    Protocol
):
    def now(
        self,
    ) -> datetime:
        """返回带时区的当前时间。"""


class PlanExecutorIdFactory(
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
class UTCPlanExecutorClock:
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
class UUIDPlanExecutorIdFactory:
    def new_id(
        self,
        prefix: str,
    ) -> str:
        return (
            f"{prefix}_{uuid4().hex}"
        )


# ============================================================
# 保持原顺序去重。
#
# Runtime 中不要随便 set(...)，
# 因为顺序也是审计信息。
# ============================================================


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


_SUPPORTED_TOOL_NAMES = {
    "query_financial_data",
    "execute_calculation",
    "retrieve_documents",
}


# ============================================================
# AgentRuntime:
#
#     控制整个 Agent 生命周期
#
# RuntimePlanExecutor:
#
#     真正解释并执行 RuntimePlan
#
#
# RuntimePlan
#      ↓
# current_step
#      ↓
# PlanStep
#      ↓
# tool_by_step_id
#      ↓
# ToolExecutor
#      ↓
# AgentState
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class RuntimePlanExecutor:
    tool_executor: ToolExecutorProvider

    granted_permissions: frozenset[
        ToolPermission
    ]

    financial_max_results: int = 5

    document_top_k: int = 5

    clock: PlanExecutorClock = field(
        default_factory=(
            UTCPlanExecutorClock
        )
    )

    id_factory: PlanExecutorIdFactory = (
        field(
            default_factory=(
                UUIDPlanExecutorIdFactory
            )
        )
    )

    def __post_init__(
        self,
    ) -> None:
        if not (
            1
            <= self.financial_max_results
            <= 50
        ):
            raise ValueError(
                "financial_max_results "
                "必须位于 [1, 50]"
            )

        if not (
            1
            <= self.document_top_k
            <= 50
        ):
            raise ValueError(
                "document_top_k "
                "必须位于 [1, 50]"
            )

    # ========================================================
    # 现在 8B 已经支持三个 Tool：
    #
    # query_financial_data
    # execute_calculation
    # retrieve_documents
    #
    #
    # 但仍然不会执行：
    #
    # compare
    # rank
    # synthesize
    #
    # 因为它们不是 Tool Step。
    # 它们属于下一步 8C。
    # ========================================================

    def execute_available_tool_steps(
        self,
        state: AgentState,
    ) -> AgentState:
        current_state = state

        while (
            self
            ._has_remaining_plan_step(
                current_state
            )
        ):
            step = (
                self._current_plan_step(
                    current_state
                )
            )

            tool_name = (
                self._tool_name_for_step(
                    current_state,
                    step,
                )
            )

            # =================================================
            # 没有 Tool Binding：
            #
            # compare / rank / synthesize
            #
            # 留给 8C。
            # =================================================

            if tool_name is None:
                break

            if (
                tool_name
                not in _SUPPORTED_TOOL_NAMES
            ):
                raise RuntimePlanExecutorError(
                    "RuntimePlan 绑定了"
                    "不支持执行的工具："
                    f"{tool_name}"
                )

            current_state = (
                self.execute_next_tool_step(
                    current_state
                )
            )

        return current_state

    # ========================================================
    # Runtime 最小执行粒度：
    #
    #     one Plan Step
    #
    # 而不是：
    #
    #     run whole plan
    #
    # 后面才能做到：
    #
    # Step
    # ↓
    # Checkpoint
    # ↓
    # Step
    # ↓
    # Checkpoint
    # ========================================================

    def execute_next_tool_step(
        self,
        state: AgentState,
    ) -> AgentState:
        self._validate_executable_state(
            state
        )

        step = self._current_plan_step(
            state
        )

        self._validate_dependencies(
            state=state,
            step=step,
        )

        tool_name = (
            self._tool_name_for_step(
                state,
                step,
            )
        )

        if tool_name is None:
            raise RuntimePlanExecutorError(
                f"{step.step_id} "
                "不是 Tool Step"
            )

        if (
            tool_name
            == "query_financial_data"
        ):
            return (
                self
                ._execute_financial_retrieval(
                    state=state,
                    step=step,
                )
            )

        if (
            tool_name
            == "execute_calculation"
        ):
            return (
                self
                ._execute_calculation(
                    state=state,
                    step=step,
                )
            )

        if (
            tool_name
            == "retrieve_documents"
        ):
            return (
                self
                ._execute_document_retrieval(
                    state=state,
                    step=step,
                )
            )

        raise RuntimePlanExecutorError(
            "RuntimePlan 绑定了"
            "不支持执行的工具："
            f"{tool_name}"
        )

    # ========================================================
    # Financial Retrieval
    # ========================================================

    def _execute_financial_retrieval(
        self,
        *,
        state: AgentState,
        step: ComplexPlanStepOutput,
    ) -> AgentState:
        runtime_plan = (
            self._require_runtime_plan(
                state
            )
        )

        retrieval_query_id = (
            step.retrieval_query_id
        )

        if retrieval_query_id is None:
            raise RuntimePlanExecutorError(
                "Financial Retrieval Step "
                "缺少 retrieval_query_id"
            )

        query = (
            self
            ._find_financial_query(
                runtime_plan=(
                    runtime_plan
                ),
                query_id=(
                    retrieval_query_id
                ),
            )
        )

        arguments = (
            QueryFinancialDataInput(
                query=query,
                max_results=(
                    self
                    .financial_max_results
                ),
            )
            .model_dump(
                mode="json"
            )
        )

        node_started_at = (
            self.clock.now()
        )

        timer_start = (
            perf_counter()
        )

        # ====================================================
        # Runtime 永远通过 ToolExecutor，
        # 绝不能直接：
        #
        # tool.handle(...)
        #
        # ====================================================

        execution_result = (
            self.tool_executor.execute(
                tool_name=(
                    "query_financial_data"
                ),
                arguments=arguments,
                request_id=(
                    state.request_id
                ),
                run_id=(
                    state.run_id
                ),
                step_id=(
                    step.step_id
                ),
                granted_permissions=(
                    self
                    .granted_permissions
                ),
            )
        )

        output = (
            QueryFinancialDataOutput
            .model_validate(
                execution_result.output
            )
        )

        trace = output.trace

        if (
            trace.query_id
            != retrieval_query_id
        ):
            raise RuntimePlanExecutorError(
                "Financial Retrieval Trace "
                "query_id 与 Plan 不一致"
            )

        if trace.status != "completed":
            raise RuntimePlanExecutorError(
                "Financial Retrieval "
                "领域执行失败："
                f"{trace.error_message}"
            )

        if not trace.retrieved_fact_ids:
            raise RuntimePlanExecutorError(
                "Financial Retrieval "
                "没有返回可用 fact_id"
            )

        node_completed_at = (
            self.clock.now()
        )

        return self._commit_tool_step(
            state=state,
            step=step,
            tool_name=(
                "query_financial_data"
            ),
            execution_result=(
                execution_result
            ),
            output_ref_values=(
                trace.retrieved_fact_ids
            ),
            node_started_at=(
                node_started_at
            ),
            node_completed_at=(
                node_completed_at
            ),
            timer_start=(
                timer_start
            ),
            input_summary={
                "query_id": (
                    retrieval_query_id
                ),
            },
            output_summary={
                "fact_count": len(
                    trace.retrieved_fact_ids
                ),
                "evidence_count": len(
                    trace
                    .retrieved_evidence_ids
                ),
                "chunk_count": len(
                    trace
                    .retrieved_chunk_ids
                ),
            },
            retrieval_trace=trace,
            resolved_fact_ids=(
                trace.retrieved_fact_ids
            ),
            evidence_ids=(
                trace
                .retrieved_evidence_ids
            ),
        )

    # ========================================================
    # 8B-2
    #
    # Calculation Execution
    # ========================================================

    def _execute_calculation(
        self,
        *,
        state: AgentState,
        step: ComplexPlanStepOutput,
    ) -> AgentState:
        if (
            step.calculation_id is None
            or step.formula_id is None
        ):
            raise RuntimePlanExecutorError(
                "Calculation Step "
                "缺少 calculation_id "
                "或 formula_id"
            )

        # ====================================================
        # Planner 给的是：
        #
        # input_refs = (
        #     "retrieval_result_q1",
        #     "retrieval_result_q2",
        # )
        #
        # Calculator 需要的是：
        #
        # input_fact_ids = (
        #     "fact_xxx_revenue",
        #     "fact_xxx_cost",
        # )
        #
        #
        # runtime_refs 就负责完成：
        #
        # logical reference
        #       ↓
        # physical runtime ID
        # ====================================================

        input_fact_ids: list[
            str
        ] = []

        for input_ref in (
            step.input_refs
        ):
            resolved_ids = (
                state.runtime_refs.get(
                    input_ref
                )
            )

            if resolved_ids is None:
                raise RuntimePlanExecutorError(
                    "Calculation 输入引用"
                    "尚未解析："
                    f"{input_ref}"
                )

            # =================================================
            # Calculation 一个输入槽位
            # 必须唯一对应一个 FinancialFact。
            #
            # 如果一个 Query 返回两个 Fact：
            #
            # Runtime 不能偷偷选第一个。
            #
            # 必须把“歧义”暴露出来。
            # =================================================

            if len(resolved_ids) != 1:
                raise RuntimePlanExecutorError(
                    "Calculation 每个 input_ref "
                    "必须唯一解析为一个 fact_id："
                    f"{input_ref} -> "
                    f"{resolved_ids}"
                )

            fact_id = resolved_ids[0]

            if not fact_id.startswith(
                "fact_"
            ):
                raise RuntimePlanExecutorError(
                    "Calculation input_ref "
                    "必须解析为 fact_id："
                    f"{input_ref} -> {fact_id}"
                )

            # =================================================
            # 不要排序！
            #
            # step.input_refs 的顺序
            # =
            # Formula 参数顺序
            # =================================================

            input_fact_ids.append(
                fact_id
            )

        calculation_input = (
            ExecuteCalculationInput(
                calculation_id=(
                    step.calculation_id
                ),
                formula_id=(
                    step.formula_id
                ),
                input_fact_ids=tuple(
                    input_fact_ids
                ),
            )
        )

        arguments = (
            calculation_input
            .model_dump(
                mode="json"
            )
        )

        node_started_at = (
            self.clock.now()
        )

        timer_start = (
            perf_counter()
        )

        execution_result = (
            self.tool_executor.execute(
                tool_name=(
                    "execute_calculation"
                ),
                arguments=arguments,
                request_id=(
                    state.request_id
                ),
                run_id=(
                    state.run_id
                ),
                step_id=(
                    step.step_id
                ),
                granted_permissions=(
                    self
                    .granted_permissions
                ),
            )
        )

        output = (
            ExecuteCalculationOutput
            .model_validate(
                execution_result.output
            )
        )

        trace = output.trace

        # ====================================================
        # Defense in depth：
        #
        # Tool 本身已经检查一次，
        # Runtime 边界仍然再确认。
        # ====================================================

        if (
            trace.calculation_id
            != step.calculation_id
        ):
            raise RuntimePlanExecutorError(
                "Calculation Trace "
                "calculation_id 与 Plan 不一致"
            )

        if (
            trace.formula_id
            != step.formula_id
        ):
            raise RuntimePlanExecutorError(
                "Calculation Trace "
                "formula_id 与 Plan 不一致"
            )

        if (
            trace.input_fact_ids
            != tuple(input_fact_ids)
        ):
            raise RuntimePlanExecutorError(
                "Calculation Trace "
                "input_fact_ids 与 Runtime "
                "解析结果不一致"
            )

        # ====================================================
        # Tool 成功 != Calculation 成功
        #
        # ToolExecutor 可以成功返回：
        #
        # ComplexCalculationTrace(
        #     status="failed",
        # )
        #
        # 这是领域失败，不是 Tool Infrastructure Crash。
        #
        # 8B 先显式报错。
        # Step 10 会把它接入 Failure Recovery。
        # ====================================================

        if trace.status != "completed":
            raise RuntimePlanExecutorError(
                "Calculation 领域执行失败："
                f"{trace.error_message}"
            )

        node_completed_at = (
            self.clock.now()
        )

        return self._commit_tool_step(
            state=state,
            step=step,
            tool_name=(
                "execute_calculation"
            ),
            execution_result=(
                execution_result
            ),

            # =================================================
            # Calculation Output Ref：
            #
            # calculation_xxx
            #       ↓
            # calculation_xxx
            #
            # 后面 8C Compare 可以通过这个 ID
            # 找到 calculation_traces 中的结果值。
            # =================================================

            output_ref_values=(
                trace.calculation_id,
            ),
            node_started_at=(
                node_started_at
            ),
            node_completed_at=(
                node_completed_at
            ),
            timer_start=(
                timer_start
            ),
            input_summary={
                "calculation_id": (
                    trace.calculation_id
                ),
                "formula_id": (
                    trace.formula_id
                ),
                "input_fact_count": len(
                    trace.input_fact_ids
                ),
            },
            output_summary={
                "calculation_status": (
                    trace.status
                ),
                "result_unit": (
                    trace.result_unit
                ),
            },
            calculation_trace=(
                trace
            ),
            calculation_ids=(
                trace.calculation_id,
            ),
        )

    # ========================================================
    # 8B-3
    #
    # Document Retrieval Execution
    # ========================================================

    def _execute_document_retrieval(
        self,
        *,
        state: AgentState,
        step: ComplexPlanStepOutput,
    ) -> AgentState:
        runtime_plan = (
            self._require_runtime_plan(
                state
            )
        )

        retrieval_query_id = (
            step.retrieval_query_id
        )

        if retrieval_query_id is None:
            raise RuntimePlanExecutorError(
                "Document Retrieval Step "
                "缺少 retrieval_query_id"
            )

        query = (
            self
            ._find_document_query(
                runtime_plan=(
                    runtime_plan
                ),
                query_id=(
                    retrieval_query_id
                ),
            )
        )

        document_input = (
            RetrieveDocumentsInput(
                query=query,
                top_k=(
                    self.document_top_k
                ),
            )
        )

        arguments = (
            document_input
            .model_dump(
                mode="json"
            )
        )

        node_started_at = (
            self.clock.now()
        )

        timer_start = (
            perf_counter()
        )

        execution_result = (
            self.tool_executor.execute(
                tool_name=(
                    "retrieve_documents"
                ),
                arguments=arguments,
                request_id=(
                    state.request_id
                ),
                run_id=(
                    state.run_id
                ),
                step_id=(
                    step.step_id
                ),
                granted_permissions=(
                    self
                    .granted_permissions
                ),
            )
        )

        output = (
            RetrieveDocumentsOutput
            .model_validate(
                execution_result.output
            )
        )

        if (
            output.query_id
            != retrieval_query_id
        ):
            raise RuntimePlanExecutorError(
                "Document Retrieval Output "
                "query_id 与 Plan 不一致"
            )

        # ====================================================
        # Document Tool 的 Schema 允许：
        #
        # documents=()
        #
        # 因为“搜不到”本身是合法 Tool 输出。
        #
        # 但 Runtime 不能把：
        #
        # zero documents
        #
        # 当成：
        #
        # successful evidence
        #
        # ====================================================

        if not output.documents:
            raise RuntimePlanExecutorError(
                "Document Retrieval "
                "没有返回可用文档"
            )

        chunk_ids = tuple(
            document.chunk_id
            for document
            in output.documents
        )

        node_completed_at = (
            self.clock.now()
        )

        # ====================================================
        # Financial Retrieval：
        #
        # output_ref
        #    ↓
        # fact_ids
        #
        #
        # Document Retrieval：
        #
        # output_ref
        #    ↓
        # chunk_ids
        #
        #
        # Calculation：
        #
        # output_ref
        #    ↓
        # calculation_id
        #
        #
        # runtime_refs 是“统一引用表”，
        # 不要求所有 Value 都是同一种 ID。
        # ====================================================

        return self._commit_tool_step(
            state=state,
            step=step,
            tool_name=(
                "retrieve_documents"
            ),
            execution_result=(
                execution_result
            ),
            output_ref_values=(
                chunk_ids
            ),
            node_started_at=(
                node_started_at
            ),
            node_completed_at=(
                node_completed_at
            ),
            timer_start=(
                timer_start
            ),
            input_summary={
                "query_id": (
                    retrieval_query_id
                ),
            },
            output_summary={
                "document_count": len(
                    output.documents
                ),
                "chunk_count": len(
                    chunk_ids
                ),
            },
            retrieved_documents=(
                output.documents
            ),
        )

    # ========================================================
    # 三类 Tool 最后都会做一模一样的一批事情：
    #
    # 1. 写 runtime_refs
    # 2. current_step + 1
    # 3. completed_step_ids
    # 4. tool_results
    # 5. tool_call_traces
    # 6. NodeSpan
    # 7. 更新 State
    #
    # 所以这些公共行为集中在一个地方，
    # 避免三份复制代码。
    # ========================================================

    def _commit_tool_step(
        self,
        *,
        state: AgentState,
        step: ComplexPlanStepOutput,
        tool_name: str,
        execution_result: ToolExecutionResult,
        output_ref_values: tuple[
            str,
            ...
        ],
        node_started_at: datetime,
        node_completed_at: datetime,
        timer_start: float,
        input_summary: dict[
            str,
            object,
        ],
        output_summary: dict[
            str,
            object,
        ],
        retrieval_trace: (
            ComplexRetrievalTrace
            | None
        ) = None,
        calculation_trace: (
            ComplexCalculationTrace
            | None
        ) = None,
        retrieved_documents: tuple[
            RetrievedDocument,
            ...
        ] = (),
        resolved_fact_ids: tuple[
            str,
            ...
        ] = (),
        evidence_ids: tuple[
            str,
            ...
        ] = (),
        calculation_ids: tuple[
            str,
            ...
        ] = (),
    ) -> AgentState:
        runtime_plan = (
            self._require_runtime_plan(
                state
            )
        )

        if not output_ref_values:
            raise RuntimePlanExecutorError(
                "Tool Step 必须产生"
                "至少一个 Runtime Reference"
            )

        output_ref_values = (
            _unique_in_order(
                output_ref_values
            )
        )

        runtime_refs = dict(
            state.runtime_refs
        )

        if (
            step.output_ref
            in runtime_refs
        ):
            raise RuntimePlanExecutorError(
                "Runtime output_ref "
                "已经存在，不能覆盖："
                f"{step.output_ref}"
            )

        runtime_refs[
            step.output_ref
        ] = output_ref_values

        next_current_step = (
            state.current_step + 1
        )

        all_plan_steps_finished = (
            next_current_step
            >= len(
                runtime_plan.plan.steps
            )
        )

        next_node = (
            "verify_evidence"
            if all_plan_steps_finished
            else "execute_plan"
        )

        merged_output_summary = {
            **output_summary,
            "tool_reused": (
                execution_result.reused
            ),
            "tool_attempt_count": len(
                execution_result.traces
            ),
        }

        span = NodeSpan(
            span_id=(
                self.id_factory.new_id(
                    "span"
                )
            ),
            node_name="execute_plan",
            attempt=1,
            status="completed",
            input_summary={
                "step_id": (
                    step.step_id
                ),
                "action": (
                    step.action
                ),
                "tool_name": (
                    tool_name
                ),
                **input_summary,
            },
            output_summary=(
                merged_output_summary
            ),
            started_at=(
                node_started_at
            ),
            completed_at=(
                node_completed_at
            ),
            latency_ms=max(
                (
                    perf_counter()
                    - timer_start
                )
                * 1000.0,
                0.0,
            ),
            checkpoint_revision=(
                state
                .checkpoint_revision
            ),
            error_type=None,
            error_message=None,
        )

        new_retrieval_traces = (
            state.retrieval_traces
        )

        if retrieval_trace is not None:
            new_retrieval_traces = (
                new_retrieval_traces
                + (
                    retrieval_trace,
                )
            )

        new_calculation_traces = (
            state.calculation_traces
        )

        if (
            calculation_trace
            is not None
        ):
            new_calculation_traces = (
                new_calculation_traces
                + (
                    calculation_trace,
                )
            )

        # ====================================================
        # 所有变化一次 model_validate。
        #
        # 继续保持 8A 的：
        #
        # Atomic State Transition
        # ====================================================

        return self._replace_state(
            state,
            status="executing",
            current_step=(
                next_current_step
            ),
            completed_step_ids=(
                state.completed_step_ids
                + (
                    step.step_id,
                )
            ),
            runtime_refs=(
                runtime_refs
            ),
            tool_results=(
                state.tool_results
                + (
                    execution_result,
                )
            ),
            tool_call_traces=(
                state.tool_call_traces
                + execution_result.traces
            ),
            retrieval_traces=(
                new_retrieval_traces
            ),
            calculation_traces=(
                new_calculation_traces
            ),
            retrieved_documents=(
                state.retrieved_documents
                + retrieved_documents
            ),
            resolved_fact_ids=(
                _unique_in_order(
                    state.resolved_fact_ids
                    + resolved_fact_ids
                )
            ),
            evidence_ids=(
                _unique_in_order(
                    state.evidence_ids
                    + evidence_ids
                )
            ),
            calculation_ids=(
                _unique_in_order(
                    state.calculation_ids
                    + calculation_ids
                )
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
            current_node=(
                "execute_plan"
            ),
            next_node=(
                next_node
            ),
            updated_at=(
                node_completed_at
            ),
        )

    # ========================================================
    # Query Lookup
    # ========================================================

    @staticmethod
    def _find_financial_query(
        *,
        runtime_plan: RuntimePlan,
        query_id: str,
    ) -> ComplexRetrievalQueryOutput:
        for query in (
            runtime_plan
            .financial_queries
        ):
            if query.query_id == query_id:
                return query

        raise RuntimePlanExecutorError(
            "找不到 Financial Query："
            f"{query_id}"
        )

    @staticmethod
    def _find_document_query(
        *,
        runtime_plan: RuntimePlan,
        query_id: str,
    ) -> DocumentEvidenceQuery:
        for query in (
            runtime_plan
            .document_queries
        ):
            if query.query_id == query_id:
                return query

        raise RuntimePlanExecutorError(
            "找不到 Document Query："
            f"{query_id}"
        )

    # ========================================================
    # Guards
    # ========================================================

    @staticmethod
    def _validate_executable_state(
        state: AgentState,
    ) -> None:
        if (
            state.status
            not in {
                "planned",
                "executing",
            }
        ):
            raise RuntimePlanExecutorError(
                "只有 planned / executing "
                "状态可以执行 RuntimePlan，"
                f"当前状态：{state.status}"
            )

        if state.runtime_plan is None:
            raise RuntimePlanExecutorError(
                "AgentState 缺少 runtime_plan"
            )

        if (
            state.current_step
            >= len(
                state
                .runtime_plan
                .plan
                .steps
            )
        ):
            raise RuntimePlanExecutorError(
                "RuntimePlan 已经没有"
                "剩余步骤"
            )

        if (
            state.step_count
            >= state.max_steps
        ):
            raise RuntimePlanExecutorError(
                "Runtime 已达到 max_steps"
            )

    @staticmethod
    def _validate_dependencies(
        *,
        state: AgentState,
        step: ComplexPlanStepOutput,
    ) -> None:
        missing_dependencies = (
            set(step.depends_on)
            - set(
                state.completed_step_ids
            )
        )

        if missing_dependencies:
            raise RuntimePlanExecutorError(
                f"{step.step_id} "
                "存在尚未完成的依赖："
                f"{sorted(missing_dependencies)}"
            )

        if (
            step.step_id
            in state.completed_step_ids
        ):
            raise RuntimePlanExecutorError(
                "当前 Plan Step "
                "已经完成，不能重复提交："
                f"{step.step_id}"
            )

    @staticmethod
    def _require_runtime_plan(
        state: AgentState,
    ) -> RuntimePlan:
        runtime_plan = (
            state.runtime_plan
        )

        if runtime_plan is None:
            raise RuntimePlanExecutorError(
                "AgentState 缺少 runtime_plan"
            )

        return runtime_plan

    @staticmethod
    def _has_remaining_plan_step(
        state: AgentState,
    ) -> bool:
        runtime_plan = (
            state.runtime_plan
        )

        if runtime_plan is None:
            return False

        return (
            state.current_step
            < len(
                runtime_plan.plan.steps
            )
        )

    def _current_plan_step(
        self,
        state: AgentState,
    ) -> ComplexPlanStepOutput:
        runtime_plan = (
            self._require_runtime_plan(
                state
            )
        )

        if (
            state.current_step
            >= len(
                runtime_plan.plan.steps
            )
        ):
            raise RuntimePlanExecutorError(
                "RuntimePlan 已经没有"
                "剩余步骤"
            )

        return (
            runtime_plan
            .plan
            .steps[
                state.current_step
            ]
        )

    @staticmethod
    def _tool_name_for_step(
        state: AgentState,
        step: ComplexPlanStepOutput,
    ) -> str | None:
        runtime_plan = (
            state.runtime_plan
        )

        if runtime_plan is None:
            return None

        return (
            runtime_plan
            .tool_by_step_id
            .get(
                step.step_id
            )
        )

    # ========================================================
    # Atomic AgentState Replacement
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