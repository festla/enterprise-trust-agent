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
from decimal import Decimal
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
from app.services.registry import (
    RegistryBundle,
)
from app.services.runtime_access_control import (
    RuntimeAccessController,
)

class RuntimePlanExecutorError(
    ValueError
):
    """RuntimePlan 执行阶段基础异常。"""

    def __init__(
        self,
        message: str,
        *,
        execution_result: (
            ToolExecutionResult | None
        ) = None,
        calculation_trace: (
            ComplexCalculationTrace | None
        ) = None,
    ) -> None:
        super().__init__(message)

        self.execution_result = (
            execution_result
        )

        self.calculation_trace = (
            calculation_trace
        )


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
        """返回当前时间。"""


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

_INTERNAL_ACTIONS = {
    "compare",
    "rank",
    "synthesize",
}


@dataclass(
    frozen=True,
    slots=True,
)
class RuntimePlanExecutor:
    tool_executor: ToolExecutorProvider

    # ========================================================
    # Legacy Runtime Permission Snapshot
    #
    # Week7 Step4.3 开始：
    # 这个字段不再是权限判断的最终 Authority。
    #
    # 暂时保留它是为了避免一次性破坏已有
    # Runtime / Test 的构造接口。
    #
    # 真正权限来自：
    #
    # AgentState.user_role
    #       ↓
    # RuntimeAccessController
    # ========================================================

    granted_permissions: frozenset[
        ToolPermission
    ]

    access_controller: (
        RuntimeAccessController
    ) = field(
        default_factory=(
            RuntimeAccessController
        )
    )

    registry_bundle: (
        RegistryBundle | None
    ) = None

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

    def _effective_permissions_for_state(
        self,
        state: AgentState,
    ) -> frozenset[
        ToolPermission
    ]:
        """根据当前 UserRole 解析真正有效的 Tool Permission。

        Week7 Step4.3：

        不信任 RuntimePlanExecutor 构造时传入的固定权限，
        也不直接把 AgentState.granted_permissions 当成
        唯一安全边界。

        Authority：

            state.user_role
                ↓
            RuntimeAccessController
                ↓
            effective permissions

        state.granted_permissions 只作为审计快照，
        并在这里检查它是否被篡改或发生漂移。
        """

        effective_permissions = (
            self.access_controller
            .permissions_for_role(
                state.user_role
            )
        )

        recorded_permissions = (
            frozenset(
                state.granted_permissions
            )
        )

        if (
            recorded_permissions
            != effective_permissions
        ):
            raise RuntimePlanExecutorError(
                "AgentState RBAC 权限快照"
                "与 user_role 不一致："
                f"role={state.user_role}"
            )

        return effective_permissions

    # 兼容 8B：
    # 只连续执行 Tool Step，遇到非 Tool Step 停止。
    def execute_available_tool_steps(
        self,
        state: AgentState,
    ) -> AgentState:
        current_state = state

        while self._has_remaining_plan_step(
            current_state
        ):
            step = self._current_plan_step(
                current_state
            )

            tool_name = (
                self._tool_name_for_step(
                    current_state,
                    step,
                )
            )

            if tool_name is None:
                break

            current_state = (
                self.execute_next_tool_step(
                    current_state
                )
            )

        return current_state

    # Step 8 Final：
    # 连续执行 Tool Step 和 Runtime Internal Step。
    def execute_all_steps(
        self,
        state: AgentState,
    ) -> AgentState:
        current_state = state

        while self._has_remaining_plan_step(
            current_state
        ):
            current_state = (
                self.execute_next_step(
                    current_state
                )
            )

        return current_state

    def execute_next_step(
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

        if tool_name is not None:
            return self._dispatch_tool_step(
                state=state,
                step=step,
                tool_name=tool_name,
            )

        return self._execute_internal_step(
            state=state,
            step=step,
        )

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

        return self._dispatch_tool_step(
            state=state,
            step=step,
            tool_name=tool_name,
        )

    def _dispatch_tool_step(
        self,
        *,
        state: AgentState,
        step: ComplexPlanStepOutput,
        tool_name: str,
    ) -> AgentState:
        if (
            tool_name
            not in _SUPPORTED_TOOL_NAMES
        ):
            raise RuntimePlanExecutorError(
                "RuntimePlan 绑定了"
                "不支持执行的工具："
                f"{tool_name}"
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

        return (
            self
            ._execute_document_retrieval(
                state=state,
                step=step,
            )
        )

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

        query_id = (
            step.retrieval_query_id
        )

        if query_id is None:
            raise RuntimePlanExecutorError(
                "Financial Retrieval Step "
                "缺少 retrieval_query_id"
            )

        query = self._find_financial_query(
            runtime_plan=runtime_plan,
            query_id=query_id,
        )

        arguments = (
            QueryFinancialDataInput(
                query=query,
                max_results=(
                    self.financial_max_results
                ),
            )
            .model_dump(
                mode="json"
            )
        )

        started_at = self.clock.now()
        timer_start = perf_counter()

        result = (
            self.tool_executor.execute(
                tool_name=(
                    "query_financial_data"
                ),
                arguments=arguments,
                request_id=(
                    state.request_id
                ),
                run_id=state.run_id,
                step_id=step.step_id,
                granted_permissions=(
                    self
                    ._effective_permissions_for_state(
                        state
                    )
                ),
            )
        )

        output = (
            QueryFinancialDataOutput
            .model_validate(
                result.output
            )
        )

        trace = output.trace

        if trace.query_id != query_id:
            raise RuntimePlanExecutorError(
                "Financial Retrieval Trace "
                "query_id 与 Plan 不一致",
                execution_result=result,
            )

        if trace.status != "completed":
            raise RuntimePlanExecutorError(
                "Financial Retrieval "
                "领域执行失败："
                f"{trace.error_message}",
                execution_result=result,
            )

        if not trace.retrieved_fact_ids:
            raise RuntimePlanExecutorError(
                "Financial Retrieval "
                "没有返回可用 fact_id",
                execution_result=result,
            )

        completed_at = self.clock.now()

        return self._commit_tool_step(
            state=state,
            step=step,
            tool_name=(
                "query_financial_data"
            ),
            execution_result=result,
            output_ref_values=(
                trace.retrieved_fact_ids
            ),
            started_at=started_at,
            completed_at=completed_at,
            timer_start=timer_start,
            input_summary={
                "query_id": query_id,
            },
            output_summary={
                "fact_count": len(
                    trace.retrieved_fact_ids
                ),
                "evidence_count": len(
                    trace
                    .retrieved_evidence_ids
                ),
            },
            retrieval_trace=trace,
            resolved_fact_ids=(
                trace.retrieved_fact_ids
            ),
            evidence_ids=(
                trace.retrieved_evidence_ids
            ),
        )

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

        input_fact_ids: list[str] = []

        # input_refs 的顺序就是公式参数顺序，
        # 这里不能排序。
        for input_ref in step.input_refs:
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

            input_fact_ids.append(
                fact_id
            )

        arguments = (
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
            .model_dump(
                mode="json"
            )
        )

        started_at = self.clock.now()
        timer_start = perf_counter()

        result = (
            self.tool_executor.execute(
                tool_name=(
                    "execute_calculation"
                ),
                arguments=arguments,
                request_id=(
                    state.request_id
                ),
                run_id=state.run_id,
                step_id=step.step_id,
                granted_permissions=(
                    self
                    ._effective_permissions_for_state(
                        state
                    )
                ),
            )
        )

        output = (
            ExecuteCalculationOutput
            .model_validate(
                result.output
            )
        )

        trace = output.trace

        if (
            trace.calculation_id
            != step.calculation_id
        ):
            raise RuntimePlanExecutorError(
                "Calculation Trace "
                "calculation_id 与 Plan 不一致",
                execution_result=result,
                calculation_trace=trace,
            )

        if (
            trace.formula_id
            != step.formula_id
        ):
            raise RuntimePlanExecutorError(
                "Calculation Trace "
                "formula_id 与 Plan 不一致",
                execution_result=result,
                calculation_trace=trace,
            )

        if (
            trace.input_fact_ids
            != tuple(input_fact_ids)
        ):
            raise RuntimePlanExecutorError(
                "Calculation Trace "
                "input_fact_ids 与 Runtime "
                "解析结果不一致",
                execution_result=result,
                calculation_trace=trace,
            )

        if trace.status != "completed":
            raise RuntimePlanExecutorError(
                "Calculation 领域执行失败："
                f"{trace.error_message}",
                execution_result=result,
                calculation_trace=trace,
            )

        completed_at = self.clock.now()

        return self._commit_tool_step(
            state=state,
            step=step,
            tool_name=(
                "execute_calculation"
            ),
            execution_result=result,
            output_ref_values=(
                trace.calculation_id,
            ),
            started_at=started_at,
            completed_at=completed_at,
            timer_start=timer_start,
            input_summary={
                "calculation_id": (
                    trace.calculation_id
                ),
                "formula_id": (
                    trace.formula_id
                ),
            },
            output_summary={
                "result_unit": (
                    trace.result_unit
                ),
            },
            calculation_trace=trace,
            calculation_ids=(
                trace.calculation_id,
            ),
        )

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

        query_id = (
            step.retrieval_query_id
        )

        if query_id is None:
            raise RuntimePlanExecutorError(
                "Document Retrieval Step "
                "缺少 retrieval_query_id"
            )

        query = self._find_document_query(
            runtime_plan=runtime_plan,
            query_id=query_id,
        )

        arguments = (
            RetrieveDocumentsInput(
                query=query,
                top_k=self.document_top_k,
            )
            .model_dump(
                mode="json"
            )
        )

        started_at = self.clock.now()
        timer_start = perf_counter()

        result = (
            self.tool_executor.execute(
                tool_name="retrieve_documents",
                arguments=arguments,
                request_id=(
                    state.request_id
                ),
                run_id=state.run_id,
                step_id=step.step_id,
                granted_permissions=(
                    self
                    ._effective_permissions_for_state(
                        state
                    )
                ),
            )
        )

        output = (
            RetrieveDocumentsOutput
            .model_validate(
                result.output
            )
        )

        if output.query_id != query_id:
            raise RuntimePlanExecutorError(
                "Document Retrieval Output "
                "query_id 与 Plan 不一致",
                execution_result=result,
            )

        if not output.documents:
            raise RuntimePlanExecutorError(
                "Document Retrieval "
                "没有返回可用文档",
                execution_result=result,
            )

        chunk_ids = tuple(
            document.chunk_id
            for document
            in output.documents
        )

        completed_at = self.clock.now()

        return self._commit_tool_step(
            state=state,
            step=step,
            tool_name=(
                "retrieve_documents"
            ),
            execution_result=result,
            output_ref_values=chunk_ids,
            started_at=started_at,
            completed_at=completed_at,
            timer_start=timer_start,
            input_summary={
                "query_id": query_id,
            },
            output_summary={
                "document_count": len(
                    output.documents
                ),
            },
            retrieved_documents=(
                output.documents
            ),
        )

    # compare / rank / synthesize 是 Runtime 内部步骤，
    # 不调用外部 Tool。
    def _execute_internal_step(
        self,
        *,
        state: AgentState,
        step: ComplexPlanStepOutput,
    ) -> AgentState:
        if (
            step.action
            not in _INTERNAL_ACTIONS
        ):
            raise RuntimePlanExecutorError(
                "当前 Runtime 不支持"
                "非工具步骤："
                f"{step.action}"
            )

        input_artifacts = (
            self._resolve_input_artifacts(
                state=state,
                step=step,
            )
        )

        if step.action == "rank":
            output_artifacts = tuple(
                sorted(
                    input_artifacts,
                    key=lambda artifact_id: (
                        self._numeric_value(
                            state=state,
                            artifact_id=(
                                artifact_id
                            ),
                        )
                    ),
                    reverse=True,
                )
            )

        else:
            # compare 和 synthesize 都保留输入顺序。
            output_artifacts = (
                input_artifacts
            )

        return self._commit_internal_step(
            state=state,
            step=step,
            output_ref_values=(
                output_artifacts
            ),
        )

    def _resolve_input_artifacts(
        self,
        *,
        state: AgentState,
        step: ComplexPlanStepOutput,
    ) -> tuple[str, ...]:
        result: list[str] = []

        for input_ref in step.input_refs:
            values = (
                state.runtime_refs.get(
                    input_ref
                )
            )

            if values is None:
                raise RuntimePlanExecutorError(
                    "Runtime Internal Step "
                    "使用了尚未解析的引用："
                    f"{input_ref}"
                )

            result.extend(values)

        resolved = _unique_in_order(
            tuple(result)
        )

        if not resolved:
            raise RuntimePlanExecutorError(
                "Runtime Internal Step "
                "没有可用输入"
            )

        return resolved

    def _numeric_value(
        self,
        *,
        state: AgentState,
        artifact_id: str,
    ) -> Decimal:
        if artifact_id.startswith(
            "calculation_"
        ):
            for trace in (
                state.calculation_traces
            ):
                if (
                    trace.calculation_id
                    != artifact_id
                ):
                    continue

                if (
                    trace.status
                    != "completed"
                    or trace.result_value
                    is None
                ):
                    raise RuntimePlanExecutorError(
                        "Ranking 使用了"
                        "无效 Calculation："
                        f"{artifact_id}"
                    )

                return trace.result_value

            raise RuntimePlanExecutorError(
                "找不到 Ranking "
                "需要的 Calculation："
                f"{artifact_id}"
            )

        if artifact_id.startswith(
            "fact_"
        ):
            if self.registry_bundle is None:
                raise RuntimePlanExecutorError(
                    "Ranking FinancialFact "
                    "需要 registry_bundle"
                )

            fact = (
                self.registry_bundle
                .financial_facts
                .require(artifact_id)
            )

            return fact.normalized_value

        raise RuntimePlanExecutorError(
            "Ranking 只支持 "
            "FinancialFact 或 Calculation："
            f"{artifact_id}"
        )

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
        started_at: datetime,
        completed_at: datetime,
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
            ComplexRetrievalTrace | None
        ) = None,
        calculation_trace: (
            ComplexCalculationTrace | None
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

        output_ref_values = (
            _unique_in_order(
                output_ref_values
            )
        )

        if not output_ref_values:
            raise RuntimePlanExecutorError(
                "Tool Step 必须产生"
                "至少一个 Runtime Reference"
            )

        runtime_refs = dict(
            state.runtime_refs
        )

        if step.output_ref in runtime_refs:
            raise RuntimePlanExecutorError(
                "Runtime output_ref "
                "已经存在："
                f"{step.output_ref}"
            )

        runtime_refs[
            step.output_ref
        ] = output_ref_values

        next_step = (
            state.current_step + 1
        )

        next_node = (
            "verify_evidence"
            if next_step
            >= len(
                runtime_plan.plan.steps
            )
            else "execute_plan"
        )

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
                "step_id": step.step_id,
                "action": step.action,
                "tool_name": tool_name,
                **input_summary,
            },
            output_summary={
                **output_summary,
                "tool_reused": (
                    execution_result.reused
                ),
                "tool_attempt_count": len(
                    execution_result.traces
                ),
            },
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
                state.checkpoint_revision
            ),
            error_type=None,
            error_message=None,
        )

        retrieval_traces = (
            state.retrieval_traces
        )

        if retrieval_trace is not None:
            retrieval_traces += (
                retrieval_trace,
            )

        calculation_traces = (
            state.calculation_traces
        )

        if calculation_trace is not None:
            calculation_traces += (
                calculation_trace,
            )

        retry_increment = max(
            len(
                execution_result.traces
            )
            - 1,
            0,
        )

        return self._replace_state(
            state,
            status="executing",
            current_step=next_step,
            completed_step_ids=(
                state.completed_step_ids
                + (
                    step.step_id,
                )
            ),
            runtime_refs=runtime_refs,
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
                retrieval_traces
            ),
            calculation_traces=(
                calculation_traces
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
            retry_count=(
                state.retry_count
                + retry_increment
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
            current_node="execute_plan",
            next_node=next_node,
            updated_at=completed_at,
        )

    def _commit_internal_step(
        self,
        *,
        state: AgentState,
        step: ComplexPlanStepOutput,
        output_ref_values: tuple[
            str,
            ...
        ],
    ) -> AgentState:
        runtime_plan = (
            self._require_runtime_plan(
                state
            )
        )

        output_ref_values = (
            _unique_in_order(
                output_ref_values
            )
        )

        if not output_ref_values:
            raise RuntimePlanExecutorError(
                "Internal Step "
                "必须产生 Runtime Reference"
            )

        runtime_refs = dict(
            state.runtime_refs
        )

        if step.output_ref in runtime_refs:
            raise RuntimePlanExecutorError(
                "Runtime output_ref "
                "已经存在："
                f"{step.output_ref}"
            )

        runtime_refs[
            step.output_ref
        ] = output_ref_values

        started_at = self.clock.now()
        timer_start = perf_counter()
        completed_at = self.clock.now()

        next_step = (
            state.current_step + 1
        )

        next_node = (
            "verify_evidence"
            if next_step
            >= len(
                runtime_plan.plan.steps
            )
            else "execute_plan"
        )

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
                "step_id": step.step_id,
                "action": step.action,
                "input_ref_count": len(
                    step.input_refs
                ),
            },
            output_summary={
                "artifact_count": len(
                    output_ref_values
                ),
                "output_ref": (
                    step.output_ref
                ),
            },
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
                state.checkpoint_revision
            ),
            error_type=None,
            error_message=None,
        )

        return self._replace_state(
            state,
            status="executing",
            current_step=next_step,
            completed_step_ids=(
                state.completed_step_ids
                + (
                    step.step_id,
                )
            ),
            runtime_refs=runtime_refs,
            node_spans=(
                state.node_spans
                + (
                    span,
                )
            ),
            step_count=(
                state.step_count + 1
            ),
            current_node="execute_plan",
            next_node=next_node,
            updated_at=completed_at,
        )

    @staticmethod
    def _find_financial_query(
        *,
        runtime_plan: RuntimePlan,
        query_id: str,
    ) -> ComplexRetrievalQueryOutput:
        for query in (
            runtime_plan.financial_queries
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
            runtime_plan.document_queries
        ):
            if query.query_id == query_id:
                return query

        raise RuntimePlanExecutorError(
            "找不到 Document Query："
            f"{query_id}"
        )

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
                state.runtime_plan
                .plan.steps
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
        missing = (
            set(step.depends_on)
            - set(
                state.completed_step_ids
            )
        )

        if missing:
            raise RuntimePlanExecutorError(
                f"{step.step_id} "
                "存在尚未完成的依赖："
                f"{sorted(missing)}"
            )

        if (
            step.step_id
            in state.completed_step_ids
        ):
            raise RuntimePlanExecutorError(
                "当前 Plan Step "
                "已经完成："
                f"{step.step_id}"
            )

    @staticmethod
    def _require_runtime_plan(
        state: AgentState,
    ) -> RuntimePlan:
        if state.runtime_plan is None:
            raise RuntimePlanExecutorError(
                "AgentState 缺少 runtime_plan"
            )

        return state.runtime_plan

    @staticmethod
    def _has_remaining_plan_step(
        state: AgentState,
    ) -> bool:
        if state.runtime_plan is None:
            return False

        return (
            state.current_step
            < len(
                state.runtime_plan
                .plan.steps
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
            runtime_plan.plan.steps[
                state.current_step
            ]
        )

    @staticmethod
    def _tool_name_for_step(
        state: AgentState,
        step: ComplexPlanStepOutput,
    ) -> str | None:
        if state.runtime_plan is None:
            return None

        return (
            state.runtime_plan
            .tool_by_step_id
            .get(step.step_id)
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