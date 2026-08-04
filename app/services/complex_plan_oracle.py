from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Protocol

from app.schemas.complex_plan_eval import (
    ComplexFinancialEvalCase,
)
from app.schemas.complex_plan_eval_result import (
    ComplexCalculationTrace,
    ComplexFinalAnswerOutput,
    ComplexPlanOutput,
    ComplexPlanRunResult,
    ComplexPlanStepOutput,
    ComplexRetrievalQueryOutput,
    ComplexRetrievalTrace,
    ComplexRewriteOutput,
)
from app.schemas.enums import ValidationStatus


class GoldOracleExecutionError(ValueError):
    """Gold Oracle 执行失败。"""


class GoldOracleRetriever(Protocol):
    """复杂问题原子查询检索器接口。"""

    @property
    def retriever_id(self) -> str:
        """返回检索器及配置标识。"""

    def retrieve(
        self,
        *,
        query: ComplexRetrievalQueryOutput,
        top_k: int,
    ) -> ComplexRetrievalTrace:
        """执行一个原子查询。"""


class GoldOracleCalculator(Protocol):
    """确定性计算器接口。"""

    @property
    def calculator_id(self) -> str:
        """返回计算器及版本标识。"""

    def calculate(
        self,
        *,
        calculation_id: str,
        formula_id: str,
        input_fact_ids: tuple[str, ...],
    ) -> ComplexCalculationTrace:
        """执行一个确定性计算步骤。"""


class GoldOracleAnswerGenerator(Protocol):
    """复杂问题回答生成器接口。"""

    @property
    def generator_id(self) -> str:
        """返回回答生成器标识。"""

    def generate(
        self,
        *,
        question: str,
        rewrite: ComplexRewriteOutput,
        plan: ComplexPlanOutput,
        retrieval_traces: tuple[
            ComplexRetrievalTrace,
            ...,
        ],
        calculation_traces: tuple[
            ComplexCalculationTrace,
            ...,
        ],
    ) -> ComplexFinalAnswerOutput:
        """只根据实际执行轨迹生成答案。"""


def build_gold_oracle_rewrite(
    case: ComplexFinancialEvalCase,
) -> ComplexRewriteOutput:
    """从 Gold Case 构造无答案泄漏的 Oracle Rewrite。"""

    queries = tuple(
        ComplexRetrievalQueryOutput(
            query_id=query.query_id,
            semantic_query=query.semantic_query,
            company_id=query.company_id,
            report_id=query.report_id,
            metric_id=query.metric_id,
            fiscal_year=query.fiscal_year,
            report_type=query.report_type,
            statement_type=query.statement_type,
            statement_scope=query.statement_scope,
        )
        for query
        in case.gold_rewrite.retrieval_queries
    )

    return ComplexRewriteOutput(
        normalized_question=(
            case.gold_rewrite
            .normalized_question
        ),
        retrieval_queries=queries,
    )


def build_gold_oracle_plan(
    case: ComplexFinancialEvalCase,
) -> ComplexPlanOutput:
    """构造不暴露目标 Fact ID 的 Oracle Plan。"""

    query_id_by_fact_id = {
        query.target_fact_id: query.query_id
        for query
        in case.gold_rewrite.retrieval_queries
    }

    runtime_ref_by_fact_id = {
        fact_id: (
            "retrieval_result_"
            f"{query_id}"
        )
        for fact_id, query_id
        in query_id_by_fact_id.items()
    }

    steps: list[
        ComplexPlanStepOutput
    ] = []

    for gold_step in case.gold_plan.steps:
        if gold_step.action == "retrieve":
            target_fact_id = (
                gold_step.target_fact_ids[0]
            )

            query_id = (
                query_id_by_fact_id.get(
                    target_fact_id
                )
            )

            if query_id is None:
                raise GoldOracleExecutionError(
                    "Gold Plan 的 retrieve 步骤"
                    "找不到对应 Retrieval Query："
                    f"{gold_step.step_id}"
                )

            output_ref = (
                runtime_ref_by_fact_id[
                    target_fact_id
                ]
            )

            steps.append(
                ComplexPlanStepOutput(
                    step_id=gold_step.step_id,
                    action="retrieve",
                    description=(
                        gold_step.description
                    ),
                    input_refs=(),
                    depends_on=(),
                    output_ref=output_ref,
                    retrieval_query_id=query_id,
                )
            )

            continue

        mapped_input_refs = tuple(
            runtime_ref_by_fact_id.get(
                input_ref,
                input_ref,
            )
            for input_ref
            in gold_step.input_refs
        )

        mapped_output_ref = (
            runtime_ref_by_fact_id.get(
                gold_step.output_ref,
                gold_step.output_ref,
            )
        )

        steps.append(
            ComplexPlanStepOutput(
                step_id=gold_step.step_id,
                action=gold_step.action,
                description=(
                    gold_step.description
                ),
                input_refs=mapped_input_refs,
                depends_on=(
                    gold_step.depends_on
                ),
                output_ref=(
                    mapped_output_ref
                ),
                calculation_id=(
                    gold_step.calculation_id
                ),
                formula_id=(
                    gold_step.formula_id
                ),
            )
        )

    return ComplexPlanOutput(
        steps=tuple(steps),
        final_step_id=(
            case.gold_plan.final_step_id
        ),
    )


def execute_gold_oracle_case(
    *,
    run_id: str,
    case: ComplexFinancialEvalCase,
    retriever: GoldOracleRetriever,
    generator: GoldOracleAnswerGenerator,
    calculator: (
        GoldOracleCalculator | None
    ) = None,
    top_k: int = 5,
) -> ComplexPlanRunResult:
    """按照 Gold Rewrite/Plan 执行一条复杂问题。

    Oracle 只提供结构化意图与步骤，不向检索器提供
    target_fact_id、Gold Evidence、Gold 页码或 Gold Answer。
    """

    if top_k <= 0:
        raise GoldOracleExecutionError(
            "top_k 必须大于 0"
        )

    started_at = datetime.now(
        timezone.utc
    )

    timer_start = perf_counter()

    rewrite: ComplexRewriteOutput | None = None
    plan: ComplexPlanOutput | None = None

    retrieval_traces: list[
        ComplexRetrievalTrace
    ] = []

    calculation_traces: list[
        ComplexCalculationTrace
    ] = []

    runtime_refs: dict[
        str,
        tuple[str, ...],
    ] = {}

    retriever_id = (
        retriever.retriever_id.strip()
    )

    generator_id = (
        generator.generator_id.strip()
    )

    calculator_id = (
        calculator.calculator_id.strip()
        if calculator is not None
        else None
    )

    def build_failed_result(
        *,
        error_stage: str,
        error_message: str,
    ) -> ComplexPlanRunResult:
        completed_at = datetime.now(
            timezone.utc
        )

        latency_ms = (
            perf_counter() - timer_start
        ) * 1000

        return ComplexPlanRunResult(
            run_id=run_id,
            case_id=case.case_id,
            execution_mode="gold_oracle",
            status="failed",
            rewrite=rewrite,
            plan=plan,
            retrieval_traces=tuple(
                retrieval_traces
            ),
            calculation_traces=tuple(
                calculation_traces
            ),
            answer=None,
            planner_id="gold_oracle_v1",
            retriever_id=(
                retriever_id or None
            ),
            calculator_id=(
                calculator_id or None
            ),
            generator_id=(
                generator_id or None
            ),
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=latency_ms,
            error_stage=error_stage,
            error_message=error_message,
        )

    if (
        case.validation_status
        is not ValidationStatus.VERIFIED
    ):
        return build_failed_result(
            error_stage="planning",
            error_message=(
                f"Case '{case.case_id}' "
                "尚未 verified"
            ),
        )

    if not retriever_id:
        return build_failed_result(
            error_stage="system",
            error_message=(
                "retriever_id 不能为空"
            ),
        )

    if not generator_id:
        return build_failed_result(
            error_stage="system",
            error_message=(
                "generator_id 不能为空"
            ),
        )

    try:
        rewrite = build_gold_oracle_rewrite(
            case
        )
    except Exception as exc:
        return build_failed_result(
            error_stage="rewrite",
            error_message=str(exc),
        )

    try:
        plan = build_gold_oracle_plan(
            case
        )
    except Exception as exc:
        return build_failed_result(
            error_stage="planning",
            error_message=str(exc),
        )

    query_by_id = {
        query.query_id: query
        for query
        in rewrite.retrieval_queries
    }

    for step in plan.steps:
        if step.action == "retrieve":
            query_id = (
                step.retrieval_query_id
            )

            assert query_id is not None

            query = query_by_id.get(
                query_id
            )

            if query is None:
                return build_failed_result(
                    error_stage="planning",
                    error_message=(
                        "Plan 引用了不存在的 "
                        f"query_id：{query_id}"
                    ),
                )

            try:
                trace = (
                    ComplexRetrievalTrace
                    .model_validate(
                        retriever.retrieve(
                            query=query,
                            top_k=top_k,
                        )
                    )
                )
            except Exception as exc:
                return build_failed_result(
                    error_stage="retrieval",
                    error_message=(
                        f"{query_id} 检索异常："
                        f"{exc}"
                    ),
                )

            retrieval_traces.append(
                trace
            )

            if trace.query_id != query_id:
                return build_failed_result(
                    error_stage="retrieval",
                    error_message=(
                        "Retriever 返回了错误的 "
                        "query_id："
                        f"expected={query_id}, "
                        f"actual={trace.query_id}"
                    ),
                )

            if trace.top_k != top_k:
                return build_failed_result(
                    error_stage="retrieval",
                    error_message=(
                        "Retriever 未使用要求的 "
                        f"top_k：expected={top_k}, "
                        f"actual={trace.top_k}"
                    ),
                )

            if trace.status == "failed":
                return build_failed_result(
                    error_stage="retrieval",
                    error_message=(
                        trace.error_message
                        or f"{query_id} 检索失败"
                    ),
                )

            if not trace.retrieved_fact_ids:
                return build_failed_result(
                    error_stage="retrieval",
                    error_message=(
                        f"{query_id} 没有解析出 "
                        "FinancialFact"
                    ),
                )

            # 有序列表中的第一项是当前检索器选择的 Fact。
            runtime_refs[
                step.output_ref
            ] = (
                trace.retrieved_fact_ids[0],
            )

            continue

        try:
            resolved_inputs = (
                _resolve_runtime_inputs(
                    step=step,
                    runtime_refs=runtime_refs,
                )
            )
        except GoldOracleExecutionError as exc:
            return build_failed_result(
                error_stage="planning",
                error_message=str(exc),
            )

        if step.action in {
            "calculate",
            "normalize_unit",
        }:
            if calculator is None:
                return build_failed_result(
                    error_stage="calculation",
                    error_message=(
                        f"{step.step_id} 需要 "
                        "Calculator，但未提供"
                    ),
                )

            if not calculator_id:
                return build_failed_result(
                    error_stage="system",
                    error_message=(
                        "calculator_id 不能为空"
                    ),
                )

            invalid_input_ids = [
                item
                for item in resolved_inputs
                if not item.startswith("fact_")
            ]

            if invalid_input_ids:
                return build_failed_result(
                    error_stage="calculation",
                    error_message=(
                        f"{step.step_id} 收到非 Fact "
                        "计算输入："
                        f"{invalid_input_ids}"
                    ),
                )

            assert step.calculation_id is not None
            assert step.formula_id is not None

            try:
                calculation_trace = (
                    ComplexCalculationTrace
                    .model_validate(
                        calculator.calculate(
                            calculation_id=(
                                step.calculation_id
                            ),
                            formula_id=(
                                step.formula_id
                            ),
                            input_fact_ids=(
                                resolved_inputs
                            ),
                        )
                    )
                )
            except Exception as exc:
                return build_failed_result(
                    error_stage="calculation",
                    error_message=(
                        f"{step.step_id} 计算异常："
                        f"{exc}"
                    ),
                )

            calculation_traces.append(
                calculation_trace
            )

            if (
                calculation_trace.calculation_id
                != step.calculation_id
            ):
                return build_failed_result(
                    error_stage="calculation",
                    error_message=(
                        "Calculator 返回了错误的 "
                        "calculation_id"
                    ),
                )

            if (
                calculation_trace.formula_id
                != step.formula_id
            ):
                return build_failed_result(
                    error_stage="calculation",
                    error_message=(
                        "Calculator 返回了错误的 "
                        "formula_id"
                    ),
                )

            if (
                calculation_trace.input_fact_ids
                != resolved_inputs
            ):
                return build_failed_result(
                    error_stage="calculation",
                    error_message=(
                        "Calculator 使用的 "
                        "input_fact_ids 顺序不一致"
                    ),
                )

            if (
                calculation_trace.status
                == "failed"
            ):
                return build_failed_result(
                    error_stage="calculation",
                    error_message=(
                        calculation_trace
                        .error_message
                        or f"{step.step_id} 计算失败"
                    ),
                )

            runtime_refs[
                step.output_ref
            ] = (
                calculation_trace
                .calculation_id,
            )

            continue

        # compare、rank 和 synthesize 在第一版中
        # 只传递已解析引用；最终语义由 Generator 完成。
        runtime_refs[
            step.output_ref
        ] = resolved_inputs

    try:
        answer = (
            ComplexFinalAnswerOutput
            .model_validate(
                generator.generate(
                    question=case.question,
                    rewrite=rewrite,
                    plan=plan,
                    retrieval_traces=tuple(
                        retrieval_traces
                    ),
                    calculation_traces=tuple(
                        calculation_traces
                    ),
                )
            )
        )
    except Exception as exc:
        return build_failed_result(
            error_stage="answer",
            error_message=(
                f"回答生成失败：{exc}"
            ),
        )

    completed_at = datetime.now(
        timezone.utc
    )

    latency_ms = (
        perf_counter() - timer_start
    ) * 1000

    try:
        return ComplexPlanRunResult(
            run_id=run_id,
            case_id=case.case_id,
            execution_mode="gold_oracle",
            status="completed",
            rewrite=rewrite,
            plan=plan,
            retrieval_traces=tuple(
                retrieval_traces
            ),
            calculation_traces=tuple(
                calculation_traces
            ),
            answer=answer,
            planner_id="gold_oracle_v1",
            retriever_id=retriever_id,
            calculator_id=(
                calculator_id
                if calculation_traces
                else None
            ),
            generator_id=generator_id,
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=latency_ms,
            error_stage=None,
            error_message=None,
        )
    except Exception as exc:
        return build_failed_result(
            error_stage="answer",
            error_message=(
                "最终运行结果未通过审计约束："
                f"{exc}"
            ),
        )


def _resolve_runtime_inputs(
    *,
    step: ComplexPlanStepOutput,
    runtime_refs: dict[
        str,
        tuple[str, ...],
    ],
) -> tuple[str, ...]:
    resolved: list[str] = []

    for input_ref in step.input_refs:
        values = runtime_refs.get(
            input_ref
        )

        if values is None:
            raise GoldOracleExecutionError(
                f"{step.step_id} 引用了尚未执行的 "
                f"input_ref：{input_ref}"
            )

        resolved.extend(values)

    return tuple(resolved)