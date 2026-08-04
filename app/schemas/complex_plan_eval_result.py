from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.schemas.enums import (
    ReportType,
    StatementScope,
    StatementType,
)


ExecutionMode = Literal[
    "gold_oracle",
    "agent",
]

RunStatus = Literal[
    "completed",
    "refused",
    "failed",
]

StepAction = Literal[
    "retrieve",
    "normalize_unit",
    "calculate",
    "compare",
    "rank",
    "synthesize",
]

StepExecutionStatus = Literal[
    "completed",
    "failed",
]

ErrorStage = Literal[
    "rewrite",
    "planning",
    "retrieval",
    "calculation",
    "answer",
    "citation",
    "system",
]


_ID_PATTERN = r"^[a-z0-9_]+$"
_CASE_ID_PATTERN = r"^complex_[0-9]{3}$"
_QUERY_ID_PATTERN = r"^q[1-9][0-9]*$"
_STEP_ID_PATTERN = r"^s[1-9][0-9]*$"
_FACT_ID_PATTERN = r"^fact_[a-z0-9_]+$"
_EVIDENCE_ID_PATTERN = (
    r"^evidence_[a-z0-9_]+$"
)
_CALCULATION_ID_PATTERN = (
    r"^calculation_[a-z0-9_]+$"
)
_RUN_ID_PATTERN = (
    r"^complex_run_[a-z0-9_]+$"
)


def _validate_unique_ids(
    value: tuple[str, ...],
    *,
    field_name: str,
    pattern: str,
) -> tuple[str, ...]:
    if len(value) != len(set(value)):
        raise ValueError(
            f"{field_name} 不能包含重复 ID"
        )

    for item in value:
        if re.fullmatch(pattern, item) is None:
            raise ValueError(
                f"{field_name} 包含非法 ID："
                f"{item}"
            )

    return value


class ComplexRetrievalQueryOutput(
    BaseModel
):
    """Rewrite 阶段生成的一个原子检索请求。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    query_id: str = Field(
        pattern=_QUERY_ID_PATTERN,
    )

    semantic_query: str = Field(
        min_length=1,
    )

    company_id: str = Field(
        pattern=_ID_PATTERN,
    )

    report_id: str = Field(
        pattern=_ID_PATTERN,
    )

    metric_id: str = Field(
        pattern=_ID_PATTERN,
    )

    fiscal_year: int = Field(
        ge=2000,
        le=2100,
    )

    report_type: ReportType = (
        ReportType.ANNUAL_REPORT
    )

    statement_type: StatementType
    statement_scope: StatementScope


class ComplexRewriteOutput(BaseModel):
    """复杂问题实际生成的结构化 Rewrite。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    normalized_question: str = Field(
        min_length=1,
    )

    retrieval_queries: tuple[
        ComplexRetrievalQueryOutput,
        ...,
    ] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_query_ids(self) -> Self:
        query_ids = [
            query.query_id
            for query in self.retrieval_queries
        ]

        if len(query_ids) != len(set(query_ids)):
            raise ValueError(
                "retrieval_queries 中的 "
                "query_id 必须唯一"
            )

        return self


class ComplexPlanStepOutput(BaseModel):
    """复杂问题实际生成的结构化计划步骤。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    step_id: str = Field(
        pattern=_STEP_ID_PATTERN,
    )

    action: StepAction

    description: str = Field(
        min_length=1,
    )

    input_refs: tuple[str, ...] = ()

    depends_on: tuple[str, ...] = ()

    output_ref: str = Field(
        pattern=_ID_PATTERN,
    )

    retrieval_query_id: str | None = Field(
        default=None,
        pattern=_QUERY_ID_PATTERN,
    )

    calculation_id: str | None = Field(
        default=None,
        pattern=_CALCULATION_ID_PATTERN,
    )

    formula_id: str | None = Field(
        default=None,
        pattern=_ID_PATTERN,
    )

    @field_validator("input_refs")
    @classmethod
    def validate_input_refs(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        return _validate_unique_ids(
            value,
            field_name="input_refs",
            pattern=_ID_PATTERN,
        )

    @field_validator("depends_on")
    @classmethod
    def validate_depends_on(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        return _validate_unique_ids(
            value,
            field_name="depends_on",
            pattern=_STEP_ID_PATTERN,
        )

    @model_validator(mode="after")
    def validate_action_contract(self) -> Self:
        if self.action == "retrieve":
            if self.retrieval_query_id is None:
                raise ValueError(
                    "retrieve 步骤必须填写 "
                    "retrieval_query_id"
                )

            if self.input_refs:
                raise ValueError(
                    "retrieve 步骤不能填写 "
                    "input_refs"
                )

            if self.depends_on:
                raise ValueError(
                    "retrieve 步骤不能依赖 "
                    "其他步骤"
                )

            if (
                self.calculation_id is not None
                or self.formula_id is not None
            ):
                raise ValueError(
                    "retrieve 步骤不能填写 "
                    "计算字段"
                )

            return self

        if self.retrieval_query_id is not None:
            raise ValueError(
                "只有 retrieve 步骤可以填写 "
                "retrieval_query_id"
            )

        if not self.input_refs:
            raise ValueError(
                f"{self.action} 步骤必须填写 "
                "input_refs"
            )

        if self.action in {
            "calculate",
            "normalize_unit",
        }:
            if self.calculation_id is None:
                raise ValueError(
                    f"{self.action} 步骤必须填写 "
                    "calculation_id"
                )

            if self.formula_id is None:
                raise ValueError(
                    f"{self.action} 步骤必须填写 "
                    "formula_id"
                )

            if (
                self.output_ref
                != self.calculation_id
            ):
                raise ValueError(
                    "计算步骤的 output_ref "
                    "必须等于 calculation_id"
                )

            return self

        if (
            self.calculation_id is not None
            or self.formula_id is not None
        ):
            raise ValueError(
                f"{self.action} 步骤不能填写 "
                "计算字段"
            )

        return self


class ComplexPlanOutput(BaseModel):
    """复杂问题实际生成的可执行结构化计划。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    steps: tuple[
        ComplexPlanStepOutput,
        ...,
    ] = Field(
        min_length=1,
    )

    final_step_id: str = Field(
        pattern=_STEP_ID_PATTERN,
    )

    @model_validator(mode="after")
    def validate_plan_topology(self) -> Self:
        step_ids = [
            step.step_id
            for step in self.steps
        ]

        if len(step_ids) != len(set(step_ids)):
            raise ValueError(
                "Plan 中的 step_id 必须唯一"
            )

        output_refs = [
            step.output_ref
            for step in self.steps
        ]

        if len(output_refs) != len(
            set(output_refs)
        ):
            raise ValueError(
                "Plan 中的 output_ref 必须唯一"
            )

        previous_step_ids: set[str] = set()
        available_output_refs: set[str] = set()

        for step in self.steps:
            missing_dependencies = (
                set(step.depends_on)
                - previous_step_ids
            )

            if missing_dependencies:
                raise ValueError(
                    f"{step.step_id} 依赖了尚未执行"
                    "的步骤："
                    f"{sorted(missing_dependencies)}"
                )

            missing_input_refs = (
                set(step.input_refs)
                - available_output_refs
            )

            if missing_input_refs:
                raise ValueError(
                    f"{step.step_id} 使用了尚未生成"
                    "的 input_refs："
                    f"{sorted(missing_input_refs)}"
                )

            previous_step_ids.add(
                step.step_id
            )

            available_output_refs.add(
                step.output_ref
            )

        if (
            self.final_step_id
            != self.steps[-1].step_id
        ):
            raise ValueError(
                "final_step_id 必须是最后一个步骤"
            )

        return self


class ComplexRetrievalTrace(BaseModel):
    """一个原子查询的实际检索轨迹。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    query_id: str = Field(
        pattern=_QUERY_ID_PATTERN,
    )

    status: StepExecutionStatus

    retrieved_fact_ids: tuple[str, ...] = ()
    retrieved_evidence_ids: tuple[str, ...] = ()
    retrieved_chunk_ids: tuple[str, ...] = ()

    top_k: int = Field(
        ge=1,
    )

    latency_ms: float = Field(
        ge=0,
    )

    error_message: str | None = None

    @field_validator("retrieved_fact_ids")
    @classmethod
    def validate_fact_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        return _validate_unique_ids(
            value,
            field_name="retrieved_fact_ids",
            pattern=_FACT_ID_PATTERN,
        )

    @field_validator(
        "retrieved_evidence_ids"
    )
    @classmethod
    def validate_evidence_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        return _validate_unique_ids(
            value,
            field_name="retrieved_evidence_ids",
            pattern=_EVIDENCE_ID_PATTERN,
        )

    @field_validator("retrieved_chunk_ids")
    @classmethod
    def validate_chunk_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        return _validate_unique_ids(
            value,
            field_name="retrieved_chunk_ids",
            pattern=_ID_PATTERN,
        )

    @model_validator(mode="after")
    def validate_status_contract(self) -> Self:
        if self.status == "completed":
            if self.error_message is not None:
                raise ValueError(
                    "completed Retrieval Trace "
                    "不能填写 error_message"
                )
        elif not self.error_message:
            raise ValueError(
                "failed Retrieval Trace "
                "必须填写 error_message"
            )

        return self


class ComplexCalculationTrace(BaseModel):
    """一个计算步骤的实际执行结果。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    calculation_id: str = Field(
        pattern=_CALCULATION_ID_PATTERN,
    )

    metric_id: str = Field(
        pattern=_ID_PATTERN,
    )

    formula_id: str = Field(
        pattern=_ID_PATTERN,
    )

    input_fact_ids: tuple[str, ...] = Field(
        min_length=1,
    )

    status: StepExecutionStatus

    result_value: Decimal | None = None

    result_unit: str | None = None

    latency_ms: float = Field(
        ge=0,
    )

    error_message: str | None = None

    @field_validator("input_fact_ids")
    @classmethod
    def validate_input_fact_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        return _validate_unique_ids(
            value,
            field_name="input_fact_ids",
            pattern=_FACT_ID_PATTERN,
        )

    @model_validator(mode="after")
    def validate_status_contract(self) -> Self:
        if self.status == "completed":
            if self.result_value is None:
                raise ValueError(
                    "completed Calculation Trace "
                    "必须填写 result_value"
                )

            if self.result_unit is None:
                raise ValueError(
                    "completed Calculation Trace "
                    "必须填写 result_unit"
                )

            if self.error_message is not None:
                raise ValueError(
                    "completed Calculation Trace "
                    "不能填写 error_message"
                )

            return self

        if not self.error_message:
            raise ValueError(
                "failed Calculation Trace "
                "必须填写 error_message"
            )

        return self


class ComplexFinalAnswerOutput(BaseModel):
    """复杂问题的实际回答及引用。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    answer_text: str = Field(
        min_length=1,
    )

    supporting_fact_ids: tuple[
        str,
        ...,
    ] = Field(
        min_length=1,
    )

    supporting_calculation_ids: tuple[
        str,
        ...,
    ] = ()

    citation_evidence_ids: tuple[
        str,
        ...,
    ] = Field(
        min_length=1,
    )

    @field_validator("supporting_fact_ids")
    @classmethod
    def validate_supporting_fact_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        return _validate_unique_ids(
            value,
            field_name="supporting_fact_ids",
            pattern=_FACT_ID_PATTERN,
        )

    @field_validator(
        "supporting_calculation_ids"
    )
    @classmethod
    def validate_calculation_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        return _validate_unique_ids(
            value,
            field_name=(
                "supporting_calculation_ids"
            ),
            pattern=_CALCULATION_ID_PATTERN,
        )

    @field_validator(
        "citation_evidence_ids"
    )
    @classmethod
    def validate_citation_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        return _validate_unique_ids(
            value,
            field_name=(
                "citation_evidence_ids"
            ),
            pattern=_EVIDENCE_ID_PATTERN,
        )


class ComplexPlanRunResult(BaseModel):
    """一次复杂问题执行的完整、可审计结果。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    run_id: str = Field(
        pattern=_RUN_ID_PATTERN,
    )

    case_id: str = Field(
        pattern=_CASE_ID_PATTERN,
    )

    execution_mode: ExecutionMode
    status: RunStatus

    rewrite: ComplexRewriteOutput | None = None
    plan: ComplexPlanOutput | None = None

    retrieval_traces: tuple[
        ComplexRetrievalTrace,
        ...,
    ] = ()

    calculation_traces: tuple[
        ComplexCalculationTrace,
        ...,
    ] = ()

    answer: ComplexFinalAnswerOutput | None = None

    planner_id: str | None = None
    retriever_id: str | None = None
    calculator_id: str | None = None
    generator_id: str | None = None

    started_at: datetime
    completed_at: datetime

    latency_ms: float = Field(
        ge=0,
    )

    error_stage: ErrorStage | None = None
    error_message: str | None = None

    @field_validator(
        "started_at",
        "completed_at",
    )
    @classmethod
    def validate_timezone(
        cls,
        value: datetime,
    ) -> datetime:
        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "datetime 必须包含时区信息"
            )

        return value

    @model_validator(mode="after")
    def validate_run_contract(self) -> Self:
        if self.completed_at < self.started_at:
            raise ValueError(
                "completed_at 不能早于 started_at"
            )

        retrieval_query_ids = [
            trace.query_id
            for trace in self.retrieval_traces
        ]

        if len(retrieval_query_ids) != len(
            set(retrieval_query_ids)
        ):
            raise ValueError(
                "retrieval_traces 中的 query_id "
                "必须唯一"
            )

        calculation_ids = [
            trace.calculation_id
            for trace in self.calculation_traces
        ]

        if len(calculation_ids) != len(
            set(calculation_ids)
        ):
            raise ValueError(
                "calculation_traces 中的 "
                "calculation_id 必须唯一"
            )

        if self.status == "completed":
            self._validate_completed_run()
        else:
            self._validate_incomplete_run()

        return self

    def _validate_completed_run(self) -> None:
        if self.rewrite is None:
            raise ValueError(
                "completed Run 必须包含 rewrite"
            )

        if self.plan is None:
            raise ValueError(
                "completed Run 必须包含 plan"
            )

        if self.answer is None:
            raise ValueError(
                "completed Run 必须包含 answer"
            )

        if not self.planner_id:
            raise ValueError(
                "completed Run 必须填写 planner_id"
            )

        if not self.retriever_id:
            raise ValueError(
                "completed Run 必须填写 retriever_id"
            )

        if (
            self.calculation_traces
            and not self.calculator_id
        ):
            raise ValueError(
                "包含计算步骤的 completed Run "
                "必须填写 calculator_id"
            )

        if not self.generator_id:
            raise ValueError(
                "completed Run 必须填写 generator_id"
            )

        if (
            self.error_stage is not None
            or self.error_message is not None
        ):
            raise ValueError(
                "completed Run 不能填写错误字段"
            )

        if any(
            trace.status != "completed"
            for trace in self.retrieval_traces
        ):
            raise ValueError(
                "completed Run 不能包含失败的 "
                "Retrieval Trace"
            )

        if any(
            trace.status != "completed"
            for trace in self.calculation_traces
        ):
            raise ValueError(
                "completed Run 不能包含失败的 "
                "Calculation Trace"
            )

        self._validate_rewrite_plan_traces()
        self._validate_answer_references()

    def _validate_incomplete_run(self) -> None:
        if self.error_stage is None:
            raise ValueError(
                "refused 或 failed Run "
                "必须填写 error_stage"
            )

        if not self.error_message:
            raise ValueError(
                "refused 或 failed Run "
                "必须填写 error_message"
            )

        if self.answer is not None:
            raise ValueError(
                "refused 或 failed Run "
                "不能包含 final answer"
            )

    def _validate_rewrite_plan_traces(
        self,
    ) -> None:
        assert self.rewrite is not None
        assert self.plan is not None

        rewrite_query_ids = {
            query.query_id
            for query
            in self.rewrite.retrieval_queries
        }

        plan_query_ids = {
            step.retrieval_query_id
            for step in self.plan.steps
            if step.action == "retrieve"
        }

        if rewrite_query_ids != plan_query_ids:
            raise ValueError(
                "Rewrite Query 与 Plan Retrieve "
                "步骤必须一一对应"
            )

        trace_query_ids = {
            trace.query_id
            for trace in self.retrieval_traces
        }

        if rewrite_query_ids != trace_query_ids:
            raise ValueError(
                "Rewrite Query 与 Retrieval Trace "
                "必须一一对应"
            )

        plan_calculation_ids = {
            step.calculation_id
            for step in self.plan.steps
            if step.action in {
                "calculate",
                "normalize_unit",
            }
        }

        trace_calculation_ids = {
            trace.calculation_id
            for trace in self.calculation_traces
        }

        if (
            plan_calculation_ids
            != trace_calculation_ids
        ):
            raise ValueError(
                "Plan Calculation 与 "
                "Calculation Trace 必须一一对应"
            )

    def _validate_answer_references(
        self,
    ) -> None:
        assert self.answer is not None

        retrieved_fact_ids = {
            fact_id
            for trace in self.retrieval_traces
            for fact_id
            in trace.retrieved_fact_ids
        }

        retrieved_evidence_ids = {
            evidence_id
            for trace in self.retrieval_traces
            for evidence_id
            in trace.retrieved_evidence_ids
        }

        completed_calculation_ids = {
            trace.calculation_id
            for trace in self.calculation_traces
            if trace.status == "completed"
        }

        missing_fact_ids = (
            set(self.answer.supporting_fact_ids)
            - retrieved_fact_ids
        )

        if missing_fact_ids:
            raise ValueError(
                "Answer 引用了未检索到的 Fact："
                f"{sorted(missing_fact_ids)}"
            )

        missing_evidence_ids = (
            set(
                self.answer
                .citation_evidence_ids
            )
            - retrieved_evidence_ids
        )

        if missing_evidence_ids:
            raise ValueError(
                "Answer 引用了未检索到的 Evidence："
                f"{sorted(missing_evidence_ids)}"
            )

        missing_calculation_ids = (
            set(
                self.answer
                .supporting_calculation_ids
            )
            - completed_calculation_ids
        )

        if missing_calculation_ids:
            raise ValueError(
                "Answer 引用了未完成的 Calculation："
                f"{sorted(missing_calculation_ids)}"
            )

        for trace in self.calculation_traces:
            missing_inputs = (
                set(trace.input_fact_ids)
                - retrieved_fact_ids
            )

            if missing_inputs:
                raise ValueError(
                    "Calculation 使用了未检索到的 "
                    "Fact："
                    f"{sorted(missing_inputs)}"
                )