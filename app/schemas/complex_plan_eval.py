from __future__ import annotations

import re
from datetime import datetime
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .enums import (
    ReportType,
    StatementScope,
    StatementType,
    ValidationStatus,
)


QuestionType = Literal[
    "single_company_multi_metric",
    "cross_company_comparison",
    "multi_company_ranking",
]

Difficulty = Literal[
    "medium",
    "hard",
]

AliasType = Literal[
    "company",
    "metric",
    "fiscal_year",
    "statement_scope",
]

PlanAction = Literal[
    "retrieve",
    "normalize_unit",
    "calculate",
    "compare",
    "rank",
    "synthesize",
]


_ID_PATTERN = r"^[a-z0-9_]+$"
_FACT_ID_PATTERN = r"^fact_[a-z0-9_]+$"
_EVIDENCE_ID_PATTERN = r"^evidence_[a-z0-9_]+$"
_CALCULATION_ID_PATTERN = r"^calculation_[a-z0-9_]+$"
_QUERY_ID_PATTERN = r"^q[1-9][0-9]*$"
_STEP_ID_PATTERN = r"^s[1-9][0-9]*$"
_CASE_ID_PATTERN = r"^complex_[0-9]{3}$"


class GoldResolvedAlias(BaseModel):
    """问题中的原始表达及其标准化结果。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    source_text: str = Field(min_length=1)
    alias_type: AliasType
    normalized_value: str = Field(min_length=1)


class GoldRetrievalQuery(BaseModel):
    """Gold Rewrite 拆解出的一个原子事实检索请求。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    query_id: str = Field(pattern=_QUERY_ID_PATTERN)

    # 该检索请求预期得到的 FinancialFact ID。
    target_fact_id: str = Field(pattern=_FACT_ID_PATTERN)

    # baseline_query 必须保留用户原始问题。
    baseline_query: str = Field(min_length=1)

    # semantic_query 是面向单个原子事实的重写查询。
    semantic_query: str = Field(min_length=1)

    company_id: str = Field(pattern=_ID_PATTERN)
    report_id: str = Field(pattern=_ID_PATTERN)
    metric_id: str = Field(pattern=_ID_PATTERN)
    fiscal_year: int = Field(ge=2000, le=2100)

    report_type: ReportType = ReportType.ANNUAL_REPORT
    statement_type: StatementType
    statement_scope: StatementScope

    # 页码用于人工审核，不作为检索器必须命中的唯一条件。
    gold_pdf_pages: tuple[int, ...] = Field(
        min_length=1,
    )

    @field_validator("gold_pdf_pages")
    @classmethod
    def validate_gold_pdf_pages(
        cls,
        value: tuple[int, ...],
    ) -> tuple[int, ...]:
        if any(page <= 0 for page in value):
            raise ValueError(
                "gold_pdf_pages 中的页码必须大于 0"
            )

        if len(value) != len(set(value)):
            raise ValueError(
                "gold_pdf_pages 不能包含重复页码"
            )

        return tuple(sorted(value))

    @model_validator(mode="after")
    def validate_report_identity(self) -> Self:
        expected_prefix = f"{self.company_id}_"

        if not self.report_id.startswith(expected_prefix):
            raise ValueError(
                "report_id 必须属于 company_id 对应的公司"
            )

        match = re.search(
            r"_(20[0-9]{2})_",
            self.report_id,
        )

        if match is not None:
            report_year = int(match.group(1))

            # 允许从 2024 年年报中检索 2023 年比较数据。
            if report_year < self.fiscal_year:
                raise ValueError(
                    "report_id 的报告年度不能早于 fiscal_year"
                )


        if self.statement_scope is StatementScope.UNKNOWN:
            raise ValueError(
                "Gold Query 的 statement_scope 不能是 unknown"
            )

        return self


class GoldRewrite(BaseModel):
    """复杂问题的人工标准化结果和原子查询拆解。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    normalized_question: str = Field(min_length=1)

    resolved_aliases: tuple[GoldResolvedAlias, ...] = ()

    retrieval_queries: tuple[
        GoldRetrievalQuery,
        ...
    ] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_retrieval_queries(self) -> Self:
        query_ids = [
            query.query_id
            for query in self.retrieval_queries
        ]

        if len(query_ids) != len(set(query_ids)):
            raise ValueError(
                "retrieval_queries 中的 query_id 必须唯一"
            )

        target_fact_ids = [
            query.target_fact_id
            for query in self.retrieval_queries
        ]

        if len(target_fact_ids) != len(set(target_fact_ids)):
            raise ValueError(
                "每个 retrieval_query 必须对应不同的 "
                "target_fact_id"
            )

        return self


class GoldPlanStep(BaseModel):
    """Gold Plan 中的一个可执行步骤。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    step_id: str = Field(pattern=_STEP_ID_PATTERN)
    action: PlanAction
    description: str = Field(min_length=1)

    # retrieve 步骤需要检索的事实。
    target_fact_ids: tuple[str, ...] = ()

    # 当前步骤消费的事实、计算结果或中间结果 ID。
    input_refs: tuple[str, ...] = ()

    # 当前步骤依赖的前序 step_id。
    depends_on: tuple[str, ...] = ()

    # 当前步骤生成的结果 ID。
    output_ref: str = Field(pattern=_ID_PATTERN)

    # calculate 和 normalize_unit 步骤需要填写。
    calculation_id: str | None = Field(
        default=None,
        pattern=_CALCULATION_ID_PATTERN,
    )

    formula_id: str | None = Field(
        default=None,
        pattern=_ID_PATTERN,
    )

    @field_validator("target_fact_ids")
    @classmethod
    def validate_target_fact_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError(
                "target_fact_ids 不能包含重复 ID"
            )

        for fact_id in value:
            if re.fullmatch(
                _FACT_ID_PATTERN,
                fact_id,
            ) is None:
                raise ValueError(
                    f"非法 fact_id：{fact_id}"
                )

        return value

    @field_validator("input_refs")
    @classmethod
    def validate_input_refs(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError(
                "input_refs 不能包含重复 ID"
            )

        for input_ref in value:
            if re.fullmatch(
                _ID_PATTERN,
                input_ref,
            ) is None:
                raise ValueError(
                    f"非法 input_ref：{input_ref}"
                )

        return value

    @field_validator("depends_on")
    @classmethod
    def validate_depends_on(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError(
                "depends_on 不能包含重复 step_id"
            )

        for step_id in value:
            if re.fullmatch(
                _STEP_ID_PATTERN,
                step_id,
            ) is None:
                raise ValueError(
                    f"非法 depends_on step_id：{step_id}"
                )

        return value

    @model_validator(mode="after")
    def validate_action_contract(self) -> Self:
        if self.action == "retrieve":
            if len(self.target_fact_ids) != 1:
                raise ValueError(
                    "retrieve 步骤必须且只能包含一个 "
                    "target_fact_id"
                )

            if self.input_refs:
                raise ValueError(
                    "retrieve 步骤不能包含 input_refs"
                )

            if self.depends_on:
                raise ValueError(
                    "retrieve 步骤不能依赖其他步骤"
                )

            if self.output_ref != self.target_fact_ids[0]:
                raise ValueError(
                    "retrieve 步骤的 output_ref 必须等于 "
                    "target_fact_id"
                )

            if (
                self.calculation_id is not None
                or self.formula_id is not None
            ):
                raise ValueError(
                    "retrieve 步骤不能包含计算字段"
                )

            return self

        if self.target_fact_ids:
            raise ValueError(
                "只有 retrieve 步骤可以填写 "
                "target_fact_ids"
            )

        if not self.input_refs:
            raise ValueError(
                f"{self.action} 步骤必须包含 input_refs"
            )

        if self.action in {
            "calculate",
            "normalize_unit",
        }:
            if self.calculation_id is None:
                raise ValueError(
                    f"{self.action} 步骤必须包含 "
                    "calculation_id"
                )

            if self.formula_id is None:
                raise ValueError(
                    f"{self.action} 步骤必须包含 "
                    "formula_id"
                )

            if self.output_ref != self.calculation_id:
                raise ValueError(
                    "计算步骤的 output_ref 必须等于 "
                    "calculation_id"
                )

            return self

        if (
            self.calculation_id is not None
            or self.formula_id is not None
        ):
            raise ValueError(
                f"{self.action} 步骤不能包含计算字段"
            )

        return self


class GoldPlan(BaseModel):
    """按拓扑顺序排列的 Gold 执行计划。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    steps: tuple[GoldPlanStep, ...] = Field(
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
                "Gold Plan 中的 step_id 必须唯一"
            )

        output_refs = [
            step.output_ref
            for step in self.steps
        ]

        if len(output_refs) != len(set(output_refs)):
            raise ValueError(
                "Gold Plan 中的 output_ref 必须唯一"
            )

        previous_step_ids: set[str] = set()

        for step in self.steps:
            missing_dependencies = (
                set(step.depends_on)
                - previous_step_ids
            )

            if missing_dependencies:
                raise ValueError(
                    f"{step.step_id} 依赖了尚未执行的步骤："
                    f"{sorted(missing_dependencies)}"
                )

            previous_step_ids.add(step.step_id)

        if self.final_step_id != self.steps[-1].step_id:
            raise ValueError(
                "final_step_id 必须是 steps 中最后一个步骤"
            )

        return self


class GoldAnswer(BaseModel):
    """人工确认的最终答案及其来源引用。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    answer_text: str = Field(min_length=1)
    conclusion: str = Field(min_length=1)

    supporting_fact_ids: tuple[str, ...] = Field(
        min_length=1,
    )

    evidence_ids: tuple[str, ...] = Field(
        min_length=1,
    )

    supporting_calculation_ids: tuple[str, ...] = ()

    @field_validator("supporting_fact_ids")
    @classmethod
    def validate_supporting_fact_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError(
                "supporting_fact_ids 不能包含重复 ID"
            )

        for fact_id in value:
            if re.fullmatch(
                _FACT_ID_PATTERN,
                fact_id,
            ) is None:
                raise ValueError(
                    f"非法 supporting fact_id：{fact_id}"
                )

        return value

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError(
                "evidence_ids 不能包含重复 ID"
            )

        for evidence_id in value:
            if re.fullmatch(
                _EVIDENCE_ID_PATTERN,
                evidence_id,
            ) is None:
                raise ValueError(
                    f"非法 evidence_id：{evidence_id}"
                )

        return value

    @field_validator("supporting_calculation_ids")
    @classmethod
    def validate_supporting_calculation_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError(
                "supporting_calculation_ids 不能包含重复 ID"
            )

        for calculation_id in value:
            if re.fullmatch(
                _CALCULATION_ID_PATTERN,
                calculation_id,
            ) is None:
                raise ValueError(
                    "非法 supporting calculation_id："
                    f"{calculation_id}"
                )

        return value


class ComplexFinancialEvalCase(BaseModel):
    """复杂财务问题的 Rewrite、Plan 和 Answer Gold Case。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    case_id: str = Field(
        pattern=_CASE_ID_PATTERN,
    )

    question: str = Field(min_length=1)

    question_type: QuestionType
    difficulty: Difficulty

    company_ids: tuple[str, ...] = Field(
        min_length=1,
    )

    report_ids: tuple[str, ...] = Field(
        min_length=1,
    )

    fiscal_years: tuple[int, ...] = Field(
        min_length=1,
    )

    gold_rewrite: GoldRewrite
    gold_plan: GoldPlan

    # 引用现有 FinancialFact、SourceEvidence 和
    # DerivedCalculation，不在这里重复定义它们。
    gold_fact_ids: tuple[str, ...] = Field(
        min_length=2,
    )

    gold_evidence_ids: tuple[str, ...] = Field(
        min_length=1,
    )

    gold_calculation_ids: tuple[str, ...] = ()

    gold_answer: GoldAnswer

    validation_status: ValidationStatus = (
        ValidationStatus.PENDING
    )

    validated_by: str | None = None
    validated_at: datetime | None = None

    source_version: str = Field(
        min_length=1,
        pattern=_ID_PATTERN,
    )

    created_at: datetime
    updated_at: datetime

    review_notes: str | None = None

    @field_validator(
        "company_ids",
        "report_ids",
    )
    @classmethod
    def validate_basic_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError(
                "ID 列表不能包含重复值"
            )

        for item in value:
            if re.fullmatch(
                _ID_PATTERN,
                item,
            ) is None:
                raise ValueError(
                    f"非法 ID：{item}"
                )

        return value

    @field_validator("gold_fact_ids")
    @classmethod
    def validate_gold_fact_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError(
                "gold_fact_ids 不能包含重复 ID"
            )

        for fact_id in value:
            if re.fullmatch(
                _FACT_ID_PATTERN,
                fact_id,
            ) is None:
                raise ValueError(
                    f"非法 gold fact_id：{fact_id}"
                )

        return value

    @field_validator("gold_evidence_ids")
    @classmethod
    def validate_gold_evidence_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError(
                "gold_evidence_ids 不能包含重复 ID"
            )

        for evidence_id in value:
            if re.fullmatch(
                _EVIDENCE_ID_PATTERN,
                evidence_id,
            ) is None:
                raise ValueError(
                    f"非法 gold evidence_id：{evidence_id}"
                )

        return value

    @field_validator("gold_calculation_ids")
    @classmethod
    def validate_gold_calculation_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError(
                "gold_calculation_ids 不能包含重复 ID"
            )

        for calculation_id in value:
            if re.fullmatch(
                _CALCULATION_ID_PATTERN,
                calculation_id,
            ) is None:
                raise ValueError(
                    "非法 gold calculation_id："
                    f"{calculation_id}"
                )

        return value

    @field_validator("fiscal_years")
    @classmethod
    def validate_fiscal_years(
        cls,
        value: tuple[int, ...],
    ) -> tuple[int, ...]:
        if any(
            year < 2000 or year > 2100
            for year in value
        ):
            raise ValueError(
                "fiscal_years 必须在 2000 到 2100 之间"
            )

        if len(value) != len(set(value)):
            raise ValueError(
                "fiscal_years 不能包含重复年份"
            )

        return tuple(sorted(value))

    @field_validator(
        "created_at",
        "updated_at",
        "validated_at",
    )
    @classmethod
    def require_timezone(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError(
                "时间字段必须包含时区信息"
            )

        return value

    @model_validator(mode="after")
    def validate_case_contract(self) -> Self:
        company_count = len(self.company_ids)

        if (
            self.question_type
            == "single_company_multi_metric"
            and company_count != 1
        ):
            raise ValueError(
                "single_company_multi_metric "
                "必须且只能包含一家公司"
            )

        if (
            self.question_type
            == "cross_company_comparison"
            and company_count < 2
        ):
            raise ValueError(
                "cross_company_comparison "
                "至少需要两家公司"
            )

        if (
            self.question_type
            == "multi_company_ranking"
            and company_count < 3
        ):
            raise ValueError(
                "multi_company_ranking "
                "至少需要三家公司"
            )

        if (
            self.difficulty == "hard"
            and len(self.gold_fact_ids) < 4
            and company_count < 3
        ):
            raise ValueError(
                "hard 问题至少需要四个事实，"
                "或者至少涉及三家公司"
            )

        company_id_set = set(self.company_ids)
        report_id_set = set(self.report_ids)
        fiscal_year_set = set(self.fiscal_years)
        gold_fact_id_set = set(self.gold_fact_ids)
        gold_evidence_id_set = set(
            self.gold_evidence_ids
        )
        gold_calculation_id_set = set(
            self.gold_calculation_ids
        )

        companies_with_reports: set[str] = set()

        for report_id in self.report_ids:
            matched_company_ids = {
                company_id
                for company_id in company_id_set
                if report_id.startswith(
                    f"{company_id}_"
                )
            }

            if len(matched_company_ids) != 1:
                raise ValueError(
                    f"report_id 无法唯一归属于公司："
                    f"{report_id}"
                )

            companies_with_reports.update(
                matched_company_ids
            )

        if companies_with_reports != company_id_set:
            missing_companies = (
                company_id_set
                - companies_with_reports
            )

            raise ValueError(
                "以下公司没有对应报告："
                f"{sorted(missing_companies)}"
            )

        retrieval_queries = (
            self.gold_rewrite.retrieval_queries
        )

        query_fact_ids = {
            query.target_fact_id
            for query in retrieval_queries
        }

        if query_fact_ids != gold_fact_id_set:
            raise ValueError(
                "Gold Rewrite 的 target_fact_id "
                "必须与 gold_fact_ids 完全一致"
            )

        for query in retrieval_queries:
            if query.baseline_query != self.question:
                raise ValueError(
                    f"{query.query_id} 的 baseline_query "
                    "必须等于原始 question"
                )

            if query.company_id not in company_id_set:
                raise ValueError(
                    f"{query.query_id} 引用了未声明的公司"
                )

            if query.report_id not in report_id_set:
                raise ValueError(
                    f"{query.query_id} 引用了未声明的报告"
                )

            if query.fiscal_year not in fiscal_year_set:
                raise ValueError(
                    f"{query.query_id} 引用了未声明的年份"
                )

        plan_retrieved_fact_ids = {
            fact_id
            for step in self.gold_plan.steps
            if step.action == "retrieve"
            for fact_id in step.target_fact_ids
        }

        if plan_retrieved_fact_ids != gold_fact_id_set:
            raise ValueError(
                "Gold Plan 必须检索全部且仅检索 "
                "gold_fact_ids 中的事实"
            )

        plan_calculation_ids = {
            step.calculation_id
            for step in self.gold_plan.steps
            if step.calculation_id is not None
        }

        if (
            plan_calculation_ids
            != gold_calculation_id_set
        ):
            raise ValueError(
                "Gold Plan 的 calculation_id "
                "必须与 gold_calculation_ids 完全一致"
            )

        produced_by: dict[str, str] = {}

        for step in self.gold_plan.steps:
            missing_inputs = {
                input_ref
                for input_ref in step.input_refs
                if input_ref not in produced_by
            }

            if missing_inputs:
                raise ValueError(
                    f"{step.step_id} 使用了尚未生成的输入："
                    f"{sorted(missing_inputs)}"
                )

            required_dependencies = {
                produced_by[input_ref]
                for input_ref in step.input_refs
            }

            missing_dependency_steps = (
                required_dependencies
                - set(step.depends_on)
            )

            if missing_dependency_steps:
                raise ValueError(
                    f"{step.step_id} 的 depends_on "
                    "缺少输入生产步骤："
                    f"{sorted(missing_dependency_steps)}"
                )

            produced_by[
                step.output_ref
            ] = step.step_id

        answer_fact_ids = set(
            self.gold_answer.supporting_fact_ids
        )

        if not answer_fact_ids.issubset(
            gold_fact_id_set
        ):
            raise ValueError(
                "Gold Answer 引用了未声明的 fact_id"
            )

        answer_evidence_ids = set(
            self.gold_answer.evidence_ids
        )

        if not answer_evidence_ids.issubset(
            gold_evidence_id_set
        ):
            raise ValueError(
                "Gold Answer 引用了未声明的 evidence_id"
            )

        answer_calculation_ids = set(
            self.gold_answer.supporting_calculation_ids
        )

        if not answer_calculation_ids.issubset(
            gold_calculation_id_set
        ):
            raise ValueError(
                "Gold Answer 引用了未声明的 "
                "calculation_id"
            )

        if (
            self.validation_status
            is ValidationStatus.VERIFIED
        ):
            if self.validated_by is None:
                raise ValueError(
                    "verified Case 必须填写 validated_by"
                )

            if self.validated_at is None:
                raise ValueError(
                    "verified Case 必须填写 validated_at"
                )
        elif (
            self.validated_by is not None
            or self.validated_at is not None
        ):
            raise ValueError(
                "只有 verified Case 才能填写 "
                "validated_by 和 validated_at"
            )

        if self.updated_at < self.created_at:
            raise ValueError(
                "updated_at 不能早于 created_at"
            )

        return self