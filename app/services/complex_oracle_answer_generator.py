from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.schemas.complex_plan_eval_result import (
    ComplexCalculationTrace,
    ComplexFinalAnswerOutput,
    ComplexPlanOutput,
    ComplexRetrievalTrace,
    ComplexRewriteOutput,
)
from app.schemas.enums import (
    UnitCode,
    ValidationStatus,
)
from app.schemas.financial_fact import (
    FinancialFact,
)
from app.services.registry import RegistryBundle


class ComplexOracleAnswerGeneratorError(
    ValueError
):
    """复杂问题确定性答案生成失败。"""


_UNIT_SUFFIXES = {
    UnitCode.CNY.value: "元",
    UnitCode.CNY_THOUSAND.value: "千元",
    UnitCode.CNY_TEN_THOUSAND.value: "万元",
    UnitCode.CNY_MILLION.value: "百万元",
    UnitCode.CNY_HUNDRED_MILLION.value: "亿元",
    UnitCode.PERCENT.value: "%",
    UnitCode.PERCENTAGE_POINT.value: "个百分点",
    UnitCode.RATIO.value: "",
    UnitCode.CNY_PER_SHARE.value: "元/股",
    UnitCode.COUNT.value: "",
    UnitCode.TEXT.value: "",
}

_SCOPE_LABELS = {
    "consolidated": "合并口径",
    "parent_company": "母公司口径",
}


@dataclass(slots=True)
class ComplexOracleAnswerGenerator:
    """只根据实际执行结果生成确定性答案。"""

    registry_bundle: RegistryBundle

    generator_id: str = (
        "deterministic_financial_"
        "answer_generator_v1"
    )

    def __post_init__(self) -> None:
        normalized_generator_id = (
            self.generator_id.strip()
        )

        if not normalized_generator_id:
            raise (
                ComplexOracleAnswerGeneratorError(
                    "generator_id 不能为空"
                )
            )

        self.generator_id = (
            normalized_generator_id
        )

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
        """根据实际 Fact、Evidence 和计算结果生成答案。"""

        if not question.strip():
            raise (
                ComplexOracleAnswerGeneratorError(
                    "question 不能为空"
                )
            )

        facts = self._load_retrieved_facts(
            rewrite=rewrite,
            retrieval_traces=(
                retrieval_traces
            ),
        )

        citation_evidence_ids = (
            self._load_citation_evidence_ids(
                facts=facts,
            )
        )

        self._validate_calculations(
            facts=facts,
            calculation_traces=(
                calculation_traces
            ),
        )

        actions = {
            step.action
            for step in plan.steps
        }

        if calculation_traces:
            if not actions.intersection(
                {
                    "calculate",
                    "normalize_unit",
                }
            ):
                raise (
                    ComplexOracleAnswerGeneratorError(
                        "存在 Calculation Trace，"
                        "但 Plan 中没有计算步骤"
                    )
                )

            answer_text = (
                self._render_calculation_answer(
                    facts=facts,
                    calculation_traces=(
                        calculation_traces
                    ),
                )
            )

        elif "rank" in actions:
            answer_text = (
                self._render_ranking_answer(
                    facts=facts,
                )
            )

        elif "compare" in actions:
            answer_text = (
                self._render_comparison_answer(
                    facts=facts,
                )
            )

        else:
            answer_text = (
                self._render_synthesis_answer(
                    facts=facts,
                )
            )

        return ComplexFinalAnswerOutput(
            answer_text=answer_text,
            supporting_fact_ids=tuple(
                fact.fact_id
                for fact in facts
            ),
            supporting_calculation_ids=tuple(
                trace.calculation_id
                for trace
                in calculation_traces
            ),
            citation_evidence_ids=(
                citation_evidence_ids
            ),
        )

    def _load_retrieved_facts(
        self,
        *,
        rewrite: ComplexRewriteOutput,
        retrieval_traces: tuple[
            ComplexRetrievalTrace,
            ...,
        ],
    ) -> tuple[FinancialFact, ...]:
        if not retrieval_traces:
            raise (
                ComplexOracleAnswerGeneratorError(
                    "没有 Retrieval Trace"
                )
            )

        query_by_id = {
            query.query_id: query
            for query
            in rewrite.retrieval_queries
        }

        trace_query_ids = {
            trace.query_id
            for trace
            in retrieval_traces
        }

        if trace_query_ids != set(
            query_by_id
        ):
            raise (
                ComplexOracleAnswerGeneratorError(
                    "Rewrite Query 与 "
                    "Retrieval Trace 不一致"
                )
            )

        facts: list[FinancialFact] = []
        seen_fact_ids: set[str] = set()

        for trace in retrieval_traces:
            if trace.status != "completed":
                raise (
                    ComplexOracleAnswerGeneratorError(
                        f"{trace.query_id} "
                        "检索尚未完成"
                    )
                )

            if not trace.retrieved_fact_ids:
                raise (
                    ComplexOracleAnswerGeneratorError(
                        f"{trace.query_id} "
                        "没有检索到 Fact"
                    )
                )

            query = query_by_id[
                trace.query_id
            ]

            for fact_id in (
                trace.retrieved_fact_ids
            ):
                fact = (
                    self.registry_bundle
                    .financial_facts
                    .get(fact_id)
                )

                if fact is None:
                    raise (
                        ComplexOracleAnswerGeneratorError(
                            "Retrieval Trace 引用了"
                            "不存在的 Fact："
                            f"{fact_id}"
                        )
                    )

                if (
                    fact.validation_status
                    is not
                    ValidationStatus.VERIFIED
                ):
                    raise (
                        ComplexOracleAnswerGeneratorError(
                            "答案不能使用未 verified "
                            f"的 Fact：{fact_id}"
                        )
                    )

                self._validate_fact_identity(
                    fact=fact,
                    query=query,
                )

                if (
                    fact.primary_evidence_id
                    not in
                    trace.retrieved_evidence_ids
                ):
                    raise (
                        ComplexOracleAnswerGeneratorError(
                            f"Fact '{fact_id}' "
                            "没有检索到其主证据："
                            f"{fact.primary_evidence_id}"
                        )
                    )

                if fact_id in seen_fact_ids:
                    continue

                seen_fact_ids.add(fact_id)
                facts.append(fact)

        if not facts:
            raise (
                ComplexOracleAnswerGeneratorError(
                    "没有可用于回答的 Fact"
                )
            )

        return tuple(facts)

    def _validate_fact_identity(
        self,
        *,
        fact: FinancialFact,
        query,
    ) -> None:
        identity_matches = (
            fact.company_id
            == query.company_id
            and fact.report_id
            == query.report_id
            and fact.metric_id
            == query.metric_id
            and fact.fiscal_year
            == query.fiscal_year
            and fact.statement_type
            is query.statement_type
            and fact.statement_scope
            is query.statement_scope
        )

        if not identity_matches:
            raise (
                ComplexOracleAnswerGeneratorError(
                    "实际检索 Fact 与 Query "
                    "结构化身份不一致："
                    f"query_id={query.query_id}, "
                    f"fact_id={fact.fact_id}"
                )
            )

    def _load_citation_evidence_ids(
        self,
        *,
        facts: tuple[
            FinancialFact,
            ...,
        ],
    ) -> tuple[str, ...]:
        evidence_ids = tuple(
            dict.fromkeys(
                fact.primary_evidence_id
                for fact in facts
            )
        )

        for evidence_id in evidence_ids:
            evidence = (
                self.registry_bundle
                .evidences
                .get(evidence_id)
            )

            if evidence is None:
                raise (
                    ComplexOracleAnswerGeneratorError(
                        "Fact 引用了不存在的 "
                        f"Evidence：{evidence_id}"
                    )
                )

            if (
                evidence.validation_status
                is not
                ValidationStatus.VERIFIED
            ):
                raise (
                    ComplexOracleAnswerGeneratorError(
                        "答案不能引用未 verified "
                        f"的 Evidence：{evidence_id}"
                    )
                )

        return evidence_ids

    def _validate_calculations(
        self,
        *,
        facts: tuple[
            FinancialFact,
            ...,
        ],
        calculation_traces: tuple[
            ComplexCalculationTrace,
            ...,
        ],
    ) -> None:
        retrieved_fact_ids = {
            fact.fact_id
            for fact in facts
        }

        for trace in calculation_traces:
            if trace.status != "completed":
                raise (
                    ComplexOracleAnswerGeneratorError(
                        "答案不能使用未完成的 "
                        "Calculation："
                        f"{trace.calculation_id}"
                    )
                )

            missing_input_fact_ids = (
                set(trace.input_fact_ids)
                - retrieved_fact_ids
            )

            if missing_input_fact_ids:
                raise (
                    ComplexOracleAnswerGeneratorError(
                        "Calculation 使用了未检索"
                        "到的 Fact："
                        f"{sorted(missing_input_fact_ids)}"
                    )
                )

            if not (
                self.registry_bundle
                .metrics
                .contains(trace.metric_id)
            ):
                raise (
                    ComplexOracleAnswerGeneratorError(
                        "Calculation 引用了不存在"
                        "的 Metric："
                        f"{trace.metric_id}"
                    )
                )

    def _render_synthesis_answer(
        self,
        *,
        facts: tuple[
            FinancialFact,
            ...,
        ],
    ) -> str:
        company_id = self._require_same(
            [
                fact.company_id
                for fact in facts
            ],
            "company_id",
        )

        fiscal_year = self._require_same(
            [
                fact.fiscal_year
                for fact in facts
            ],
            "fiscal_year",
        )

        scope = self._require_same(
            [
                fact.statement_scope
                for fact in facts
            ],
            "statement_scope",
        )

        prefix = (
            f"{self._company_name(company_id)}"
            f"{fiscal_year}年"
            f"{self._scope_label(scope)}"
        )

        clauses = [
            (
                f"{self._metric_name(fact.metric_id)}"
                f"为{self._format_fact_value(fact)}"
            )
            for fact in facts
        ]

        return prefix + "，".join(clauses) + "。"

    def _render_calculation_answer(
        self,
        *,
        facts: tuple[
            FinancialFact,
            ...,
        ],
        calculation_traces: tuple[
            ComplexCalculationTrace,
            ...,
        ],
    ) -> str:
        if len(calculation_traces) != 1:
            raise (
                ComplexOracleAnswerGeneratorError(
                    "当前答案生成器只支持一次"
                    "计算对应一个最终答案"
                )
            )

        calculation = (
            calculation_traces[0]
        )

        if set(
            calculation.input_fact_ids
        ) != {
            fact.fact_id
            for fact in facts
        }:
            raise (
                ComplexOracleAnswerGeneratorError(
                    "计算输入 Fact 与实际检索 "
                    "Fact 不完全一致"
                )
            )

        fact_by_id = {
            fact.fact_id: fact
            for fact in facts
        }

        input_facts = tuple(
            fact_by_id[fact_id]
            for fact_id
            in calculation.input_fact_ids
        )

        company_id = self._require_same(
            [
                fact.company_id
                for fact in input_facts
            ],
            "company_id",
        )

        fiscal_year = self._require_same(
            [
                fact.fiscal_year
                for fact in input_facts
            ],
            "fiscal_year",
        )

        scope = self._require_same(
            [
                fact.statement_scope
                for fact in input_facts
            ],
            "statement_scope",
        )

        input_clauses = [
            (
                f"{self._metric_name(fact.metric_id)}"
                f"为{self._format_fact_value(fact)}"
            )
            for fact in input_facts
        ]

        result_value_text = (
            self._format_quantity(
                calculation.result_value,
                calculation.result_unit,
            )
        )

        result_clause = (
            f"{self._metric_name(calculation.metric_id)}"
            f"为{result_value_text}"
        )

        prefix = (
            f"{self._company_name(company_id)}"
            f"{fiscal_year}年"
            f"{self._scope_label(scope)}"
        )

        return (
            prefix
            + "，".join(
                [
                    *input_clauses,
                    result_clause,
                ]
            )
            + "。"
        )

    def _render_comparison_answer(
        self,
        *,
        facts: tuple[
            FinancialFact,
            ...,
        ],
    ) -> str:
        if len(facts) != 2:
            raise (
                ComplexOracleAnswerGeneratorError(
                    "当前 compare 答案要求"
                    "恰好两个 Fact"
                )
            )

        self._require_same(
            [
                fact.metric_id
                for fact in facts
            ],
            "metric_id",
        )

        fiscal_year = self._require_same(
            [
                fact.fiscal_year
                for fact in facts
            ],
            "fiscal_year",
        )

        scope = self._require_same(
            [
                fact.statement_scope
                for fact in facts
            ],
            "statement_scope",
        )

        self._require_same(
            [
                fact.normalized_unit
                for fact in facts
            ],
            "normalized_unit",
        )

        first, second = facts

        first_clause = (
            f"{self._company_name(first.company_id)}"
            f"{fiscal_year}年"
            f"{self._scope_label(scope)}"
            f"{self._metric_name(first.metric_id)}"
            f"为{self._format_fact_value(first)}"
        )

        second_clause = (
            f"{self._company_name(second.company_id)}"
            f"为{self._format_fact_value(second)}"
        )

        if (
            first.normalized_value
            > second.normalized_value
        ):
            conclusion = (
                f"{self._company_name(first.company_id)}"
                "更高"
            )
        elif (
            second.normalized_value
            > first.normalized_value
        ):
            conclusion = (
                f"{self._company_name(second.company_id)}"
                "更高"
            )
        else:
            conclusion = "两家公司相同"

        return (
            f"{first_clause}，"
            f"{second_clause}；"
            f"{conclusion}。"
        )

    def _render_ranking_answer(
        self,
        *,
        facts: tuple[
            FinancialFact,
            ...,
        ],
    ) -> str:
        if len(facts) < 2:
            raise (
                ComplexOracleAnswerGeneratorError(
                    "rank 至少需要两个 Fact"
                )
            )

        metric_id = self._require_same(
            [
                fact.metric_id
                for fact in facts
            ],
            "metric_id",
        )

        fiscal_year = self._require_same(
            [
                fact.fiscal_year
                for fact in facts
            ],
            "fiscal_year",
        )

        scope = self._require_same(
            [
                fact.statement_scope
                for fact in facts
            ],
            "statement_scope",
        )

        self._require_same(
            [
                fact.normalized_unit
                for fact in facts
            ],
            "normalized_unit",
        )

        ranked_facts = sorted(
            facts,
            key=lambda fact: (
                fact.normalized_value
            ),
            reverse=True,
        )

        ranking_items = [
            (
                f"{rank}. "
                f"{self._company_name(fact.company_id)}"
                f"{self._format_fact_value(fact)}"
            )
            for rank, fact
            in enumerate(
                ranked_facts,
                start=1,
            )
        ]

        return (
            f"按{fiscal_year}年"
            f"{self._scope_label(scope)}"
            f"{self._metric_name(metric_id)}"
            "从高到低排序："
            + "；".join(ranking_items)
            + "。"
        )

    def _company_name(
        self,
        company_id: str,
    ) -> str:
        company = (
            self.registry_bundle
            .companies
            .get(company_id)
        )

        if company is None:
            raise (
                ComplexOracleAnswerGeneratorError(
                    "不存在的 Company："
                    f"{company_id}"
                )
            )

        return company.short_name_cn

    def _metric_name(
        self,
        metric_id: str,
    ) -> str:
        metric = (
            self.registry_bundle
            .metrics
            .get(metric_id)
        )

        if metric is None:
            raise (
                ComplexOracleAnswerGeneratorError(
                    "不存在的 Metric："
                    f"{metric_id}"
                )
            )

        return metric.display_name_cn

    def _scope_label(
        self,
        scope,
    ) -> str:
        scope_value = getattr(
            scope,
            "value",
            str(scope),
        )

        return _SCOPE_LABELS.get(
            scope_value,
            f"{scope_value}口径",
        )

    def _format_fact_value(
        self,
        fact: FinancialFact,
    ) -> str:
        return self._format_quantity(
            fact.normalized_value,
            fact.normalized_unit,
        )

    def _format_quantity(
        self,
        value: Decimal | None,
        unit,
    ) -> str:
        if value is None:
            raise (
                ComplexOracleAnswerGeneratorError(
                    "待格式化数值不能为空"
                )
            )

        unit_value = getattr(
            unit,
            "value",
            str(unit),
        )

        suffix = _UNIT_SUFFIXES.get(
            unit_value,
        )

        formatted_value = format(
            value,
            ",f",
        )

        if suffix is None:
            return (
                f"{formatted_value} "
                f"{unit_value}"
            )

        return formatted_value + suffix

    def _require_same(
        self,
        values,
        field_name: str,
    ):
        normalized_values = tuple(values)

        if not normalized_values:
            raise (
                ComplexOracleAnswerGeneratorError(
                    f"{field_name} 不能为空"
                )
            )

        first_value = normalized_values[0]

        if any(
            value != first_value
            for value in normalized_values[1:]
        ):
            raise (
                ComplexOracleAnswerGeneratorError(
                    "答案所依赖的 Fact 具有"
                    f"不一致的 {field_name}"
                )
            )

        return first_value