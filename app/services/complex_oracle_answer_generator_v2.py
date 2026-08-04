from __future__ import annotations

from dataclasses import dataclass

from app.schemas.complex_plan_eval_result import (
    ComplexCalculationTrace,
    ComplexFinalAnswerOutput,
    ComplexPlanOutput,
    ComplexRetrievalTrace,
    ComplexRewriteOutput,
)
from app.schemas.financial_fact import (
    FinancialFact,
)
from app.services.complex_oracle_answer_generator import (
    ComplexOracleAnswerGenerator,
    ComplexOracleAnswerGeneratorError,
)


@dataclass(slots=True)
class ComplexOracleAnswerGeneratorV2(
    ComplexOracleAnswerGenerator
):
    """
    支持多指标比较、多计算比较及计算结果排名的
    确定性答案生成器。
    """

    generator_id: str = (
        "deterministic_financial_"
        "answer_generator_v2"
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
        if not question.strip():
            raise (
                ComplexOracleAnswerGeneratorError(
                    "question 不能为空"
                )
            )

        facts = self._load_retrieved_facts(
            rewrite=rewrite,
            retrieval_traces=retrieval_traces,
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

            if "rank" in actions:
                answer_text = (
                    self
                    ._render_calculation_ranking_answer(
                        facts=facts,
                        calculation_traces=(
                            calculation_traces
                        ),
                    )
                )

            elif "compare" in actions:
                answer_text = (
                    self
                    ._render_calculation_comparison_answer(
                        facts=facts,
                        calculation_traces=(
                            calculation_traces
                        ),
                    )
                )

            elif len(calculation_traces) == 1:
                answer_text = (
                    ComplexOracleAnswerGenerator
                    ._render_calculation_answer(
                        self,
                        facts=facts,
                        calculation_traces=(
                            calculation_traces
                        ),
                    )
                )

            else:
                answer_text = (
                    self
                    ._render_multiple_calculation_synthesis_answer(
                        facts=facts,
                        calculation_traces=(
                            calculation_traces
                        ),
                    )
                )

        elif "rank" in actions:
            answer_text = (
                self._render_fact_ranking_answer(
                    facts=facts,
                )
            )

        elif "compare" in actions:
            answer_text = (
                self
                ._render_enhanced_comparison_answer(
                    facts=facts,
                )
            )

        else:
            answer_text = (
                ComplexOracleAnswerGenerator
                ._render_synthesis_answer(
                    self,
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

    def _render_enhanced_comparison_answer(
        self,
        *,
        facts: tuple[
            FinancialFact,
            ...,
        ],
    ) -> str:
        company_ids = tuple(
            dict.fromkeys(
                fact.company_id
                for fact in facts
            )
        )

        metric_ids = tuple(
            dict.fromkeys(
                fact.metric_id
                for fact in facts
            )
        )

        # 原有能力：两家公司比较同一个指标
        if (
            len(facts) == 2
            and len(metric_ids) == 1
        ):
            return (
                ComplexOracleAnswerGenerator
                ._render_comparison_answer(
                    self,
                    facts=facts,
                )
            )

        # 新能力：同一家公司比较多个指标
        if len(company_ids) == 1:
            return (
                self
                ._render_same_company_comparison(
                    facts=facts,
                )
            )

        # 新能力：两家公司同时比较多个指标
        if (
            len(company_ids) == 2
            and len(metric_ids) >= 2
        ):
            return (
                self
                ._render_multi_metric_company_comparison(
                    facts=facts,
                    company_ids=company_ids,
                    metric_ids=metric_ids,
                )
            )

        raise ComplexOracleAnswerGeneratorError(
            "不支持当前 Fact 组合的比较答案"
        )

    def _render_same_company_comparison(
        self,
        *,
        facts: tuple[
            FinancialFact,
            ...,
        ],
    ) -> str:
        self._require_same(
            [
                fact.normalized_unit
                for fact in facts
            ],
            "normalized_unit",
        )

        prefix = self._build_period_prefix(
            facts=facts,
        )

        cash_flow_metric_ids = {
            (
                "net_cash_flow_from_"
                "operating_activities"
            ),
            (
                "net_cash_flow_from_"
                "investing_activities"
            ),
            (
                "net_cash_flow_from_"
                "financing_activities"
            ),
        }

        if all(
            fact.metric_id
            in cash_flow_metric_ids
            for fact in facts
        ):
            clauses = []

            for fact in facts:
                if fact.normalized_value > 0:
                    direction = "净流入"
                elif fact.normalized_value < 0:
                    direction = "净流出"
                else:
                    direction = (
                        "无净流入或净流出"
                    )

                clauses.append(
                    f"{self._metric_name(fact.metric_id)}"
                    f"为{self._format_fact_value(fact)}，"
                    f"为{direction}"
                )

            return (
                prefix
                + "；".join(clauses)
                + "。"
            )

        clauses = [
            (
                f"{self._metric_name(fact.metric_id)}"
                f"为{self._format_fact_value(fact)}"
            )
            for fact in facts
        ]

        ranked_facts = sorted(
            facts,
            key=lambda fact: (
                fact.normalized_value
            ),
            reverse=True,
        )

        if (
            ranked_facts[0].normalized_value
            == ranked_facts[1].normalized_value
        ):
            conclusion = "两项金额相同"
        else:
            highest_metric_name = (
                self._metric_name(
                    ranked_facts[0].metric_id
                )
            )

            conclusion = (
                highest_metric_name
                + "更高"
            )

        return (
            prefix
            + "，".join(clauses)
            + f"；{conclusion}。"
        )

    def _render_multi_metric_company_comparison(
        self,
        *,
        facts: tuple[
            FinancialFact,
            ...,
        ],
        company_ids: tuple[str, ...],
        metric_ids: tuple[str, ...],
    ) -> str:
        answers: list[str] = []

        for metric_id in metric_ids:
            metric_facts = tuple(
                fact
                for fact in facts
                if fact.metric_id == metric_id
            )

            if (
                len(metric_facts) != 2
                or {
                    fact.company_id
                    for fact in metric_facts
                }
                != set(company_ids)
            ):
                raise (
                    ComplexOracleAnswerGeneratorError(
                        "多指标跨公司比较要求"
                        "每个指标各有两家公司 Fact"
                    )
                )

            answers.append(
                ComplexOracleAnswerGenerator
                ._render_comparison_answer(
                    self,
                    facts=metric_facts,
                )
            )

        return "".join(answers)

    def _render_multiple_calculation_synthesis_answer(
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

        used_fact_ids = {
            fact_id
            for trace in calculation_traces
            for fact_id in trace.input_fact_ids
        }

        actual_fact_ids = {
            fact.fact_id
            for fact in facts
        }

        if used_fact_ids != actual_fact_ids:
            raise (
                ComplexOracleAnswerGeneratorError(
                    "多计算汇总的输入 Fact "
                    "与检索 Fact 不一致"
                )
            )

        expected_context = (
            company_id,
            fiscal_year,
            scope,
        )

        for trace in calculation_traces:
            if (
                self._calculation_context(
                    facts=facts,
                    calculation=trace,
                )
                != expected_context
            ):
                raise (
                    ComplexOracleAnswerGeneratorError(
                        "多计算汇总包含不一致的"
                        "公司、年份或口径"
                    )
                )

        prefix = (
            f"{self._company_name(company_id)}"
            f"{fiscal_year}年"
            f"{self._scope_label(scope)}"
        )

        fact_clauses = [
            (
                f"{self._metric_name(fact.metric_id)}"
                f"为{self._format_fact_value(fact)}"
            )
            for fact in facts
        ]

        calculation_clauses = [
            (
                self._metric_name(
                    trace.metric_id
                )
                + "为"
                + self._format_quantity(
                    trace.result_value,
                    trace.result_unit,
                )
            )
            for trace in calculation_traces
        ]

        return (
            prefix
            + "，".join(fact_clauses)
            + "；"
            + "，".join(calculation_clauses)
            + "。"
        )

    def _render_calculation_comparison_answer(
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
        if len(calculation_traces) != 2:
            raise (
                ComplexOracleAnswerGeneratorError(
                    "计算结果比较要求恰好两个 "
                    "Calculation Trace"
                )
            )

        metric_id = self._require_same(
            [
                trace.metric_id
                for trace
                in calculation_traces
            ],
            "calculation.metric_id",
        )

        result_unit = self._require_same(
            [
                trace.result_unit
                for trace
                in calculation_traces
            ],
            "calculation.result_unit",
        )

        contexts = [
            self._calculation_context(
                facts=facts,
                calculation=trace,
            )
            for trace
            in calculation_traces
        ]

        fiscal_year = self._require_same(
            [
                context[1]
                for context in contexts
            ],
            "fiscal_year",
        )

        scope = self._require_same(
            [
                context[2]
                for context in contexts
            ],
            "statement_scope",
        )

        company_ids = [
            context[0]
            for context in contexts
        ]

        if (
            len(company_ids)
            != len(set(company_ids))
        ):
            raise (
                ComplexOracleAnswerGeneratorError(
                    "计算结果比较要求来自"
                    "不同公司"
                )
            )

        first, second = calculation_traces
        first_company_id = company_ids[0]
        second_company_id = company_ids[1]

        first_value = self._format_quantity(
            first.result_value,
            result_unit,
        )

        second_value = self._format_quantity(
            second.result_value,
            result_unit,
        )

        if (
            first.result_value
            > second.result_value
        ):
            conclusion = (
                self._company_name(
                    first_company_id
                )
                + "更高"
            )
        elif (
            second.result_value
            > first.result_value
        ):
            conclusion = (
                self._company_name(
                    second_company_id
                )
                + "更高"
            )
        else:
            conclusion = "两家公司相同"

        return (
            f"{self._company_name(first_company_id)}"
            f"{fiscal_year}年"
            f"{self._scope_label(scope)}"
            f"{self._metric_name(metric_id)}"
            f"为{first_value}，"
            f"{self._company_name(second_company_id)}"
            f"为{second_value}；"
            f"{conclusion}。"
        )

    def _render_calculation_ranking_answer(
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
        if len(calculation_traces) < 2:
            raise (
                ComplexOracleAnswerGeneratorError(
                    "计算结果排名至少需要两个 "
                    "Calculation Trace"
                )
            )

        metric_id = self._require_same(
            [
                trace.metric_id
                for trace
                in calculation_traces
            ],
            "calculation.metric_id",
        )

        result_unit = self._require_same(
            [
                trace.result_unit
                for trace
                in calculation_traces
            ],
            "calculation.result_unit",
        )

        contexts = [
            self._calculation_context(
                facts=facts,
                calculation=trace,
            )
            for trace
            in calculation_traces
        ]

        fiscal_year = self._require_same(
            [
                context[1]
                for context in contexts
            ],
            "fiscal_year",
        )

        scope = self._require_same(
            [
                context[2]
                for context in contexts
            ],
            "statement_scope",
        )

        company_ids = [
            context[0]
            for context in contexts
        ]

        if (
            len(company_ids)
            != len(set(company_ids))
        ):
            raise (
                ComplexOracleAnswerGeneratorError(
                    "计算结果排名包含重复公司"
                )
            )

        records = [
            (
                contexts[index][0],
                trace,
            )
            for index, trace
            in enumerate(calculation_traces)
        ]

        ranked_records = sorted(
            records,
            key=lambda record: (
                record[1].result_value
            ),
            reverse=True,
        )

        ranking_items = [
            (
                f"{rank}. "
                + self._company_name(
                    company_id
                )
                + self._format_quantity(
                    trace.result_value,
                    result_unit,
                )
            )
            for rank, (
                company_id,
                trace,
            )
            in enumerate(
                ranked_records,
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

    def _render_fact_ranking_answer(
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

        period = (
            f"{fiscal_year}年末"
            if all(
                fact.statement_type.value
                == "balance_sheet"
                for fact in facts
            )
            else f"{fiscal_year}年"
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
            f"按{period}"
            f"{self._scope_label(scope)}"
            f"{self._metric_name(metric_id)}"
            "从高到低排序："
            + "；".join(ranking_items)
            + "。"
        )

    def _calculation_context(
        self,
        *,
        facts: tuple[
            FinancialFact,
            ...,
        ],
        calculation: ComplexCalculationTrace,
    ):
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

        return (
            company_id,
            fiscal_year,
            scope,
        )

    def _build_period_prefix(
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

        period = (
            f"{fiscal_year}年末"
            if all(
                fact.statement_type.value
                == "balance_sheet"
                for fact in facts
            )
            else f"{fiscal_year}年"
        )

        return (
            f"{self._company_name(company_id)}"
            f"{period}"
            f"{self._scope_label(scope)}"
        )