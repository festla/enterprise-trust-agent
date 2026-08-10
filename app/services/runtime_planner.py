from __future__ import annotations

from dataclasses import dataclass

from app.schemas.agent_runtime import (
    AgentIntent,
    ParsedFinancialQuery,
    RuntimePlan,
)
from app.schemas.complex_plan_eval_result import (
    ComplexPlanOutput,
    ComplexPlanStepOutput,
    ComplexRetrievalQueryOutput,
)
from app.schemas.enums import (
    MetricOrigin,
    StatementScope,
)
from app.schemas.tool_registry import (
    DocumentEvidenceQuery,
)
from app.services.registry import (
    RegistryBundle,
)


class RuntimePlannerError(
    ValueError
):
    """Runtime Planner 基础异常。"""


# ============================================================
# Formula 决定“计算需要哪些输入 Metric”。
#
# 注意这里保存的是：
#
#     metric_id
#
# 而不是：
#
#     fact_id
#
# 因为 Planner 只能规划“要找什么”，
# 不能提前知道真正会检索到哪个 Fact。
#
#
# 例如：
#
# gross_profit_margin_formula
#
#          ↓
#
# revenue
# operating_cost
#
#          ↓
#
# Runtime 执行时才真正得到：
#
# fact_xxx_revenue
# fact_xxx_operating_cost
#
# ============================================================


_FORMULA_INPUT_METRIC_IDS = {
    "gross_profit_margin_formula": (
        "revenue",
        "operating_cost",
    ),

    (
        "selling_and_r_and_d_"
        "expense_ratio_formula"
    ): (
        "revenue",
        "selling_expenses",
        (
            "research_and_"
            "development_expenses"
        ),
    ),

    (
        "operating_cash_flow_to_"
        "net_profit_ratio_formula"
    ): (
        (
            "net_cash_flow_from_"
            "operating_activities"
        ),
        "net_profit",
    ),

    "current_ratio_formula": (
        "current_assets",
        "current_liabilities",
    ),

    "debt_to_equity_ratio_formula": (
        "total_liabilities",
        "total_equity",
    ),

    (
        "effective_income_tax_rate_formula"
    ): (
        "income_tax_expense",
        "total_profit",
    ),
}


# ============================================================
# 保持原顺序去重。
#
# Calculation 的 input_refs / depends_on
# 都不能因为 set 而破坏原始公式顺序。
# ============================================================


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
# Parser：
#
#   用户说了什么？
#
# Router：
#
#   这是哪种任务？
#
# Planner：
#
#   具体生成哪些 Query？
#   哪些 Step？
#   每个 Step 调哪个 Tool？
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class RuntimePlanner:
    registry_bundle: RegistryBundle

    planner_version: str = (
        "deterministic_runtime_planner_v1"
    )

    def create_plan(
        self,
        *,
        parsed_query: ParsedFinancialQuery,
        intent: AgentIntent,
    ) -> RuntimePlan:
        self._validate_ready_to_plan(
            parsed_query=parsed_query,
            intent=intent,
        )

        if intent == "financial_fact":
            return (
                self
                ._build_financial_fact_plan(
                    parsed_query
                )
            )

        if (
            intent
            == "financial_calculation"
        ):
            return (
                self
                ._build_financial_calculation_plan(
                    parsed_query
                )
            )

        # ============================================================
        # Comparison / Ranking 并不是新的 Tool。
        #
        # 它们是“多个已有结果之间的 Runtime 操作”。
        #
        # 比较对象可能来自：
        #
        # query_financial_data
        #
        # 也可能来自：
        #
        # execute_calculation
        # ============================================================

        if intent == "financial_comparison":
            return (
                self
                ._build_financial_comparison_plan(
                    parsed_query
                )
            )

        if intent == "document_evidence":
            return (
                self
                ._build_document_evidence_plan(
                    parsed_query
                )
            )

        # ====================================================
        # 异常兜底了，目前会走到这里了 7C-3
        # ====================================================

        raise RuntimePlannerError(
            "当前 Planner 子阶段尚未实现 "
            f"intent：{intent}"
        )

    # ========================================================
    # Missing != Unsupported
    #
    # Router 可以识别任务类型，
    # 但 Planner 必须确认已经有足够的信息，
    # 才允许生成真正可执行的 RuntimePlan。
    # ========================================================

    @staticmethod
    def _validate_ready_to_plan(
        *,
        parsed_query: ParsedFinancialQuery,
        intent: AgentIntent,
    ) -> None:
        if intent == "unsupported":
            raise RuntimePlannerError(
                "unsupported intent "
                "不能生成 RuntimePlan"
            )

        if (
            parsed_query
            .unsupported_reason
            is not None
        ):
            raise RuntimePlannerError(
                "被标记为 unsupported 的问题"
                "不能生成 RuntimePlan"
            )

        required_identity_fields = {
            "company_ids",
            "years",
            "report_ids",
        }

        blocking_missing_fields = tuple(
            field_name
            for field_name
            in parsed_query.missing_fields
            if field_name
            in required_identity_fields
        )

        if blocking_missing_fields:
            raise RuntimePlannerError(
                "生成执行计划所需身份字段缺失："
                f"{blocking_missing_fields}"
            )

        if not parsed_query.report_ids:
            raise RuntimePlannerError(
                "生成执行计划至少需要一个 report_id"
            )

    # ========================================================
    # financial_fact
    # ========================================================

    def _build_financial_fact_plan(
        self,
        parsed_query: ParsedFinancialQuery,
    ) -> RuntimePlan:
        if not parsed_query.metric_ids:
            raise RuntimePlannerError(
                "financial_fact "
                "必须至少包含一个 reported metric"
            )

        if (
            parsed_query
            .calculation_metric_ids
        ):
            raise RuntimePlannerError(
                "financial_fact 计划不能包含 "
                "calculation_metric_ids"
            )

        financial_queries: list[
            ComplexRetrievalQueryOutput
        ] = []

        steps: list[
            ComplexPlanStepOutput
        ] = []

        tool_by_step_id: dict[
            str,
            str,
        ] = {}

        # ====================================================
        # key:
        #
        # (
        #   report_id,
        #   metric_id,
        #   statement_scope,
        # )
        #
        # value:
        #
        # (
        #   output_ref,
        #   producer_step_id,
        # )
        #
        # 后面 Calculation Planner 会依靠这个结构
        # 避免重复 Retrieval。
        # ====================================================

        retrieval_runtime_by_key: dict[
            tuple[str, str, str],
            tuple[str, str],
        ] = {}

        result_refs: list[str] = []
        producer_step_ids: list[str] = []

        for report_id in (
            parsed_query.report_ids
        ):
            report = (
                self.registry_bundle
                .reports
                .require(report_id)
            )

            self._validate_report_identity(
                parsed_query=parsed_query,
                report_id=report_id,
                company_id=(
                    report.company_id
                ),
                fiscal_year=(
                    report.fiscal_year
                ),
            )

            for metric_id in (
                parsed_query.metric_ids
            ):
                metric = (
                    self.registry_bundle
                    .metrics
                    .require(metric_id)
                )

                if (
                    metric.metric_origin
                    is not MetricOrigin.REPORTED
                ):
                    raise RuntimePlannerError(
                        "financial_fact "
                        "只能直接查询 reported metric："
                        f"{metric_id}"
                    )

                statement_scope = (
                    self._resolve_statement_scope(
                        parsed_query=(
                            parsed_query
                        ),
                        metric_id=metric_id,
                        allowed_scopes=tuple(
                            metric.allowed_scopes
                        ),
                    )
                )

                (
                    output_ref,
                    step_id,
                ) = (
                    self
                    ._ensure_financial_retrieval(
                        report_id=(
                            report_id
                        ),
                        metric_id=(
                            metric_id
                        ),
                        statement_scope=(
                            statement_scope
                        ),
                        financial_queries=(
                            financial_queries
                        ),
                        steps=steps,
                        tool_by_step_id=(
                            tool_by_step_id
                        ),
                        retrieval_runtime_by_key=(
                            retrieval_runtime_by_key
                        ),
                    )
                )

                result_refs.append(
                    output_ref
                )

                producer_step_ids.append(
                    step_id
                )

        if not steps:
            raise RuntimePlannerError(
                "financial_fact "
                "没有生成任何 Retrieval Step"
            )

        if len(result_refs) > 1:
            synthesize_step_id = (
                f"s{len(steps) + 1}"
            )

            steps.append(
                ComplexPlanStepOutput(
                    step_id=(
                        synthesize_step_id
                    ),
                    action="synthesize",
                    description=(
                        "汇总多个财务事实查询结果"
                    ),
                    input_refs=tuple(
                        result_refs
                    ),
                    depends_on=tuple(
                        producer_step_ids
                    ),
                    output_ref=(
                        "synthesized_result"
                    ),
                )
            )

        plan = ComplexPlanOutput(
            steps=tuple(steps),
            final_step_id=(
                steps[-1].step_id
            ),
        )

        return RuntimePlan(
            intent="financial_fact",
            planner_version=(
                self.planner_version
            ),
            normalized_question=(
                parsed_query
                .normalized_question
            ),
            financial_queries=tuple(
                financial_queries
            ),
            document_queries=(),
            plan=plan,
            tool_by_step_id=(
                tool_by_step_id
            ),
        )

    # ========================================================
    # financial_calculation
    #
    # 例如：
    #
    # “美的2024年毛利率是多少？”
    #
    #
    # gross_profit_margin
    #           ↓
    # gross_profit_margin_formula
    #           ↓
    # revenue + operating_cost
    #
    #
    # q1 revenue
    #       ↓
    # s1 query_financial_data
    #
    # q2 operating_cost
    #       ↓
    # s2 query_financial_data
    #
    # s1 + s2
    #       ↓
    # s3 execute_calculation
    # ========================================================

    def _build_financial_calculation_plan(
        self,
        parsed_query: ParsedFinancialQuery,
    ) -> RuntimePlan:
        if not (
            parsed_query
            .calculation_metric_ids
        ):
            raise RuntimePlannerError(
                "financial_calculation "
                "必须至少包含一个 derived metric"
            )

        financial_queries: list[
            ComplexRetrievalQueryOutput
        ] = []

        steps: list[
            ComplexPlanStepOutput
        ] = []

        tool_by_step_id: dict[
            str,
            str,
        ] = {}

        retrieval_runtime_by_key: dict[
            tuple[str, str, str],
            tuple[str, str],
        ] = {}

        # ========================================================
        # Calculation 本身也可以复用。
        #
        # key:
        #
        # report_id
        # calculation_metric_id
        # statement_scope
        #
        # ↓
        #
        # calculation_id
        # producer_step_id
        # ========================================================

        calculation_runtime_by_key: dict[
            tuple[str, str, str],
            tuple[str, str],
        ] = {}

        requested_result_refs: list[
            str
        ] = []

        requested_producer_step_ids: list[
            str
        ] = []

        for report_id in (
            parsed_query.report_ids
        ):
            report = (
                self.registry_bundle
                .reports
                .require(report_id)
            )

            self._validate_report_identity(
                parsed_query=parsed_query,
                report_id=report_id,
                company_id=(
                    report.company_id
                ),
                fiscal_year=(
                    report.fiscal_year
                ),
            )

            # ====================================================
            # 用户可能同时要求：
            #
            # 营业收入 + 毛利率
            #
            # revenue 是最终结果之一，
            # 同时又是毛利率的计算输入。
            # ====================================================

            for metric_id in (
                parsed_query.metric_ids
            ):
                metric = (
                    self.registry_bundle
                    .metrics
                    .require(metric_id)
                )

                if (
                    metric.metric_origin
                    is not MetricOrigin.REPORTED
                ):
                    raise RuntimePlannerError(
                        "metric_ids 只能包含 "
                        "reported metric："
                        f"{metric_id}"
                    )

                statement_scope = (
                    self._resolve_statement_scope(
                        parsed_query=(
                            parsed_query
                        ),
                        metric_id=metric_id,
                        allowed_scopes=tuple(
                            metric.allowed_scopes
                        ),
                    )
                )

                (
                    output_ref,
                    step_id,
                ) = (
                    self
                    ._ensure_financial_retrieval(
                        report_id=report_id,
                        metric_id=metric_id,
                        statement_scope=(
                            statement_scope
                        ),
                        financial_queries=(
                            financial_queries
                        ),
                        steps=steps,
                        tool_by_step_id=(
                            tool_by_step_id
                        ),
                        retrieval_runtime_by_key=(
                            retrieval_runtime_by_key
                        ),
                    )
                )

                requested_result_refs.append(
                    output_ref
                )

                requested_producer_step_ids.append(
                    step_id
                )

            for calculation_metric_id in (
                parsed_query
                .calculation_metric_ids
            ):
                (
                    calculation_ref,
                    calculation_step_id,
                ) = (
                    self
                    ._ensure_financial_calculation(
                        parsed_query=(
                            parsed_query
                        ),
                        report_id=report_id,
                        calculation_metric_id=(
                            calculation_metric_id
                        ),
                        financial_queries=(
                            financial_queries
                        ),
                        steps=steps,
                        tool_by_step_id=(
                            tool_by_step_id
                        ),
                        retrieval_runtime_by_key=(
                            retrieval_runtime_by_key
                        ),
                        calculation_runtime_by_key=(
                            calculation_runtime_by_key
                        ),
                    )
                )

                requested_result_refs.append(
                    calculation_ref
                )

                requested_producer_step_ids.append(
                    calculation_step_id
                )

        if not steps:
            raise RuntimePlannerError(
                "financial_calculation "
                "没有生成任何执行步骤"
            )

        if (
            len(requested_result_refs)
            > 1
        ):
            synthesize_step_id = (
                f"s{len(steps) + 1}"
            )

            steps.append(
                ComplexPlanStepOutput(
                    step_id=(
                        synthesize_step_id
                    ),
                    action="synthesize",
                    description=(
                        "汇总财务查询与计算结果"
                    ),
                    input_refs=tuple(
                        requested_result_refs
                    ),
                    depends_on=(
                        _unique_in_order(
                            tuple(
                                requested_producer_step_ids
                            )
                        )
                    ),
                    output_ref=(
                        "synthesized_result"
                    ),
                )
            )

        plan = ComplexPlanOutput(
            steps=tuple(steps),
            final_step_id=(
                steps[-1].step_id
            ),
        )

        return RuntimePlan(
            intent="financial_calculation",
            planner_version=(
                self.planner_version
            ),
            normalized_question=(
                parsed_query
                .normalized_question
            ),
            financial_queries=tuple(
                financial_queries
            ),
            document_queries=(),
            plan=plan,
            tool_by_step_id=(
                tool_by_step_id
            ),
        )

    # ============================================================
    # financial_comparison
    #
    # 本质不是一种新的数据来源。
    #
    # 它只是：
    #
    # 先得到多个 Financial Result
    #            ↓
    #       compare / rank
    #
    #
    # Result 可以来自：
    #
    # Retrieval
    # 或
    # Calculation
    # ============================================================

    def _build_financial_comparison_plan(
        self,
        parsed_query: ParsedFinancialQuery,
    ) -> RuntimePlan:
        if not (
            parsed_query.comparison_requested
            or parsed_query.ranking_requested
        ):
            raise RuntimePlannerError(
                "financial_comparison "
                "必须包含 comparison_requested "
                "或 ranking_requested"
            )

        target_metric_count = (
            len(parsed_query.metric_ids)
            + len(
                parsed_query
                .calculation_metric_ids
            )
        )

        if target_metric_count == 0:
            raise RuntimePlannerError(
                "financial_comparison "
                "必须至少包含一个财务指标"
            )

        # ========================================================
        # Ranking 当前只支持：
        #
        # 多公司 / 多年度
        #       +
        # 同一个 Metric
        #
        # 例如：
        #
        # 美的、格力谁的营业收入最高？
        #
        #
        # 不支持：
        #
        # 收入 + 毛利率 + 总资产
        # 谁“综合最高”
        #
        # 因为这需要额外的综合评分定义。
        # ========================================================

        if (
            parsed_query.ranking_requested
            and target_metric_count != 1
        ):
            raise RuntimePlannerError(
                "ranking 当前只支持"
                "一个目标财务指标"
            )

        financial_queries: list[
            ComplexRetrievalQueryOutput
        ] = []

        steps: list[
            ComplexPlanStepOutput
        ] = []

        tool_by_step_id: dict[
            str,
            str,
        ] = {}

        retrieval_runtime_by_key: dict[
            tuple[str, str, str],
            tuple[str, str],
        ] = {}

        calculation_runtime_by_key: dict[
            tuple[str, str, str],
            tuple[str, str],
        ] = {}

        comparison_input_refs: list[
            str
        ] = []

        producer_step_ids: list[
            str
        ] = []

        for report_id in (
            parsed_query.report_ids
        ):
            report = (
                self.registry_bundle
                .reports
                .require(report_id)
            )

            self._validate_report_identity(
                parsed_query=parsed_query,
                report_id=report_id,
                company_id=(
                    report.company_id
                ),
                fiscal_year=(
                    report.fiscal_year
                ),
            )

            # ====================================================
            # Reported Metrics
            # ====================================================

            for metric_id in (
                parsed_query.metric_ids
            ):
                metric = (
                    self.registry_bundle
                    .metrics
                    .require(metric_id)
                )

                if (
                    metric.metric_origin
                    is not MetricOrigin.REPORTED
                ):
                    raise RuntimePlannerError(
                        "metric_ids 只能包含 "
                        "reported metric："
                        f"{metric_id}"
                    )

                statement_scope = (
                    self._resolve_statement_scope(
                        parsed_query=(
                            parsed_query
                        ),
                        metric_id=metric_id,
                        allowed_scopes=tuple(
                            metric.allowed_scopes
                        ),
                    )
                )

                (
                    output_ref,
                    producer_step_id,
                ) = (
                    self
                    ._ensure_financial_retrieval(
                        report_id=report_id,
                        metric_id=metric_id,
                        statement_scope=(
                            statement_scope
                        ),
                        financial_queries=(
                            financial_queries
                        ),
                        steps=steps,
                        tool_by_step_id=(
                            tool_by_step_id
                        ),
                        retrieval_runtime_by_key=(
                            retrieval_runtime_by_key
                        ),
                    )
                )

                comparison_input_refs.append(
                    output_ref
                )

                producer_step_ids.append(
                    producer_step_id
                )

            # ====================================================
            # Derived Metrics
            # ====================================================

            for calculation_metric_id in (
                parsed_query
                .calculation_metric_ids
            ):
                (
                    calculation_ref,
                    calculation_step_id,
                ) = (
                    self
                    ._ensure_financial_calculation(
                        parsed_query=(
                            parsed_query
                        ),
                        report_id=report_id,
                        calculation_metric_id=(
                            calculation_metric_id
                        ),
                        financial_queries=(
                            financial_queries
                        ),
                        steps=steps,
                        tool_by_step_id=(
                            tool_by_step_id
                        ),
                        retrieval_runtime_by_key=(
                            retrieval_runtime_by_key
                        ),
                        calculation_runtime_by_key=(
                            calculation_runtime_by_key
                        ),
                    )
                )

                comparison_input_refs.append(
                    calculation_ref
                )

                producer_step_ids.append(
                    calculation_step_id
                )

        # ========================================================
        # Compare / Rank 至少需要两个结果。
        #
        # 一个值没有“比较”的意义。
        # ========================================================

        if len(comparison_input_refs) < 2:
            raise RuntimePlannerError(
                "financial_comparison "
                "至少需要两个可比较结果"
            )

        final_action = (
            "rank"
            if parsed_query.ranking_requested
            else "compare"
        )

        final_output_ref = (
            "ranking_result"
            if final_action == "rank"
            else "comparison_result"
        )

        final_step_id = (
            f"s{len(steps) + 1}"
        )

        steps.append(
            ComplexPlanStepOutput(
                step_id=final_step_id,
                action=final_action,
                description=(
                    "对财务结果进行排名"
                    if final_action == "rank"
                    else "比较多个财务结果"
                ),
                input_refs=tuple(
                    comparison_input_refs
                ),
                depends_on=(
                    _unique_in_order(
                        tuple(
                            producer_step_ids
                        )
                    )
                ),
                output_ref=(
                    final_output_ref
                ),
            )
        )

        plan = ComplexPlanOutput(
            steps=tuple(steps),
            final_step_id=(
                final_step_id
            ),
        )

        # ========================================================
        # compare / rank 没有加入 tool_by_step_id。
        #
        # 因为它们不是 External Tool Call。
        #
        # Step 8 Runtime 会在自己的确定性执行逻辑中
        # 消费这些 Runtime Result。
        # ========================================================

        return RuntimePlan(
            intent="financial_comparison",
            planner_version=(
                self.planner_version
            ),
            normalized_question=(
                parsed_query
                .normalized_question
            ),
            financial_queries=tuple(
                financial_queries
            ),
            document_queries=(),
            plan=plan,
            tool_by_step_id=(
                tool_by_step_id
            ),
        )

    # ========================================================
    # document_evidence
    # ========================================================

    def _build_document_evidence_plan(
        self,
        parsed_query: ParsedFinancialQuery,
    ) -> RuntimePlan:
        document_queries: list[
            DocumentEvidenceQuery
        ] = []

        steps: list[
            ComplexPlanStepOutput
        ] = []

        tool_by_step_id: dict[
            str,
            str,
        ] = {}

        result_refs: list[str] = []
        producer_step_ids: list[str] = []

        for report_id in (
            parsed_query.report_ids
        ):
            report = (
                self.registry_bundle
                .reports
                .require(report_id)
            )

            self._validate_report_identity(
                parsed_query=parsed_query,
                report_id=report_id,
                company_id=(
                    report.company_id
                ),
                fiscal_year=(
                    report.fiscal_year
                ),
            )

            query_id = (
                f"q{len(document_queries) + 1}"
            )

            query = DocumentEvidenceQuery(
                query_id=query_id,
                semantic_query=(
                    parsed_query
                    .normalized_question
                ),
                company_id=(
                    report.company_id
                ),
                report_id=(
                    report.report_id
                ),
                fiscal_year=(
                    report.fiscal_year
                ),
                report_type=(
                    report.report_type
                ),
            )

            document_queries.append(
                query
            )

            step_id = (
                f"s{len(steps) + 1}"
            )

            output_ref = (
                f"retrieval_result_{query_id}"
            )

            steps.append(
                ComplexPlanStepOutput(
                    step_id=step_id,
                    action="retrieve",
                    description=(
                        "检索财报原文证据："
                        f"{report.report_id}"
                    ),
                    input_refs=(),
                    depends_on=(),
                    output_ref=output_ref,
                    retrieval_query_id=(
                        query_id
                    ),
                )
            )

            tool_by_step_id[
                step_id
            ] = "retrieve_documents"

            result_refs.append(
                output_ref
            )

            producer_step_ids.append(
                step_id
            )

        if not steps:
            raise RuntimePlannerError(
                "document_evidence "
                "没有生成任何 Retrieval Step"
            )

        if len(result_refs) > 1:
            synthesize_step_id = (
                f"s{len(steps) + 1}"
            )

            steps.append(
                ComplexPlanStepOutput(
                    step_id=(
                        synthesize_step_id
                    ),
                    action="synthesize",
                    description=(
                        "汇总多个报告的文档证据"
                    ),
                    input_refs=tuple(
                        result_refs
                    ),
                    depends_on=tuple(
                        producer_step_ids
                    ),
                    output_ref=(
                        "synthesized_result"
                    ),
                )
            )

        plan = ComplexPlanOutput(
            steps=tuple(steps),
            final_step_id=(
                steps[-1].step_id
            ),
        )

        return RuntimePlan(
            intent="document_evidence",
            planner_version=(
                self.planner_version
            ),
            normalized_question=(
                parsed_query
                .normalized_question
            ),
            financial_queries=(),
            document_queries=tuple(
                document_queries
            ),
            plan=plan,
            tool_by_step_id=(
                tool_by_step_id
            ),
        )

    # ========================================================
    # 这个 Helper 是 7C-2 很重要的重构。
    #
    # 它解决：
    #
    # “同一个 Fact Query 不要生成两遍。”
    #
    #
    # 用户：
    #
    # “营业收入和毛利率分别是多少？”
    #
    # revenue：
    #
    # 既是用户直接要求的结果，
    # 又是 gross_profit_margin 的输入。
    #
    # 正确：
    #
    # q1 revenue
    # q2 operating_cost
    # s3 calculation
    #
    # 错误：
    #
    # q1 revenue
    # q2 revenue
    # q3 operating_cost
    # ========================================================

    def _ensure_financial_retrieval(
        self,
        *,
        report_id: str,
        metric_id: str,
        statement_scope: StatementScope,
        financial_queries: list[
            ComplexRetrievalQueryOutput
        ],
        steps: list[
            ComplexPlanStepOutput
        ],
        tool_by_step_id: dict[
            str,
            str,
        ],
        retrieval_runtime_by_key: dict[
            tuple[str, str, str],
            tuple[str, str],
        ],
    ) -> tuple[str, str]:
        key = (
            report_id,
            metric_id,
            statement_scope.value,
        )

        existing = (
            retrieval_runtime_by_key
            .get(key)
        )

        if existing is not None:
            return existing

        report = (
            self.registry_bundle
            .reports
            .require(report_id)
        )

        company = (
            self.registry_bundle
            .companies
            .require(
                report.company_id
            )
        )

        metric = (
            self.registry_bundle
            .metrics
            .require(metric_id)
        )

        if (
            statement_scope
            not in metric.allowed_scopes
        ):
            raise RuntimePlannerError(
                f"Metric '{metric_id}' "
                "不支持 statement_scope："
                f"{statement_scope.value}"
            )

        query_id = (
            f"q{len(financial_queries) + 1}"
        )

        query = (
            ComplexRetrievalQueryOutput(
                query_id=query_id,
                semantic_query=(
                    self
                    ._build_financial_semantic_query(
                        company_name=(
                            company.short_name_cn
                        ),
                        fiscal_year=(
                            report.fiscal_year
                        ),
                        metric_name=(
                            metric.display_name_cn
                        ),
                        statement_scope=(
                            statement_scope
                        ),
                    )
                ),
                company_id=(
                    report.company_id
                ),
                report_id=(
                    report.report_id
                ),
                metric_id=(
                    metric.metric_id
                ),
                fiscal_year=(
                    report.fiscal_year
                ),
                report_type=(
                    report.report_type
                ),
                statement_type=(
                    metric.statement_type
                ),
                statement_scope=(
                    statement_scope
                ),
            )
        )

        financial_queries.append(
            query
        )

        step_id = (
            f"s{len(steps) + 1}"
        )

        output_ref = (
            f"retrieval_result_{query_id}"
        )

        steps.append(
            ComplexPlanStepOutput(
                step_id=step_id,
                action="retrieve",
                description=(
                    "查询"
                    f"{company.short_name_cn}"
                    f"{report.fiscal_year}年"
                    f"{metric.display_name_cn}"
                ),
                input_refs=(),
                depends_on=(),
                output_ref=(
                    output_ref
                ),
                retrieval_query_id=(
                    query_id
                ),
            )
        )

        tool_by_step_id[
            step_id
        ] = (
            "query_financial_data"
        )

        runtime_identity = (
            output_ref,
            step_id,
        )

        retrieval_runtime_by_key[
            key
        ] = runtime_identity

        return runtime_identity

    # ============================================================
    # 这个 Helper 与：
    #
    #     _ensure_financial_retrieval()
    #
    # 是一对。
    #
    #
    # _ensure_financial_retrieval
    # → 确保某个 Financial Fact Query 已经存在
    #
    #
    # _ensure_financial_calculation
    # → 确保某个 Derived Metric Calculation
    #   已经存在
    #
    #
    # Comparison / Ranking 后面可以直接复用它。
    # ============================================================

    def _ensure_financial_calculation(
        self,
        *,
        parsed_query: ParsedFinancialQuery,
        report_id: str,
        calculation_metric_id: str,
        financial_queries: list[
            ComplexRetrievalQueryOutput
        ],
        steps: list[
            ComplexPlanStepOutput
        ],
        tool_by_step_id: dict[
            str,
            str,
        ],
        retrieval_runtime_by_key: dict[
            tuple[str, str, str],
            tuple[str, str],
        ],
        calculation_runtime_by_key: dict[
            tuple[str, str, str],
            tuple[str, str],
        ],
    ) -> tuple[str, str]:
        report = (
            self.registry_bundle
            .reports
            .require(report_id)
        )

        calculation_metric = (
            self.registry_bundle
            .metrics
            .require(
                calculation_metric_id
            )
        )

        if (
            calculation_metric
            .metric_origin
            is not MetricOrigin.DERIVED
        ):
            raise RuntimePlannerError(
                "calculation_metric_ids "
                "只能包含 derived metric："
                f"{calculation_metric_id}"
            )

        formula_id = (
            calculation_metric
            .formula_id
        )

        if formula_id is None:
            raise RuntimePlannerError(
                "Derived Metric 缺少 "
                "formula_id："
                f"{calculation_metric_id}"
            )

        input_metric_ids = (
            _FORMULA_INPUT_METRIC_IDS
            .get(formula_id)
        )

        if input_metric_ids is None:
            raise RuntimePlannerError(
                "Runtime Planner "
                "不支持 formula_id："
                f"{formula_id}"
            )

        calculation_scope = (
            self._resolve_statement_scope(
                parsed_query=parsed_query,
                metric_id=(
                    calculation_metric_id
                ),
                allowed_scopes=tuple(
                    calculation_metric
                    .allowed_scopes
                ),
            )
        )

        calculation_key = (
            report_id,
            calculation_metric_id,
            calculation_scope.value,
        )

        existing = (
            calculation_runtime_by_key
            .get(calculation_key)
        )

        if existing is not None:
            return existing

        input_refs: list[str] = []

        dependency_step_ids: list[
            str
        ] = []

        for input_metric_id in (
            input_metric_ids
        ):
            input_metric = (
                self.registry_bundle
                .metrics
                .require(
                    input_metric_id
                )
            )

            if (
                input_metric.metric_origin
                is not MetricOrigin.REPORTED
            ):
                raise RuntimePlannerError(
                    "当前 Calculation "
                    "只允许直接使用 "
                    "reported input metric："
                    f"{input_metric_id}"
                )

            if (
                calculation_scope
                not in input_metric
                .allowed_scopes
            ):
                raise RuntimePlannerError(
                    f"Calculation "
                    f"'{calculation_metric_id}' "
                    "的输入 Metric "
                    f"'{input_metric_id}' "
                    "不支持目标口径："
                    f"{calculation_scope.value}"
                )

            (
                input_ref,
                producer_step_id,
            ) = (
                self
                ._ensure_financial_retrieval(
                    report_id=report_id,
                    metric_id=(
                        input_metric_id
                    ),
                    statement_scope=(
                        calculation_scope
                    ),
                    financial_queries=(
                        financial_queries
                    ),
                    steps=steps,
                    tool_by_step_id=(
                        tool_by_step_id
                    ),
                    retrieval_runtime_by_key=(
                        retrieval_runtime_by_key
                    ),
                )
            )

            # ====================================================
            # 【重点理解】
            #
            # 绝对不能 sorted(input_refs)。
            #
            # Formula 的参数顺序本身就是 Contract。
            # ====================================================

            input_refs.append(
                input_ref
            )

            dependency_step_ids.append(
                producer_step_id
            )

        calculation_id = (
            "calculation_"
            f"{report.company_id}_"
            f"{report.fiscal_year}_"
            f"{calculation_metric.metric_id}"
        )

        calculation_step_id = (
            f"s{len(steps) + 1}"
        )

        company = (
            self.registry_bundle
            .companies
            .require(
                report.company_id
            )
        )

        steps.append(
            ComplexPlanStepOutput(
                step_id=(
                    calculation_step_id
                ),
                action="calculate",
                description=(
                    "计算"
                    f"{company.short_name_cn}"
                    f"{report.fiscal_year}年"
                    f"{calculation_metric.display_name_cn}"
                ),
                input_refs=tuple(
                    input_refs
                ),
                depends_on=(
                    _unique_in_order(
                        tuple(
                            dependency_step_ids
                        )
                    )
                ),
                output_ref=(
                    calculation_id
                ),
                calculation_id=(
                    calculation_id
                ),
                formula_id=(
                    formula_id
                ),
            )
        )

        tool_by_step_id[
            calculation_step_id
        ] = "execute_calculation"

        runtime_identity = (
            calculation_id,
            calculation_step_id,
        )

        calculation_runtime_by_key[
            calculation_key
        ] = runtime_identity

        return runtime_identity




    # ========================================================
    # Report Identity Validation
    # ========================================================

    @staticmethod
    def _validate_report_identity(
        *,
        parsed_query: ParsedFinancialQuery,
        report_id: str,
        company_id: str,
        fiscal_year: int,
    ) -> None:
        if (
            parsed_query.company_ids
            and company_id
            not in parsed_query.company_ids
        ):
            raise RuntimePlannerError(
                "Report 与 Parsed Query "
                "company_id 不一致："
                f"{report_id}"
            )

        if (
            parsed_query.years
            and fiscal_year
            not in parsed_query.years
        ):
            raise RuntimePlannerError(
                "Report 与 Parsed Query "
                "fiscal_year 不一致："
                f"{report_id}"
            )

    # ========================================================
    # Statement Scope Policy
    # ========================================================

    @staticmethod
    def _resolve_statement_scope(
        *,
        parsed_query: ParsedFinancialQuery,
        metric_id: str,
        allowed_scopes: tuple[
            StatementScope,
            ...
        ],
    ) -> StatementScope:
        requested_scope = (
            parsed_query
            .statement_scope
        )

        if requested_scope is not None:
            if (
                requested_scope
                not in allowed_scopes
            ):
                raise RuntimePlannerError(
                    f"Metric '{metric_id}' "
                    "不支持请求的 statement_scope："
                    f"{requested_scope.value}"
                )

            return requested_scope

        if (
            StatementScope.CONSOLIDATED
            in allowed_scopes
        ):
            return (
                StatementScope.CONSOLIDATED
            )

        if len(allowed_scopes) == 1:
            return allowed_scopes[0]

        raise RuntimePlannerError(
            f"Metric '{metric_id}' "
            "无法唯一确定 statement_scope"
        )

    @staticmethod
    def _build_financial_semantic_query(
        *,
        company_name: str,
        fiscal_year: int,
        metric_name: str,
        statement_scope: StatementScope,
    ) -> str:
        if (
            statement_scope
            is StatementScope.CONSOLIDATED
        ):
            scope_text = "合并口径"

        elif (
            statement_scope
            is StatementScope.PARENT_COMPANY
        ):
            scope_text = "母公司口径"

        else:
            scope_text = (
                statement_scope.value
            )

        return (
            f"{company_name}"
            f"{fiscal_year}年"
            f"{scope_text}"
            f"{metric_name}"
        )