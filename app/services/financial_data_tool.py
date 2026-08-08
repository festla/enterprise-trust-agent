from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from pydantic import BaseModel

from app.schemas.enums import (
    ValidationStatus,
)
from app.schemas.tool_registry import (
    QueryFinancialDataInput,
    QueryFinancialDataOutput,
    ToolDefinition,
)
from app.services.registry import (
    RegistryBundle,
)
from app.services.tool_registry import (
    ToolRegistry,
)


@dataclass(
    frozen=True,
    slots=True,
)
class QueryFinancialDataTool:
    """把 RegistryBundle 财务事实查询包装成 Agent Tool。"""

    registry_bundle: RegistryBundle

    tool_name: str = "query_financial_data"

    tool_version: str = "1.0.0"

    def build_definition(
        self,
    ) -> ToolDefinition:
        return ToolDefinition(
            tool_name=self.tool_name,
            description=(
                "根据结构化公司、报告、指标、"
                "年度和报表口径查询已核验财务事实。"
            ),
            version=self.tool_version,
            input_schema=(
                QueryFinancialDataInput
                .model_json_schema()
            ),
            output_schema=(
                QueryFinancialDataOutput
                .model_json_schema()
            ),
            permission="read_financial_data",

            # Registry 查询是本地内存操作，
            # 正常情况下应该很快。
            timeout_seconds=2.0,

            # 这里设置 0，不是因为 ToolExecutor
            # 不支持 Retry。
            #
            # 而是这个 Handler 当前只是本地 Registry 查询，
            # 没有 HTTP 503、网络抖动之类的临时失败。
            #
            # “是否重试”应该根据工具失败类型决定，
            # 不能所有 Tool 都机械配置 retry。
            max_retries=0,

            # 查询没有副作用，所以是幂等操作。
            idempotent=True,

            max_result_bytes=64 * 1024,
        )
    # ========================================================
    # ToolExecutor 在执行 Handler 前已经完成：
    #
    # 1. 权限检查
    # 2. Input Schema 校验
    # 3. 参数 Hash
    # 4. Idempotency Cache 查询
    #
    # 所以 Handler 的职责应该保持纯粹：
    #
    #        “执行领域业务”
    #
    # 不要在这里重新做权限、Retry、Timeout。
    # ========================================================

    def handle(
        self,
        input_value: BaseModel,
    ) -> QueryFinancialDataOutput:
        if not isinstance(
            input_value,
            QueryFinancialDataInput,
        ):
            raise TypeError(
                "query_financial_data "
                "必须接受 QueryFinancialDataInput"
            )

        query = input_value.query

        timer_start = perf_counter()

        candidate_facts = (
            self.registry_bundle
            .financial_facts
            .find(
                company_id=query.company_id,
                report_id=query.report_id,
                metric_id=query.metric_id,
                fiscal_year=query.fiscal_year,
                statement_scope=(
                    query.statement_scope.value
                ),
            )
        )

        verified_supports = []

        for fact in candidate_facts:
            if (
                fact.statement_type
                is not query.statement_type
            ):
                continue

            if (
                fact.validation_status
                is not ValidationStatus.VERIFIED
            ):
                continue

            evidence = (
                self.registry_bundle
                .evidences
                .get(
                    fact.primary_evidence_id
                )
            )
            

            if evidence is None:
                continue

            if (
                evidence.validation_status
                is not ValidationStatus.VERIFIED
            ):
                continue

            verified_supports.append(
                (
                    fact,
                    evidence,
                )
            )

        verified_supports.sort(
            key=lambda item: item[0].fact_id
        )

        selected_supports = (
            verified_supports[
                :input_value.max_results
            ]
        )

        retrieved_fact_ids = tuple(
            fact.fact_id
            for fact, _
            in selected_supports
        )

        retrieved_evidence_ids = (
            _unique_in_order(
                evidence.evidence_id
                for _, evidence
                in selected_supports
            )
        )

        retrieved_chunk_ids = (
            _unique_in_order(
                evidence.chunk_id
                for _, evidence
                in selected_supports
                if evidence.chunk_id is not None
            )
        )

        latency_ms = (
            perf_counter()
            - timer_start
        ) * 1000.0

        from app.schemas.complex_plan_eval_result import (
            ComplexRetrievalTrace,
        )

        trace = ComplexRetrievalTrace(
            query_id=query.query_id,
            status="completed",
            retrieved_fact_ids=(
                retrieved_fact_ids
            ),
            retrieved_evidence_ids=(
                retrieved_evidence_ids
            ),
            retrieved_chunk_ids=(
                retrieved_chunk_ids
            ),
            top_k=input_value.max_results,
            latency_ms=latency_ms,
            error_message=None,
        )

        return QueryFinancialDataOutput(
            trace=trace
        )


def _unique_in_order(
    values,
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
# Runtime 的其他模块不应该知道：
#
#   QueryFinancialDataTool
#   build_definition
#   handle
#
# 应该通过一个注册入口完成依赖注入。
#
# 最终 Runtime Factory 会类似：
#
#   registry = ToolRegistry()
#
#   register_query_financial_data_tool(...)
#   register_retrieve_documents_tool(...)
#   register_execute_calculation_tool(...)
#
# ============================================================

def register_query_financial_data_tool(
    *,
    tool_registry: ToolRegistry,
    registry_bundle: RegistryBundle,
) -> QueryFinancialDataTool:
    tool = QueryFinancialDataTool(
        registry_bundle=registry_bundle
    )

    tool_registry.register(
        definition=(
            tool.build_definition()
        ),
        input_model=(
            QueryFinancialDataInput
        ),
        output_model=(
            QueryFinancialDataOutput
        ),
        handler=tool.handle,
    )

    return tool