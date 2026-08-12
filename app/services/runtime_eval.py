from __future__ import annotations

import hashlib
import json

from collections import (
    Counter,
)
from dataclasses import (
    dataclass,
)
from datetime import (
    date,
    datetime,
    timezone,
)
from decimal import Decimal
from pathlib import Path

from app.schemas.agent_runtime import (
    AgentState,
    RuntimePlan,
)
from app.schemas.enums import (
    AttributionType,
    EvidenceType,
    PeriodType,
    RestatementStatus,
    StatementType,
    UnitCode,
    ValidationStatus,
)
from app.schemas.evidence import (
    SourceEvidence,
)
from app.schemas.financial_fact import (
    FinancialFact,
)
from app.schemas.reranker import (
    RerankedRetrievalHit,
)
from app.schemas.runtime_eval import (
    RuntimeEvalCase,
    RuntimeEvalCaseResult,
    RuntimeEvalSummary,
)
from app.services.agent_runtime import (
    AgentRuntime,
)
from app.services.calculation_tool import (
    register_execute_calculation_tool,
)
from app.services.checkpoint_store import (
    InMemoryCheckpointStore,
)
from app.services.complex_oracle_calculator_adapter import (
    ComplexOracleCalculatorAdapter,
)
from app.services.document_retrieval_tool import (
    register_retrieve_documents_tool,
)
from app.services.financial_data_tool import (
    register_query_financial_data_tool,
)
from app.services.registry import (
    RegistryBundle,
)
from app.services.registry_loader import (
    load_registry_bundle,
)
from app.services.runtime_completion import (
    RuntimeAnswerGenerator,
    RuntimeEvidenceVerifier,
)
from app.services.runtime_intent_router import (
    RuntimeIntentRouter,
)
from app.services.runtime_plan_executor import (
    RuntimePlanExecutor,
)
from app.services.runtime_planner import (
    RuntimePlanner,
)
from app.services.runtime_query_parser import (
    RuntimeQueryParser,
)
from app.schemas.tool_registry import (
    DocumentEvidenceQuery,
)

from app.services.tool_registry import (
    ToolExecutor,
    ToolRegistry,
)
from app.services.trajectory_store import (
    TrajectoryStore,
)


class RuntimeEvalError(
    ValueError
):
    """Runtime Control Eval 基础异常。"""


@dataclass(
    slots=True,
)
class RuntimeEvalEnvironment:
    runtime: AgentRuntime

    registry_bundle: RegistryBundle


class RuntimeEvalDocumentProvider:
    """控制流评测专用确定性文档 Provider。"""

    def search(
        self,
        *,
        query: DocumentEvidenceQuery,
        top_k: int,
    ) -> tuple[
        RerankedRetrievalHit,
        ...
    ]:
        digest = hashlib.sha256(
            query.semantic_query.encode(
                "utf-8"
            )
        ).hexdigest()[:10]

        document_id = (
            f"doc_{query.report_id}_"
            f"runtime_eval"
        )

        page_id = (
            f"{document_id}_page_0001"
        )

        text = (
            "Runtime Control Dev 模拟文档证据："
            f"{query.report_id} 的管理层披露了"
            "经营风险、战略、业务布局、竞争格局"
            "以及未来展望。"
            "该文本只用于验证 Agent Runtime "
            "的控制流、工具顺序和轨迹回放。"
        )

        hit = (
            RerankedRetrievalHit
            .model_construct(
                rank=1,
                score=1.0,
                chunk_id=(
                    f"chunk_runtime_eval_"
                    f"{query.report_id}_"
                    f"{digest}"
                ),
                document_id=document_id,
                page_id=page_id,
                company_id=(
                    query.company_id
                ),
                report_id=(
                    query.report_id
                ),
                fiscal_year=(
                    query.fiscal_year
                ),
                report_type=(
                    query.report_type
                ),
                pdf_page=1,
                printed_page=1,
                section_path=(
                    "Runtime Control Eval",
                ),
                text=text,
            )
        )

        return (
            hit,
        )


def _stable_synthetic_value(
    *,
    company_id: str,
    fiscal_year: int,
    metric_id: str,
) -> Decimal:
    identity = (
        f"{company_id}:"
        f"{fiscal_year}"
    )

    digest = hashlib.sha256(
        identity.encode(
            "utf-8"
        )
    ).hexdigest()

    offset = (
        int(
            digest[:8],
            16,
        )
        % 5_000_000_000
    )

    revenue_value = Decimal(
        100_000_000_000
        + offset
    )

    if metric_id == "revenue":
        return revenue_value

    if metric_id == "operating_cost":
        return (
            revenue_value
            * Decimal("0.70")
        )

    return (
        Decimal(
            10_000_000_000
            + offset
        )
    )


def _has_verified_support(
    *,
    bundle: RegistryBundle,
    query,
) -> bool:
    candidates = (
        bundle.financial_facts.find(
            company_id=(
                query.company_id
            ),
            report_id=(
                query.report_id
            ),
            metric_id=(
                query.metric_id
            ),
            fiscal_year=(
                query.fiscal_year
            ),
            statement_scope=(
                query.statement_scope.value
            ),
        )
    )

    for fact in candidates:
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
            bundle.evidences.get(
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

        return True

    return False


def _add_synthetic_support(
    *,
    bundle: RegistryBundle,
    query,
) -> None:
    metric = (
        bundle.metrics.require(
            query.metric_id
        )
    )

    now = datetime.now(
        timezone.utc
    )

    scope_value = (
        query.statement_scope.value
    )

    fact_id = (
        "fact_runtime_eval_"
        f"{query.company_id}_"
        f"{query.fiscal_year}_"
        f"{query.metric_id}_"
        f"{scope_value}"
    )

    evidence_id = (
        "evidence_runtime_eval_"
        f"{query.company_id}_"
        f"{query.fiscal_year}_"
        f"{query.metric_id}_"
        f"{scope_value}"
    )

    if (
        bundle.financial_facts
        .contains(fact_id)
    ):
        return

    value = (
        _stable_synthetic_value(
            company_id=(
                query.company_id
            ),
            fiscal_year=(
                query.fiscal_year
            ),
            metric_id=(
                query.metric_id
            ),
        )
    )

    if (
        metric.period_type
        is PeriodType.INSTANT
    ):
        period_start = None
        period_end = None

        as_of_date = date(
            query.fiscal_year,
            12,
            31,
        )

    else:
        period_start = date(
            query.fiscal_year,
            1,
            1,
        )

        period_end = date(
            query.fiscal_year,
            12,
            31,
        )

        as_of_date = None

    normalized_unit = (
        metric.default_unit
    )

    currency = (
        "CNY"
        if normalized_unit
        is UnitCode.CNY
        else None
    )

    evidence_text = (
        "Runtime Control Dev synthetic "
        f"support：{query.company_id} "
        f"{query.fiscal_year} "
        f"{query.metric_id} = {value}"
    )

    source_hash = (
        hashlib.sha256(
            evidence_text.encode(
                "utf-8"
            )
        ).hexdigest()
    )

    document_id = (
        f"doc_{query.report_id}_"
        "runtime_eval"
    )

    evidence = SourceEvidence(
        evidence_id=evidence_id,
        report_id=query.report_id,
        document_id=document_id,
        page_id=(
            f"{document_id}_page_0001"
        ),
        chunk_id=(
            "chunk_runtime_eval_"
            f"{query.company_id}_"
            f"{query.fiscal_year}_"
            f"{query.metric_id}"
        ),
        evidence_type=(
            EvidenceType
            .FINANCIAL_STATEMENT_CELL
        ),
        attribution_type=(
            AttributionType
            .REPORT_DISCLOSURE
        ),
        statement_type=(
            query.statement_type
        ),
        statement_scope=(
            query.statement_scope
        ),
        section_title=(
            "Runtime Control Eval"
        ),
        table_name=(
            "Runtime Control Eval Table"
        ),
        row_label=(
            metric.display_name_cn
        ),
        column_label=(
            f"{query.fiscal_year}年度"
        ),
        printed_page=1,
        pdf_page=1,
        evidence_text=(
            evidence_text
        ),
        cell_value=str(value),
        source_hash=source_hash,
        validation_status=(
            ValidationStatus.VERIFIED
        ),
        validated_by=(
            "runtime_control_eval_fixture"
        ),
        created_at=now,
    )

    fact = FinancialFact(
        fact_id=fact_id,
        company_id=(
            query.company_id
        ),
        report_id=(
            query.report_id
        ),
        metric_id=(
            query.metric_id
        ),
        fiscal_year=(
            query.fiscal_year
        ),
        statement_type=(
            query.statement_type
        ),
        statement_scope=(
            query.statement_scope
        ),
        period_type=(
            metric.period_type
        ),
        period_start=(
            period_start
        ),
        period_end=(
            period_end
        ),
        as_of_date=(
            as_of_date
        ),
        raw_value=value,
        raw_unit=(
            normalized_unit
        ),
        unit_multiplier=(
            Decimal("1")
        ),
        normalized_value=value,
        normalized_unit=(
            normalized_unit
        ),
        currency=currency,
        table_name=(
            "Runtime Control Eval Table"
        ),
        row_label=(
            metric.display_name_cn
        ),
        column_label=(
            f"{query.fiscal_year}年度"
        ),
        is_comparative_value=False,
        restatement_status=(
            RestatementStatus
            .NOT_APPLICABLE
        ),
        primary_evidence_id=(
            evidence_id
        ),
        validation_status=(
            ValidationStatus.VERIFIED
        ),
        validated_by=(
            "runtime_control_eval_fixture"
        ),
        validated_at=now,
        source_version=(
            "runtime_control_eval_v1"
        ),
        created_at=now,
        updated_at=now,
    )

    bundle.evidences.add(
        evidence
    )

    bundle.financial_facts.add(
        fact
    )


def ensure_runtime_plan_support(
    *,
    bundle: RegistryBundle,
    runtime_plan: RuntimePlan,
) -> None:
    """为控制流评测补足缺失的结构化输入。"""

    for query in (
        runtime_plan.financial_queries
    ):
        if _has_verified_support(
            bundle=bundle,
            query=query,
        ):
            continue

        _add_synthetic_support(
            bundle=bundle,
            query=query,
        )


def build_runtime_eval_environment(
    *,
    project_root: Path,
    trajectory_root: Path,
) -> RuntimeEvalEnvironment:
    """构建无需真实 Embedding 的 Runtime Control 环境。"""

    registry_root = (
        project_root
        / "data"
        / "processed"
        / "registries"
    )

    (
        bundle,
        _,
        _,
        _,
    ) = load_registry_bundle(
        companies_path=(
            registry_root
            / "companies.yaml"
        ),
        reports_path=(
            registry_root
            / "reports.yaml"
        ),
        metrics_path=(
            registry_root
            / "metrics.yaml"
        ),
        evidences_path=(
            registry_root
            / "evidences.yaml"
        ),
        financial_facts_path=(
            registry_root
            / "financial_facts.yaml"
        ),
    )

    tool_registry = (
        ToolRegistry()
    )

    register_query_financial_data_tool(
        tool_registry=(
            tool_registry
        ),
        registry_bundle=bundle,
    )

    register_execute_calculation_tool(
        tool_registry=(
            tool_registry
        ),
        calculation_provider=(
            ComplexOracleCalculatorAdapter(
                registry_bundle=bundle
            )
        ),
    )

    register_retrieve_documents_tool(
        tool_registry=(
            tool_registry
        ),
        hit_provider=(
            RuntimeEvalDocumentProvider()
        ),
    )

    tool_executor = (
        ToolExecutor(
            tool_registry,
            retry_backoff_seconds=0,
        )
    )

    plan_executor = (
        RuntimePlanExecutor(
            tool_executor=(
                tool_executor
            ),
            granted_permissions=(
                frozenset(
                    {
                        "read_financial_data",
                        "read_documents",
                        "execute_calculation",
                    }
                )
            ),
            registry_bundle=bundle,

            # 控制流评测要求一个 Retrieval Slot
            # 唯一解析成一个 Fact。
            financial_max_results=1,

            document_top_k=1,
        )
    )

    runtime = AgentRuntime(
        query_parser=(
            RuntimeQueryParser(
                registry_bundle=bundle
            )
        ),
        intent_router=(
            RuntimeIntentRouter()
        ),
        planner=(
            RuntimePlanner(
                registry_bundle=bundle
            )
        ),
        plan_executor=(
            plan_executor
        ),
        verifier=(
            RuntimeEvidenceVerifier(
                registry_bundle=bundle
            )
        ),
        answer_generator=(
            RuntimeAnswerGenerator(
                registry_bundle=bundle
            )
        ),
        checkpoint_store=(
            InMemoryCheckpointStore()
        ),
        trajectory_store=(
            TrajectoryStore(
                trajectory_root
            )
        ),
    )

    return RuntimeEvalEnvironment(
        runtime=runtime,
        registry_bundle=bundle,
    )


def _logical_tool_sequence(
    state: AgentState,
) -> tuple[str, ...]:
    """Retry 不应被误算成额外逻辑 Tool Step。"""

    result: list[str] = []

    seen_step_ids: set[
        str
    ] = set()

    for trace in (
        state.tool_call_traces
    ):
        if (
            trace.step_id
            in seen_step_ids
        ):
            continue

        seen_step_ids.add(
            trace.step_id
        )

        result.append(
            trace.tool_name
        )

    return tuple(result)


def _plan_actions(
    state: AgentState,
):
    if state.runtime_plan is None:
        return ()

    return tuple(
        step.action
        for step
        in state.runtime_plan
        .plan.steps
    )


def _ratio(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
    )


def evaluate_runtime_cases(
    *,
    environment: RuntimeEvalEnvironment,
    cases: tuple[
        RuntimeEvalCase,
        ...
    ],
) -> tuple[
    tuple[
        RuntimeEvalCaseResult,
        ...
    ],
    RuntimeEvalSummary,
]:
    runtime = environment.runtime

    bundle = (
        environment.registry_bundle
    )

    results: list[
        RuntimeEvalCaseResult
    ] = []

    for case in cases:
        state: AgentState | None = None

        error_message: str | None = None

        replay_ok: bool | None = None

        try:
            run_id = (
                f"run_{case.case_id}"
            )

            thread_id = (
                f"thread_{case.case_id}"
            )

            state = runtime.prepare(
                query=case.question,
                run_id=run_id,
                thread_id=thread_id,
                max_steps=64,
            )

            if (
                state.runtime_plan
                is not None
            ):
                ensure_runtime_plan_support(
                    bundle=bundle,
                    runtime_plan=(
                        state.runtime_plan
                    ),
                )

            # resume() 从 prepare 已写入的
            # Checkpoint 开始继续。
            #
            # 因此这个评测同时走过了
            # Step 10 的恢复路径。
            state = runtime.resume(
                run_id=run_id,
                thread_id=thread_id,
            )

            actual_plan_actions = (
                _plan_actions(
                    state
                )
            )

            actual_tool_sequence = (
                _logical_tool_sequence(
                    state
                )
            )

            intent_ok = (
                state.intent
                == case.expected_intent
            )

            argument_ok = (
                state.company_ids
                == (
                    case
                    .expected_company_ids
                )
                and state.years
                == case.expected_years
                and state.metric_ids
                == (
                    case
                    .expected_metric_ids
                )
            )

            plan_ok = (
                actual_plan_actions
                == (
                    case
                    .expected_plan_actions
                )
            )

            tool_ok = (
                Counter(
                    actual_tool_sequence
                )
                == Counter(
                    case
                    .expected_tool_sequence
                )
            )

            tool_sequence_ok = (
                actual_tool_sequence
                == (
                    case
                    .expected_tool_sequence
                )
            )

            termination_ok = (
                state.status
                == (
                    case
                    .expected_final_status
                )
                and state.stop_reason
                == (
                    case
                    .expected_stop_reason
                )
            )

            if case.replay_required:
                if (
                    runtime.trajectory_store
                    is None
                ):
                    replay_ok = False

                else:
                    replay = (
                        runtime
                        .trajectory_store
                        .replay(
                            state.run_id
                        )
                    )

                    replay_ok = (
                        replay.final_status
                        == state.status
                        and replay.stop_reason
                        == state.stop_reason
                        and replay.tools
                        == (
                            actual_tool_sequence
                        )
                    )

            else:
                replay_ok = None

            case_pass = all(
                (
                    intent_ok,
                    argument_ok,
                    plan_ok,
                    tool_ok,
                    tool_sequence_ok,
                    termination_ok,
                    (
                        replay_ok
                        is not False
                    ),
                )
            )

        except Exception as exc:
            error_message = (
                f"{exc.__class__.__name__}: "
                f"{exc}"
            )[:2000]

            actual_plan_actions = (
                _plan_actions(state)
                if state is not None
                else ()
            )

            actual_tool_sequence = (
                _logical_tool_sequence(
                    state
                )
                if state is not None
                else ()
            )

            intent_ok = False
            argument_ok = False
            plan_ok = False
            tool_ok = False
            tool_sequence_ok = False
            termination_ok = False

            if case.replay_required:
                replay_ok = False

            case_pass = False

        result = (
            RuntimeEvalCaseResult(
                case_id=(
                    case.case_id
                ),
                question=(
                    case.question
                ),
                expected_intent=(
                    case.expected_intent
                ),
                actual_intent=(
                    state.intent
                    if state is not None
                    else None
                ),
                expected_company_ids=(
                    case
                    .expected_company_ids
                ),
                actual_company_ids=(
                    state.company_ids
                    if state is not None
                    else ()
                ),
                expected_years=(
                    case.expected_years
                ),
                actual_years=(
                    state.years
                    if state is not None
                    else ()
                ),
                expected_metric_ids=(
                    case
                    .expected_metric_ids
                ),
                actual_metric_ids=(
                    state.metric_ids
                    if state is not None
                    else ()
                ),
                expected_plan_actions=(
                    case
                    .expected_plan_actions
                ),
                actual_plan_actions=(
                    actual_plan_actions
                ),
                expected_tool_sequence=(
                    case
                    .expected_tool_sequence
                ),
                actual_tool_sequence=(
                    actual_tool_sequence
                ),
                expected_final_status=(
                    case
                    .expected_final_status
                ),
                actual_final_status=(
                    state.status
                    if state is not None
                    else None
                ),
                expected_stop_reason=(
                    case
                    .expected_stop_reason
                ),
                actual_stop_reason=(
                    state.stop_reason
                    if state is not None
                    else None
                ),
                intent_ok=intent_ok,
                argument_ok=(
                    argument_ok
                ),
                plan_ok=plan_ok,
                tool_ok=tool_ok,
                tool_sequence_ok=(
                    tool_sequence_ok
                ),
                termination_ok=(
                    termination_ok
                ),
                replay_ok=replay_ok,
                case_pass=case_pass,
                error_message=(
                    error_message
                ),
            )
        )

        results.append(
            result
        )

    result_tuple = tuple(
        results
    )

    case_count = len(
        result_tuple
    )

    replay_applicable_count = sum(
        1
        for case in cases
        if case.replay_required
    )

    replay_success_count = sum(
        1
        for result
        in result_tuple
        if result.replay_ok is True
    )

    summary = RuntimeEvalSummary(
        case_count=case_count,
        passed_count=sum(
            result.case_pass
            for result
            in result_tuple
        ),
        completed_count=sum(
            result.actual_final_status
            == "completed"
            for result
            in result_tuple
        ),
        refused_count=sum(
            result.actual_final_status
            == "refused"
            for result
            in result_tuple
        ),
        awaiting_human_count=sum(
            result.actual_final_status
            == "awaiting_human"
            for result
            in result_tuple
        ),
        failed_count=sum(
            result.actual_final_status
            == "failed"
            for result
            in result_tuple
        ),
        intent_accuracy=_ratio(
            sum(
                result.intent_ok
                for result
                in result_tuple
            ),
            case_count,
        ),
        argument_accuracy=_ratio(
            sum(
                result.argument_ok
                for result
                in result_tuple
            ),
            case_count,
        ),
        plan_accuracy=_ratio(
            sum(
                result.plan_ok
                for result
                in result_tuple
            ),
            case_count,
        ),
        tool_accuracy=_ratio(
            sum(
                result.tool_ok
                for result
                in result_tuple
            ),
            case_count,
        ),
        tool_sequence_accuracy=(
            _ratio(
                sum(
                    result
                    .tool_sequence_ok
                    for result
                    in result_tuple
                ),
                case_count,
            )
        ),
        termination_accuracy=(
            _ratio(
                sum(
                    result
                    .termination_ok
                    for result
                    in result_tuple
                ),
                case_count,
            )
        ),
        task_success_rate=_ratio(
            sum(
                result.case_pass
                for result
                in result_tuple
            ),
            case_count,
        ),
        replay_applicable_count=(
            replay_applicable_count
        ),
        replay_success_count=(
            replay_success_count
        ),
        replay_success_rate=(
            _ratio(
                replay_success_count,
                replay_applicable_count,
            )
        ),
    )

    return (
        result_tuple,
        summary,
    )


def write_runtime_eval_cases(
    *,
    path: Path,
    cases: tuple[
        RuntimeEvalCase,
        ...
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    content = "\n".join(
        case.model_dump_json()
        for case in cases
    )

    path.write_text(
        content + "\n",
        encoding="utf-8",
    )


def load_runtime_eval_cases(
    path: Path,
) -> tuple[
    RuntimeEvalCase,
    ...
]:
    cases: list[
        RuntimeEvalCase
    ] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                cases.append(
                    RuntimeEvalCase
                    .model_validate_json(
                        line
                    )
                )

            except Exception as exc:
                raise RuntimeEvalError(
                    "Runtime Eval Case "
                    "解析失败："
                    f"line={line_number}"
                ) from exc

    if not cases:
        raise RuntimeEvalError(
            "Runtime Eval Dataset 不能为空"
        )

    case_ids = tuple(
        case.case_id
        for case in cases
    )

    if (
        len(case_ids)
        != len(set(case_ids))
    ):
        raise RuntimeEvalError(
            "Runtime Eval case_id "
            "不能重复"
        )

    return tuple(cases)


def write_runtime_eval_results(
    *,
    results_path: Path,
    summary_path: Path,
    results: tuple[
        RuntimeEvalCaseResult,
        ...
    ],
    summary: RuntimeEvalSummary,
) -> None:
    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_content = "\n".join(
        result.model_dump_json()
        for result in results
    )

    results_path.write_text(
        result_content + "\n",
        encoding="utf-8",
    )

    summary_path.write_text(
        json.dumps(
            summary.model_dump(
                mode="json"
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build_runtime_control_dev_v1_cases(
) -> tuple[
    RuntimeEvalCase,
    ...
]:
    """构造固定 50 条 Week 6 Runtime 控制流评测题。"""

    cases: list[
        RuntimeEvalCase
    ] = []

    def add(
        *,
        question: str,
        category,
        intent,
        companies=(),
        years=(),
        metrics=(),
        actions=(),
        tools=(),
        status="completed",
        stop_reason="completed",
        replay_required=True,
    ) -> None:
        case_id = (
            f"runtime_{len(cases) + 1:03d}"
        )

        cases.append(
            RuntimeEvalCase(
                case_id=case_id,
                question=question,
                category=category,
                expected_intent=intent,
                expected_company_ids=(
                    companies
                ),
                expected_years=years,
                expected_metric_ids=(
                    metrics
                ),
                expected_plan_actions=(
                    actions
                ),
                expected_tool_sequence=(
                    tools
                ),
                expected_final_status=(
                    status
                ),
                expected_stop_reason=(
                    stop_reason
                ),
                replay_required=(
                    replay_required
                ),
            )
        )

    fact_actions = (
        "retrieve",
    )

    fact_tools = (
        "query_financial_data",
    )

    multi_fact_actions = (
        "retrieve",
        "retrieve",
        "synthesize",
    )

    multi_fact_tools = (
        "query_financial_data",
        "query_financial_data",
    )

    calculation_actions = (
        "retrieve",
        "retrieve",
        "calculate",
    )

    calculation_tools = (
        "query_financial_data",
        "query_financial_data",
        "execute_calculation",
    )

    compare_fact_actions = (
        "retrieve",
        "retrieve",
        "compare",
    )

    rank_fact_actions = (
        "retrieve",
        "retrieve",
        "rank",
    )

    compare_fact_tools = (
        "query_financial_data",
        "query_financial_data",
    )

    compare_calculation_actions = (
        "retrieve",
        "retrieve",
        "calculate",
        "retrieve",
        "retrieve",
        "calculate",
        "compare",
    )

    rank_calculation_actions = (
        "retrieve",
        "retrieve",
        "calculate",
        "retrieve",
        "retrieve",
        "calculate",
        "rank",
    )

    compare_calculation_tools = (
        "query_financial_data",
        "query_financial_data",
        "execute_calculation",
        "query_financial_data",
        "query_financial_data",
        "execute_calculation",
    )

    document_actions = (
        "retrieve",
    )

    document_tools = (
        "retrieve_documents",
    )

    # 1~16：Financial Fact
    fact_specs = (
        (
            "美的集团2024年营业收入是多少？",
            ("midea_group",),
            (2024,),
            ("revenue",),
        ),
        (
            "美的2025年营收是多少？",
            ("midea_group",),
            (2025,),
            ("revenue",),
        ),
        (
            "格力电器2024年营业收入是多少？",
            ("gree_electric",),
            (2024,),
            ("revenue",),
        ),
        (
            "格力2025年营收是多少？",
            ("gree_electric",),
            (2025,),
            ("revenue",),
        ),
        (
            "海尔智家2024年营业收入是多少？",
            ("haier_smart_home",),
            (2024,),
            ("revenue",),
        ),
        (
            "海尔2025年营收是多少？",
            ("haier_smart_home",),
            (2025,),
            ("revenue",),
        ),
        (
            "海信家电2024年营业收入是多少？",
            ("hisense_home",),
            (2024,),
            ("revenue",),
        ),
        (
            "海信2025年营收是多少？",
            ("hisense_home",),
            (2025,),
            ("revenue",),
        ),
        (
            "老板电器2024年营业收入是多少？",
            ("robam",),
            (2024,),
            ("revenue",),
        ),
        (
            "老板2025年营收是多少？",
            ("robam",),
            (2025,),
            ("revenue",),
        ),
        (
            "苏泊尔2024年营业收入是多少？",
            ("supor",),
            (2024,),
            ("revenue",),
        ),
        (
            "苏泊尔2025年营收是多少？",
            ("supor",),
            (2025,),
            ("revenue",),
        ),
        (
            "美的集团2024年营业成本是多少？",
            ("midea_group",),
            (2024,),
            ("operating_cost",),
        ),
        (
            "格力电器2025年营业成本是多少？",
            ("gree_electric",),
            (2025,),
            ("operating_cost",),
        ),
    )

    for (
        question,
        companies,
        years,
        metrics,
    ) in fact_specs:
        add(
            question=question,
            category="financial_fact",
            intent="financial_fact",
            companies=companies,
            years=years,
            metrics=metrics,
            actions=fact_actions,
            tools=fact_tools,
        )

    add(
        question=(
            "美的集团2024年营业收入和"
            "营业成本分别是多少？"
        ),
        category="financial_fact",
        intent="financial_fact",
        companies=(
            "midea_group",
        ),
        years=(2024,),
        metrics=(
            "revenue",
            "operating_cost",
        ),
        actions=multi_fact_actions,
        tools=multi_fact_tools,
    )

    add(
        question=(
            "格力电器2024年营业收入和"
            "营业成本是多少？"
        ),
        category="financial_fact",
        intent="financial_fact",
        companies=(
            "gree_electric",
        ),
        years=(2024,),
        metrics=(
            "revenue",
            "operating_cost",
        ),
        actions=multi_fact_actions,
        tools=multi_fact_tools,
    )

    # 17~24：Calculation
    calculation_specs = (
        (
            "美的集团2024年毛利率是多少？",
            "midea_group",
            2024,
        ),
        (
            "美的2025年毛利率是多少？",
            "midea_group",
            2025,
        ),
        (
            "格力电器2024年毛利率是多少？",
            "gree_electric",
            2024,
        ),
        (
            "格力2025年毛利率是多少？",
            "gree_electric",
            2025,
        ),
        (
            "海尔智家2024年毛利率是多少？",
            "haier_smart_home",
            2024,
        ),
        (
            "海信家电2025年毛利率是多少？",
            "hisense_home",
            2025,
        ),
        (
            "老板电器2024年毛利率是多少？",
            "robam",
            2024,
        ),
        (
            "苏泊尔2025年毛利率是多少？",
            "supor",
            2025,
        ),
    )

    for (
        question,
        company_id,
        year,
    ) in calculation_specs:
        add(
            question=question,
            category=(
                "financial_calculation"
            ),
            intent=(
                "financial_calculation"
            ),
            companies=(
                company_id,
            ),
            years=(year,),
            metrics=(
                "gross_profit_margin",
            ),
            actions=(
                calculation_actions
            ),
            tools=(
                calculation_tools
            ),
        )

    # 25~30：Reported Comparison / Ranking
    add(
        question=(
            "比较美的集团2024年和"
            "2025年的营业收入"
        ),
        category="financial_comparison",
        intent="financial_comparison",
        companies=("midea_group",),
        years=(2024, 2025),
        metrics=("revenue",),
        actions=compare_fact_actions,
        tools=compare_fact_tools,
    )

    add(
        question=(
            "美的集团2024年和2025年"
            "营业收入最高的是哪一年？"
        ),
        category="financial_comparison",
        intent="financial_comparison",
        companies=("midea_group",),
        years=(2024, 2025),
        metrics=("revenue",),
        actions=rank_fact_actions,
        tools=compare_fact_tools,
    )

    add(
        question=(
            "格力电器2024年和2025年"
            "营业收入对比"
        ),
        category="financial_comparison",
        intent="financial_comparison",
        companies=("gree_electric",),
        years=(2024, 2025),
        metrics=("revenue",),
        actions=compare_fact_actions,
        tools=compare_fact_tools,
    )

    add(
        question=(
            "格力电器2024年和2025年"
            "营业收入最高的是哪一年？"
        ),
        category="financial_comparison",
        intent="financial_comparison",
        companies=("gree_electric",),
        years=(2024, 2025),
        metrics=("revenue",),
        actions=rank_fact_actions,
        tools=compare_fact_tools,
    )

    add(
        question=(
            "比较美的集团和格力电器"
            "2024年的营业收入"
        ),
        category="financial_comparison",
        intent="financial_comparison",
        companies=(
            "midea_group",
            "gree_electric",
        ),
        years=(2024,),
        metrics=("revenue",),
        actions=compare_fact_actions,
        tools=compare_fact_tools,
    )

    add(
        question=(
            "美的集团和格力电器"
            "2024年营业收入最高的是哪家？"
        ),
        category="financial_comparison",
        intent="financial_comparison",
        companies=(
            "midea_group",
            "gree_electric",
        ),
        years=(2024,),
        metrics=("revenue",),
        actions=rank_fact_actions,
        tools=compare_fact_tools,
    )

    # 31~34：Derived Comparison / Ranking
    add(
        question=(
            "比较美的集团2024年和"
            "2025年的毛利率"
        ),
        category="financial_comparison",
        intent="financial_comparison",
        companies=("midea_group",),
        years=(2024, 2025),
        metrics=(
            "gross_profit_margin",
        ),
        actions=(
            compare_calculation_actions
        ),
        tools=(
            compare_calculation_tools
        ),
    )

    add(
        question=(
            "美的集团2024年和2025年"
            "毛利率最高的是哪一年？"
        ),
        category="financial_comparison",
        intent="financial_comparison",
        companies=("midea_group",),
        years=(2024, 2025),
        metrics=(
            "gross_profit_margin",
        ),
        actions=(
            rank_calculation_actions
        ),
        tools=(
            compare_calculation_tools
        ),
    )

    add(
        question=(
            "比较海尔智家和海信家电"
            "2024年的毛利率"
        ),
        category="financial_comparison",
        intent="financial_comparison",
        companies=(
            "haier_smart_home",
            "hisense_home",
        ),
        years=(2024,),
        metrics=(
            "gross_profit_margin",
        ),
        actions=(
            compare_calculation_actions
        ),
        tools=(
            compare_calculation_tools
        ),
    )

    add(
        question=(
            "海尔智家和海信家电"
            "2024年谁的毛利率最高？"
        ),
        category="financial_comparison",
        intent="financial_comparison",
        companies=(
            "haier_smart_home",
            "hisense_home",
        ),
        years=(2024,),
        metrics=(
            "gross_profit_margin",
        ),
        actions=(
            rank_calculation_actions
        ),
        tools=(
            compare_calculation_tools
        ),
    )

    # 35~42：Document Evidence
    document_specs = (
        (
            "美的集团2024年主要经营风险有哪些？",
            "midea_group",
            2024,
            (),
        ),
        (
            "格力电器2025年管理层如何描述未来战略？",
            "gree_electric",
            2025,
            (),
        ),
        (
            "海尔智家2024年业务布局有哪些？",
            "haier_smart_home",
            2024,
            (),
        ),
        (
            "海信家电2025年竞争格局如何？",
            "hisense_home",
            2025,
            (),
        ),
        (
            "老板电器2024年有哪些经营风险？",
            "robam",
            2024,
            (),
        ),
        (
            "苏泊尔2025年未来展望是什么？",
            "supor",
            2025,
            (),
        ),
        (
            "为什么美的集团2024年营业收入增长？",
            "midea_group",
            2024,
            ("revenue",),
        ),
        (
            "为什么格力电器2025年营业收入下降？",
            "gree_electric",
            2025,
            ("revenue",),
        ),
    )

    for (
        question,
        company_id,
        year,
        metrics,
    ) in document_specs:
        add(
            question=question,
            category="document_evidence",
            intent="document_evidence",
            companies=(
                company_id,
            ),
            years=(year,),
            metrics=metrics,
            actions=document_actions,
            tools=document_tools,
        )

    # 43~46：Clarification
    add(
        question="营业收入是多少？",
        category="clarification",
        intent="financial_fact",
        metrics=("revenue",),
        status="awaiting_human",
        stop_reason=(
            "human_review_required"
        ),
        replay_required=False,
    )

    add(
        question="美的集团营业收入是多少？",
        category="clarification",
        intent="financial_fact",
        companies=("midea_group",),
        metrics=("revenue",),
        status="awaiting_human",
        stop_reason=(
            "human_review_required"
        ),
        replay_required=False,
    )

    add(
        question="2024年营业收入是多少？",
        category="clarification",
        intent="financial_fact",
        years=(2024,),
        metrics=("revenue",),
        status="awaiting_human",
        stop_reason=(
            "human_review_required"
        ),
        replay_required=False,
    )

    add(
        question="毛利率是多少？",
        category="clarification",
        intent=(
            "financial_calculation"
        ),
        metrics=(
            "gross_profit_margin",
        ),
        status="awaiting_human",
        stop_reason=(
            "human_review_required"
        ),
        replay_required=False,
    )

    # 47~50：Unsupported
    for question in (
        "帮我写一首诗",
        "解释一下快速排序",
        "今天天气怎么样？",
        "给我推荐一部电影",
    ):
        add(
            question=question,
            category="unsupported",
            intent="unsupported",
            status="refused",
            stop_reason="unsupported",
            replay_required=True,
        )

    if len(cases) != 50:
        raise RuntimeEvalError(
            "runtime_control_dev_v1 "
            "必须恰好包含 50 条："
            f"actual={len(cases)}"
        )

    return tuple(cases)