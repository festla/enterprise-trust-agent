from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
from decimal import Decimal
from pathlib import Path

from app.schemas.company import (
    Company,
)
from app.schemas.complex_plan_eval_result import (
    ComplexCalculationTrace,
)
from app.schemas.enums import (
    MetricOrigin,
    ReportType,
    StatementScope,
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
from app.schemas.metric import (
    FinancialMetric,
)
from app.schemas.report import (
    Report,
)
from app.services.agent_runtime import (
    AgentRuntime,
)
from app.services.calculation_tool import (
    register_execute_calculation_tool,
)
from app.services.checkpoint_store import (
    InMemoryCheckpointStore,
    SQLiteCheckpointStore,
)
from app.services.financial_data_tool import (
    register_query_financial_data_tool,
)
from app.services.registry import (
    RegistryBundle,
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
from app.services.runtime_policy import (
    RuntimeRiskPolicy,
)
from app.services.runtime_planner import (
    RuntimePlanner,
)
from app.services.runtime_query_parser import (
    RuntimeQueryParser,
)
from app.services.tool_registry import (
    InMemoryToolResultCache,
    ToolExecutor,
    ToolRegistry,
)
from app.services.trajectory_store import (
    TrajectoryStore,
)
from app.services.runtime_answer_draft import (
    RuntimeAnswerDraftBuilder,
)
from app.services.runtime_trust_verifier import (
    RuntimeTrustVerifier,
)

class FixedClock:
    def now(
        self,
    ) -> datetime:
        return datetime(
            2026,
            8,
            12,
            8,
            0,
            tzinfo=timezone.utc,
        )


class SequentialIdFactory:
    def __init__(
        self,
    ) -> None:
        self._values: dict[
            str,
            int,
        ] = {}

    def new_id(
        self,
        prefix: str,
    ) -> str:
        next_value = (
            self._values.get(
                prefix,
                0,
            )
            + 1
        )

        self._values[
            prefix
        ] = next_value

        return (
            f"{prefix}_{next_value}"
        )


class FakeCalculationProvider:
    def calculate(
        self,
        *,
        calculation_id: str,
        formula_id: str,
        input_fact_ids: tuple[
            str,
            ...
        ],
    ) -> ComplexCalculationTrace:
        return ComplexCalculationTrace(
            calculation_id=(
                calculation_id
            ),
            metric_id=(
                "gross_profit_margin"
            ),
            formula_id=formula_id,
            input_fact_ids=(
                input_fact_ids
            ),
            status="completed",
            result_value=(
                Decimal("20.7768")
            ),
            result_unit="percent",
            latency_ms=1.0,
            error_message=None,
        )


class CountingToolExecutor:
    """记录 Runtime 实际请求执行了哪些 Tool Step。"""

    def __init__(
        self,
        delegate: ToolExecutor,
    ) -> None:
        self.delegate = delegate

        self.calls: list[
            tuple[str, str]
        ] = []

    def execute(
        self,
        **kwargs,
    ):
        self.calls.append(
            (
                kwargs["tool_name"],
                kwargs["step_id"],
            )
        )

        return self.delegate.execute(
            **kwargs
        )

    def count_step(
        self,
        step_id: str,
    ) -> int:
        return sum(
            1
            for _, recorded_step_id
            in self.calls
            if recorded_step_id
            == step_id
        )


def _add_fact(
    bundle: RegistryBundle,
    *,
    metric_id: str,
    value: str,
) -> None:
    fact_id = (
        f"fact_midea_group_"
        f"2024_{metric_id}"
    )

    evidence_id = (
        f"evidence_midea_group_"
        f"2024_{metric_id}"
    )

    bundle.financial_facts.add(
        FinancialFact.model_construct(
            fact_id=fact_id,
            company_id="midea_group",
            report_id=(
                "midea_group_2024"
            ),
            metric_id=metric_id,
            fiscal_year=2024,
            statement_type=(
                StatementType
                .INCOME_STATEMENT
            ),
            statement_scope=(
                StatementScope
                .CONSOLIDATED
            ),
            normalized_value=(
                Decimal(value)
            ),
            normalized_unit=(
                UnitCode.CNY
            ),
            primary_evidence_id=(
                evidence_id
            ),
            validation_status=(
                ValidationStatus
                .VERIFIED
            ),
        )
    )

    bundle.evidences.add(
        SourceEvidence.model_construct(
            evidence_id=evidence_id,
            report_id=(
                "midea_group_2024"
            ),
            document_id=(
                "document_midea_2024"
            ),
            page_id=(
                f"page_midea_2024_"
                f"{metric_id}"
            ),
            chunk_id=(
                f"chunk_midea_2024_"
                f"{metric_id}"
            ),
            pdf_page=158,
            printed_page=157,
            evidence_text=(
                f"美的集团2024年"
                f"{metric_id}为{value}元。"
            ),
            validation_status=(
                ValidationStatus
                .VERIFIED
            ),
        )
    )


def _build_bundle(
) -> RegistryBundle:
    bundle = RegistryBundle()

    bundle.companies.add(
        Company.model_construct(
            company_id="midea_group",
            legal_name_cn=(
                "美的集团股份有限公司"
            ),
            short_name_cn="美的集团",
            stock_code="000333",
        )
    )

    bundle.reports.add(
        Report.model_construct(
            report_id=(
                "midea_group_2024"
            ),
            company_id="midea_group",
            fiscal_year=2024,
            report_type=(
                ReportType.ANNUAL_REPORT
            ),
        )
    )

    bundle.metrics.add(
        FinancialMetric.model_construct(
            metric_id="revenue",
            display_name_cn="营业收入",
            display_name_en="Revenue",
            metric_origin=(
                MetricOrigin.REPORTED
            ),
            statement_type=(
                StatementType
                .INCOME_STATEMENT
            ),
            allowed_scopes=[
                StatementScope
                .CONSOLIDATED,
                StatementScope
                .PARENT_COMPANY,
            ],
            formula_id=None,
        )
    )

    bundle.metrics.add(
        FinancialMetric.model_construct(
            metric_id="operating_cost",
            display_name_cn="营业成本",
            display_name_en=(
                "Operating Cost"
            ),
            metric_origin=(
                MetricOrigin.REPORTED
            ),
            statement_type=(
                StatementType
                .INCOME_STATEMENT
            ),
            allowed_scopes=[
                StatementScope
                .CONSOLIDATED,
                StatementScope
                .PARENT_COMPANY,
            ],
            formula_id=None,
        )
    )

    bundle.metrics.add(
        FinancialMetric.model_construct(
            metric_id=(
                "gross_profit_margin"
            ),
            display_name_cn="毛利率",
            display_name_en=(
                "Gross Profit Margin"
            ),
            metric_origin=(
                MetricOrigin.DERIVED
            ),
            statement_type=(
                StatementType
                .INCOME_STATEMENT
            ),
            allowed_scopes=[
                StatementScope
                .CONSOLIDATED,
                StatementScope
                .PARENT_COMPANY,
            ],
            formula_id=(
                "gross_profit_margin_formula"
            ),
        )
    )

    _add_fact(
        bundle,
        metric_id="revenue",
        value="407149600000",
    )

    _add_fact(
        bundle,
        metric_id="operating_cost",
        value="322560000000",
    )

    return bundle


_DEFAULT_TRAJECTORY = object()


def _build_runtime(
    tmp_path: Path,
    *,
    checkpoint_store,
    trajectory_store=(
        _DEFAULT_TRAJECTORY
    ),
    result_cache=None,
):
    bundle = _build_bundle()

    clock = FixedClock()

    id_factory = (
        SequentialIdFactory()
    )

    tool_registry = ToolRegistry()

    register_query_financial_data_tool(
        tool_registry=tool_registry,
        registry_bundle=bundle,
    )

    register_execute_calculation_tool(
        tool_registry=tool_registry,
        calculation_provider=(
            FakeCalculationProvider()
        ),
    )

    base_executor = ToolExecutor(
        tool_registry,
        result_cache=result_cache,
        retry_backoff_seconds=0,
    )

    counting_executor = (
        CountingToolExecutor(
            base_executor
        )
    )

    plan_executor = (
        RuntimePlanExecutor(
            tool_executor=(
                counting_executor
            ),
            granted_permissions=(
                frozenset(
                    {
                        "read_financial_data",
                        "execute_calculation",
                    }
                )
            ),
            registry_bundle=bundle,
            clock=clock,
            id_factory=id_factory,
        )
    )

    if (
        trajectory_store
        is _DEFAULT_TRAJECTORY
    ):
        trajectory_store = (
            TrajectoryStore(
                tmp_path
                / "trajectories"
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
                registry_bundle=bundle,
                clock=clock,
                id_factory=id_factory,
            )
        ),
        answer_draft_builder=(
            RuntimeAnswerDraftBuilder(
                registry_bundle=bundle
            )
        ),
        trust_verifier=(
            RuntimeTrustVerifier(
                registry_bundle=bundle
            )
        ),

        risk_policy=(
            RuntimeRiskPolicy(
                id_factory=id_factory
            )
        ),

        answer_generator=(
            RuntimeAnswerGenerator(
                registry_bundle=bundle,
                clock=clock,
                id_factory=id_factory,
            )
        ),
        checkpoint_store=(
            checkpoint_store
        ),
        trajectory_store=(
            trajectory_store
        ),
        clock=clock,
        id_factory=id_factory,
    )

    return (
        runtime,
        counting_executor,
        trajectory_store,
    )


def test_resume_skips_checkpointed_completed_step(
    tmp_path: Path,
) -> None:
    checkpoint_store = (
        InMemoryCheckpointStore()
    )

    (
        runtime,
        counter,
        _,
    ) = _build_runtime(
        tmp_path,
        checkpoint_store=(
            checkpoint_store
        ),
    )

    state = runtime.prepare(
        query=(
            "美的集团2024年"
            "营业收入和营业成本是多少？"
        ),
        run_id=(
            "run_resume_skip"
        ),
        thread_id=(
            "thread_resume_skip"
        ),
    )

    assert runtime.plan_executor is not None

    # 执行 s1，并明确保存 Checkpoint。
    state = (
        runtime.plan_executor
        .execute_next_step(
            state
        )
    )

    state = (
        runtime._persist_checkpoint(
            state
        )
    )

    assert state.current_step == 1

    assert (
        counter.count_step("s1")
        == 1
    )

    recovered = runtime.resume(
        run_id="run_resume_skip",
        thread_id=(
            "thread_resume_skip"
        ),
    )

    assert (
        recovered.status
        == "completed"
    )

    # s1 已经在 Checkpoint 中完成，
    # 所以恢复后不能重新请求 s1。
    assert (
        counter.count_step("s1")
        == 1
    )

    assert (
        counter.count_step("s2")
        == 1
    )

    assert (
        recovered.completed_step_ids
        == (
            "s1",
            "s2",
            "s3",
        )
    )


def test_resume_reuses_tool_when_success_was_not_checkpointed(
    tmp_path: Path,
) -> None:
    checkpoint_store = (
        InMemoryCheckpointStore()
    )

    result_cache = (
        InMemoryToolResultCache()
    )

    (
        runtime,
        counter,
        _,
    ) = _build_runtime(
        tmp_path,
        checkpoint_store=(
            checkpoint_store
        ),
        result_cache=result_cache,
    )

    prepared = runtime.prepare(
        query=(
            "美的集团2024年"
            "营业收入是多少？"
        ),
        run_id=(
            "run_crash_gap"
        ),
        thread_id=(
            "thread_crash_gap"
        ),
    )

    assert runtime.plan_executor is not None

    # Tool 已经真实成功，
    # 但模拟在新的 Checkpoint 写入前进程退出。
    uncheckpointed = (
        runtime.plan_executor
        .execute_next_step(
            prepared
        )
    )

    assert (
        uncheckpointed
        .tool_call_traces[-1]
        .status
        == "succeeded"
    )

    # 故意不调用 _persist_checkpoint。
    recovered = runtime.resume(
        run_id="run_crash_gap",
        thread_id=(
            "thread_crash_gap"
        ),
    )

    assert (
        recovered.status
        == "completed"
    )

    # Runtime 的确再次请求了 s1，
    # 但 ToolExecutor 根据幂等键复用了缓存。
    assert (
        counter.count_step("s1")
        == 2
    )

    s1_traces = tuple(
        trace
        for trace
        in recovered.tool_call_traces
        if trace.step_id == "s1"
    )

    assert len(s1_traces) == 1

    assert (
        s1_traces[0].status
        == "reused"
    )


def test_sqlite_checkpoint_recovers_across_runtime_instances(
    tmp_path: Path,
) -> None:
    database_path = (
        tmp_path
        / "runtime_checkpoints.db"
    )

    trajectory_store = (
        TrajectoryStore(
            tmp_path
            / "sqlite_trajectories"
        )
    )

    store_a = (
        SQLiteCheckpointStore(
            database_path
        )
    )

    (
        runtime_a,
        _,
        _,
    ) = _build_runtime(
        tmp_path,
        checkpoint_store=store_a,
        trajectory_store=(
            trajectory_store
        ),
    )

    state = runtime_a.prepare(
        query=(
            "美的集团2024年"
            "毛利率是多少？"
        ),
        run_id=(
            "run_sqlite_recovery"
        ),
        thread_id=(
            "thread_sqlite_recovery"
        ),
    )

    assert (
        runtime_a.plan_executor
        is not None
    )

    # Runtime A 只完成第一步。
    state = (
        runtime_a.plan_executor
        .execute_next_step(
            state
        )
    )

    state = (
        runtime_a
        ._persist_checkpoint(
            state
        )
    )

    assert state.current_step == 1

    # 模拟新的 Python 进程：
    # 创建新的 Store / Runtime / ToolExecutor。
    store_b = (
        SQLiteCheckpointStore(
            database_path
        )
    )

    (
        runtime_b,
        counter_b,
        _,
    ) = _build_runtime(
        tmp_path,
        checkpoint_store=store_b,
        trajectory_store=(
            trajectory_store
        ),
    )

    recovered = runtime_b.resume(
        run_id=(
            "run_sqlite_recovery"
        ),
        thread_id=(
            "thread_sqlite_recovery"
        ),
    )

    assert (
        recovered.status
        == "completed"
    )

    # 新 Runtime 直接从 s2 开始。
    assert (
        counter_b.count_step("s1")
        == 0
    )

    assert (
        counter_b.count_step("s2")
        == 1
    )

    assert (
        counter_b.count_step("s3")
        == 1
    )

    latest = store_b.load_latest(
        run_id=(
            "run_sqlite_recovery"
        ),
        thread_id=(
            "thread_sqlite_recovery"
        ),
    )

    assert (
        latest.state.status
        == "completed"
    )

    assert (
        latest.state.current_step
        == 3
    )


def test_terminal_checkpoint_resume_finalizes_missing_trajectory(
    tmp_path: Path,
) -> None:
    checkpoint_store = (
        InMemoryCheckpointStore()
    )

    # 第一个 Runtime 故意没有 TrajectoryStore。
    (
        runtime_a,
        _,
        _,
    ) = _build_runtime(
        tmp_path,
        checkpoint_store=(
            checkpoint_store
        ),
        trajectory_store=None,
    )

    completed = runtime_a.run(
        query=(
            "美的集团2024年"
            "营业收入是多少？"
        ),
        run_id=(
            "run_terminal_recovery"
        ),
        thread_id=(
            "thread_terminal_recovery"
        ),
    )

    assert (
        completed.status
        == "completed"
    )

    trajectory_store = (
        TrajectoryStore(
            tmp_path
            / "terminal_trajectories"
        )
    )

    (
        runtime_b,
        _,
        _,
    ) = _build_runtime(
        tmp_path,
        checkpoint_store=(
            checkpoint_store
        ),
        trajectory_store=(
            trajectory_store
        ),
    )

    # terminal Checkpoint 已经存在，
    # resume 只需要补写 Trajectory。
    restored = runtime_b.resume(
        run_id=(
            "run_terminal_recovery"
        ),
        thread_id=(
            "thread_terminal_recovery"
        ),
    )

    assert (
        restored.status
        == "completed"
    )

    trajectory = (
        trajectory_store.load(
            "run_terminal_recovery"
        )
    )

    assert (
        trajectory.final_status
        == "completed"
    )

    # 再次 resume 不能因为 Trajectory
    # 已经存在而报错。
    restored_again = (
        runtime_b.resume(
            run_id=(
                "run_terminal_recovery"
            ),
            thread_id=(
                "thread_terminal_recovery"
            ),
        )
    )

    assert (
        restored_again.status
        == "completed"
    )

    assert (
        trajectory_store.list_run_ids()
        == (
            "run_terminal_recovery",
        )
    )


def test_recovered_calculation_trajectory_can_be_replayed(
    tmp_path: Path,
) -> None:
    checkpoint_store = (
        InMemoryCheckpointStore()
    )

    trajectory_store = (
        TrajectoryStore(
            tmp_path
            / "replay_trajectories"
        )
    )

    (
        runtime,
        _,
        _,
    ) = _build_runtime(
        tmp_path,
        checkpoint_store=(
            checkpoint_store
        ),
        trajectory_store=(
            trajectory_store
        ),
    )

    state = runtime.prepare(
        query=(
            "美的集团2024年"
            "毛利率是多少？"
        ),
        run_id=(
            "run_replay_recovery"
        ),
        thread_id=(
            "thread_replay_recovery"
        ),
    )

    assert runtime.plan_executor is not None

    state = (
        runtime.plan_executor
        .execute_next_step(
            state
        )
    )

    state = (
        runtime._persist_checkpoint(
            state
        )
    )

    recovered = runtime.resume(
        run_id=(
            "run_replay_recovery"
        ),
        thread_id=(
            "thread_replay_recovery"
        ),
    )

    assert (
        recovered.status
        == "completed"
    )

    replay = (
        trajectory_store.replay(
            recovered.run_id
        )
    )

    assert replay.nodes == (
        "parse_query",
        "route_intent",
        "create_plan",
        "execute_plan",
        "execute_plan",
        "execute_plan",
        "verify_evidence",
        "prepare_answer",
        "verify_answer",
        "evaluate_policy",
        "generate_answer",
    )

    assert replay.tools == (
        "query_financial_data",
        "query_financial_data",
        "execute_calculation",
    )

    assert replay.supporting_fact_ids == (
        "fact_midea_group_2024_revenue",
        (
            "fact_midea_group_"
            "2024_operating_cost"
        ),
    )

    assert replay.evidence_ids == (
        (
            "evidence_midea_group_"
            "2024_revenue"
        ),
        (
            "evidence_midea_group_"
            "2024_operating_cost"
        ),
    )

    assert replay.calculation_ids == (
        (
            "calculation_midea_group_"
            "2024_gross_profit_margin"
        ),
    )

    assert (
        replay.final_status
        == "completed"
    )

    assert (
        replay.stop_reason
        == "completed"
    )