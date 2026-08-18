from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
from decimal import Decimal
from pathlib import Path

from langgraph.checkpoint.memory import (
    InMemorySaver,
)
from langgraph.types import (
    Command,
)

from app.schemas.company import (
    Company,
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
from app.services.checkpoint_store import (
    InMemoryCheckpointStore,
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
from app.services.runtime_graph import (
    build_agent_runtime_graph,
    create_runtime_graph_input,
    extract_agent_state,
    runtime_graph_config,
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
from app.services.tool_registry import (
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
from app.services.runtime_policy import (
    RuntimeRiskPolicy,
)

class FixedClock:
    def now(
        self,
    ) -> datetime:
        return datetime(
            2026,
            8,
            11,
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
        value = (
            self._values.get(
                prefix,
                0,
            )
            + 1
        )

        self._values[
            prefix
        ] = value

        return (
            f"{prefix}_{value}"
        )


def _build_bundle(
) -> RegistryBundle:
    bundle = RegistryBundle()

    bundle.companies.add(
        Company.model_construct(
            company_id=(
                "midea_group"
            ),
            legal_name_cn=(
                "美的集团股份有限公司"
            ),
            short_name_cn=(
                "美的集团"
            ),
            stock_code="000333",
        )
    )

    bundle.reports.add(
        Report.model_construct(
            report_id=(
                "midea_group_2024"
            ),
            company_id=(
                "midea_group"
            ),
            fiscal_year=2024,
            report_type=(
                ReportType
                .ANNUAL_REPORT
            ),
        )
    )

    bundle.metrics.add(
        FinancialMetric.model_construct(
            metric_id="revenue",
            display_name_cn=(
                "营业收入"
            ),
            display_name_en=(
                "Revenue"
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

    bundle.financial_facts.add(
        FinancialFact.model_construct(
            fact_id=(
                "fact_midea_group_"
                "2024_revenue"
            ),
            company_id=(
                "midea_group"
            ),
            report_id=(
                "midea_group_2024"
            ),
            metric_id="revenue",
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
                Decimal(
                    "407149600000"
                )
            ),
            normalized_unit=(
                UnitCode.CNY
            ),
            primary_evidence_id=(
                "evidence_midea_group_"
                "2024_revenue"
            ),
            validation_status=(
                ValidationStatus
                .VERIFIED
            ),
        )
    )

    bundle.evidences.add(
        SourceEvidence.model_construct(
            evidence_id=(
                "evidence_midea_group_"
                "2024_revenue"
            ),
            report_id=(
                "midea_group_2024"
            ),
            document_id=(
                "document_midea_2024"
            ),
            page_id=(
                "page_midea_2024_158"
            ),
            chunk_id=(
                "chunk_midea_2024_revenue"
            ),
            pdf_page=158,
            printed_page=157,
            evidence_text=(
                "美的集团2024年营业收入"
                "为407149600000元。"
            ),
            validation_status=(
                ValidationStatus
                .VERIFIED
            ),
        )
    )

    return bundle


def _build_runtime(
    tmp_path: Path,
):
    bundle = _build_bundle()

    clock = FixedClock()

    id_factory = (
        SequentialIdFactory()
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

    tool_executor = ToolExecutor(
        tool_registry,
        retry_backoff_seconds=0,
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
                    }
                )
            ),
            registry_bundle=bundle,
            clock=clock,
            id_factory=id_factory,
        )
    )

    checkpoint_store = (
        InMemoryCheckpointStore()
    )

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
                id_factory=(
                    id_factory
                ),
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
                id_factory=(
                    id_factory
                )
            )
        ),


        answer_generator=(
            RuntimeAnswerGenerator(
                registry_bundle=bundle,
                clock=clock,
                id_factory=(
                    id_factory
                ),
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
        checkpoint_store,
        trajectory_store,
    )


def test_graph_runs_financial_fact_end_to_end(
    tmp_path: Path,
) -> None:
    (
        runtime,
        checkpoint_store,
        trajectory_store,
    ) = _build_runtime(
        tmp_path
    )

    graph = (
        build_agent_runtime_graph(
            runtime,
            checkpointer=(
                InMemorySaver()
            ),
        )
    )

    initial = (
        create_runtime_graph_input(
            runtime,
            query=(
                "美的集团2024年"
                "营业收入是多少？"
            ),
            run_id=(
                "run_graph_fact"
            ),
            thread_id=(
                "thread_graph_fact"
            ),
        )
    )

    config = runtime_graph_config(
        "thread_graph_fact"
    )

    result = graph.invoke(
        initial,
        config=config,
    )

    state = extract_agent_state(
        result
    )

    assert (
        state.status
        == "completed"
    )

    assert (
        state.stop_reason
        == "completed"
    )

    assert state.answer is not None

    assert (
        "407149600000"
        in state.answer.answer_text
    )

    assert (
        state.answer_draft
        is not None
    )

    assert (
        state.answer_draft
        .claims[0]
        .claim_text
        ==
        "美的集团2024年"
        "营业收入为407149600000元"
    )

    assert tuple(
        span.node_name
        for span
        in state.node_spans
    ) == (
        "parse_query",
        "route_intent",
        "create_plan",
        "execute_plan",
        "verify_evidence",
        "prepare_answer",
        "verify_answer",
        "evaluate_policy",
        "generate_answer",
    )

    assert (
        state.risk_level
        == "low"
    )

    assert (
        state.policy_decision
        is not None
    )

    assert (
        state.policy_decision.action
        == "allow"
    )

    assert (
        state.verification_report
        is not None
    )

    assert (
        state.verification_report
        .passed
        is True
    )
    latest = (
        checkpoint_store
        .load_latest(
            run_id=(
                state.run_id
            ),
            thread_id=(
                state.thread_id
            ),
        )
    )

    assert (
        latest.state.status
        == "completed"
    )

    trajectory = (
        trajectory_store.load(
            state.run_id
        )
    )

    assert (
        trajectory.final_status
        == "completed"
    )


def test_graph_routes_unsupported_to_finish(
    tmp_path: Path,
) -> None:
    (
        runtime,
        _,
        trajectory_store,
    ) = _build_runtime(
        tmp_path
    )

    graph = (
        build_agent_runtime_graph(
            runtime,
            checkpointer=(
                InMemorySaver()
            ),
        )
    )

    initial = (
        create_runtime_graph_input(
            runtime,
            query="帮我写一首诗",
            run_id=(
                "run_graph_unsupported"
            ),
            thread_id=(
                "thread_graph_unsupported"
            ),
        )
    )

    result = graph.invoke(
        initial,
        config=(
            runtime_graph_config(
                (
                    "thread_graph_"
                    "unsupported"
                )
            )
        ),
    )

    state = extract_agent_state(
        result
    )

    assert (
        state.status
        == "refused"
    )

    assert (
        state.stop_reason
        == "unsupported"
    )

    assert (
        state.runtime_plan
        is None
    )

    trajectory = (
        trajectory_store.load(
            state.run_id
        )
    )

    assert (
        trajectory.final_status
        == "refused"
    )


def test_graph_interrupt_resumes_with_corrected_query(
    tmp_path: Path,
) -> None:
    (
        runtime,
        _,
        trajectory_store,
    ) = _build_runtime(
        tmp_path
    )

    graph = (
        build_agent_runtime_graph(
            runtime,
            checkpointer=(
                InMemorySaver()
            ),
        )
    )

    initial = (
        create_runtime_graph_input(
            runtime,
            query="营业收入是多少？",
            run_id=(
                "run_graph_human_resume"
            ),
            thread_id=(
                "thread_graph_human_resume"
            ),
        )
    )

    config = runtime_graph_config(
        "thread_graph_human_resume"
    )

    interrupted = graph.invoke(
        initial,
        config=config,
    )

    assert (
        "__interrupt__"
        in interrupted
    )

    waiting_state = (
        extract_agent_state(
            interrupted
        )
    )

    assert (
        waiting_state.status
        == "awaiting_human"
    )

    assert (
        waiting_state
        .pending_human_review
        is True
    )

    resumed = graph.invoke(
        Command(
            resume={
                "approved": True,
                "corrected_query": (
                    "美的集团2024年"
                    "营业收入是多少？"
                ),
                "reason": (
                    "补充公司和年份"
                ),
                "reviewer_id": (
                    "test_user"
                ),
            }
        ),
        config=config,
    )

    state = extract_agent_state(
        resumed
    )

    assert (
        state.status
        == "completed"
    )

    assert (
        state.run_id
        == "run_graph_human_resume"
    )

    assert (
        state.thread_id
        == (
            "thread_graph_human_resume"
        )
    )

    assert (
        state.human_decision
        is not None
    )

    assert (
        state.human_decision
        .approved
        is True
    )

    assert (
        state.query
        == (
            "美的集团2024年"
            "营业收入是多少？"
        )
    )

    assert tuple(
        span.node_name
        for span
        in state.node_spans
    ) == (
        "parse_query",
        "route_intent",
        "await_human",
        "parse_query",
        "route_intent",
        "create_plan",
        "execute_plan",
        "verify_evidence",
        "prepare_answer",
        "verify_answer",
        "evaluate_policy",
        "generate_answer",
    )

    assert (
        state.verification_report
        is not None
    )

    assert (
        state.verification_report
        .passed
        is True
    )

    trajectory = (
        trajectory_store.load(
            state.run_id
        )
    )

    assert (
        trajectory.final_status
        == "completed"
    )


def test_graph_interrupt_can_be_rejected(
    tmp_path: Path,
) -> None:
    (
        runtime,
        _,
        trajectory_store,
    ) = _build_runtime(
        tmp_path
    )

    graph = (
        build_agent_runtime_graph(
            runtime,
            checkpointer=(
                InMemorySaver()
            ),
        )
    )

    initial = (
        create_runtime_graph_input(
            runtime,
            query="营业收入是多少？",
            run_id=(
                "run_graph_human_reject"
            ),
            thread_id=(
                "thread_graph_human_reject"
            ),
        )
    )

    config = runtime_graph_config(
        "thread_graph_human_reject"
    )

    interrupted = graph.invoke(
        initial,
        config=config,
    )

    assert (
        "__interrupt__"
        in interrupted
    )

    resumed = graph.invoke(
        Command(
            resume={
                "approved": False,
                "reason": (
                    "用户取消本次查询"
                ),
                "reviewer_id": (
                    "test_user"
                ),
            }
        ),
        config=config,
    )

    state = extract_agent_state(
        resumed
    )

    assert (
        state.status
        == "refused"
    )

    assert (
        state.stop_reason
        == "human_rejected"
    )

    assert (
        state.pending_human_review
        is False
    )

    assert (
        state.human_decision
        is not None
    )

    assert (
        state.human_decision
        .approved
        is False
    )

    trajectory = (
        trajectory_store.load(
            state.run_id
        )
    )

    assert (
        trajectory.final_status
        == "refused"
    )

    assert (
        trajectory.stop_reason
        == "human_rejected"
    )