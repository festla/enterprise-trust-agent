from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.memory import (
    InMemorySaver,
)

from app.services.competition_dataset import (
    build_competition_question,
    load_competition_qa_excel,
)
from app.services.competition_excel_runtime import (
    load_competition_excel_runtime_resources,
)
from app.workflow.competition_agent_graph import (
    build_competition_agent_graph,
)
from app.workflow.competition_checkpoint import (
    build_competition_checkpoint_serializer,
)
from app.workflow.competition_runtime import (
    CompetitionAgentModelResources,
    load_competition_agent_runtime_resources,
)


QA_FILE = Path(
    "data/competition/private/qa/QA数据.xlsx"
)

ATTACHMENTS_ROOT = Path(
    "data/competition/private/attachments"
)

CORPUS_DIR = Path(
    "data/competition/processed/corpora/"
    "competition_corpus_8cbec682dfce4962"
)

INDEX_DIR = Path(
    "data/competition/processed/indexes/bm25/"
    "competition_bm25_index_07bc5262330bc2a5"
)


CASE_ID = "Q003"

THREAD_ID = (
    "competition-excel-checkpoint-Q003"
)


class NeverCalledSufficiencyProvider:
    provider_id = (
        "checkpoint-test:"
        "excel-never-called-sufficiency"
    )

    def __init__(
        self,
    ) -> None:
        self._request_count = 0

    @property
    def request_count(
        self,
    ) -> int:
        return self._request_count

    def assess(
        self,
        **kwargs,
    ):
        self._request_count += 1

        raise AssertionError(
            "Excel Agent 不应该调用 "
            "Sufficiency Provider"
        )


class NeverCalledAnswerProvider:
    provider_id = (
        "checkpoint-test:"
        "excel-never-called-answer"
    )

    def __init__(
        self,
    ) -> None:
        self._request_count = 0

    @property
    def request_count(
        self,
    ) -> int:
        return self._request_count

    def generate(
        self,
        **kwargs,
    ):
        self._request_count += 1

        raise AssertionError(
            "Excel Agent 不应该调用 "
            "Answer Provider"
        )


def main() -> None:
    # ========================================================
    # Load Case
    # ========================================================

    cases = tuple(
        load_competition_qa_excel(
            QA_FILE
        )
    )

    case = next(
        case
        for case in cases
        if case.case_id == CASE_ID
    )

    question = (
        build_competition_question(
            case
        )
    )

    if (
        question.source_type
        != "excel"
    ):
        raise RuntimeError(
            f"{CASE_ID} 不是 Excel Case"
        )

    # ========================================================
    # Process Resources
    # ========================================================

    runtime_resources = (
        load_competition_agent_runtime_resources(
            attachments_root=(
                ATTACHMENTS_ROOT
            ),
            corpus_directory=(
                CORPUS_DIR
            ),
            index_directory=(
                INDEX_DIR
            ),
        )
    )

    excel_resources = (
        load_competition_excel_runtime_resources(
            attachments_root=(
                ATTACHMENTS_ROOT
            )
        )
    )

    sufficiency_provider = (
        NeverCalledSufficiencyProvider()
    )

    answer_provider = (
        NeverCalledAnswerProvider()
    )

    model_resources = (
        CompetitionAgentModelResources(
            sufficiency_provider=(
                sufficiency_provider
            ),
            answer_provider=(
                answer_provider
            ),
        )
    )

    # ========================================================
    # Strict-safe Checkpointer
    # ========================================================

    serializer = (
        build_competition_checkpoint_serializer()
    )

    checkpointer = InMemorySaver(
        serde=serializer,
    )

    graph = (
        build_competition_agent_graph(
            runtime_resources=(
                runtime_resources
            ),
            excel_resources=(
                excel_resources
            ),
            model_resources=(
                model_resources
            ),
            checkpointer=(
                checkpointer
            ),
        )
    )

    config = {
        "configurable": {
            "thread_id": THREAD_ID,
        }
    }

    # ========================================================
    # Execute Excel Path
    # ========================================================

    final_state = graph.invoke(
        {
            "question": question,
        },
        config=config,
    )

    result = final_state[
        "result"
    ]

    # ========================================================
    # Force Checkpoint Deserialization
    # ========================================================

    snapshot = graph.get_state(
        config
    )

    history = list(
        graph.get_state_history(
            config
        )
    )

    trace = (
        snapshot
        .values
        .get(
            "trace",
            (),
        )
    )

    trace_stages = tuple(
        event.stage
        for event in trace
    )

    excel_runtime = (
        snapshot
        .values
        .get(
            "excel_runtime"
        )
    )

    evidence_bundle = (
        snapshot
        .values
        .get(
            "evidence_bundle"
        )
    )

    # ========================================================
    # Console
    # ========================================================

    print()
    print(
        "===== Competition Excel "
        "Checkpoint Smoke ====="
    )

    print(
        "Case:",
        CASE_ID,
    )

    print(
        "Thread:",
        THREAD_ID,
    )

    print(
        "QA type:",
        case.qa_type,
    )

    print(
        "Final status:",
        result.status,
    )

    print(
        "Agent answer:",
        (
            result.answer.answer
            if result.status
            == "answered"
            else None
        ),
    )

    print(
        "Gold answer:",
        case.answer,
    )

    print(
        "Trace:",
        trace_stages,
    )

    print(
        "Checkpoint count:",
        len(history),
    )

    print(
        "Latest next:",
        snapshot.next,
    )

    print(
        "Excel solver:",
        (
            excel_runtime.solver_type
            if excel_runtime
            is not None
            else None
        ),
    )

    print(
        "Evidence count:",
        (
            len(
                evidence_bundle.evidences
            )
            if evidence_bundle
            is not None
            else 0
        ),
    )

    print(
        "Sufficiency requests:",
        sufficiency_provider
        .request_count,
    )

    print(
        "Answer requests:",
        answer_provider
        .request_count,
    )

    # ========================================================
    # Acceptance
    # ========================================================

    if (
        result.status
        != "answered"
    ):
        raise RuntimeError(
            "Excel Checkpoint "
            "没有正常回答"
        )

    if (
        result.answer.answer
        != case.answer
    ):
        raise RuntimeError(
            "Excel Checkpoint "
            "答案错误"
        )

    expected_trace = (
        "source_resolution",
        "excel_solver",
        "citation",
    )

    if (
        trace_stages
        != expected_trace
    ):
        raise RuntimeError(
            "Excel Checkpoint Trace "
            "异常："
            f"{trace_stages}"
        )

    if snapshot.next:
        raise RuntimeError(
            "Excel Graph 完成后仍有 "
            "pending node："
            f"{snapshot.next}"
        )

    if (
        len(history)
        < 2
    ):
        raise RuntimeError(
            "Excel Graph 没有产生"
            "足够 checkpoint"
        )

    if (
        sufficiency_provider
        .request_count
        != 0
    ):
        raise RuntimeError(
            "Excel 路径错误调用了 "
            "Sufficiency LLM"
        )

    if (
        answer_provider
        .request_count
        != 0
    ):
        raise RuntimeError(
            "Excel 路径错误调用了 "
            "Answer LLM"
        )

    if (
        excel_runtime
        is None
    ):
        raise RuntimeError(
            "Excel Runtime State 缺失"
        )

    if (
        excel_runtime.solver_type
        != "lookup"
    ):
        raise RuntimeError(
            "Q003 Solver 路由异常："
            f"{excel_runtime.solver_type}"
        )

    if (
        evidence_bundle
        is None
        or not (
            evidence_bundle.evidences
        )
    ):
        raise RuntimeError(
            "Excel Checkpoint "
            "缺少 Evidence"
        )

    print()
    print(
        "EXCEL CHECKPOINT PASS"
    )


if __name__ == "__main__":
    main()