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
from app.workflow.competition_runtime import (
    load_competition_agent_model_resources_from_environment,
    load_competition_agent_runtime_resources,
)
from app.workflow.competition_checkpoint import (
    build_competition_checkpoint_serializer,
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


CASE_ID = "Q105"

THREAD_ID = (
    "competition-checkpoint-Q105"
)


def main() -> None:
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

    model_resources = (
        load_competition_agent_model_resources_from_environment()
    )

    # ==========================================
    # Checkpointer
    # ==========================================

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

    # ==========================================
    # Execute
    # ==========================================

    final_state = graph.invoke(
        {
            "question": question,
        },
        config=config,
    )

    result = final_state[
        "result"
    ]

    if (
        result.status
        != "answered"
    ):
        raise RuntimeError(
            "Checkpoint Smoke "
            "没有正常回答"
        )

    if (
        result.answer.answer
        != case.answer
    ):
        raise RuntimeError(
            "Checkpoint Smoke "
            "答案回归失败"
        )

    # ==========================================
    # Latest Snapshot
    # ==========================================

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

    print()
    print(
        "===== Competition Agent "
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
        "Final status:",
        result.status,
    )

    print(
        "Agent answer:",
        result.answer.answer,
    )

    print(
        "Gold answer:",
        case.answer,
    )

    print(
        "Latest next:",
        snapshot.next,
    )

    print(
        "Checkpoint count:",
        len(history),
    )

    print(
        "Trace:",
        trace_stages,
    )

    print(
        "Latest state keys:",
        tuple(
            sorted(
                snapshot
                .values
                .keys()
            )
        ),
    )

    if snapshot.next:
        raise RuntimeError(
            "完成后的 Graph "
            "不应该仍有 pending node"
        )

    if (
        len(history)
        < 2
    ):
        raise RuntimeError(
            "没有生成足够的 "
            "checkpoint history"
        )

    expected_trace = (
        "source_resolution",
        "retrieval",
        "evidence_assembly",
        "readiness",
        "sufficiency",
        "answer",
        "citation",
    )

    if (
        trace_stages
        != expected_trace
    ):
        raise RuntimeError(
            "Checkpoint 后的 Trace "
            "发生回归："
            f"{trace_stages}"
        )

    print()
    print(
        "CHECKPOINT PERSISTENCE PASS"
    )


if __name__ == "__main__":
    main()