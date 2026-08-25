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
    CompetitionAgentModelResources,
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
    "competition-resume-Q105"
)


class InjectedCheckpointFailure(
    Exception
):
    """
    专门用于验证 LangGraph
    checkpoint/resume。

    这不是业务 Failure Code，
    也不是 Retry Policy。
    """
    pass


class FailOnceSufficiencyProvider:
    def __init__(
        self,
        delegate,
    ) -> None:
        self._delegate = delegate
        self._attempt_count = 0

    @property
    def provider_id(
        self,
    ) -> str:
        return (
            "checkpoint-test:"
            f"{self._delegate.provider_id}"
        )

    @property
    def request_count(
        self,
    ) -> int:
        return self._attempt_count

    def assess(
        self,
        *,
        question: str,
        options: dict[str, str],
        generation_context: str,
        allowed_citation_ids: tuple[
            str,
            ...,
        ],
    ):
        self._attempt_count += 1

        if (
            self._attempt_count
            == 1
        ):
            raise (
                InjectedCheckpointFailure(
                    "Injected failure "
                    "before sufficiency "
                    "provider execution"
                )
            )

        return self._delegate.assess(
            question=question,
            options=options,
            generation_context=(
                generation_context
            ),
            allowed_citation_ids=(
                allowed_citation_ids
            ),
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

    base_model_resources = (
        load_competition_agent_model_resources_from_environment()
    )

    flaky_sufficiency = (
        FailOnceSufficiencyProvider(
            base_model_resources
            .sufficiency_provider
        )
    )

    model_resources = (
        CompetitionAgentModelResources(
            sufficiency_provider=(
                flaky_sufficiency
            ),
            answer_provider=(
                base_model_resources
                .answer_provider
            ),
        )
    )

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

    print()
    print(
        "===== Competition Agent "
        "Checkpoint Resume ====="
    )

    # ==========================================
    # First Run: Fail at Sufficiency
    # ==========================================

    try:
        graph.invoke(
            {
                "question": question,
            },
            config=config,
        )

    except (
        InjectedCheckpointFailure
    ) as exc:
        print()
        print(
            "Injected failure caught:"
        )

        print(
            type(exc).__name__,
            str(exc),
        )

    else:
        raise RuntimeError(
            "第一次执行应该在 "
            "Sufficiency 发生注入失败"
        )

    # ==========================================
    # Inspect failed checkpoint
    # ==========================================

    failed_snapshot = (
        graph.get_state(
            config
        )
    )

    failed_trace = (
        failed_snapshot
        .values
        .get(
            "trace",
            (),
        )
    )

    failed_stages = tuple(
        event.stage
        for event
        in failed_trace
    )

    print()
    print(
        "After failure:"
    )

    print(
        "  next:",
        failed_snapshot.next,
    )

    print(
        "  trace:",
        failed_stages,
    )

    expected_before_failure = (
        "source_resolution",
        "retrieval",
        "evidence_assembly",
        "readiness",
    )

    if (
        failed_stages
        != expected_before_failure
    ):
        raise RuntimeError(
            "失败前已完成阶段"
            "不符合预期："
            f"{failed_stages}"
        )

    if (
        "sufficiency"
        not in failed_snapshot.next
    ):
        raise RuntimeError(
            "Checkpoint 没有停在 "
            "sufficiency"
        )

    # ==========================================
    # Resume
    #
    # 关键：
    # 不再传 question，
    # 使用同一个 thread_id。
    # ==========================================

    final_state = graph.invoke(
        None,
        config=config,
    )

    result = final_state[
        "result"
    ]

    final_trace = (
        final_state.get(
            "trace",
            (),
        )
    )

    final_stages = tuple(
        event.stage
        for event
        in final_trace
    )

    print()
    print(
        "After resume:"
    )

    print(
        "  status:",
        result.status,
    )

    print(
        "  answer:",
        (
            result.answer.answer
            if (
                result.status
                == "answered"
            )
            else None
        ),
    )

    print(
        "  gold:",
        case.answer,
    )

    print(
        "  trace:",
        final_stages,
    )

    print(
        "  sufficiency attempts:",
        flaky_sufficiency
        .request_count,
    )

    history = list(
        graph.get_state_history(
            config
        )
    )

    print(
        "  checkpoint count:",
        len(history),
    )

    if (
        result.status
        != "answered"
    ):
        raise RuntimeError(
            "Resume 后没有正常回答"
        )

    if (
        result.answer.answer
        != case.answer
    ):
        raise RuntimeError(
            "Resume 后答案错误"
        )

    expected_final_trace = (
        "source_resolution",
        "retrieval",
        "evidence_assembly",
        "readiness",
        "sufficiency",
        "answer",
        "citation",
    )

    if (
        final_stages
        != expected_final_trace
    ):
        raise RuntimeError(
            "Resume 后 Trace "
            "存在重复执行或缺失："
            f"{final_stages}"
        )

    if (
        flaky_sufficiency
        .request_count
        != 2
    ):
        raise RuntimeError(
            "Sufficiency 执行次数异常："
            f"{flaky_sufficiency.request_count}"
        )

    print()
    print(
        "CHECKPOINT RESUME PASS"
    )


if __name__ == "__main__":
    main()