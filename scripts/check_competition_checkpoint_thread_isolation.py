from __future__ import annotations

from pathlib import Path

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
    open_competition_sqlite_checkpointer,
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

CHECKPOINT_DB = Path(
    "data/competition/processed/checkpoints/"
    "competition_agent_thread_isolation.sqlite3"
)


CASE_A = "Q003"
CASE_B = "Q035"

THREAD_A = (
    "competition-thread-A-Q003"
)

THREAD_B = (
    "competition-thread-B-Q035"
)


class NeverCalledSufficiencyProvider:
    provider_id = (
        "thread-isolation:"
        "never-called-sufficiency"
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
            "Excel thread isolation "
            "不应该调用 Sufficiency LLM"
        )


class NeverCalledAnswerProvider:
    provider_id = (
        "thread-isolation:"
        "never-called-answer"
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
            "Excel thread isolation "
            "不应该调用 Answer LLM"
        )


def _remove_checkpoint_files() -> None:
    paths = [
        CHECKPOINT_DB,
        Path(
            f"{CHECKPOINT_DB}-wal"
        ),
        Path(
            f"{CHECKPOINT_DB}-shm"
        ),
    ]

    for path in paths:
        if path.exists():
            path.unlink()


def _history_case_ids(
    history,
) -> tuple[
    str,
    ...,
]:
    """
    提取一个 thread 的 checkpoint history 中
    实际出现过的 case_id。

    某些最早期 checkpoint 可能尚未写入
    question，因此忽略没有 question 的 snapshot。
    """

    result: list[str] = []

    for snapshot in history:
        question = (
            snapshot
            .values
            .get(
                "question"
            )
        )

        if question is None:
            continue

        case_id = question.case_id

        if case_id not in result:
            result.append(
                case_id
            )

    return tuple(result)


def main() -> None:
    # ========================================================
    # Fresh DB
    # ========================================================

    _remove_checkpoint_files()

    # ========================================================
    # Dataset
    # ========================================================

    cases = tuple(
        load_competition_qa_excel(
            QA_FILE
        )
    )

    case_by_id = {
        case.case_id: case
        for case in cases
    }

    case_a = case_by_id[
        CASE_A
    ]

    case_b = case_by_id[
        CASE_B
    ]

    question_a = (
        build_competition_question(
            case_a
        )
    )

    question_b = (
        build_competition_question(
            case_b
        )
    )

    if (
        question_a.source_type
        != "excel"
    ):
        raise RuntimeError(
            f"{CASE_A} 不是 Excel Case"
        )

    if (
        question_b.source_type
        != "excel"
    ):
        raise RuntimeError(
            f"{CASE_B} 不是 Excel Case"
        )

    # ========================================================
    # Process-level Resources
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
    # Two independent thread configs
    # ========================================================

    config_a = {
        "configurable": {
            "thread_id": THREAD_A,
        }
    }

    config_b = {
        "configurable": {
            "thread_id": THREAD_B,
        }
    }

    # ========================================================
    # Same DB / Same Graph / Different Threads
    # ========================================================

    with (
        open_competition_sqlite_checkpointer(
            CHECKPOINT_DB
        )
        as checkpointer
    ):
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

        # ----------------------------------------------------
        # Thread A
        # ----------------------------------------------------

        final_a = graph.invoke(
            {
                "question": (
                    question_a
                ),
            },
            config=config_a,
        )

        # ----------------------------------------------------
        # Thread B
        # ----------------------------------------------------

        final_b = graph.invoke(
            {
                "question": (
                    question_b
                ),
            },
            config=config_b,
        )

        # ----------------------------------------------------
        # Reload BOTH states from SQLite
        # ----------------------------------------------------

        state_a = graph.get_state(
            config_a
        )

        state_b = graph.get_state(
            config_b
        )

        history_a = list(
            graph.get_state_history(
                config_a
            )
        )

        history_b = list(
            graph.get_state_history(
                config_b
            )
        )

        # ====================================================
        # Extract A
        # ====================================================

        result_a = final_a[
            "result"
        ]

        question_state_a = (
            state_a
            .values[
                "question"
            ]
        )

        runtime_a = (
            state_a
            .values
            .get(
                "excel_runtime"
            )
        )

        trace_a = tuple(
            event.stage
            for event
            in state_a
            .values
            .get(
                "trace",
                (),
            )
        )

        history_case_ids_a = (
            _history_case_ids(
                history_a
            )
        )

        # ====================================================
        # Extract B
        # ====================================================

        result_b = final_b[
            "result"
        ]

        question_state_b = (
            state_b
            .values[
                "question"
            ]
        )

        runtime_b = (
            state_b
            .values
            .get(
                "excel_runtime"
            )
        )

        trace_b = tuple(
            event.stage
            for event
            in state_b
            .values
            .get(
                "trace",
                (),
            )
        )

        history_case_ids_b = (
            _history_case_ids(
                history_b
            )
        )

        # ====================================================
        # Console
        # ====================================================

        print()
        print(
            "===== Competition Checkpoint "
            "Thread Isolation ====="
        )

        print()
        print(
            "Shared DB:",
            CHECKPOINT_DB,
        )

        print()

        print(
            "Thread A:"
        )

        print(
            "  thread_id:",
            THREAD_A,
        )

        print(
            "  case:",
            question_state_a
            .case_id,
        )

        print(
            "  answer:",
            result_a.answer.answer,
        )

        print(
            "  gold:",
            case_a.answer,
        )

        print(
            "  solver:",
            (
                runtime_a.solver_type
                if runtime_a
                is not None
                else None
            ),
        )

        print(
            "  trace:",
            trace_a,
        )

        print(
            "  next:",
            state_a.next,
        )

        print(
            "  checkpoints:",
            len(history_a),
        )

        print(
            "  history case_ids:",
            history_case_ids_a,
        )

        print()

        print(
            "Thread B:"
        )

        print(
            "  thread_id:",
            THREAD_B,
        )

        print(
            "  case:",
            question_state_b
            .case_id,
        )

        print(
            "  answer:",
            result_b.answer.answer,
        )

        print(
            "  gold:",
            case_b.answer,
        )

        print(
            "  solver:",
            (
                runtime_b.solver_type
                if runtime_b
                is not None
                else None
            ),
        )

        print(
            "  trace:",
            trace_b,
        )

        print(
            "  next:",
            state_b.next,
        )

        print(
            "  checkpoints:",
            len(history_b),
        )

        print(
            "  history case_ids:",
            history_case_ids_b,
        )

        print()

        print(
            "LLM requests:"
        )

        print(
            "  sufficiency:",
            sufficiency_provider
            .request_count,
        )

        print(
            "  answer:",
            answer_provider
            .request_count,
        )

        # ====================================================
        # Acceptance Gates
        # ====================================================

        expected_trace = (
            "source_resolution",
            "excel_solver",
            "citation",
        )

        # ------------------------
        # Thread A
        # ------------------------

        if (
            result_a.status
            != "answered"
        ):
            raise RuntimeError(
                "Thread A 未 answered"
            )

        if (
            result_a.answer.answer
            != case_a.answer
        ):
            raise RuntimeError(
                "Thread A 答案错误"
            )

        if (
            question_state_a
            .case_id
            != CASE_A
        ):
            raise RuntimeError(
                "Thread A State "
                "读取到了错误 case"
            )

        if (
            runtime_a is None
            or runtime_a.solver_type
            != "lookup"
        ):
            raise RuntimeError(
                "Thread A Solver 异常"
            )

        if (
            trace_a
            != expected_trace
        ):
            raise RuntimeError(
                "Thread A Trace 异常："
                f"{trace_a}"
            )

        if state_a.next:
            raise RuntimeError(
                "Thread A 仍有 "
                "pending node"
            )

        if (
            history_case_ids_a
            != (CASE_A,)
        ):
            raise RuntimeError(
                "Thread A checkpoint history "
                "发生串线："
                f"{history_case_ids_a}"
            )

        # ------------------------
        # Thread B
        # ------------------------

        if (
            result_b.status
            != "answered"
        ):
            raise RuntimeError(
                "Thread B 未 answered"
            )

        if (
            result_b.answer.answer
            != case_b.answer
        ):
            raise RuntimeError(
                "Thread B 答案错误"
            )

        if (
            question_state_b
            .case_id
            != CASE_B
        ):
            raise RuntimeError(
                "Thread B State "
                "读取到了错误 case"
            )

        if (
            runtime_b is None
            or runtime_b.solver_type
            != "compare"
        ):
            raise RuntimeError(
                "Thread B Solver 异常"
            )

        if (
            trace_b
            != expected_trace
        ):
            raise RuntimeError(
                "Thread B Trace 异常："
                f"{trace_b}"
            )

        if state_b.next:
            raise RuntimeError(
                "Thread B 仍有 "
                "pending node"
            )

        if (
            history_case_ids_b
            != (CASE_B,)
        ):
            raise RuntimeError(
                "Thread B checkpoint history "
                "发生串线："
                f"{history_case_ids_b}"
            )

        # ------------------------
        # Cross-thread isolation
        # ------------------------

        if (
            CASE_B
            in history_case_ids_a
        ):
            raise RuntimeError(
                "Thread A history "
                "出现 Thread B Case"
            )

        if (
            CASE_A
            in history_case_ids_b
        ):
            raise RuntimeError(
                "Thread B history "
                "出现 Thread A Case"
            )

        if (
            sufficiency_provider
            .request_count
            != 0
        ):
            raise RuntimeError(
                "Excel Thread Test "
                "错误调用 Sufficiency LLM"
            )

        if (
            answer_provider
            .request_count
            != 0
        ):
            raise RuntimeError(
                "Excel Thread Test "
                "错误调用 Answer LLM"
            )

    print()
    print(
        "THREAD ISOLATION PASS"
    )


if __name__ == "__main__":
    main()