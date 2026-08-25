from __future__ import annotations

from pathlib import Path

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
    load_competition_agent_model_resources_from_environment,
    load_competition_agent_runtime_resources,
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
    "competition_agent_resume.sqlite3"
)

THREAD_ID = (
    "competition-sqlite-resume-Q105"
)

EXPECTED_ANSWER = "A"


def main() -> None:
    if (
        not CHECKPOINT_DB.exists()
    ):
        raise RuntimeError(
            "SQLite checkpoint DB "
            "不存在，请先运行 Process A"
        )

    # ========================================================
    # Fresh Python Process:
    # reload ALL process-level resources
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

    model_resources = (
        load_competition_agent_model_resources_from_environment()
    )

    config = {
        "configurable": {
            "thread_id": THREAD_ID,
        }
    }

    # ========================================================
    # Process B
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

        # --------------------------------------------
        # Inspect BEFORE resume
        # --------------------------------------------

        before = graph.get_state(
            config
        )

        before_trace = tuple(
            event.stage
            for event
            in before
            .values
            .get(
                "trace",
                (),
            )
        )

        print()
        print(
            "===== SQLite Process B ====="
        )

        print(
            "Thread:",
            THREAD_ID,
        )

        print(
            "DB:",
            CHECKPOINT_DB,
        )

        print()
        print(
            "Before resume:"
        )

        print(
            "  next:",
            before.next,
        )

        print(
            "  trace:",
            before_trace,
        )

        expected_before = (
            "source_resolution",
            "retrieval",
            "evidence_assembly",
            "readiness",
        )

        if (
            before_trace
            != expected_before
        ):
            raise RuntimeError(
                "Process B 读取到的 "
                "持久化 Trace 异常："
                f"{before_trace}"
            )

        if (
            "sufficiency"
            not in before.next
        ):
            raise RuntimeError(
                "Process B 没有从 SQLite "
                "读取到 pending sufficiency"
            )

        # ==========================================
        # Resume:
        #
        # 关键：
        # 没有重新传 Question
        # ==========================================

        before_sufficiency = (
            model_resources
            .sufficiency_provider
            .request_count
        )

        before_answer = (
            model_resources
            .answer_provider
            .request_count
        )

        final_state = graph.invoke(
            None,
            config=config,
        )

        after_sufficiency = (
            model_resources
            .sufficiency_provider
            .request_count
        )

        after_answer = (
            model_resources
            .answer_provider
            .request_count
        )

        result = final_state[
            "result"
        ]

        final_trace = tuple(
            event.stage
            for event
            in final_state.get(
                "trace",
                (),
            )
        )

        latest = graph.get_state(
            config
        )

        sufficiency_requests = (
            after_sufficiency
            - before_sufficiency
        )

        answer_requests = (
            after_answer
            - before_answer
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
                if result.status
                == "answered"
                else None
            ),
        )

        print(
            "  expected:",
            EXPECTED_ANSWER,
        )

        print(
            "  trace:",
            final_trace,
        )

        print(
            "  latest next:",
            latest.next,
        )

        print(
            "  sufficiency requests:",
            sufficiency_requests,
        )

        print(
            "  answer requests:",
            answer_requests,
        )

        expected_final = (
            "source_resolution",
            "retrieval",
            "evidence_assembly",
            "readiness",
            "sufficiency",
            "answer",
            "citation",
        )

        if (
            result.status
            != "answered"
        ):
            raise RuntimeError(
                "跨进程 Resume 后 "
                "没有回答"
            )

        if (
            result.answer.answer
            != EXPECTED_ANSWER
        ):
            raise RuntimeError(
                "跨进程 Resume "
                "答案错误"
            )

        if (
            final_trace
            != expected_final
        ):
            raise RuntimeError(
                "跨进程 Resume "
                "发生重复/缺失执行："
                f"{final_trace}"
            )

        if latest.next:
            raise RuntimeError(
                "Resume 完成后仍有 "
                "pending node"
            )

        if (
            sufficiency_requests
            != 1
        ):
            raise RuntimeError(
                "Resume 应只执行一次 "
                "Sufficiency："
                f"{sufficiency_requests}"
            )

        if (
            answer_requests
            != 1
        ):
            raise RuntimeError(
                "Resume 应只执行一次 "
                "Answer："
                f"{answer_requests}"
            )

    print()
    print(
        "SQLITE CROSS-PROCESS "
        "RESUME PASS"
    )


if __name__ == "__main__":
    main()