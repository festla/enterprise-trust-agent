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
    load_competition_agent_model_resources_from_environment,
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
    "competition_agent_resume.sqlite3"
)

CASE_ID = "Q105"

THREAD_ID = (
    "competition-sqlite-resume-Q105"
)


class InjectedProcessFailure(
    RuntimeError
):
    pass


class AlwaysFailSufficiencyProvider:
    """
    Process A 专用。

    用于证明：
    source / retrieval / evidence / readiness
    已经保存进 SQLite，
    而 sufficiency 尚未完成。
    """

    provider_id = (
        "checkpoint-test:"
        "always-fail-sufficiency"
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

        raise InjectedProcessFailure(
            "Injected process failure "
            "at sufficiency"
        )


def main() -> None:
    # ========================================================
    # Fresh DB
    # ========================================================

    if CHECKPOINT_DB.exists():
        CHECKPOINT_DB.unlink()

    # SQLite WAL / SHM leftovers
    for suffix in (
        "-wal",
        "-shm",
    ):
        path = Path(
            f"{CHECKPOINT_DB}{suffix}"
        )

        if path.exists():
            path.unlink()

    # ========================================================
    # Load Question
    # ========================================================

    cases = tuple(
        load_competition_qa_excel(
            QA_FILE
        )
    )

    case = next(
        item
        for item in cases
        if item.case_id == CASE_ID
    )

    question = (
        build_competition_question(
            case
        )
    )

    # ========================================================
    # Runtime Resources
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

    base_models = (
        load_competition_agent_model_resources_from_environment()
    )

    failing_provider = (
        AlwaysFailSufficiencyProvider()
    )

    model_resources = (
        CompetitionAgentModelResources(
            sufficiency_provider=(
                failing_provider
            ),
            answer_provider=(
                base_models
                .answer_provider
            ),
        )
    )

    config = {
        "configurable": {
            "thread_id": THREAD_ID,
        }
    }

    # ========================================================
    # Process A
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

        try:
            graph.invoke(
                {
                    "question": question,
                },
                config=config,
            )

        except InjectedProcessFailure as exc:
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
                "Process A 应该在 "
                "sufficiency 失败"
            )

        snapshot = graph.get_state(
            config
        )

        trace_stages = tuple(
            event.stage
            for event
            in snapshot
            .values
            .get(
                "trace",
                (),
            )
        )

        print()
        print(
            "===== SQLite Process A ====="
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
            "DB:",
            CHECKPOINT_DB,
        )

        print(
            "DB exists:",
            CHECKPOINT_DB.exists(),
        )

        print(
            "Next:",
            snapshot.next,
        )

        print(
            "Trace:",
            trace_stages,
        )

        print(
            "Sufficiency attempts:",
            failing_provider
            .request_count,
        )

        expected_trace = (
            "source_resolution",
            "retrieval",
            "evidence_assembly",
            "readiness",
        )

        if (
            trace_stages
            != expected_trace
        ):
            raise RuntimeError(
                "SQLite Failure checkpoint "
                "Trace 异常："
                f"{trace_stages}"
            )

        if (
            "sufficiency"
            not in snapshot.next
        ):
            raise RuntimeError(
                "SQLite checkpoint "
                "没有停在 sufficiency"
            )

        if (
            not CHECKPOINT_DB.exists()
        ):
            raise RuntimeError(
                "SQLite checkpoint "
                "文件不存在"
            )

    # 到这里 connection 已经关闭。
    # Process B 必须重新打开数据库。

    print()
    print(
        "SQLITE FAILURE CHECKPOINT SAVED"
    )

    print()
    print(
        "Now run Process B:"
    )

    print(
        "uv run --env-file .env "
        "python -m "
        "scripts."
        "check_competition_sqlite_checkpoint_resume"
    )


if __name__ == "__main__":
    main()