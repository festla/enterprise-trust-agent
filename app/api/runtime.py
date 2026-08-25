from __future__ import annotations

import os

from collections.abc import (
    Iterator,
)
from contextlib import (
    contextmanager,
)
from dataclasses import (
    dataclass,
)
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


@dataclass(
    frozen=True,
    slots=True,
)
class CompetitionApiSettings:
    """
    Competition 部署运行配置。

    所有路径均允许通过环境变量覆盖，
    避免部署代码写死 Windows / Linux 路径。
    """

    attachments_root: Path

    corpus_directory: Path

    index_directory: Path

    checkpoint_database: Path

    @classmethod
    def from_environment(
        cls,
    ) -> "CompetitionApiSettings":
        return cls(
            attachments_root=Path(
                os.getenv(
                    "COMPETITION_ATTACHMENTS_ROOT",
                    (
                        "data/competition/"
                        "private/attachments"
                    ),
                )
            ),
            corpus_directory=Path(
                os.getenv(
                    "COMPETITION_CORPUS_DIR",
                    (
                        "data/competition/"
                        "processed/corpora/"
                        "competition_corpus_"
                        "8cbec682dfce4962"
                    ),
                )
            ),
            index_directory=Path(
                os.getenv(
                    "COMPETITION_INDEX_DIR",
                    (
                        "data/competition/"
                        "processed/indexes/bm25/"
                        "competition_bm25_index_"
                        "07bc5262330bc2a5"
                    ),
                )
            ),
            checkpoint_database=Path(
                os.getenv(
                    "COMPETITION_CHECKPOINT_DB",
                    (
                        "data/runtime/checkpoints/"
                        "competition_agent.sqlite3"
                    ),
                )
            ),
        )


@contextmanager
def open_competition_api_graph(
    settings: CompetitionApiSettings,
) -> Iterator[object]:
    """
    构建 Competition API 使用的长期 Graph。

    Runtime Resources 只在服务启动时加载一次，
    后续 HTTP 请求直接复用。

    SQLite Checkpointer 生命周期与服务一致。
    """

    runtime_resources = (
        load_competition_agent_runtime_resources(
            attachments_root=(
                settings.attachments_root
            ),
            corpus_directory=(
                settings.corpus_directory
            ),
            index_directory=(
                settings.index_directory
            ),
        )
    )

    excel_resources = (
        load_competition_excel_runtime_resources(
            attachments_root=(
                settings.attachments_root
            )
        )
    )

    model_resources = (
        load_competition_agent_model_resources_from_environment()
    )

    with (
        open_competition_sqlite_checkpointer(
            settings.checkpoint_database
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

        yield graph