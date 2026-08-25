from __future__ import annotations

from langgraph.checkpoint.serde.jsonplus import (
    JsonPlusSerializer,
)
import sqlite3

from contextlib import (
    contextmanager,
)
from pathlib import Path
from collections.abc import (
    Iterator,
)

from langgraph.checkpoint.sqlite import (
    SqliteSaver,
)

COMPETITION_CHECKPOINT_ALLOWED_TYPES: tuple[
    tuple[str, str],
    ...,
] = (
    # ========================================================
    # Question / Source
    # ========================================================

    (
        "app.schemas.competition",
        "CompetitionQuestion",
    ),
    (
        "app.schemas.competition",
        "CompetitionSourceResolution",
    ),

    # ========================================================
    # Execution Trace
    # ========================================================

    (
        "app.schemas.competition_agent_execution",
        "CompetitionAgentTraceEvent",
    ),

    # ========================================================
    # Retrieval
    # ========================================================

    (
        "app.schemas.competition_retrieval",
        "CompetitionBM25Hit",
    ),
    (
        "app.schemas.competition_option_anchored_merge",
        "CompetitionBM25OptionAnchoredMergeHit",
    ),
    (
        "app.schemas.competition_context_expansion",
        "CompetitionContextExpandedHit",
    ),
    (
        "app.services.competition_text_retrieval",
        "CompetitionTextRetrievalResult",
    ),

    # ========================================================
    # Evidence / Readiness
    # ========================================================

    (
        "app.schemas.competition_evidence",
        "CompetitionEvidenceBundle",
    ),
    (
        "app.schemas.competition_runtime_result",
        "CompetitionEvidenceReadinessDecision",
    ),

    # ========================================================
    # Semantic Decision
    # ========================================================

    (
        "app.schemas.competition_sufficiency",
        "CompetitionEvidenceSufficiencyAssessment",
    ),

    # ========================================================
    # Answer / Citation
    # ========================================================

    (
        "app.schemas.competition_answer",
        "CompetitionFinalAnswer",
    ),
    (
        "app.schemas.competition_citation",
        "CompetitionCitationBundle",
    ),

    # ========================================================
    # Final Result
    # ========================================================

    (
        "app.schemas.competition_runtime_result",
        "CompetitionAnsweredResult",
    ),
    (
        "app.schemas.competition_runtime_result",
        "CompetitionRefusedResult",
    ),

    # ========================================================
    # Excel Runtime
    # ========================================================

    (
        "app.services.competition_excel_runtime",
        "CompetitionExcelRuntimeResult",
    ),
    (
        "app.schemas.competition_excel_solver",
        "CompetitionExcelLookupResult",
    ),
    (
        "app.schemas.competition_excel_solver",
        "CompetitionExcelCompareResult",
    ),
    (
        "app.schemas.competition_excel_solver",
        "CompetitionExcelCalculationResult",
    ),
)


def build_competition_checkpoint_serializer(
) -> JsonPlusSerializer:
    """
    Competition Agent 专用 checkpoint serializer。

    只允许显式列出的自定义业务类型从
    msgpack checkpoint 中被恢复。

    不使用：
        allowed_msgpack_modules=True

    避免 checkpoint 恢复过程中无边界地
    import / reconstruct 任意 Python 类型。
    """

    return JsonPlusSerializer(
        pickle_fallback=False,
        allowed_msgpack_modules=(
            COMPETITION_CHECKPOINT_ALLOWED_TYPES
        ),
    )

@contextmanager
def open_competition_sqlite_checkpointer(
    database_path: Path,
) -> Iterator[
    SqliteSaver
]:
    """
    打开 Competition Agent 的 SQLite Checkpointer。

    特性：
    - checkpoint 持久化到磁盘
    - 使用显式 msgpack allowlist
    - 支持跨 Python 进程恢复
    - connection 生命周期由 context manager 管理
    """

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serializer = (
        build_competition_checkpoint_serializer()
    )

    connection = sqlite3.connect(
        database_path,
        check_same_thread=False,
    )

    try:
        checkpointer = SqliteSaver(
            connection,
            serde=serializer,
        )

        yield checkpointer

    finally:
        connection.close()