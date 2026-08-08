from __future__ import annotations

import sqlite3

from datetime import (
    datetime,
    timedelta,
    timezone,
)
from pathlib import Path
from typing import Iterator

import pytest

from app.schemas.agent_runtime import (
    AgentState,
)
from app.services.checkpoint_store import (
    CheckpointConflictError,
    CheckpointNotFoundError,
    CheckpointStore,
    CorruptCheckpointError,
    InMemoryCheckpointStore,
    IncompatibleCheckpointVersionError,
    InvalidCheckpointRevisionError,
    SQLiteCheckpointStore,
)


@pytest.fixture(
    params=(
        "memory",
        "sqlite",
    )
)
def checkpoint_store(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> Iterator[CheckpointStore]:
    if request.param == "memory":
        yield InMemoryCheckpointStore()
        return

    database_path = (
        tmp_path / "checkpoints.db"
    )

    yield SQLiteCheckpointStore(
        database_path
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _build_state(
    *,
    run_id: str = "run_1",
    thread_id: str = "thread_1",
    step_count: int = 0,
) -> AgentState:
    now = _now()

    status = (
        "created"
        if step_count == 0
        else "parsing"
    )

    next_node = (
        "parse_query"
        if step_count == 0
        else "route_intent"
    )

    return AgentState(
        request_id="request_1",
        trace_id="trace_1",
        run_id=run_id,
        thread_id=thread_id,
        query="美的集团2024年营业收入是多少？",
        status=status,
        current_node="parse_query",
        next_node=next_node,
        step_count=step_count,
        max_steps=32,
        started_at=now,
        updated_at=(
            now
            + timedelta(
                seconds=step_count
            )
        ),
    )


def test_first_save_and_load_latest(
    checkpoint_store: CheckpointStore,
) -> None:
    record = checkpoint_store.save(
        _build_state(),
        expected_revision=0,
    )

    assert record.revision == 1
    assert record.parent_revision is None
    assert (
        record.state.checkpoint_revision
        == 1
    )

    loaded = checkpoint_store.load_latest(
        run_id="run_1",
        thread_id="thread_1",
    )

    assert loaded == record


def test_store_appends_revisions(
    checkpoint_store: CheckpointStore,
) -> None:
    first_record = checkpoint_store.save(
        _build_state(
            step_count=0,
        ),
        expected_revision=0,
    )

    second_record = checkpoint_store.save(
        _build_state(
            step_count=1,
        ),
        expected_revision=1,
    )

    assert first_record.revision == 1
    assert second_record.revision == 2
    assert second_record.parent_revision == 1

    old_record = (
        checkpoint_store.load_revision(
            run_id="run_1",
            thread_id="thread_1",
            revision=1,
        )
    )

    assert old_record.state.step_count == 0

    latest_record = (
        checkpoint_store.load_latest(
            run_id="run_1",
            thread_id="thread_1",
        )
    )

    assert latest_record.state.step_count == 1


def test_expected_revision_conflict(
    checkpoint_store: CheckpointStore,
) -> None:
    checkpoint_store.save(
        _build_state(),
        expected_revision=0,
    )

    with pytest.raises(
        CheckpointConflictError,
        match="revision 冲突",
    ):
        checkpoint_store.save(
            _build_state(
                step_count=1,
            ),
            expected_revision=0,
        )

    metadata = (
        checkpoint_store.list_checkpoints(
            run_id="run_1",
            thread_id="thread_1",
        )
    )

    assert len(metadata) == 1


def test_missing_checkpoint_raises(
    checkpoint_store: CheckpointStore,
) -> None:
    with pytest.raises(
        CheckpointNotFoundError,
    ):
        checkpoint_store.load_latest(
            run_id="missing_run",
            thread_id="missing_thread",
        )

    with pytest.raises(
        CheckpointNotFoundError,
    ):
        checkpoint_store.load_revision(
            run_id="missing_run",
            thread_id="missing_thread",
            revision=1,
        )


def test_invalid_revision_raises(
    checkpoint_store: CheckpointStore,
) -> None:
    with pytest.raises(
        InvalidCheckpointRevisionError,
    ):
        checkpoint_store.load_revision(
            run_id="run_1",
            thread_id="thread_1",
            revision=0,
        )

    with pytest.raises(
        InvalidCheckpointRevisionError,
    ):
        checkpoint_store.save(
            _build_state(),
            expected_revision=-1,
        )


def test_list_checkpoints_returns_ordered_metadata(
    checkpoint_store: CheckpointStore,
) -> None:
    checkpoint_store.save(
        _build_state(
            step_count=0,
        ),
    )

    checkpoint_store.save(
        _build_state(
            step_count=1,
        ),
    )

    metadata = (
        checkpoint_store.list_checkpoints(
            run_id="run_1",
            thread_id="thread_1",
        )
    )

    assert tuple(
        item.revision
        for item in metadata
    ) == (
        1,
        2,
    )

    assert metadata[0].parent_revision is None
    assert metadata[1].parent_revision == 1
    assert metadata[1].status == "parsing"


def test_sqlite_store_survives_new_instance(
    tmp_path: Path,
) -> None:
    database_path = (
        tmp_path / "persistent.db"
    )

    first_store = SQLiteCheckpointStore(
        database_path
    )

    saved = first_store.save(
        _build_state(),
        expected_revision=0,
    )

    second_store = SQLiteCheckpointStore(
        database_path
    )

    restored = second_store.load_latest(
        run_id="run_1",
        thread_id="thread_1",
    )

    assert restored == saved


def test_sqlite_rejects_incompatible_version(
    tmp_path: Path,
) -> None:
    database_path = (
        tmp_path / "incompatible.db"
    )

    store = SQLiteCheckpointStore(
        database_path
    )

    store.save(
        _build_state(),
    )

    with sqlite3.connect(
        database_path
    ) as connection:
        connection.execute(
            """
            UPDATE agent_checkpoints
            SET checkpoint_schema_version = 99
            WHERE
                run_id = 'run_1'
                AND thread_id = 'thread_1'
                AND revision = 1
            """
        )

    with pytest.raises(
        IncompatibleCheckpointVersionError,
        match="Checkpoint Schema",
    ):
        store.load_latest(
            run_id="run_1",
            thread_id="thread_1",
        )


def test_sqlite_detects_corrupt_state(
    tmp_path: Path,
) -> None:
    database_path = (
        tmp_path / "corrupt.db"
    )

    store = SQLiteCheckpointStore(
        database_path
    )

    store.save(
        _build_state(),
    )

    with sqlite3.connect(
        database_path
    ) as connection:
        connection.execute(
            """
            UPDATE agent_checkpoints
            SET state_json = '{"corrupted":true}'
            WHERE
                run_id = 'run_1'
                AND thread_id = 'thread_1'
                AND revision = 1
            """
        )

    with pytest.raises(
        CorruptCheckpointError,
        match="SHA-256",
    ):
        store.load_latest(
            run_id="run_1",
            thread_id="thread_1",
        )