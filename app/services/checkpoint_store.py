from __future__ import annotations

import hashlib
import json
import sqlite3
import threading

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from app.schemas.agent_runtime import (
    AgentState,
)
from app.schemas.checkpoint import (
    CheckpointMetadata,
    CheckpointRecord,
)


SUPPORTED_CHECKPOINT_SCHEMA_VERSION = 1
SUPPORTED_STATE_SCHEMA_VERSION = 1


class CheckpointStoreError(RuntimeError):
    """CheckpointStore 基础异常"""

class CheckpointNotFoundError(
    CheckpointStoreError
):
    """没有找到请求的 Checkpoint。"""

class CheckpointConflictError(
    CheckpointStoreError
):
    """保存时发生乐观并发冲突。"""

class IncompatibleCheckpointVersionError(
    CheckpointStoreError
):
    """持久化版本与当前 Runtime 不兼容。"""

class CorruptCheckpointError(
    CheckpointStoreError
):
    """Checkpoint 内容损坏或哈希不一致。"""

class InvalidCheckpointRevisionError(
    CheckpointStoreError
):
    """Checkpoint revision 非法。"""


# Protocol 用来定义一个接口约定。

# 它表示任何存储类只要实现下面四个方法，
# 就可以被 Runtime 当作 CheckpointStore 使用
class CheckpointStore(Protocol):
    """Runtime 使用的 CheckpointStore 接口。"""

    def save(
        self,
        state: AgentState,
        *,
        expected_revision: int | None = None,
    ) -> CheckpointRecord:
        """保存新的追加式 Checkpoint。"""

    def load_latest(
        self,
        *,
        run_id: str,
        thread_id: str,
    ) -> CheckpointRecord:
        """恢复指定运行的最新 Checkpoint。"""

    def load_revision(
        self,
        *,
        run_id: str,
        thread_id: str,
        revision: int,
    ) -> CheckpointRecord:
        """恢复指定 revision。"""

    def list_checkpoints(
        self,
        *,
        run_id: str,
        thread_id: str,
    ) -> tuple[CheckpointMetadata, ...]:
        """按 revision 升序列出 Checkpoint。"""
        """这里返回的是轻量的 CheckpointMetadata"""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _validate_requested_revision(
    revision: int,
) -> None:
    if revision < 1:
        raise InvalidCheckpointRevisionError(
            "revision 必须大于等于 1"
        )


def _validate_expected_revision(
    expected_revision: int | None,
) -> None:
    if (
        expected_revision is not None
        and expected_revision < 0
    ):
        raise InvalidCheckpointRevisionError(
            "expected_revision 不能小于 0"
        )

def _cannonical_state_json(
    state: AgentState,
) -> str:
    payload = state.model_dump(
        mode="json",
    )

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

def _calculate_sha256(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _prepare_state(
    state: AgentState,
    *,
    revision: int,
) -> AgentState:
    """重新验证状态并写入实际 revision。"""

    payload = state.model_dump(
        mode="python",
    )

    payload["checkpoint_revision"] = (
        revision
    )

    try:
        return AgentState.model_validate(
            payload
        )
    except ValidationError as exc:
        raise CorruptCheckpointError(
            "待保存的 AgentState 未通过重新校验"
        ) from exc


def _build_record(
    *,
    state: AgentState,
    revision: int,
    created_at: datetime,
) -> tuple[CheckpointRecord, str]:
    prepared_state = _prepare_state(
        state,
        revision=revision,
    )

    state_json = _cannonical_state_json(
        prepared_state
    )

    state_sha256 = _calculate_sha256(
        state_json
    )

    parent_revision = (
        None
        if revision == 1
        else revision - 1
    )

    record = CheckpointRecord(
        run_id=prepared_state.run_id,
        thread_id=prepared_state.thread_id,
        revision=revision,
        parent_revision=parent_revision,
        state_schema_version=(
            prepared_state.schema_version
        ),
        state_sha256=state_sha256,
        created_at=created_at,
        state=prepared_state,
    )

    return record, state_json


def _deserialize_record(
    *,
    checkpoint_schema_version: int,
    run_id: str,
    thread_id: str,
    revision: int,
    parent_revision: int | None,
    state_schema_version: int,
    state_json: str,
    state_sha256: str,
    created_at: str | datetime,
) -> CheckpointRecord:
    if (
        checkpoint_schema_version
        != SUPPORTED_CHECKPOINT_SCHEMA_VERSION
    ):
        raise (
            IncompatibleCheckpointVersionError(
                "不支持的 Checkpoint Schema "
                f"版本：{checkpoint_schema_version}"
            )
        )

    if (
        state_schema_version
        != SUPPORTED_STATE_SCHEMA_VERSION
    ):
        raise (
            IncompatibleCheckpointVersionError(
                "不支持的 AgentState Schema "
                f"版本：{state_schema_version}"
            )
        )

    calculated_sha256 = (
        _calculate_sha256(state_json)
    )

    if calculated_sha256 != state_sha256:
        raise CorruptCheckpointError(
            "Checkpoint SHA-256 校验失败"
        )

    try:
        raw_state = json.loads(
            state_json
        )
    except json.JSONDecodeError as exc:
        raise CorruptCheckpointError(
            "Checkpoint state_json 不是合法 JSON"
        ) from exc

    if not isinstance(raw_state, dict):
        raise CorruptCheckpointError(
            "Checkpoint state_json 顶层必须是对象"
        )

    embedded_schema_version = (
        raw_state.get("schema_version")
    )

    if (
        embedded_schema_version
        != state_schema_version
    ):
        raise CorruptCheckpointError(
            "state_json 内部版本与数据库版本不一致"
        )

    try:
        state = AgentState.model_validate(
            raw_state
        )
    except ValidationError as exc:
        raise CorruptCheckpointError(
            "Checkpoint 中的 AgentState "
            "未通过 Schema 校验"
        ) from exc

    if isinstance(created_at, str):
        try:
            parsed_created_at = (
                datetime.fromisoformat(
                    created_at
                )
            )
        except ValueError as exc:
            raise CorruptCheckpointError(
                "Checkpoint created_at 非法"
            ) from exc

    else:
        parsed_created_at = created_at

    try:
        return CheckpointRecord(
            schema_version=(
                checkpoint_schema_version
            ),
            run_id=run_id,
            thread_id=thread_id,
            revision=revision,
            parent_revision=parent_revision,
            state_schema_version=(
                state_schema_version
            ),
            state_sha256=state_sha256,
            created_at=parsed_created_at,
            state = state,
        )
    except ValidationError as exc:
        raise CorruptCheckpointError(
            "Checkpoint Record 未通过校验"
        ) from exc


def _check_expected_revision(
    *,
    latest_revision: int,
    expected_revision: int | None,
) -> None:
    if expected_revision is None:
        return 

    if expected_revision != latest_revision:
        raise CheckpointConflictError(
            "checkpoint revision 冲突："
            f"expected={expected_revision}, "
            f"actual={latest_revision}"
        )


class InMemoryCheckpointStore:
    """线程安全、追加式的内存 CheckpointStore。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()

        self._records: dict[
            tuple[str, str],
            list[dict[str, Any]],
        ] = {}

    def save(
        self,
        state: AgentState,
        *,
        expected_revision: int | None = None,
    ) -> CheckpointRecord:
        _validate_expected_revision(
            expected_revision
        )

        key = (
            state.run_id,
            state.thread_id,
        )

        with self._lock:
            rows = self._records.setdefault(
                key,
                [],
            )

            latest_revision = len(rows)

            _check_expected_revision(
                latest_revision=latest_revision,
                expected_revision=(
                    expected_revision
                ),
            )

            revision = latest_revision + 1
            created_at = _utc_now()

            record, state_json = (
                _build_record(
                    state=state,
                    revision=revision,
                    created_at=created_at,
                )
            )

            rows.append(
                {
                    "checkpoint_schema_version": (
                        record.schema_version
                    ),
                    "run_id": record.run_id,
                    "thread_id": (
                        record.thread_id
                    ),
                    "revision": record.revision,
                    "parent_revision": (
                        record.parent_revision
                    ),
                    "state_schema_version": (
                        record.state_schema_version
                    ),
                    "state_json": state_json,
                    "state_sha256": (
                        record.state_sha256
                    ),
                    "created_at": (
                        record.created_at.isoformat()
                    ),
                }
            )

            return record

    def load_latest(
        self,
        *,
        run_id: str,
        thread_id: str,
    ) -> CheckpointRecord:
        key = (
            run_id,
            thread_id,
        )

        with self._lock:
            rows = self._records.get(key)

            if not rows:
                raise CheckpointNotFoundError(
                    "没有找到 Checkpoint："
                    f"run_id={run_id}, "
                    f"thread_id={thread_id}"
                )

            row = dict(rows[-1])

        return _deserialize_record(**row)

    def load_revision(
        self,
        *,
        run_id: str,
        thread_id: str,
        revision: int,
    ) -> CheckpointRecord:
        _validate_requested_revision(
            revision
        )

        key = (
            run_id,
            thread_id,
        )

        with self._lock:
            rows = self._records.get(key)

            if (
                not rows
                or revision > len(rows)
            ):
                raise CheckpointNotFoundError(
                    "没有找到指定 revision："
                    f"run_id={run_id}, "
                    f"thread_id={thread_id}, "
                    f"revision={revision}"
                )

            row = dict(
                rows[revision - 1]
            )

        return _deserialize_record(**row)

    def list_checkpoints(
        self,
        *,
        run_id: str,
        thread_id: str,
    ) -> tuple[CheckpointMetadata, ...]:
        key = (
            run_id,
            thread_id,
        )

        with self._lock:
            rows = tuple(
                dict(row)
                for row in self._records.get(
                    key,
                    [],
                )
            )

        return tuple(
            _deserialize_record(
                **row
            ).to_metadata()
            for row in rows
        )


class SQLiteCheckpointStore:
    """使用 SQLite 事务保存追加式 CheckpointStore。"""

    def __init__(
        self,
        database_path: str | Path,
    ) -> None:
        self._database_path = Path(
            database_path
        )

        if self._database_path.exists():
            if self._database_path.is_dir():
                raise ValueError(
                    "database_path 不能是目录"
                )

        self._database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = threading.RLock()

        self._initialize_database()

    @property
    def database_path(self) -> Path:
        return self._database_path

    def _connect(
        self,
    ) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._database_path,
            timeout=30.0,
            isolation_level=None,
        )

        connection.row_factory = (
            sqlite3.Row
        )

        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        connection.execute(
            "PRAGMA busy_timeout = 30000"
        )

        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "PRAGMA journal_mode = WAL"
            )

            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS
                agent_checkpoints (
                    run_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    parent_revision INTEGER,
                    checkpoint_schema_version INTEGER NOT NULL,
                    state_schema_version INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    state_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (
                        run_id,
                        thread_id,
                        revision
                    )
                );

                CREATE INDEX IF NOT EXISTS
                idx_agent_checkpoints_latest
                ON agent_checkpoints (
                    run_id,
                    thread_id,
                    revision DESC
                );
                """
            )

    def save(
        self,
        state: AgentState,
        *,
        expected_revision: int | None = None,
    ) -> CheckpointRecord:
        _validate_expected_revision(
            expected_revision
        )

        with self._lock:
            connection = self._connect()

            try:
                connection.execute(
                    "BEGIN IMMEDIATE"
                ) # 尽早取得写事务锁。

                latest_revision = (
                    self._read_latest_revision(
                        connection,
                        run_id=state.run_id,
                        thread_id=(
                            state.thread_id
                        ),
                    )
                )

                _check_expected_revision(
                    latest_revision=(
                        latest_revision
                    ),
                    expected_revision=(
                        expected_revision
                    )
                )

                revision = (
                    latest_revision + 1
                )

                created_at = _utc_now()

                record, state_json = (
                    _build_record(
                        state=state,
                        revision=revision,
                        created_at=created_at,
                    )
                )

                connection.execute(
                    """
                    INSERT INTO agent_checkpoints (
                        run_id,
                        thread_id,
                        revision,
                        parent_revision,
                        checkpoint_schema_version,
                        state_schema_version,
                        state_json,
                        state_sha256,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.run_id,
                        record.thread_id,
                        record.revision,
                        record.parent_revision,
                        record.schema_version,
                        (
                            record
                            .state_schema_version
                        ),
                        state_json,
                        record.state_sha256,
                        (
                            record.created_at
                            .isoformat()
                        ),
                    ),
                )

                connection.commit()

                return record

            except Exception:
                connection.rollback()
                raise

            finally:
                connection.close()

    def load_latest(
        self,
        *,
        run_id: str,
        thread_id: str,
    ) -> CheckpointRecord:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    checkpoint_schema_version,
                    run_id,
                    thread_id,
                    revision,
                    parent_revision,
                    state_schema_version,
                    state_json,
                    state_sha256,
                    created_at
                FROM agent_checkpoints
                WHERE
                    run_id = ?
                    AND thread_id = ?
                ORDER BY revision DESC
                LIMIT 1
                """,
                (
                    run_id,
                    thread_id,
                ),
            ).fetchone()

        if row is None:
            raise CheckpointNotFoundError(
                "没有找到 Checkpoint："
                f"run_id={run_id}, "
                f"thread_id={thread_id}"
            )

        return _deserialize_record(
            **dict(row)
        )

    def load_revision(
        self,
        *,
        run_id: str,
        thread_id: str,
        revision: int,
    ) -> CheckpointRecord:
        _validate_requested_revision(
            revision
        )

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    checkpoint_schema_version,
                    run_id,
                    thread_id,
                    revision,
                    parent_revision,
                    state_schema_version,
                    state_json,
                    state_sha256,
                    created_at
                FROM agent_checkpoints
                WHERE
                    run_id = ?
                    AND thread_id = ?
                    AND revision = ?
                """,
                (
                    run_id,
                    thread_id,
                    revision,
                ),
            ).fetchone()

        if row is None:
            raise CheckpointNotFoundError(
                "没有找到指定 revision："
                f"run_id={run_id}, "
                f"thread_id={thread_id}, "
                f"revision={revision}"
            )

        return _deserialize_record(
            **dict(row)
        )

    def list_checkpoints(
        self,
        *,
        run_id: str,
        thread_id: str,
    ) -> tuple[CheckpointMetadata, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    checkpoint_schema_version,
                    run_id,
                    thread_id,
                    revision,
                    parent_revision,
                    state_schema_version,
                    state_json,
                    state_sha256,
                    created_at
                FROM agent_checkpoints
                WHERE
                    run_id = ?
                    AND thread_id = ?
                ORDER BY revision ASC
                """,
                (
                    run_id,
                    thread_id,
                ),
            ).fetchall()

        return tuple(
            _deserialize_record(
                **dict(row)
            ).to_metadata()
            for row in rows
        )

    @staticmethod
    def _read_latest_revision(
        connection: sqlite3.Connection,
        *,
        run_id: str,
        thread_id: str,
    ) -> int:
        row = connection.execute(
            """
            SELECT MAX(revision) AS revision
            FROM agent_checkpoints
            WHERE
                run_id = ?
                AND thread_id = ?
            """,
            (
                run_id,
                thread_id,
            ),
        ).fetchone()

        if row is None:
            return 0

        revision = row["revision"]

        if revision is None:
            return 0

        return int(revision)