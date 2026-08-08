from __future__ import annotations

import hashlib
import json
import os
import re
import threading

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.schemas.agent_runtime import (
    AgentTrajectory,
    TrajectoryReplay,
)


SUPPORTED_TRAJECTORY_SCHEMA_VERSION = 1

_TRAJECTORY_FILE_PREFIX = "trajectory_"
_TRAJECTORY_FILE_SUFFIX = ".jsonl"


class TrajectoryStoreError(RuntimeError):
    """Trajectory 持久化层基础异常。"""


class TrajectoryAlreadyExistsError(
    TrajectoryStoreError
):
    """同一 run_id 的轨迹已经存在。"""


class TrajectoryNotFoundError(
    TrajectoryStoreError
):
    """没有找到指定运行轨迹。"""


class CorruptTrajectoryError(
    TrajectoryStoreError
):
    """轨迹文件损坏或完整性检查失败。"""


class IncompatibleTrajectoryVersionError(
    TrajectoryStoreError
):
    """轨迹文件版本与当前 Runtime 不兼容。"""


class TrajectoryExportExistsError(
    TrajectoryStoreError
):
    """批量导出目标已经存在。"""


def _canonical_json(
    value: object,
) -> str:
    """生成稳定 JSON。

    用于：
    1. SHA-256；
    2. JSONL 持久化；
    3. 保证相同对象得到相同序列化结果。
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _trajectory_payload(
    trajectory: AgentTrajectory,
) -> dict[str, Any]:
    return trajectory.model_dump(
        mode="json"
    )


def _trajectory_sha256(
    trajectory: AgentTrajectory,
) -> str:
    return _sha256_text(
        _canonical_json(
            _trajectory_payload(
                trajectory
            )
        )
    )


def _safe_run_slug(
    run_id: str,
) -> str:
    """生成 Windows / Linux 都安全的文件名片段。

    AgentState 的 run_id 允许出现冒号，
    但 Windows 文件名不能包含冒号，所以不能直接：

        Path(f"{run_id}.jsonl")

    这里将非法字符替换为下划线，并额外附加哈希，
    防止不同 run_id 清洗后发生文件名冲突。
    """

    safe = re.sub(
        r"[^a-zA-Z0-9_.-]+",
        "_",
        run_id,
    )

    safe = safe.strip("._-")

    if not safe:
        safe = "run"

    safe = safe[:80]

    digest = _sha256_text(
        run_id
    )[:12]

    return f"{safe}_{digest}"


def _trajectory_filename(
    run_id: str,
) -> str:
    return (
        f"{_TRAJECTORY_FILE_PREFIX}"
        f"{_safe_run_slug(run_id)}"
        f"{_TRAJECTORY_FILE_SUFFIX}"
    )


def _build_envelope(
    trajectory: AgentTrajectory,
) -> dict[str, Any]:
    """构建磁盘中的不可变存储 Envelope。"""

    return {
        "schema_version": (
            SUPPORTED_TRAJECTORY_SCHEMA_VERSION
        ),
        "run_id": trajectory.run_id,
        "trajectory_sha256": (
            _trajectory_sha256(
                trajectory
            )
        ),
        "trajectory": (
            _trajectory_payload(
                trajectory
            )
        ),
    }


def _parse_envelope(
    raw_value: object,
    *,
    expected_run_id: str | None,
) -> AgentTrajectory:
    """校验 Envelope、Hash 和 AgentTrajectory。"""

    if not isinstance(
        raw_value,
        dict,
    ):
        raise CorruptTrajectoryError(
            "Trajectory Envelope 顶层必须是对象"
        )

    required_keys = {
        "schema_version",
        "run_id",
        "trajectory_sha256",
        "trajectory",
    }

    actual_keys = set(
        raw_value
    )

    if actual_keys != required_keys:
        missing_keys = sorted(
            required_keys - actual_keys
        )

        unexpected_keys = sorted(
            actual_keys - required_keys
        )

        raise CorruptTrajectoryError(
            "Trajectory Envelope 字段不合法："
            f"missing={missing_keys}, "
            f"unexpected={unexpected_keys}"
        )

    schema_version = raw_value[
        "schema_version"
    ]

    if (
        schema_version
        != SUPPORTED_TRAJECTORY_SCHEMA_VERSION
    ):
        raise (
            IncompatibleTrajectoryVersionError(
                "不支持的 Trajectory Schema "
                f"版本：{schema_version}"
            )
        )

    envelope_run_id = raw_value[
        "run_id"
    ]

    if not isinstance(
        envelope_run_id,
        str,
    ):
        raise CorruptTrajectoryError(
            "Trajectory run_id 必须是字符串"
        )

    if (
        expected_run_id is not None
        and envelope_run_id
        != expected_run_id
    ):
        raise CorruptTrajectoryError(
            "Trajectory Envelope run_id "
            "与请求的 run_id 不一致"
        )

    stored_sha256 = raw_value[
        "trajectory_sha256"
    ]

    if not isinstance(
        stored_sha256,
        str,
    ):
        raise CorruptTrajectoryError(
            "trajectory_sha256 必须是字符串"
        )

    trajectory_payload = raw_value[
        "trajectory"
    ]

    if not isinstance(
        trajectory_payload,
        dict,
    ):
        raise CorruptTrajectoryError(
            "trajectory 必须是对象"
        )

    calculated_sha256 = _sha256_text(
        _canonical_json(
            trajectory_payload
        )
    )

    if calculated_sha256 != stored_sha256:
        raise CorruptTrajectoryError(
            "Trajectory SHA-256 校验失败"
        )

    try:
        trajectory = (
            AgentTrajectory.model_validate(
                trajectory_payload
            )
        )
    except ValidationError as exc:
        raise CorruptTrajectoryError(
            "Trajectory 未通过 "
            "AgentTrajectory Schema 校验"
        ) from exc

    if (
        trajectory.run_id
        != envelope_run_id
    ):
        raise CorruptTrajectoryError(
            "Envelope run_id 与 "
            "AgentTrajectory.run_id 不一致"
        )

    return trajectory


def _read_single_jsonl_line(
    path: Path,
) -> object:
    """读取单运行 JSONL。

    每个 run 文件必须且只能保存一条非空 JSONL 记录。
    """

    try:
        text = path.read_text(
            encoding="utf-8"
        )
    except OSError as exc:
        raise CorruptTrajectoryError(
            f"无法读取 Trajectory 文件：{path}"
        ) from exc

    lines = [
        line
        for line in text.splitlines()
        if line.strip()
    ]

    if len(lines) != 1:
        raise CorruptTrajectoryError(
            "单次运行 Trajectory 文件必须且只能"
            "包含一条非空 JSONL 记录"
        )

    try:
        return json.loads(
            lines[0]
        )
    except json.JSONDecodeError as exc:
        raise CorruptTrajectoryError(
            "Trajectory 文件不是合法 JSON"
        ) from exc


def build_trajectory_replay(
    trajectory: AgentTrajectory,
) -> TrajectoryReplay:
    """从完整 Trajectory 生成适合人工审计的回放摘要。"""

    nodes = tuple(
        span.node_name
        for span in trajectory.node_spans
    )

    tools = tuple(
        trace.tool_name
        for trace
        in trajectory.tool_call_traces
    )

    tool_arguments = tuple(
        dict(trace.argument_summary)
        for trace
        in trajectory.tool_call_traces
    )

    retries = tuple(
        (
            f"{trace.tool_name}:"
            f"{trace.step_id}:"
            f"attempt={trace.attempt}:"
            f"status={trace.status}"
        )
        for trace
        in trajectory.tool_call_traces
        if trace.attempt > 1
    )

    failures: list[str] = []

    for span in trajectory.node_spans:
        if span.status == "completed":
            continue

        error_type = (
            span.error_type
            or "UnknownError"
        )

        failures.append(
            (
                f"node:{span.node_name}:"
                f"status={span.status}:"
                f"error={error_type}"
            )
        )

    for trace in (
        trajectory.tool_call_traces
    ):
        if trace.status in {
            "succeeded",
            "reused",
        }:
            continue

        error_type = (
            trace.error_type
            or "UnknownError"
        )

        failures.append(
            (
                f"tool:{trace.tool_name}:"
                f"{trace.step_id}:"
                f"attempt={trace.attempt}:"
                f"status={trace.status}:"
                f"error={error_type}"
            )
        )

    return TrajectoryReplay(
        run_id=trajectory.run_id,
        nodes=nodes,
        tools=tools,
        tool_arguments=tool_arguments,
        retries=retries,
        failures=tuple(failures),
        supporting_fact_ids=(
            trajectory.resolved_fact_ids
        ),
        evidence_ids=(
            trajectory.evidence_ids
        ),
        calculation_ids=(
            trajectory.calculation_ids
        ),
        final_status=(
            trajectory.final_status
        ),
        stop_reason=(
            trajectory.stop_reason
        ),
    )


class TrajectoryStore:
    """不可覆盖的 Agent Trajectory JSONL Store。

    设计原则：
    - 一个 run_id 对应一个文件；
    - 每个文件只有一条 JSONL；
    - 第一次保存后禁止覆盖；
    - 保存完整 AgentTrajectory；
    - 保存 SHA-256 用于完整性检查；
    - Replay 只读取经过脱敏的 argument_summary，
      不读取原始工具参数。
    """

    def __init__(
        self,
        root_directory: str | Path,
    ) -> None:
        self._root_directory = Path(
            root_directory
        )

        if self._root_directory.exists():
            if not self._root_directory.is_dir():
                raise ValueError(
                    "root_directory 必须是目录"
                )

        self._root_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = threading.RLock()

    @property
    def root_directory(self) -> Path:
        return self._root_directory

    def trajectory_path(
        self,
        run_id: str,
    ) -> Path:
        return (
            self._root_directory
            / _trajectory_filename(
                run_id
            )
        )

    def save(
        self,
        trajectory: AgentTrajectory,
    ) -> Path:
        """保存一次终止运行。

        使用文件创建模式 x：
        如果目标文件已经存在则直接失败，
        不允许静默覆盖历史轨迹。
        """

        target_path = self.trajectory_path(
            trajectory.run_id
        )

        envelope = _build_envelope(
            trajectory
        )

        serialized = (
            _canonical_json(
                envelope
            )
            + "\n"
        )

        with self._lock:
            try:
                with target_path.open(
                    "x",
                    encoding="utf-8",
                    newline="\n",
                ) as file:
                    file.write(serialized)
                    file.flush()

                    os.fsync(
                        file.fileno()
                    )

            except FileExistsError as exc:
                raise (
                    TrajectoryAlreadyExistsError(
                        "Trajectory 已经存在，"
                        "禁止覆盖："
                        f"{trajectory.run_id}"
                    )
                ) from exc

        return target_path

    def load(
        self,
        run_id: str,
    ) -> AgentTrajectory:
        """读取并校验一个运行轨迹。"""

        path = self.trajectory_path(
            run_id
        )

        if not path.exists():
            raise TrajectoryNotFoundError(
                "没有找到 Trajectory："
                f"{run_id}"
            )

        raw_envelope = (
            _read_single_jsonl_line(
                path
            )
        )

        return _parse_envelope(
            raw_envelope,
            expected_run_id=run_id,
        )

    def replay(
        self,
        run_id: str,
    ) -> TrajectoryReplay:
        """加载并生成一次人工可读回放摘要。"""

        trajectory = self.load(
            run_id
        )

        return build_trajectory_replay(
            trajectory
        )

    def list_run_ids(
        self,
    ) -> tuple[str, ...]:
        """列出 Store 中所有运行 ID。

        会读取并验证每一个 Envelope，
        因而损坏文件不会被悄悄忽略。
        """

        run_ids: list[str] = []

        paths = sorted(
            self._root_directory.glob(
                (
                    f"{_TRAJECTORY_FILE_PREFIX}"
                    f"*"
                    f"{_TRAJECTORY_FILE_SUFFIX}"
                )
            )
        )

        for path in paths:
            raw_envelope = (
                _read_single_jsonl_line(
                    path
                )
            )

            trajectory = _parse_envelope(
                raw_envelope,
                expected_run_id=None,
            )

            expected_path = (
                self.trajectory_path(
                    trajectory.run_id
                )
            )

            if (
                expected_path.resolve()
                != path.resolve()
            ):
                raise CorruptTrajectoryError(
                    "Trajectory 文件名与 "
                    "内部 run_id 不匹配"
                )

            run_ids.append(
                trajectory.run_id
            )

        if len(run_ids) != len(
            set(run_ids)
        ):
            raise CorruptTrajectoryError(
                "Trajectory Store 中存在 "
                "重复 run_id"
            )

        return tuple(
            sorted(run_ids)
        )

    def export_all(
        self,
        output_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> int:
        """将全部 AgentTrajectory 导出为标准 JSONL。

        与 Store 内部 Envelope 不同，
        导出的每一行直接是 AgentTrajectory JSON，
        更方便后续：

        - pandas 分析；
        - 评测；
        - 数据审计；
        - 离线 Replay；
        - 人工抽样。
        """

        output = Path(
            output_path
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if (
            output.exists()
            and not overwrite
        ):
            raise TrajectoryExportExistsError(
                "Trajectory 导出文件已经存在："
                f"{output}"
            )

        run_ids = self.list_run_ids()

        trajectories = tuple(
            self.load(run_id)
            for run_id in run_ids
        )

        mode = (
            "w"
            if overwrite
            else "x"
        )

        try:
            with output.open(
                mode,
                encoding="utf-8",
                newline="\n",
            ) as file:
                for trajectory in trajectories:
                    payload = (
                        _trajectory_payload(
                            trajectory
                        )
                    )

                    file.write(
                        _canonical_json(
                            payload
                        )
                    )

                    file.write("\n")

                file.flush()

                os.fsync(
                    file.fileno()
                )

        except FileExistsError as exc:
            raise (
                TrajectoryExportExistsError(
                    "Trajectory 导出文件已经存在："
                    f"{output}"
                )
            ) from exc

        return len(trajectories)