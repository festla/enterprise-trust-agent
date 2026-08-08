from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid

from collections.abc import (
    Callable,
    Collection,
    Mapping,
)
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from pydantic import (
    BaseModel,
    ValidationError,
)

from app.schemas.tool_registry import (
    ToolCallTrace,
    ToolDefinition,
    ToolExecutionResult,
    ToolPermission,
)


ToolHandler = Callable[
    [BaseModel],
    BaseModel | Mapping[str, Any],
]


_SENSITIVE_ARGUMENT_KEYS = {
    "api_key",
    "apikey",
    "password",
    "secret",
    "access_token",
    "refresh_token",
    "authorization",
    "cookie",
}


class ToolRegistryError(RuntimeError):
    """工具注册与执行层基础异常。"""


class DuplicateToolError(ToolRegistryError):
    """注册重复工具。"""


class ToolNotFoundError(ToolRegistryError):
    """请求了不存在的工具。"""


class ToolSchemaMismatchError(ToolRegistryError):
    """ToolDefinition 与 Pydantic 模型不一致。"""


class ToolExecutionFailedError(
    ToolRegistryError
):
    """一次工具调用最终失败。

    traces 保留全部执行尝试，供 Agent Runtime
    写入 Trajectory 和 Checkpoint。
    """

    def __init__(
        self,
        message: str,
        *,
        traces: tuple[
            ToolCallTrace,
            ...,
        ],
    ) -> None:
        super().__init__(message)
        self.traces = traces


class ToolPermissionDeniedError(
    ToolRegistryError
):
    """调用者没有工具执行权限。"""


class ToolInputValidationError(
    ToolRegistryError
):
    """工具输入没有通过 Schema 校验。"""


class ToolOutputValidationError(
    ToolRegistryError
):
    """工具返回值没有通过 Schema 校验。"""


class ToolResultTooLargeError(
    ToolRegistryError
):
    """工具返回值超过大小限制。"""


class RetryableToolError(
    ToolRegistryError
):
    """工具发生可重试的临时错误。"""


class PermanentToolError(
    ToolRegistryError
):
    """工具发生不可重试错误。"""


class CorruptToolCacheError(
    ToolRegistryError
):
    """幂等缓存中的结果无法通过输出 Schema 校验。"""


@dataclass(
    frozen=True,
    slots=True,
)
class RegisteredTool:
    """运行时真正注册的工具。"""

    definition: ToolDefinition # 说明书

    input_model: type[BaseModel] # 输入规范

    output_model: type[BaseModel] # 输出规范

    handler: ToolHandler # 真实干活的函数


class ToolResultCache(Protocol): # 面向接口编程
    """工具幂等结果缓存接口。"""

    def get(
        self,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        """查询已经成功执行的工具结果。"""

    def set(
        self,
        idempotency_key: str,
        output: dict[str, Any],
    ) -> None:
        """缓存已经成功验证的工具结果。"""


class InMemoryToolResultCache:
    """线程安全的内存幂等缓存。

    第 4 步先实现缓存协议和行为。
    后续 Runtime 可以把该协议替换成持久化实现。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

        self._values: dict[
            str,
            dict[str, Any],
        ] = {}

    def get(
        self,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            value = self._values.get(
                idempotency_key
            )

            if value is None:
                return None

            return deepcopy(value)

    def set(
        self,
        idempotency_key: str,
        output: dict[str, Any],
    ) -> None:
        with self._lock:
            self._values[
                idempotency_key
            ] = deepcopy(output)


class ToolRegistry:
    """显式 Python Tool Registry。

    不使用 eval、exec 或字符串动态导入。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock() # 保护注册表，

        self._tools: dict[
            str,
            RegisteredTool,
        ] = {} # 真正保存工具，tool_name -> RegisteredTool

    def register(
        self,
        *,
        definition: ToolDefinition,
        input_model: type[BaseModel],
        output_model: type[BaseModel],
        handler: ToolHandler,
    ) -> None:
        """注册一个工具。"""

        expected_input_schema = (
            input_model.model_json_schema()
        )

        expected_output_schema = (
            output_model.model_json_schema()
        )

        if (
            definition.input_schema
            != expected_input_schema
        ):
            raise ToolSchemaMismatchError(
                "ToolDefinition.input_schema "
                "与 input_model 不一致："
                f"{definition.tool_name}"
            )

        if (
            definition.output_schema
            != expected_output_schema
        ):
            raise ToolSchemaMismatchError(
                "ToolDefinition.output_schema "
                "与 output_model 不一致："
                f"{definition.tool_name}"
            )

        registered_tool = RegisteredTool(
            definition=definition,
            input_model=input_model,
            output_model=output_model,
            handler=handler,
        )

        with self._lock:
            if (
                definition.tool_name
                in self._tools
            ):
                raise DuplicateToolError(
                    "工具已经注册："
                    f"{definition.tool_name}"
                )

            self._tools[
                definition.tool_name
            ] = registered_tool

    def get(
        self,
        tool_name: str,
    ) -> RegisteredTool:
        """获取指定工具。"""

        with self._lock: # 线程安全地查字典
            tool = self._tools.get(
                tool_name
            )

        if tool is None:
            raise ToolNotFoundError(
                f"工具不存在：{tool_name}"
            )

        return tool

    def definitions(
        self,
    ) -> tuple[ToolDefinition, ...]:
        """按照工具名稳定返回注册定义。"""

        with self._lock:
            tool_names = sorted(
                self._tools
            )

            return tuple(
                self._tools[
                    tool_name
                ].definition
                for tool_name in tool_names
            )

    # 实现了这个就能写 if "retrieve_document" in registry:
    def __contains__(
        self,
        tool_name: object,
    ) -> bool:
        if not isinstance(
            tool_name,
            str,
        ):
            return False

        with self._lock:
            return tool_name in self._tools

    # 返回的是工具数量
    def __len__(self) -> int:
        with self._lock:
            return len(self._tools)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_json(
    value: object,
) -> str:
    """生成用于审计与哈希的稳定 JSON。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _calculate_argument_sha256(
    arguments: Mapping[str, Any],
) -> str:
    return _sha256_text(
        _canonical_json(
            dict(arguments)
        )
    )


def _calculate_idempotency_key(
    *,
    tool_name: str,
    tool_version: str,
    run_id: str,
    step_id: str,
    arguments_sha256: str,
) -> str:
    """生成逻辑工具调用的幂等键。

    同一个 run + step + tool + arguments
    在恢复或重放时得到相同的 key。
    """

    payload = {
        "tool_name": tool_name,
        "tool_version": tool_version,
        "run_id": run_id,
        "step_id": step_id,
        "arguments_sha256": (
            arguments_sha256
        ),
    }

    return _sha256_text(
        _canonical_json(payload)
    )


def _truncate_string(
    value: str,
    *,
    max_length: int = 300,
) -> str:
    if len(value) <= max_length:
        return value

    return (
        value[:max_length]
        + "...<truncated>"
    )


def _sanitize_argument_value(
    value: object,
    *,
    path: str,
    redacted_paths: list[str],
) -> object:
    """创建适合进入 Trace 的参数摘要。"""

    if isinstance(value, Mapping):
        sanitized: dict[str, object] = {}

        for raw_key, child in value.items():
            key = str(raw_key)
            normalized_key = (
                key.strip().lower()
            )

            child_path = (
                f"{path}.{key}"
            )

            if (
                normalized_key
                in _SENSITIVE_ARGUMENT_KEYS
            ):
                redacted_paths.append(
                    child_path
                )
                continue

            sanitized[key] = (
                _sanitize_argument_value(
                    child,
                    path=child_path,
                    redacted_paths=(
                        redacted_paths
                    ),
                )
            )

        return sanitized

    if isinstance(value, (list, tuple)):
        limited_values = value[:20]

        result = [
            _sanitize_argument_value(
                child,
                path=f"{path}[{index}]",
                redacted_paths=(
                    redacted_paths
                ),
            )
            for index, child
            in enumerate(limited_values)
        ]

        if len(value) > 20:
            result.append(
                "<remaining items truncated>"
            )

        return result

    if isinstance(value, str):
        return _truncate_string(value)

    if isinstance(
        value,
        (
            int,
            float,
            bool,
            type(None),
        ),
    ):
        return value

    return _truncate_string(
        str(value)
    )


def _build_argument_summary(
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    redacted_paths: list[str] = []

    sanitized = _sanitize_argument_value(
        dict(arguments),
        path="arguments",
        redacted_paths=redacted_paths,
    )

    if not isinstance(
        sanitized,
        dict,
    ):
        sanitized = {
            "value": sanitized,
        }

    if redacted_paths:
        sanitized[
            "_redacted_fields"
        ] = sorted(redacted_paths)

    return sanitized # 记录“这里曾经有敏感字段”，但不会记录敏感值本身。


def _result_size_bytes(
    output: Mapping[str, Any],
) -> int:
    return len(
        _canonical_json(
            dict(output)
        ).encode("utf-8")
    ) # 返回的是真实字节数，不是字符数


class ToolExecutor:
    """负责安全执行 Registry 中的工具。"""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        result_cache: (
            ToolResultCache | None
        ) = None,
        retry_backoff_seconds: float = 0.05,
        sleep_fn: Callable[
            [float],
            None,
        ] = time.sleep,
    ) -> None:
        if retry_backoff_seconds < 0:
            raise ValueError(
                "retry_backoff_seconds "
                "不能小于 0"
            )

        self._registry = registry

        self._result_cache = (
            result_cache
            or InMemoryToolResultCache()
        )

        self._retry_backoff_seconds = (
            retry_backoff_seconds
        )

        self._sleep_fn = sleep_fn

    def execute(
        self,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        request_id: str,
        run_id: str,
        step_id: str,
        granted_permissions: Collection[
            ToolPermission
        ],
    ) -> ToolExecutionResult:
        """执行一次受控工具调用。"""

        tool = self._registry.get(
            tool_name
        )

        definition = tool.definition

        arguments_dict = dict(
            arguments
        )

        arguments_sha256 = (
            _calculate_argument_sha256(
                arguments_dict
            )
        )

        idempotency_key = (
            _calculate_idempotency_key(
                tool_name=(
                    definition.tool_name
                ),
                tool_version=(
                    definition.version
                ),
                run_id=run_id,
                step_id=step_id,
                arguments_sha256=(
                    arguments_sha256
                ),
            )
        )

        raw_argument_summary = (
            _build_argument_summary(
                arguments_dict
            )
        )

        if (
            definition.permission
            not in granted_permissions
        ):
            error = (
                ToolPermissionDeniedError(
                    "缺少工具权限："
                    f"{definition.permission}"
                )
            )

            trace = self._build_failure_trace(
                request_id=request_id,
                run_id=run_id,
                step_id=step_id,
                definition=definition,
                argument_summary=(
                    raw_argument_summary
                ),
                arguments_sha256=(
                    arguments_sha256
                ),
                idempotency_key=(
                    idempotency_key
                ),
                attempt=1,
                status="permanent_error",
                error=error,
                started_at=_utc_now(),
                started_monotonic=(
                    time.perf_counter()
                ),
            )

            raise ToolExecutionFailedError(
                str(error),
                traces=(trace,),
            ) from error

        try:
            input_value = (
                tool.input_model
                .model_validate(
                    arguments_dict
                )
            )
        except ValidationError as exc:
            error = (
                ToolInputValidationError(
                    "工具输入未通过 Schema "
                    f"校验：{definition.tool_name}"
                )
            )

            trace = self._build_failure_trace(
                request_id=request_id,
                run_id=run_id,
                step_id=step_id,
                definition=definition,
                argument_summary=(
                    raw_argument_summary
                ),
                arguments_sha256=(
                    arguments_sha256
                ),
                idempotency_key=(
                    idempotency_key
                ),
                attempt=1,
                status="permanent_error",
                error=error,
                started_at=_utc_now(),
                started_monotonic=(
                    time.perf_counter()
                ),
            )

            raise ToolExecutionFailedError(
                str(error),
                traces=(trace,),
            ) from exc

        normalized_arguments = (
            input_value.model_dump(
                mode="json"
            )
        )

        argument_summary = (
            _build_argument_summary(
                normalized_arguments
            )
        )

        arguments_sha256 = (
            _calculate_argument_sha256(
                normalized_arguments
            )
        )

        idempotency_key = (
            _calculate_idempotency_key(
                tool_name=(
                    definition.tool_name
                ),
                tool_version=(
                    definition.version
                ),
                run_id=run_id,
                step_id=step_id,
                arguments_sha256=(
                    arguments_sha256
                ),
            )
        )

        cached_result = (
            self._load_cached_result(
                tool=tool,
                idempotency_key=(
                    idempotency_key
                ),
            )
        )

        if cached_result is not None:
            started_at = _utc_now()
            started_monotonic = (
                time.perf_counter()
            )

            trace = ToolCallTrace(
                tool_call_id=(
                    self._new_tool_call_id()
                ),
                request_id=request_id,
                run_id=run_id,
                step_id=step_id,
                tool_name=(
                    definition.tool_name
                ),
                tool_version=(
                    definition.version
                ),
                argument_summary=(
                    argument_summary
                ),
                arguments_sha256=(
                    arguments_sha256
                ),
                idempotency_key=(
                    idempotency_key
                ),
                attempt=1,
                status="reused",
                started_at=started_at,
                completed_at=_utc_now(),
                latency_ms=(
                    (
                        time.perf_counter()
                        - started_monotonic
                    )
                    * 1000.0
                ),
                result_size_bytes=(
                    _result_size_bytes(
                        cached_result
                    )
                ),
                error_type=None,
                error_message=None,
            )

            return ToolExecutionResult(
                output=cached_result,
                traces=(trace,),
                reused=True,
            )

        traces: list[
            ToolCallTrace
        ] = []

        total_attempts = (
            definition.max_retries + 1
        )

        for attempt in range(
            1,
            total_attempts + 1,
        ):
            started_at = _utc_now()

            started_monotonic = (
                time.perf_counter()
            )

            try:
                raw_output = (
                    self._invoke_with_timeout(
                        tool,
                        input_value,
                    )
                )

                output = (
                    self._validate_output(
                        tool,
                        raw_output,
                    )
                )

                result_size_bytes = (
                    _result_size_bytes(
                        output
                    )
                )

                if (
                    result_size_bytes
                    > definition.max_result_bytes
                ):
                    raise (
                        ToolResultTooLargeError(
                            "工具输出超过最大限制："
                            f"{result_size_bytes} > "
                            f"{definition.max_result_bytes}"
                        )
                    )

                trace = ToolCallTrace(
                    tool_call_id=(
                        self._new_tool_call_id()
                    ),
                    request_id=request_id,
                    run_id=run_id,
                    step_id=step_id,
                    tool_name=(
                        definition.tool_name
                    ),
                    tool_version=(
                        definition.version
                    ),
                    argument_summary=(
                        argument_summary
                    ),
                    arguments_sha256=(
                        arguments_sha256
                    ),
                    idempotency_key=(
                        idempotency_key
                    ),
                    attempt=attempt,
                    status="succeeded",
                    started_at=started_at,
                    completed_at=_utc_now(),
                    latency_ms=(
                        (
                            time.perf_counter()
                            - started_monotonic
                        )
                        * 1000.0
                    ),
                    result_size_bytes=(
                        result_size_bytes
                    ),
                    error_type=None,
                    error_message=None,
                )

                traces.append(trace)

                if definition.idempotent:
                    self._result_cache.set(
                        idempotency_key,
                        output,
                    )

                return ToolExecutionResult(
                    output=output,
                    traces=tuple(traces),
                    reused=False,
                )

            except FutureTimeoutError as exc:
                trace = (
                    self._build_failure_trace(
                        request_id=(
                            request_id
                        ),
                        run_id=run_id,
                        step_id=step_id,
                        definition=definition,
                        argument_summary=(
                            argument_summary
                        ),
                        arguments_sha256=(
                            arguments_sha256
                        ),
                        idempotency_key=(
                            idempotency_key
                        ),
                        attempt=attempt,
                        status="timed_out",
                        error=exc,
                        started_at=started_at,
                        started_monotonic=(
                            started_monotonic
                        ),
                    )
                )

                traces.append(trace)

                if attempt < total_attempts:
                    self._sleep_before_retry(
                        attempt
                    )
                    continue

                raise (
                    ToolExecutionFailedError(
                        "工具执行超时且重试已耗尽："
                        f"{definition.tool_name}",
                        traces=tuple(traces),
                    )
                ) from exc

            except RetryableToolError as exc:
                trace = (
                    self._build_failure_trace(
                        request_id=(
                            request_id
                        ),
                        run_id=run_id,
                        step_id=step_id,
                        definition=definition,
                        argument_summary=(
                            argument_summary
                        ),
                        arguments_sha256=(
                            arguments_sha256
                        ),
                        idempotency_key=(
                            idempotency_key
                        ),
                        attempt=attempt,
                        status=(
                            "retryable_error"
                        ),
                        error=exc,
                        started_at=started_at,
                        started_monotonic=(
                            started_monotonic
                        ),
                    )
                )

                traces.append(trace)

                if attempt < total_attempts:
                    self._sleep_before_retry(
                        attempt
                    )
                    continue

                raise (
                    ToolExecutionFailedError(
                        "工具可重试错误已耗尽："
                        f"{definition.tool_name}",
                        traces=tuple(traces),
                    )
                ) from exc

            except (
                ToolOutputValidationError,
                ToolResultTooLargeError,
                PermanentToolError,
            ) as exc:
                trace = (
                    self._build_failure_trace(
                        request_id=(
                            request_id
                        ),
                        run_id=run_id,
                        step_id=step_id,
                        definition=definition,
                        argument_summary=(
                            argument_summary
                        ),
                        arguments_sha256=(
                            arguments_sha256
                        ),
                        idempotency_key=(
                            idempotency_key
                        ),
                        attempt=attempt,
                        status="permanent_error",
                        error=exc,
                        started_at=started_at,
                        started_monotonic=(
                            started_monotonic
                        ),
                    )
                )

                traces.append(trace)

                raise (
                    ToolExecutionFailedError(
                        str(exc),
                        traces=tuple(traces),
                    )
                ) from exc

            except Exception as exc:
                error = PermanentToolError(
                    "工具发生未分类异常："
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                trace = (
                    self._build_failure_trace(
                        request_id=(
                            request_id
                        ),
                        run_id=run_id,
                        step_id=step_id,
                        definition=definition,
                        argument_summary=(
                            argument_summary
                        ),
                        arguments_sha256=(
                            arguments_sha256
                        ),
                        idempotency_key=(
                            idempotency_key
                        ),
                        attempt=attempt,
                        status="permanent_error",
                        error=error,
                        started_at=started_at,
                        started_monotonic=(
                            started_monotonic
                        ),
                    )
                )

                traces.append(trace)

                raise (
                    ToolExecutionFailedError(
                        str(error),
                        traces=tuple(traces),
                    )
                ) from exc

        raise AssertionError(
            "ToolExecutor 到达不可达分支"
        )

    def _load_cached_result(
        self,
        *,
        tool: RegisteredTool,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        if not tool.definition.idempotent:
            return None

        cached = self._result_cache.get(
            idempotency_key
        )

        if cached is None:
            return None

        try:
            validated = (
                tool.output_model
                .model_validate(cached)
            )
        except ValidationError as exc:
            raise CorruptToolCacheError(
                "幂等缓存结果无法通过输出 "
                f"Schema：{tool.definition.tool_name}"
            ) from exc

        return validated.model_dump(
            mode="json"
        )

    @staticmethod
    def _validate_output(
        tool: RegisteredTool,
        raw_output: (
            BaseModel
            | Mapping[str, Any]
        ),
    ) -> dict[str, Any]:
        try:
            validated_output = (
                tool.output_model
                .model_validate(raw_output)
            )
        except ValidationError as exc:
            raise ToolOutputValidationError(
                "工具输出未通过 Schema 校验："
                f"{tool.definition.tool_name}"
            ) from exc

        return validated_output.model_dump(
            mode="json"
        )

    @staticmethod
    def _invoke_with_timeout(
        tool: RegisteredTool,
        input_value: BaseModel,
    ) -> (
        BaseModel
        | Mapping[str, Any]
    ):
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=(
                "agent-tool"
            ),
        )

        future = executor.submit(
            tool.handler,
            input_value,
        )

        try:
            return future.result(
                timeout=(
                    tool.definition
                    .timeout_seconds
                )
            )
        except FutureTimeoutError:
            future.cancel()
            raise
        finally:
            executor.shutdown(
                wait=False,
                cancel_futures=True,
            )

    def _sleep_before_retry(
        self,
        attempt: int,
    ) -> None:
        delay = (
            self._retry_backoff_seconds
            * (2 ** (attempt - 1))
        )

        delay = min(
            delay,
            5.0,
        )

        if delay > 0:
            self._sleep_fn(delay)

    @staticmethod
    def _new_tool_call_id() -> str:
        return (
            "toolcall:"
            f"{uuid.uuid4().hex}"
        )

    @staticmethod
    def _build_failure_trace(
        *,
        request_id: str,
        run_id: str,
        step_id: str,
        definition: ToolDefinition,
        argument_summary: dict[
            str,
            Any,
        ],
        arguments_sha256: str,
        idempotency_key: str,
        attempt: int,
        status: str,
        error: BaseException,
        started_at: datetime,
        started_monotonic: float,
    ) -> ToolCallTrace:
        error_message = str(error)

        if not error_message:
            error_message = (
                type(error).__name__
            )

        return ToolCallTrace(
            tool_call_id=(
                "toolcall:"
                f"{uuid.uuid4().hex}"
            ),
            request_id=request_id,
            run_id=run_id,
            step_id=step_id,
            tool_name=(
                definition.tool_name
            ),
            tool_version=(
                definition.version
            ),
            argument_summary=(
                argument_summary
            ),
            arguments_sha256=(
                arguments_sha256
            ),
            idempotency_key=(
                idempotency_key
            ),
            attempt=attempt,
            status=status,
            started_at=started_at,
            completed_at=_utc_now(),
            latency_ms=(
                (
                    time.perf_counter()
                    - started_monotonic
                )
                * 1000.0
            ),
            result_size_bytes=0,
            error_type=(
                type(error).__name__
            ),
            error_message=(
                _truncate_string(
                    error_message,
                    max_length=2000,
                )
            ),
        )