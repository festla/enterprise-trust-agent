from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.schemas.agent_runtime import (
    AgentState,
    AgentStatus,
    RuntimeNode,
)


_RUNTIME_ID_PATTERN = (
    r"^[a-zA-Z0-9_.:-]{1,160}$"
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"

class CheckpointMetadata(BaseModel):
    """Checkpoint 列表中使用的轻量元数据。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    run_id: str = Field(
        pattern=_RUNTIME_ID_PATTERN,
    )

    thread_id: str = Field(
        pattern=_RUNTIME_ID_PATTERN,
    )

    revision: int = Field(
        ge=1,
    )

    parent_revision: int | None = Field(
        default=None,
        ge=1,
    )

    current_node: RuntimeNode

    next_node: RuntimeNode

    status: AgentStatus

    state_sha256: str = Field(
        pattern=_SHA256_PATTERN,
    )

    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def validate_timezone(
        cls,
        value: datetime,
    ) -> datetime:
        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "created_at 必须包含时区信息"
            )

        return value

    @model_validator(mode="after")
    def validate_revision_chain(self) -> Self:
        if self.revision == 1:
            if self.parent_revision is not None:
                raise ValueError(
                    "revision=1 时 "
                    "parent_revision 必须为空"
                )

        elif (
            self.parent_revision
            != self.revision - 1
        ):
            raise ValueError(
                "parent_revision 必须等于 "
                "revision - 1"
            )

        return self


class CheckpointRecord(BaseModel):
    """一次完整的、不可变的 Agent 状态快照。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    run_id: str = Field(
        pattern=_RUNTIME_ID_PATTERN,
    )

    thread_id: str = Field(
        pattern=_RUNTIME_ID_PATTERN,
    )

    revision: int = Field(
        ge=1,
    )

    parent_revision: int | None = Field(
        default=None,
        ge=1,
    )

    state_schema_version: Literal[1] = 1

    state_sha256: str = Field(
        pattern=_SHA256_PATTERN
    )

    created_at: datetime

    state: AgentState

    @field_validator("created_at")
    @classmethod
    def validate_timezone(
        cls,
        value: datetime,
    ) -> datetime:
        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "created_at 必须包含时区信息"
            )

        return value

    @model_validator(mode="after")
    def validate_checkpoint_contract(
        self,
    ) -> Self:
        if self.revision == 1:
            if self.parent_revision is not None:
                raise ValueError(
                    "revision=1 时"
                    "parent_revision 必须为空"
                )

        elif (
            self.parent_revision
            != self.revision - 1
        ):
            raise ValueError(
                "parent_revision 必须等于 "
                "revision - 1"
            )

        if self.state.run_id != self.run_id:
            raise ValueError(
                "Checkpoint run_id 必须与 "
                "AgentState.run_id 一致"
            )

        if (
            self.state.thread_id
            != self.thread_id
        ):
            raise ValueError(
                "Checkpoint thread_id 必须与 "
                "AgentState.thread_id 一致"
            )

        if (
            self.state.schema_version
            != self.state_schema_version
        ):
            raise ValueError(
                "state_schema_version 必须与 "
                "AgentState.schema_version 一致"
            )

        if (
            self.state.checkpoint_revision
            != self.revision
        ):
            raise ValueError(
                "AgentState.checkpoint_revision "
                "必须与 Checkpoint revision 一致"
            )

        return self

    def to_metadata(self) -> CheckpointMetadata:
        return CheckpointMetadata(
            run_id=self.run_id,
            thread_id=self.thread_id,
            revision=self.revision,
            parent_revision=(
                self.parent_revision
            ),
            current_node=(
                self.state.current_node
            ),
            next_node=self.state.next_node,
            status=self.state.status,
            state_sha256=self.state_sha256,
            created_at=self.created_at,
        )