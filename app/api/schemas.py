from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
)

from app.schemas.competition_runtime_result import (
    CompetitionExecutionResult,
)


class CompetitionAnswerResponse(
    BaseModel
):
    """
    Competition Agent HTTP API 返回结构。

    thread_id:
        本次 Agent 执行对应的 LangGraph thread。

    result:
        answered / refused / failed
        三种业务终态之一。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    thread_id: str

    result: CompetitionExecutionResult