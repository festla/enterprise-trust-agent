from __future__ import annotations

from uuid import (
    uuid4,
)

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
)

from app.api.schemas import (
    CompetitionAnswerResponse,
)
from app.schemas.competition import (
    CompetitionQuestion,
)


router = APIRouter()


@router.get(
    "/health"
)
def health_check(
    request: Request,
) -> dict[
    str,
    str | bool,
]:
    """
    最小健康检查。

    ready=True 表示 Competition Graph
    已完成 startup 初始化。
    """

    ready = (
        getattr(
            request.app.state,
            "competition_graph",
            None,
        )
        is not None
    )

    return {
        "status": (
            "ok"
            if ready
            else "starting"
        ),
        "service": (
            "enterprise-trust-agent"
        ),
        "ready": ready,
    }


@router.post(
    "/api/competition/answer",
    response_model=(
        CompetitionAnswerResponse
    ),
)
def answer_competition_question(
    question: CompetitionQuestion,
    request: Request,
) -> CompetitionAnswerResponse:
    """
    Competition Unified Agent API。

    Word / PDF:
        RAG → Evidence → Sufficiency
        → Answer → Citation

    Excel:
        deterministic solver
        → Evidence → Citation
    """

    graph = getattr(
        request.app.state,
        "competition_graph",
        None,
    )

    if graph is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Competition Agent "
                "runtime is not ready."
            ),
        )

    # 每个 HTTP Question 默认对应
    # 一个独立 LangGraph Thread。
    #
    # case_id 方便排查，
    # uuid 防止重复请求互相覆盖。
    thread_id = (
        f"api-{question.case_id}-"
        f"{uuid4().hex}"
    )

    config = {
        "configurable": {
            "thread_id": (
                thread_id
            ),
        }
    }

    try:
        final_state = graph.invoke(
            {
                "question": (
                    question
                ),
            },
            config=config,
        )

    except Exception as exc:
        # 只有真正逃出 Agent Failure
        # Boundary 的未知系统异常才变成 HTTP 500。
        #
        # 正常的 Agent failed / refused
        # 会作为 result 返回 HTTP 200。
        raise HTTPException(
            status_code=500,
            detail=(
                "Competition Agent "
                "execution failed: "
                f"{type(exc).__name__}"
            ),
        ) from exc

    result = final_state.get(
        "result"
    )

    if result is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Competition Agent "
                "finished without result."
            ),
        )

    return (
        CompetitionAnswerResponse(
            thread_id=thread_id,
            result=result,
        )
    )