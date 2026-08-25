from __future__ import annotations

from collections.abc import (
    Sequence,
)

from typing import (
    Annotated,
    Required,
    TypedDict,
)

from app.schemas.competition import (
    CompetitionQuestion,
    CompetitionSourceResolution,
)
from app.schemas.competition_answer import (
    CompetitionFinalAnswer,
)
from app.schemas.competition_citation import (
    CompetitionCitationBundle,
)
from app.schemas.competition_evidence import (
    CompetitionEvidenceBundle,
)
from app.schemas.competition_runtime_result import (
    CompetitionAnsweredResult,
    CompetitionEvidenceReadinessDecision,
    CompetitionRefusedResult,
)
from app.schemas.competition_sufficiency import (
    CompetitionEvidenceSufficiencyAssessment,
)
from app.schemas.competition_agent_execution import (
    CompetitionAgentFailure,
    CompetitionAgentTraceEvent,
)


from app.services.competition_excel_runtime import (
    CompetitionExcelRuntimeResult,
)
from app.services.competition_text_retrieval import (
    CompetitionTextRetrievalResult,
)

def merge_competition_trace(
    left: (
        Sequence[
            CompetitionAgentTraceEvent
        ]
        | None
    ),
    right: (
        Sequence[
            CompetitionAgentTraceEvent
        ]
        | None
    ),
) -> tuple[
    CompetitionAgentTraceEvent,
    ...,
]:
    """
    LangGraph Trace reducer。

    Checkpoint serializer 在恢复状态时，
    sequence 可能以 list 形式返回，
    而 Node Runtime 当前使用 tuple 写入。

    因此 reducer 不直接使用 operator.add，
    而是在合并前统一转换为 tuple：

        list  + tuple
        tuple + list
        list  + list
        tuple + tuple

    都得到稳定 tuple。
    """

    return (
        tuple(
            left or ()
        )
        + tuple(
            right or ()
        )
    )

class CompetitionAgentState(
    TypedDict,
    total=False,
):
    """
    Competition Agent 的 Request-level State。

    State 只保存一次 Question 执行过程中
    不断产生和变化的业务状态。

    Process-level Resources：
        Resolver
        Corpus
        BM25
        Tokenizer
        LLM Providers

    不进入 State。
    """

    # =========================
    # Input
    # =========================

    question: Required[
        CompetitionQuestion
    ]

    # =========================
    # Source
    # =========================

    source_resolution: (
        CompetitionSourceResolution
    )

    source_resolution_error: str

    # =========================
    # Retrieval
    # =========================

    retrieval: (
        CompetitionTextRetrievalResult
    )

    # =========================
    # Evidence
    # =========================

    evidence_bundle: (
        CompetitionEvidenceBundle
    )

    readiness: (
        CompetitionEvidenceReadinessDecision
    )

    # =========================
    # Semantic Decision
    # =========================

    sufficiency: (
        CompetitionEvidenceSufficiencyAssessment
    )

    # =========================
    # Answer
    # =========================

    answer: (
        CompetitionFinalAnswer
    )

    citations: (
        CompetitionCitationBundle
    )

    # =========================
    # Final Result
    # =========================

    result: (
        CompetitionAnsweredResult
        | CompetitionRefusedResult
    )

    trace: Annotated[
        tuple[
            CompetitionAgentTraceEvent,
            ...
        ],
        merge_competition_trace,
    ]

    failure: CompetitionAgentFailure

    # =========================
    # Excel
    # =========================

    excel_runtime: (
        CompetitionExcelRuntimeResult
    )