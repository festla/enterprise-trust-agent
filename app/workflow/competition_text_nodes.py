from __future__ import annotations

from collections.abc import (
    Callable,
)

from app.services.competition_evidence_assembler import (
    assemble_competition_evidence_bundle,
)
from app.services.competition_text_retrieval import (
    retrieve_competition_text,
)
from app.workflow.competition_runtime import (
    CompetitionAgentRuntimeResources,
)
from app.workflow.competition_state import (
    CompetitionAgentState,
)
from app.services.competition_source_resolver import (
    CompetitionSourceResolverError,
)

CompetitionAgentNode = Callable[
    [CompetitionAgentState],
    dict[str, object],
]


def _require_question(
    state: CompetitionAgentState,
):
    question = state.get(
        "question"
    )

    if question is None:
        raise RuntimeError(
            "Agent State 缺少 question"
        )

    return question


def build_source_resolution_node(
    resources: (
        CompetitionAgentRuntimeResources
    ),
) -> CompetitionAgentNode:
    """
    Question
        ↓
    Source Resolution
    """

    def source_resolution_node(
        state: CompetitionAgentState,
    ) -> dict[str, object]:
        question = (
            _require_question(
                state
            )
        )

        try:
            resolution = (
                resources
                .source_resolver
                .resolve(
                    question
                )
            )

        except CompetitionSourceResolverError as exc:
            return {
                "source_resolution_error": (
                    str(exc)
                ),
            }

        return {
            "source_resolution": (
                resolution
            ),
        }

    return source_resolution_node


def build_retrieval_node(
    resources: (
        CompetitionAgentRuntimeResources
    ),
) -> CompetitionAgentNode:
    """
    Question + Source Resolution
        ↓
    Frozen Text Retrieval
    """

    def retrieval_node(
        state: CompetitionAgentState,
    ) -> dict[str, object]:
        question = (
            _require_question(
                state
            )
        )

        resolution = state.get(
            "source_resolution"
        )

        if resolution is None:
            raise RuntimeError(
                "Retrieval Node 缺少 "
                "source_resolution"
            )

        if (
            resolution.case_id
            != question.case_id
        ):
            raise RuntimeError(
                "Question 与 Source Resolution "
                "case_id 不一致"
            )

        retrieval = (
            retrieve_competition_text(
                question=question,
                resolved_source_id=(
                    resolution
                    .source_id
                ),
                index=(
                    resources
                    .bm25_index
                ),
                tokenizer=(
                    resources
                    .tokenizer
                ),
                corpus_chunks=(
                    resources
                    .corpus_chunks
                ),
                config=(
                    resources
                    .retrieval_config
                ),
            )
        )

        return {
            "retrieval": retrieval,
        }

    return retrieval_node


def build_evidence_assembly_node(
    resources: (
        CompetitionAgentRuntimeResources
    ),
) -> CompetitionAgentNode:
    """
    Retrieval Result
        ↓
    Evidence Bundle

    注意：
    Retrieval 没有命中时不抛出异常。

    因为：
        no hits
    是一个正常业务状态，

    后续 Evidence Readiness / Abstention
    会把它路由到 Refusal。
    """

    def evidence_assembly_node(
        state: CompetitionAgentState,
    ) -> dict[str, object]:
        retrieval = state.get(
            "retrieval"
        )

        if retrieval is None:
            raise RuntimeError(
                "Evidence Node 缺少 retrieval"
            )

        if not retrieval.expanded_hits:
            return {}

        evidence_bundle = (
            assemble_competition_evidence_bundle(
                hits=(
                    retrieval
                    .expanded_hits
                ),
                source_records=(
                    resources
                    .source_records
                ),
                config=(
                    resources
                    .evidence_config
                ),
            )
        )

        return {
            "evidence_bundle": (
                evidence_bundle
            ),
        }

    return evidence_assembly_node