from __future__ import annotations

from pathlib import Path

from app.rag.bailian_answer_provider import (
    BailianAnswerProviderRequestError,
)
from app.schemas.competition import (
    CompetitionQuestion,
)
from app.schemas.competition_answer import (
    CompetitionGeneratedAnswer,
)
from app.schemas.competition_sufficiency import (
    CompetitionEvidenceSufficiencyAssessment,
)
from app.services.competition_dataset import (
    load_competition_qa_excel,
)
from app.workflow.competition_agent_graph import (
    build_competition_agent_graph,
)
from app.workflow.competition_runtime import (
    CompetitionAgentModelResources,
    load_competition_agent_runtime_resources,
)


QA_FILE = Path(
    "data/competition/private/qa/QA数据.xlsx"
)

ATTACHMENTS_ROOT = Path(
    "data/competition/private/attachments"
)

CORPUS_DIR = Path(
    "data/competition/processed/corpora/"
    "competition_corpus_8cbec682dfce4962"
)

INDEX_DIR = Path(
    "data/competition/processed/indexes/bm25/"
    "competition_bm25_index_07bc5262330bc2a5"
)


def _to_safe_question(
    case,
) -> CompetitionQuestion:
    return CompetitionQuestion(
        case_id=case.case_id,
        source_type=case.source_type,
        qa_type=case.qa_type,
        question=case.question,
        option_a=case.option_a,
        option_b=case.option_b,
        option_c=case.option_c,
        option_d=case.option_d,
        source_title=case.source_title,
        file_label=case.file_label,
    )


class FailingSufficiencyProvider:
    """
    故意模拟 Sufficiency API 请求失败。

    注意：
    这是 Failure Injection，
    不是生产 Provider。
    """

    def __init__(
        self,
    ) -> None:
        self.request_count = 0

    @property
    def provider_id(
        self,
    ) -> str:
        return (
            "failure-injection:"
            "sufficiency-timeout"
        )

    def assess(
        self,
        *,
        question: str,
        options: dict[str, str],
        generation_context: str,
        allowed_citation_ids: tuple[
            str,
            ...
        ],
    ) -> (
        CompetitionEvidenceSufficiencyAssessment
    ):
        self.request_count += 1

        raise BailianAnswerProviderRequestError(
            "Injected sufficiency "
            "provider timeout"
        )


class NeverCalledAnswerProvider:
    """
    Sufficiency 失败后，
    Answer Provider 绝对不能被调用。

    如果被调用，直接让脚本失败。
    """

    def __init__(
        self,
    ) -> None:
        self.request_count = 0

    @property
    def provider_id(
        self,
    ) -> str:
        return (
            "failure-injection:"
            "answer-must-not-run"
        )

    def generate(
        self,
        *,
        question: str,
        options: dict[str, str],
        generation_context: str,
        allowed_citation_ids: tuple[
            str,
            ...
        ],
    ) -> CompetitionGeneratedAnswer:
        self.request_count += 1

        raise AssertionError(
            "Sufficiency 已失败，"
            "Answer Provider 不应被调用"
        )


def main() -> None:
    cases = (
        load_competition_qa_excel(
            QA_FILE
        )
    )

    case = next(
        item
        for item in cases
        if item.case_id == "Q105"
    )

    question = (
        _to_safe_question(
            case
        )
    )

    runtime_resources = (
        load_competition_agent_runtime_resources(
            attachments_root=(
                ATTACHMENTS_ROOT
            ),
            corpus_directory=(
                CORPUS_DIR
            ),
            index_directory=(
                INDEX_DIR
            ),
        )
    )

    sufficiency_provider = (
        FailingSufficiencyProvider()
    )

    answer_provider = (
        NeverCalledAnswerProvider()
    )

    model_resources = (
        CompetitionAgentModelResources(
            sufficiency_provider=(
                sufficiency_provider
            ),
            answer_provider=(
                answer_provider
            ),
        )
    )

    graph = build_competition_agent_graph(
        runtime_resources=(
            runtime_resources
        ),
        model_resources=(
            model_resources
        ),
    )

    final_state = graph.invoke(
        {
            "question": question,
        }
    )

    result = final_state.get(
        "result"
    )

    if result is None:
        raise RuntimeError(
            "Failure Routing Graph "
            "没有 final result"
        )

    print()
    print(
        "===== Competition Agent "
        "Failure Injection ====="
    )

    print(
        "Case:",
        question.case_id,
    )

    print(
        "Final status:",
        result.status,
    )

    if result.status != "failed":
        raise RuntimeError(
            "Sufficiency Provider "
            "失败后必须返回 failed，"
            f"实际为 {result.status}"
        )

    print(
        "Failure stage:",
        result.failure_stage,
    )

    print(
        "Failure code:",
        result.failure_code,
    )

    print(
        "Retryable:",
        result.retryable,
    )

    print(
        "Error type:",
        result.error_type,
    )

    if (
        result.failure_stage
        != "sufficiency"
    ):
        raise RuntimeError(
            "failure_stage 应为 sufficiency"
        )

    if (
        result.failure_code
        != "provider_request_error"
    ):
        raise RuntimeError(
            "failure_code 应为 "
            "provider_request_error"
        )

    if result.retryable is not True:
        raise RuntimeError(
            "Provider request error "
            "应标记为 retryable"
        )

    if (
        sufficiency_provider
        .request_count
        != 1
    ):
        raise RuntimeError(
            "Sufficiency Provider "
            "必须且只能调用一次"
        )

    if (
        answer_provider
        .request_count
        != 0
    ):
        raise RuntimeError(
            "上游 Sufficiency 失败后，"
            "Answer Provider 不允许执行"
        )

    trace = final_state.get(
        "trace",
        (),
    )

    print()
    print(
        "Execution Trace:"
    )

    for index, event in enumerate(
        trace,
        start=1,
    ):
        print(
            f"  {index}. "
            f"{event.stage}: "
            f"{event.outcome}"
        )

    expected_stages = (
        "source_resolution",
        "retrieval",
        "evidence_assembly",
        "readiness",
        "sufficiency",
    )

    actual_stages = tuple(
        event.stage
        for event in trace
    )

    if (
        actual_stages
        != expected_stages
    ):
        raise RuntimeError(
            "Failure Trace 与预期不一致："
            f"{actual_stages}"
        )

    if (
        trace[-1].outcome
        != "system_failed"
    ):
        raise RuntimeError(
            "Sufficiency Failure "
            "必须记录为 system_failed"
        )

    print()
    print(
        "Sufficiency requests:",
        sufficiency_provider
        .request_count,
    )

    print(
        "Answer requests:",
        answer_provider
        .request_count,
    )

    print()
    print(
        "FAILURE ROUTING PASS"
    )


if __name__ == "__main__":
    main()