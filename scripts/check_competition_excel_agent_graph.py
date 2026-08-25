from __future__ import annotations

from pathlib import Path

from app.services.competition_dataset import (
    build_competition_question,
    load_competition_qa_excel,
)
from app.services.competition_excel_runtime import (
    load_competition_excel_runtime_resources,
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


TARGET_CASE_IDS = (
    "Q003",  # 表格取数
    "Q035",  # 表格比较
    "Q081",  # 表格计算
)


class NeverCalledSufficiencyProvider:
    provider_id = (
        "test:never-called-sufficiency"
    )

    def assess(
        self,
        **kwargs,
    ):
        raise AssertionError(
            "Excel Agent 不应该调用 "
            "Sufficiency Provider"
        )


class NeverCalledAnswerProvider:
    provider_id = (
        "test:never-called-answer"
    )

    def generate(
        self,
        **kwargs,
    ):
        raise AssertionError(
            "Excel Agent 不应该调用 "
            "Answer Provider"
        )


def main() -> None:
    cases = tuple(
        load_competition_qa_excel(
            QA_FILE
        )
    )

    case_by_id = {
        case.case_id: case
        for case in cases
    }

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

    excel_resources = (
        load_competition_excel_runtime_resources(
            attachments_root=(
                ATTACHMENTS_ROOT
            )
        )
    )

    model_resources = (
        CompetitionAgentModelResources(
            sufficiency_provider=(
                NeverCalledSufficiencyProvider()
            ),
            answer_provider=(
                NeverCalledAnswerProvider()
            ),
        )
    )

    graph = (
        build_competition_agent_graph(
            runtime_resources=(
                runtime_resources
            ),
            excel_resources=(
                excel_resources
            ),
            model_resources=(
                model_resources
            ),
        )
    )

    print()
    print(
        "===== Excel Agent Graph Smoke ====="
    )

    for case_id in TARGET_CASE_IDS:
        case = case_by_id[
            case_id
        ]

        question = (
            build_competition_question(
                case
            )
        )

        final_state = graph.invoke(
            {
                "question": (
                    question
                ),
            }
        )

        result = final_state.get(
            "result"
        )

        if result is None:
            raise RuntimeError(
                f"{case_id}: "
                "Graph 没有 Final Result"
            )

        if (
            result.status
            != "answered"
        ):
            raise RuntimeError(
                f"{case_id}: "
                "Excel Agent 未正常回答，"
                f"status={result.status}"
            )

        predicted = (
            result.answer.answer
        )

        correct = (
            predicted
            == case.answer
        )

        excel_runtime = (
            final_state.get(
                "excel_runtime"
            )
        )

        if excel_runtime is None:
            raise RuntimeError(
                f"{case_id}: "
                "缺少 excel_runtime"
            )

        trace = final_state.get(
            "trace",
            (),
        )

        print()
        print(
            "Case:",
            case_id,
        )

        print(
            "QA type:",
            case.qa_type,
        )

        print(
            "Solver:",
            excel_runtime
            .solver_type,
        )

        print(
            "Prediction:",
            predicted,
        )

        print(
            "Gold:",
            case.answer,
        )

        print(
            "Citations:",
            result
            .answer
            .citation_ids,
        )

        print(
            "Evidence IDs:",
            result
            .answer
            .evidence_ids,
        )

        print(
            "Trace:",
            tuple(
                (
                    event.stage,
                    event.outcome,
                )
                for event
                in trace
            ),
        )

        print(
            "Correct:",
            correct,
        )

        if not correct:
            raise RuntimeError(
                f"{case_id}: "
                "Excel Agent 回归失败"
            )

        stages = tuple(
            event.stage
            for event in trace
        )

        expected_stages = (
            "source_resolution",
            "excel_solver",
            "citation",
        )

        if (
            stages
            != expected_stages
        ):
            raise RuntimeError(
                f"{case_id}: "
                "Excel Agent Trace "
                "不符合预期："
                f"{stages}"
            )

    print()
    print(
        "Workbook cache size:",
        len(
            excel_resources
            .workbook_cache
        ),
    )

    print()
    print(
        "EXCEL AGENT GRAPH PASS"
    )


if __name__ == "__main__":
    main()