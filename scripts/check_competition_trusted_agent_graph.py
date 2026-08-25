from __future__ import annotations

import argparse
from pathlib import Path

from app.schemas.competition import (
    CompetitionQuestion,
)
from app.services.competition_dataset import (
    load_competition_qa_excel,
)
from app.workflow.competition_runtime import (
    load_competition_agent_model_resources_from_environment,
    load_competition_agent_runtime_resources,
)
from app.workflow.competition_text_graph import (
    build_competition_text_preparation_graph,
)
from app.workflow.competition_agent_graph import (
    build_competition_agent_graph,
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


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--case-id",
        default="Q105",
    )

    args = parser.parse_args()

    cases = (
        load_competition_qa_excel(
            QA_FILE
        )
    )

    case = next(
        (
            item
            for item in cases
            if item.case_id
            == args.case_id
        ),
        None,
    )

    if case is None:
        raise RuntimeError(
            "找不到 Case："
            f"{args.case_id}"
        )

    question = (
        _to_safe_question(
            case
        )
    )

    if (
        question.source_type
        not in {
            "word",
            "pdf",
        }
    ):
        raise RuntimeError(
            "当前 Graph Smoke "
            "仅支持 Text QA"
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

    model_resources = (
        load_competition_agent_model_resources_from_environment()
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
            "Agent Graph 没有生成 result"
        )

    print()
    print(
        "===== Competition Trusted Agent ====="
    )

    print(
        "Case:",
        question.case_id,
    )

    resolution = final_state.get(
        "source_resolution"
    )

    if resolution is not None:
        print(
            "Source:",
            resolution.source_id,
        )

    retrieval = final_state.get(
        "retrieval"
    )

    if retrieval is not None:
        print(
            "Expanded hits:",
            len(
                retrieval.expanded_hits
            ),
        )

    evidence = final_state.get(
        "evidence_bundle"
    )

    if evidence is not None:
        print(
            "Evidence:",
            len(
                evidence.evidences
            ),
        )

    readiness = final_state.get(
        "readiness"
    )

    if readiness is not None:
        print(
            "Readiness:",
            readiness.status,
        )

    sufficiency = final_state.get(
        "sufficiency"
    )

    if sufficiency is not None:
        print(
            "Sufficiency:",
            sufficiency.status,
        )

        print(
            "Judge answer:",
            sufficiency.supported_answer,
        )

    print(
        "Final status:",
        result.status,
    )

    if result.status == "answered":
        print(
            "Agent answer:",
            result.answer.answer,
        )

        print(
            "Gold answer:",
            case.answer,
        )

        print(
            "Citations:",
            result.answer.citation_ids,
        )

        if (
            result.answer.answer
            != case.answer
        ):
            raise RuntimeError(
                "Agent Answer 与 Gold 不一致"
            )

    else:
        print(
            "Refusal code:",
            result.refusal_code,
        )

        raise RuntimeError(
            "Q105 本应能够回答，"
            "但 Agent 发生拒答"
        )

    print()
    print(
        "TRUSTED AGENT GRAPH PASS"
    )

if __name__ == "__main__":
    main()