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
    load_competition_agent_runtime_resources,
)
from app.workflow.competition_text_graph import (
    build_competition_text_preparation_graph,
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

    #
    # Process startup
    #
    resources = (
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

    graph = (
        build_competition_text_preparation_graph(
            resources
        )
    )

    #
    # Request execution
    #
    result = graph.invoke(
        {
            "question": question,
        }
    )

    resolution = result.get(
        "source_resolution"
    )

    retrieval = result.get(
        "retrieval"
    )

    evidence = result.get(
        "evidence_bundle"
    )

    if resolution is None:
        raise RuntimeError(
            "Graph 缺少 "
            "source_resolution"
        )

    if retrieval is None:
        raise RuntimeError(
            "Graph 缺少 retrieval"
        )

    if evidence is None:
        raise RuntimeError(
            "Graph 缺少 evidence_bundle"
        )

    evidence_chars = sum(
        len(item.raw_content)
        for item
        in evidence.evidences
    )

    evidence_source_ids = {
        item.source.source_id
        for item
        in evidence.evidences
    }

    if evidence_source_ids != {
        resolution.source_id
    }:
        raise RuntimeError(
            "Graph Evidence "
            "出现跨 Source 泄漏："
            f"{evidence_source_ids}"
        )

    print()
    print(
        "===== Competition "
        "Text Preparation Graph ====="
    )

    print(
        "Case:",
        question.case_id,
    )

    print(
        "Source:",
        resolution.source_id,
    )

    print(
        "Resolution:",
        resolution.strategy,
    )

    print(
        "Question-only hits:",
        len(
            retrieval
            .question_only_hits
        ),
    )

    print(
        "Question+options hits:",
        len(
            retrieval
            .question_with_options_hits
        ),
    )

    print(
        "Merged seeds:",
        len(
            retrieval.seed_hits
        ),
    )

    print(
        "Expanded hits:",
        len(
            retrieval.expanded_hits
        ),
    )

    print(
        "Evidence:",
        len(
            evidence.evidences
        ),
    )

    print(
        "Evidence chars:",
        evidence_chars,
    )

    print()
    print(
        "GRAPH RUNTIME PASS"
    )


if __name__ == "__main__":
    main()