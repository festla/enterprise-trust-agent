from __future__ import annotations

import argparse
from pathlib import Path

from app.rag.tokenization import (
    DeterministicChineseBigramTokenizer,
)
from app.schemas.competition import (
    CompetitionQuestion,
)
from app.services.competition_bm25_index import (
    load_competition_bm25_index,
)
from app.services.competition_corpus import (
    load_competition_chunk_corpus,
)
from app.services.competition_dataset import (
    load_competition_qa_excel,
)
from app.services.competition_evidence_assembler import (
    CompetitionEvidenceAssemblyConfig,
    assemble_competition_evidence_bundle,
)
from app.services.competition_source_resolver import (
    CompetitionSourceResolver,
    build_competition_source_manifest,
)
from app.services.competition_text_retrieval import (
    retrieve_competition_text,
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
    """
    从 Evaluation QA Record
    显式降权为 Solver-safe Question。

    Gold 从这里开始不再进入 Runtime。
    """

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
            if (
                item.case_id
                == args.case_id
            )
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
            "当前 Runtime Check "
            "只验证 Text QA"
        )

    source_manifest = (
        build_competition_source_manifest(
            ATTACHMENTS_ROOT
        )
    )

    resolver = (
        CompetitionSourceResolver(
            source_manifest
        )
    )

    resolution = (
        resolver.resolve(
            question
        )
    )

    corpus = (
        load_competition_chunk_corpus(
            CORPUS_DIR
        )
    )

    index_result = (
        load_competition_bm25_index(
            INDEX_DIR,
            corpus_directory=(
                CORPUS_DIR
            ),
        )
    )

    tokenizer = (
        DeterministicChineseBigramTokenizer(
            spec=(
                index_result
                .manifest
                .tokenizer_spec
            )
        )
    )

    retrieval = (
        retrieve_competition_text(
            question=question,
            resolved_source_id=(
                resolution.source_id
            ),
            index=(
                index_result.index
            ),
            tokenizer=tokenizer,
            corpus_chunks=(
                corpus.chunks
            ),
        )
    )

    if not retrieval.expanded_hits:
        raise RuntimeError(
            "真实 Runtime Retrieval "
            "没有返回 Evidence Candidate"
        )

    evidence = (
        assemble_competition_evidence_bundle(
            hits=(
                retrieval.expanded_hits
            ),
            source_records=(
                corpus.manifest
                .source_records
            ),
            config=(
                CompetitionEvidenceAssemblyConfig(
                    max_evidences=48,
                    max_chars=4000,
                )
            ),
        )
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
            "Evidence 出现跨来源泄漏："
            f"{evidence_source_ids}"
        )

    print()
    print(
        "===== Competition Text "
        "Runtime Retrieval ====="
    )

    print(
        "Case:",
        question.case_id,
    )

    print(
        "Question type:",
        question.qa_type,
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
        "RUNTIME RETRIEVAL PASS"
    )


if __name__ == "__main__":
    main()