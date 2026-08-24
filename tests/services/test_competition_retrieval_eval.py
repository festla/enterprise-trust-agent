from __future__ import annotations

import pytest

from app.schemas.competition import (
    CompetitionQaCase,
)
from app.schemas.competition_evidence import (
    CompetitionKnowledgeSource,
)
from app.schemas.competition_retrieval import (
    CompetitionBM25Hit,
)
from app.schemas.competition_text import (
    CompetitionTextBlock,
    CompetitionTextDocument,
)
from app.services.competition_retrieval_eval import (
    evaluate_competition_retrieval_case,
    normalize_competition_evidence_text,
    split_competition_evidence_facts,
    summarize_competition_retrieval,
)
from app.services.competition_text_chunker import (
    build_competition_text_chunks,
)


def _case(
    *,
    case_id: str = "Q101",
    evidence: str,
    qa_type: str = "单事实检索",
) -> CompetitionQaCase:
    return CompetitionQaCase(
        case_id=case_id,
        source_type="word",
        difficulty="easy",
        difficulty_cn="简单",
        qa_type=qa_type,
        question="下列哪项表述正确？",
        option_a="选项 A",
        option_b="选项 B",
        option_c="选项 C",
        option_d="选项 D",
        answer="A",
        answer_text="选项 A",
        evidence=evidence,
        source_title="测试制度",
        file_label="测试制度.docx",
    )


def _hit(
    *,
    source_number: int,
    text: str,
    rank: int,
) -> CompetitionBM25Hit:
    source_id = (
        f"src_{source_number:016x}"
    )

    source = CompetitionKnowledgeSource(
        source_id=source_id,
        doc_id=(
            f"doc_{source_id}_"
            "0123456789abcdef01234567"
        ),
        title="检索测试来源",
        source_type="word",
        relative_path=(
            f"word/{source_id}.docx"
        ),
        sha256=(
            f"{source_number:x}"[-1]
            * 64
        ),
    )

    document = CompetitionTextDocument(
        source=source,
        blocks=(
            CompetitionTextBlock(
                block_id=(
                    f"block:{source.doc_id}:"
                    "00000"
                ),
                source_id=source.source_id,
                doc_id=source.doc_id,
                source_type="word",
                block_index=0,
                block_type="paragraph",
                paragraph_index=0,
                text=text,
            ),
        ),
    )

    chunk = (
        build_competition_text_chunks(
            document
        )[0]
    )

    return CompetitionBM25Hit(
        rank=rank,
        score=10.0 / rank,
        chunk=chunk,
    )


def test_normalizes_and_splits_evidence(
) -> None:
    assert (
        normalize_competition_evidence_text(
            "资本 充足率：12％。"
        )
        == "资本充足率12"
    )

    assert (
        split_competition_evidence_facts(
            "事实一。；事实二。\n事实三。"
        )
        == (
            "事实一。",
            "事实二。",
            "事实三。",
        )
    )


def test_case_requires_correct_source(
) -> None:
    evidence = (
        "商业银行应满足资本充足率要求。"
    )

    wrong_source_hit = _hit(
        source_number=1,
        text=evidence,
        rank=1,
    )

    correct_source_hit = _hit(
        source_number=2,
        text=(
            "本办法规定，"
            f"{evidence}"
        ),
        rank=2,
    )

    result = (
        evaluate_competition_retrieval_case(
            case=_case(
                evidence=evidence
            ),
            expected_source_id=(
                correct_source_hit.source_id
            ),
            hits=(
                wrong_source_hit,
                correct_source_hit,
            ),
        )
    )

    assert not result.source_hit_at(1)
    assert result.source_hit_at(2)

    assert not result.any_fact_hit_at(1)
    assert result.any_fact_hit_at(2)

    assert result.first_source_rank == 2
    assert result.first_evidence_rank == 2
    assert result.reciprocal_rank_at(2) == 0.5


def test_multi_fact_recall_across_chunks(
) -> None:
    first_fact = "第一项监管事实。"
    second_fact = "第二项监管事实。"

    first_hit = _hit(
        source_number=3,
        text=first_fact,
        rank=1,
    )

    unrelated_hit = _hit(
        source_number=4,
        text="无关内容。",
        rank=2,
    )

    second_hit = _hit(
        source_number=3,
        text=second_fact,
        rank=3,
    )

    result = (
        evaluate_competition_retrieval_case(
            case=_case(
                case_id="Q103",
                evidence=(
                    f"{first_fact}；"
                    f"{second_fact}"
                ),
                qa_type="多事实检索",
            ),
            expected_source_id=(
                first_hit.source_id
            ),
            hits=(
                first_hit,
                unrelated_hit,
                second_hit,
            ),
        )
    )

    assert result.fact_count == 2
    assert result.fact_recall_at(1) == 0.5
    assert result.any_fact_hit_at(1)
    assert not result.all_facts_hit_at(1)

    assert result.fact_recall_at(3) == 1.0
    assert result.all_facts_hit_at(3)


def test_summary_reports_metrics_at_k(
) -> None:
    first_fact = "第一项监管事实。"
    second_fact = "第二项监管事实。"

    first_hit = _hit(
        source_number=5,
        text=first_fact,
        rank=1,
    )

    second_hit = _hit(
        source_number=5,
        text=second_fact,
        rank=3,
    )

    successful = (
        evaluate_competition_retrieval_case(
            case=_case(
                case_id="Q105",
                evidence=(
                    f"{first_fact}；"
                    f"{second_fact}"
                ),
                qa_type="多事实检索",
            ),
            expected_source_id=(
                first_hit.source_id
            ),
            hits=(
                first_hit,
                second_hit,
            ),
        )
    )

    empty = (
        evaluate_competition_retrieval_case(
            case=_case(
                case_id="Q107",
                evidence="另一条监管事实。",
            ),
            expected_source_id=(
                "src_0000000000000006"
            ),
            hits=(),
        )
    )

    summary = (
        summarize_competition_retrieval(
            (
                successful,
                empty,
            ),
            cutoffs=(1, 3),
        )
    )

    at_one = summary.at(1)

    assert at_one.source_hit_rate == 0.5
    assert at_one.any_fact_hit_rate == 0.5
    assert at_one.all_facts_hit_rate == 0.0
    assert at_one.mean_fact_recall == 0.25
    assert (
        at_one.mean_reciprocal_rank
        == 0.5
    )

    at_three = summary.at(3)

    assert at_three.source_hit_rate == 0.5
    assert at_three.any_fact_hit_rate == 0.5
    assert at_three.all_facts_hit_rate == 0.5
    assert at_three.mean_fact_recall == 0.5
    assert (
        at_three.mean_reciprocal_rank
        == 0.5
    )


def test_rejects_excel_case(
) -> None:
    payload = _case(
        evidence="监管事实。"
    ).model_dump(
        mode="python"
    )

    payload["source_type"] = "excel"
    payload["qa_type"] = "表格取数"

    excel_case = CompetitionQaCase(
        **payload
    )

    with pytest.raises(
        ValueError,
        match="只支持 Word/PDF",
    ):
        evaluate_competition_retrieval_case(
            case=excel_case,
            expected_source_id=(
                "src_0000000000000001"
            ),
            hits=(),
        )