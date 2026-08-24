from __future__ import annotations

import hashlib

import pytest

from app.schemas.competition import (
    CompetitionQaCase,
)
from app.schemas.competition_chunk import (
    CompetitionChunkSourceSpan,
    CompetitionTextChunk,
)
from app.services.competition_evidence_coverage import (
    audit_competition_evidence_coverage_case,
    summarize_competition_evidence_coverage,
)


SOURCE_ID = "src_0123456789abcdef"
DOC_ID = (
    "doc_src_0123456789abcdef_"
    "0123456789abcdef01234567"
)


def _case(
    *,
    case_id: str,
    evidence: str,
) -> CompetitionQaCase:
    return CompetitionQaCase(
        case_id=case_id,
        source_type="pdf",
        difficulty="easy",
        difficulty_cn="简单",
        qa_type="单事实检索",
        question="下列哪项表述正确？",
        option_a="选项 A",
        option_b="选项 B",
        option_c="选项 C",
        option_d="选项 D",
        answer="A",
        answer_text="选项 A",
        evidence=evidence,
        source_title="测试监管文件",
        file_label="测试监管文件.pdf",
    )


def _chunk(
    *,
    text: str,
    chunk_index: int,
    source_id: str = SOURCE_ID,
) -> CompetitionTextChunk:
    doc_id = (
        f"doc_{source_id}_"
        "0123456789abcdef01234567"
    )

    return CompetitionTextChunk(
        chunk_id=(
            f"chunk:{source_id}:"
            f"{chunk_index:05d}"
        ),
        source_id=source_id,
        doc_id=doc_id,
        source_type="pdf",
        chunk_index=chunk_index,
        chunk_type="text",
        source_spans=(
            CompetitionChunkSourceSpan(
                block_id=(
                    f"block:{source_id}:"
                    f"{chunk_index:05d}"
                ),
                block_index=chunk_index,
                start_char=0,
                end_char=len(text),
            ),
        ),
        text=text,
        char_count=len(text),
        text_sha256=hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest(),
        page_start=chunk_index + 1,
        page_end=chunk_index + 1,
    )


def test_finds_fact_inside_single_chunk() -> None:
    fact = "资本充足率不得低于百分之八。"

    result = audit_competition_evidence_coverage_case(
        case=_case(
            case_id="Q101",
            evidence=fact,
        ),
        expected_source_id=SOURCE_ID,
        chunks=(
            _chunk(
                text=f"本办法规定，{fact}",
                chunk_index=0,
            ),
        ),
    )

    coverage = result.fact_coverages[0]

    assert coverage.single_chunk_covered
    assert coverage.two_chunk_window_covered
    assert coverage.document_covered
    assert not coverage.boundary_only
    assert not coverage.document_only
    assert not coverage.missing


def test_detects_fact_crossing_two_chunks() -> None:
    result = audit_competition_evidence_coverage_case(
        case=_case(
            case_id="Q102",
            evidence=(
                "资本充足率不得"
                "低于百分之八。"
            ),
        ),
        expected_source_id=SOURCE_ID,
        chunks=(
            _chunk(
                text="资本充足率不得",
                chunk_index=0,
            ),
            _chunk(
                text="低于百分之八。",
                chunk_index=1,
            ),
        ),
    )

    coverage = result.fact_coverages[0]

    assert not coverage.single_chunk_covered
    assert coverage.two_chunk_window_covered
    assert coverage.boundary_only
    assert coverage.document_covered
    assert not coverage.document_only


def test_detects_document_only_fact() -> None:
    result = audit_competition_evidence_coverage_case(
        case=_case(
            case_id="Q103",
            evidence=(
                "第一部分"
                "第二部分"
                "第三部分"
            ),
        ),
        expected_source_id=SOURCE_ID,
        chunks=(
            _chunk(
                text="第一部分",
                chunk_index=0,
            ),
            _chunk(
                text="第二部分",
                chunk_index=1,
            ),
            _chunk(
                text="第三部分",
                chunk_index=2,
            ),
        ),
    )

    coverage = result.fact_coverages[0]

    assert not coverage.single_chunk_covered
    assert not coverage.two_chunk_window_covered
    assert coverage.document_covered
    assert coverage.document_only
    assert not coverage.missing


def test_does_not_match_wrong_source() -> None:
    fact = "只有错误来源包含这条事实。"

    result = audit_competition_evidence_coverage_case(
        case=_case(
            case_id="Q104",
            evidence=fact,
        ),
        expected_source_id=SOURCE_ID,
        chunks=(
            _chunk(
                text="正确来源没有目标事实。",
                chunk_index=0,
            ),
            _chunk(
                text=fact,
                chunk_index=0,
                source_id=(
                    "src_fedcba9876543210"
                ),
            ),
        ),
    )

    coverage = result.fact_coverages[0]

    assert not coverage.single_chunk_covered
    assert not coverage.two_chunk_window_covered
    assert not coverage.document_covered
    assert coverage.missing


def test_summarizes_coverage_levels() -> None:
    single = audit_competition_evidence_coverage_case(
        case=_case(
            case_id="Q105",
            evidence="单块事实",
        ),
        expected_source_id=SOURCE_ID,
        chunks=(
            _chunk(
                text="单块事实",
                chunk_index=0,
            ),
        ),
    )

    boundary = audit_competition_evidence_coverage_case(
        case=_case(
            case_id="Q106",
            evidence="跨块事实",
        ),
        expected_source_id=SOURCE_ID,
        chunks=(
            _chunk(
                text="跨块",
                chunk_index=0,
            ),
            _chunk(
                text="事实",
                chunk_index=1,
            ),
        ),
    )

    document_only = (
        audit_competition_evidence_coverage_case(
            case=_case(
                case_id="Q107",
                evidence="甲乙丙",
            ),
            expected_source_id=SOURCE_ID,
            chunks=(
                _chunk(
                    text="甲",
                    chunk_index=0,
                ),
                _chunk(
                    text="乙",
                    chunk_index=1,
                ),
                _chunk(
                    text="丙",
                    chunk_index=2,
                ),
            ),
        )
    )

    missing = audit_competition_evidence_coverage_case(
        case=_case(
            case_id="Q108",
            evidence="不存在的事实",
        ),
        expected_source_id=SOURCE_ID,
        chunks=(
            _chunk(
                text="无关内容",
                chunk_index=0,
            ),
        ),
    )

    summary = summarize_competition_evidence_coverage(
        (
            single,
            boundary,
            document_only,
            missing,
        )
    )

    assert summary.case_count == 4
    assert summary.fact_count == 4

    assert (
        summary.single_chunk_covered_fact_count
        == 1
    )
    assert (
        summary.two_chunk_covered_fact_count
        == 2
    )
    assert (
        summary.document_covered_fact_count
        == 3
    )

    assert summary.boundary_only_fact_count == 1
    assert summary.document_only_fact_count == 1
    assert summary.missing_fact_count == 1

    assert summary.single_chunk_fact_coverage == 0.25
    assert summary.two_chunk_fact_coverage == 0.5
    assert summary.document_fact_coverage == 0.75


def test_rejects_missing_expected_source() -> None:
    with pytest.raises(
        ValueError,
        match="没有正确来源",
    ):
        audit_competition_evidence_coverage_case(
            case=_case(
                case_id="Q109",
                evidence="监管事实",
            ),
            expected_source_id=SOURCE_ID,
            chunks=(
                _chunk(
                    text="监管事实",
                    chunk_index=0,
                    source_id=(
                        "src_fedcba9876543210"
                    ),
                ),
            ),
        )