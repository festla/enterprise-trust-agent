import pytest
from pydantic import ValidationError

from app.schemas.competition_evidence import (
    CompetitionCalculationTrace,
    CompetitionEvidence,
    CompetitionEvidenceBundle,
    CompetitionExcelEvidenceLocation,
    CompetitionKnowledgeSource,
    CompetitionTextEvidenceLocation,
)


def test_excel_evidence_bundle(
) -> None:
    source = CompetitionKnowledgeSource(
        source_id="src_excel_001",
        doc_id="doc_excel_001",
        title="2025年9月保险业经营情况表",
        source_type="excel",
        relative_path="test.xlsx",
        source_url=(
            "https://example.com/test"
        ),
    )

    start = CompetitionEvidence(
        evidence_id="ev_start",
        evidence_type="table_cell",
        source=source,
        location=(
            CompetitionExcelEvidenceLocation(
                sheet_name="各地区数据（月度）",
                cell="C4",
                row_label="全 国",
                column_label="合计",
                unit="亿元",
            )
        ),
        raw_content=(
            "全 国｜合计=52145.77"
        ),
        numeric_value=52145.77,
        display_value="52145.77",
    )

    end = CompetitionEvidence(
        evidence_id="ev_end",
        evidence_type="table_cell",
        source=source,
        location=(
            CompetitionExcelEvidenceLocation(
                sheet_name="各地区数据（月度）",
                cell="G4",
                row_label="全 国",
                column_label="健康险",
                unit="亿元",
            )
        ),
        raw_content=(
            "全 国｜健康险=8426.99"
        ),
        numeric_value=8426.99,
        display_value="8426.99",
    )

    bundle = CompetitionEvidenceBundle(
        evidences=(
            start,
            end,
        ),
        calculation=(
            CompetitionCalculationTrace(
                operation="difference",
                formula="G4 - C4",
                input_evidence_ids=(
                    "ev_start",
                    "ev_end",
                ),
                result=-43718.78,
                resolution_mode=(
                    "semantic_headers"
                ),
            )
        ),
    )

    assert (
        len(bundle.evidences)
        == 2
    )

    assert (
        bundle.calculation.result
        == -43718.78
    )


def test_text_evidence_location(
) -> None:
    source = CompetitionKnowledgeSource(
        source_id="src_pdf_001",
        doc_id="doc_pdf_001",
        title="监管办法",
        source_type="pdf",
        relative_path="rule.pdf",
    )

    evidence = CompetitionEvidence(
        evidence_id="ev_rule_001",
        evidence_type="text",
        source=source,
        location=(
            CompetitionTextEvidenceLocation(
                page=12,
                section="第三章",
                article="第二十条",
                paragraph_index=0,
            )
        ),
        raw_content=(
            "商业银行应当……"
        ),
    )

    assert (
        evidence.location.article
        == "第二十条"
    )


def test_excel_location_requires_cell_or_range(
) -> None:
    with pytest.raises(
        ValidationError
    ):
        CompetitionExcelEvidenceLocation(
            sheet_name="Sheet1",
        )


def test_calculation_must_reference_existing_evidence(
) -> None:
    source = CompetitionKnowledgeSource(
        source_id="src_excel_001",
        doc_id="doc_excel_001",
        title="测试",
        source_type="excel",
        relative_path="test.xlsx",
    )

    evidence = CompetitionEvidence(
        evidence_id="ev_1",
        evidence_type="table_cell",
        source=source,
        location=(
            CompetitionExcelEvidenceLocation(
                sheet_name="Sheet1",
                cell="C5",
            )
        ),
        raw_content="指标=100",
        numeric_value=100,
    )

    with pytest.raises(
        ValidationError
    ):
        CompetitionEvidenceBundle(
            evidences=(
                evidence,
            ),
            calculation=(
                CompetitionCalculationTrace(
                    operation="difference",
                    formula="D5 - C5",
                    input_evidence_ids=(
                        "ev_1",
                        "ev_missing",
                    ),
                    result=20,
                )
            ),
        )