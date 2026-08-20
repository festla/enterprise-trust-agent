from __future__ import annotations

from pathlib import Path

from app.schemas.competition import (
    CompetitionQuestion,
    CompetitionSourceRecord,
)
from app.schemas.competition_excel_solver import (
    CompetitionExcelCalculationOperand,
    CompetitionExcelCalculationResult,
    CompetitionExcelCompareItem,
    CompetitionExcelCompareResult,
    CompetitionExcelEvidence,
    CompetitionExcelLookupResult,
)
from app.services.competition_excel_evidence import (
    build_excel_calculation_evidence_bundle,
    build_excel_compare_evidence_bundle,
    build_excel_lookup_evidence_bundle,
)


# ============================================================
# Test Helpers
# ============================================================


def _build_test_source(
    *,
    attachments_root: Path,
) -> CompetitionSourceRecord:
    """
    创建一个真实存在的临时附件。

    Source Catalog 会验证：
    - 文件是否存在
    - 文件名是否一致
    - 文件大小是否一致
    - SHA-256

    因此这里不能只虚构 CompetitionSourceRecord。
    """

    source_path = (
        attachments_root
        / "test.xlsx"
    )

    source_path.write_bytes(
        b"competition-excel-evidence-test"
    )

    return CompetitionSourceRecord(
        source_id=(
            "src_0123456789abcdef"
        ),
        source_type="excel",
        actual_filename="test.xlsx",
        relative_path="test.xlsx",
        extension=".xlsx",
        size_bytes=(
            source_path
            .stat()
            .st_size
        ),
    )


def _build_lookup_question(
) -> CompetitionQuestion:
    return CompetitionQuestion(
        case_id="Q001",
        source_type="excel",
        qa_type="表格取数",
        question=(
            "根据测试 Excel，"
            "指标值是多少？"
        ),
        option_a="100",
        option_b="200",
        option_c="300",
        option_d="400",
        source_title="测试监管报表",
        file_label="test.xlsx",
    )


def _build_compare_question(
) -> CompetitionQuestion:
    return CompetitionQuestion(
        case_id="Q002",
        source_type="excel",
        qa_type="表格比较",
        question=(
            "根据测试 Excel，"
            "以下哪一项数值最高？"
        ),
        option_a="指标A",
        option_b="指标B",
        option_c="指标C",
        option_d="指标D",
        source_title="测试监管报表",
        file_label="test.xlsx",
    )


def _build_calculation_question(
) -> CompetitionQuestion:
    return CompetitionQuestion(
        case_id="Q003",
        source_type="excel",
        qa_type="表格计算",
        question=(
            "“全 国”从“合计”到"
            "“健康险”的数值变化"
            "约为多少？"
        ),
        option_a="-48090.66",
        option_b="43718.78",
        option_c="-43718.78",
        option_d="-39346.90",
        source_title=(
            "测试监管报表"
        ),
        file_label="test.xlsx",
    )


# ============================================================
# Lookup Evidence
# ============================================================


def test_lookup_result_builds_single_cell_evidence(
    tmp_path: Path,
) -> None:
    attachments_root = (
        tmp_path
        / "attachments"
    )

    attachments_root.mkdir()

    source = _build_test_source(
        attachments_root=(
            attachments_root
        )
    )

    question = (
        _build_lookup_question()
    )

    result = CompetitionExcelLookupResult(
        answer_option="A",
        answer_text="100",
        evidence=(
            CompetitionExcelEvidence(
                sheet_name="Sheet1",
                value_coordinate="C5",
                value_text="100",
                numeric_value=100,
                number_format="0",
                row_label_coordinate="B5",
                row_label="测试指标",
                column_label_coordinate="C4",
                column_label="本年累计",
            )
        ),
        confidence=0.95,
    )

    bundle = (
        build_excel_lookup_evidence_bundle(
            question=question,
            source=source,
            result=result,
            attachments_root=(
                attachments_root
            ),
        )
    )

    # ========================================================
    # Evidence Bundle
    # ========================================================

    assert len(
        bundle.evidences
    ) == 1

    assert (
        bundle.calculation
        is None
    )

    evidence = (
        bundle.evidences[0]
    )

    # ========================================================
    # Source Identity
    # ========================================================

    assert (
        evidence.source.source_id
        == source.source_id
    )

    assert (
        evidence.source.title
        == "测试监管报表"
    )

    assert (
        evidence.source.source_type
        == "excel"
    )

    assert (
        evidence.source.relative_path
        == "test.xlsx"
    )

    assert (
        evidence.source.sha256
        is not None
    )

    assert len(
        evidence.source.sha256
    ) == 64

    assert (
        evidence.source.doc_id
        .startswith(
            "doc_src_0123456789abcdef_"
        )
    )

    # ========================================================
    # Cell Evidence
    # ========================================================

    assert (
        evidence.evidence_type
        == "table_cell"
    )

    assert (
        evidence.location
        .sheet_name
        == "Sheet1"
    )

    assert (
        evidence.location.cell
        == "C5"
    )

    assert (
        evidence.location
        .row_label
        == "测试指标"
    )

    assert (
        evidence.location
        .column_label
        == "本年累计"
    )

    assert (
        evidence.numeric_value
        == 100
    )

    assert (
        evidence.display_value
        == "100"
    )


# ============================================================
# Compare Evidence
# ============================================================


def test_compare_result_builds_multiple_cell_evidence(
    tmp_path: Path,
) -> None:
    attachments_root = (
        tmp_path
        / "attachments"
    )

    attachments_root.mkdir()

    source = _build_test_source(
        attachments_root=(
            attachments_root
        )
    )

    question = (
        _build_compare_question()
    )

    result = CompetitionExcelCompareResult(
        operation="max",
        answer_option="B",
        answer_text="指标B",
        winning_value=200,
        sheet_name="Sheet1",
        items=(
            CompetitionExcelCompareItem(
                option="A",
                label_text="指标A",
                label_coordinate="B5",
                value_coordinate="C5",
                value_text="100",
                numeric_value=100,
                context_labels=(
                    "本年累计",
                ),
            ),
            CompetitionExcelCompareItem(
                option="B",
                label_text="指标B",
                label_coordinate="B6",
                value_coordinate="C6",
                value_text="200",
                numeric_value=200,
                context_labels=(
                    "本年累计",
                ),
            ),
        ),
        confidence=0.95,
    )

    bundle = (
        build_excel_compare_evidence_bundle(
            question=question,
            source=source,
            result=result,
            attachments_root=(
                attachments_root
            ),
        )
    )

    assert len(
        bundle.evidences
    ) == 2

    assert (
        bundle.calculation
        is None
    )

    first = (
        bundle.evidences[0]
    )

    second = (
        bundle.evidences[1]
    )

    # ========================================================
    # 所有 Compare Evidence 必须来自同一文件版本。
    # ========================================================

    assert (
        first.source
        == second.source
    )

    assert (
        first.source.source_id
        == source.source_id
    )

    assert (
        first.source.sha256
        is not None
    )

    # ========================================================
    # 第一项 Evidence
    # ========================================================

    assert (
        first.location
        .sheet_name
        == "Sheet1"
    )

    assert (
        first.location.cell
        == "C5"
    )

    assert (
        first.location
        .row_label
        == "指标A"
    )

    assert (
        first.numeric_value
        == 100
    )

    # ========================================================
    # 第二项 Evidence
    # ========================================================

    assert (
        second.location
        .sheet_name
        == "Sheet1"
    )

    assert (
        second.location.cell
        == "C6"
    )

    assert (
        second.location
        .row_label
        == "指标B"
    )

    assert (
        second.numeric_value
        == 200
    )


# ============================================================
# Calculation Evidence
# ============================================================


def test_calculation_result_builds_trace_with_two_inputs(
    tmp_path: Path,
) -> None:
    attachments_root = (
        tmp_path
        / "attachments"
    )

    attachments_root.mkdir()

    source = _build_test_source(
        attachments_root=(
            attachments_root
        )
    )

    question = (
        _build_calculation_question()
    )

    result = (
        CompetitionExcelCalculationResult(
            operation="difference",
            entity_text="全  国",
            entity_coordinate="B4",
            sheet_name=(
                "各地区数据（月度）"
            ),
            start=(
                CompetitionExcelCalculationOperand(
                    role="start",
                    scope_text="合计",
                    coordinate="C4",
                    numeric_value=52145.77,
                    header_labels=(
                        "合计",
                    ),
                )
            ),
            end=(
                CompetitionExcelCalculationOperand(
                    role="end",
                    scope_text="健康险",
                    coordinate="G4",
                    numeric_value=8426.99,
                    header_labels=(
                        "健康险",
                    ),
                )
            ),
            formula="G4 - C4",
            result=-43718.78,
            answer_option="C",
            answer_text="-43718.78",
            resolution_mode=(
                "semantic_headers"
            ),
            confidence=0.98,
        )
    )

    bundle = (
        build_excel_calculation_evidence_bundle(
            question=question,
            source=source,
            result=result,
            attachments_root=(
                attachments_root
            ),
        )
    )

    # ========================================================
    # 两个原始 Cell Evidence。
    # ========================================================

    assert len(
        bundle.evidences
    ) == 2

    start_evidence = (
        bundle.evidences[0]
    )

    end_evidence = (
        bundle.evidences[1]
    )

    # ========================================================
    # 两个 Operand 必须来自同一实际文件版本。
    # ========================================================

    assert (
        start_evidence.source
        == end_evidence.source
    )

    assert (
        start_evidence.source.sha256
        is not None
    )

    assert (
        start_evidence.source.doc_id
        .startswith(
            "doc_src_0123456789abcdef_"
        )
    )

    # ========================================================
    # Start Evidence
    # ========================================================

    assert (
        start_evidence.location
        .sheet_name
        == "各地区数据（月度）"
    )

    assert (
        start_evidence.location.cell
        == "C4"
    )

    assert (
        start_evidence.location
        .row_label
        == "全  国"
    )

    assert (
        start_evidence.location
        .column_label
        == "合计"
    )

    assert (
        start_evidence.numeric_value
        == 52145.77
    )

    # ========================================================
    # End Evidence
    # ========================================================

    assert (
        end_evidence.location
        .sheet_name
        == "各地区数据（月度）"
    )

    assert (
        end_evidence.location.cell
        == "G4"
    )

    assert (
        end_evidence.location
        .row_label
        == "全  国"
    )

    assert (
        end_evidence.location
        .column_label
        == "健康险"
    )

    assert (
        end_evidence.numeric_value
        == 8426.99
    )

    # ========================================================
    # Calculation Trace
    #
    # Calculation 不是原始文档 Evidence，
    # 而是显式引用两个 Evidence ID 的派生结果。
    # ========================================================

    assert (
        bundle.calculation
        is not None
    )

    assert (
        bundle.calculation.operation
        == "difference"
    )

    assert (
        bundle.calculation.formula
        == "G4 - C4"
    )

    assert (
        bundle.calculation.result
        == -43718.78
    )

    assert (
        bundle.calculation
        .resolution_mode
        == "semantic_headers"
    )

    assert (
        bundle.calculation
        .input_evidence_ids
        == (
            start_evidence.evidence_id,
            end_evidence.evidence_id,
        )
    )