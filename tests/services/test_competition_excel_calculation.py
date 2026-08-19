from app.schemas.competition import (
    CompetitionQuestion,
)
from app.schemas.competition_excel import (
    CompetitionExcelCell,
    CompetitionExcelSheet,
    CompetitionExcelWorkbook,
)
from app.services.competition_excel_calculation import (
    solve_excel_table_calculation,
)


def test_difference_by_explicit_headers(
) -> None:
    sheet = CompetitionExcelSheet(
        name="各地区数据（月度）",
        sheet_index=0,
        visible=True,
        declared_max_row=4,
        declared_max_column=7,
        cells=(
            CompetitionExcelCell(
                coordinate="C3",
                row_index=3,
                column_index=3,
                cell_type="text",
                text_value="合计",
            ),
            CompetitionExcelCell(
                coordinate="G3",
                row_index=3,
                column_index=7,
                cell_type="text",
                text_value="健康险",
            ),
            CompetitionExcelCell(
                coordinate="B4",
                row_index=4,
                column_index=2,
                cell_type="text",
                text_value="全  国",
            ),
            CompetitionExcelCell(
                coordinate="C4",
                row_index=4,
                column_index=3,
                cell_type="number",
                text_value="52145.77",
                numeric_value=52145.77,
            ),
            CompetitionExcelCell(
                coordinate="G4",
                row_index=4,
                column_index=7,
                cell_type="number",
                text_value="8426.99",
                numeric_value=8426.99,
            ),
        ),
    )

    workbook = CompetitionExcelWorkbook(
        source_id=(
            "src_0123456789abcdef"
        ),
        relative_path="test.xlsx",
        excel_format="xlsx",
        sheets=(sheet,),
    )

    question = CompetitionQuestion(
        case_id="Q001",
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
        source_title="test",
        file_label="test.xlsx",
    )

    result = (
        solve_excel_table_calculation(
            question=question,
            workbook=workbook,
        )
    )

    assert (
        result.answer_option
        == "C"
    )

    assert (
        result.start.coordinate
        == "C4"
    )

    assert (
        result.end.coordinate
        == "G4"
    )

    assert (
        result.resolution_mode
        == "semantic_headers"
    )

def test_difference_uses_unique_option_fallback_when_end_header_missing(
) -> None:
    sheet = CompetitionExcelSheet(
        name="普惠型小微企业贷款",
        sheet_index=0,
        visible=True,
        declared_max_row=7,
        declared_max_column=10,
        cells=(
            CompetitionExcelCell(
                coordinate="B4",
                row_index=4,
                column_index=2,
                cell_type="text",
                text_value="一季度",
            ),
            CompetitionExcelCell(
                coordinate="C4",
                row_index=4,
                column_index=3,
                cell_type="text",
                text_value="二季度",
            ),
            CompetitionExcelCell(
                coordinate="D4",
                row_index=4,
                column_index=4,
                cell_type="text",
                text_value="三季度",
            ),
            CompetitionExcelCell(
                coordinate="E4",
                row_index=4,
                column_index=5,
                cell_type="text",
                text_value="四季度",
            ),
            CompetitionExcelCell(
                coordinate="A7",
                row_index=7,
                column_index=1,
                cell_type="text",
                text_value="股份制商业银行",
            ),
            CompetitionExcelCell(
                coordinate="B7",
                row_index=7,
                column_index=2,
                cell_type="number",
                text_value="42909.425105019",
                numeric_value=42909.425105019,
            ),
            CompetitionExcelCell(
                coordinate="E7",
                row_index=7,
                column_index=5,
                cell_type="number",
                text_value="46642.43",
                numeric_value=46642.43,
            ),
            CompetitionExcelCell(
                coordinate="J7",
                row_index=7,
                column_index=10,
                cell_type="number",
                text_value="2.48229836442131",
                numeric_value=2.48229836442131,
            ),
        ),
    )

    workbook = CompetitionExcelWorkbook(
        source_id=(
            "src_0123456789abcdef"
        ),
        relative_path="test.xlsx",
        excel_format="xlsx",
        sheets=(sheet,),
    )

    question = CompetitionQuestion(
        case_id="Q001",
        source_type="excel",
        qa_type="表格计算",
        question=(
            "“股份制商业银行”从"
            "“年-季度”到"
            "“本年累计/截至当期”"
            "的数值变化约为多少？"
        ),
        option_a="-47197.64",
        option_b="-42906.94",
        option_c="-38616.25",
        option_d="42906.94",
        source_title="test",
        file_label="test.xlsx",
    )

    result = (
        solve_excel_table_calculation(
            question=question,
            workbook=workbook,
        )
    )

    assert (
        result.answer_option
        == "B"
    )

    assert (
        result.start.coordinate
        == "B7"
    )

    assert (
        result.end.coordinate
        == "J7"
    )

    assert (
        result.resolution_mode
        == "option_constrained_fallback"
    )