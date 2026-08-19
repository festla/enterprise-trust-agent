from app.schemas.competition import (
    CompetitionQuestion,
)
from app.schemas.competition_excel import (
    CompetitionExcelCell,
    CompetitionExcelMergedRange,
    CompetitionExcelSheet,
    CompetitionExcelWorkbook,
)
from app.services.competition_excel_compare import (
    solve_excel_table_compare,
)


def test_compare_argmax_by_column_scope(
) -> None:
    sheet = CompetitionExcelSheet(
        name="各地区数据（月度）",
        sheet_index=0,
        visible=True,
        declared_max_row=7,
        declared_max_column=3,
        cells=(
            CompetitionExcelCell(
                coordinate="B3",
                row_index=3,
                column_index=2,
                cell_type="text",
                text_value="地区",
            ),
            CompetitionExcelCell(
                coordinate="C3",
                row_index=3,
                column_index=3,
                cell_type="text",
                text_value="合计",
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
                text_value="100",
                numeric_value=100,
            ),
            CompetitionExcelCell(
                coordinate="B5",
                row_index=5,
                column_index=2,
                cell_type="text",
                text_value="北 京",
            ),
            CompetitionExcelCell(
                coordinate="C5",
                row_index=5,
                column_index=3,
                cell_type="number",
                text_value="40",
                numeric_value=40,
            ),
            CompetitionExcelCell(
                coordinate="B6",
                row_index=6,
                column_index=2,
                cell_type="text",
                text_value="天 津",
            ),
            CompetitionExcelCell(
                coordinate="C6",
                row_index=6,
                column_index=3,
                cell_type="number",
                text_value="20",
                numeric_value=20,
            ),
            CompetitionExcelCell(
                coordinate="B7",
                row_index=7,
                column_index=2,
                cell_type="text",
                text_value="河 北",
            ),
            CompetitionExcelCell(
                coordinate="C7",
                row_index=7,
                column_index=3,
                cell_type="number",
                text_value="30",
                numeric_value=30,
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
        qa_type="表格比较",
        question=(
            "在“合计”口径下，"
            "以下哪一项数值最高？"
        ),
        option_a="北 京",
        option_b="天 津",
        option_c="河 北",
        option_d="全 国",
        source_title="测试",
        file_label="test.xlsx",
    )

    result = (
        solve_excel_table_compare(
            question=question,
            workbook=workbook,
        )
    )

    assert (
        result.answer_option
        == "D"
    )

    assert (
        result.answer_text
        == "全  国"
    )

    assert (
        result.winning_value
        == 100
    )

    assert (
        result.operation
        == "max"
    )


def test_compare_under_merged_header(
) -> None:
    merged = (
        CompetitionExcelMergedRange(
            range_ref="C2:D2",
            min_row=2,
            max_row=2,
            min_column=3,
            max_column=4,
            anchor_coordinate="C2",
        )
    )

    sheet = CompetitionExcelSheet(
        name="资金运用",
        sheet_index=0,
        visible=True,
        declared_max_row=7,
        declared_max_column=4,
        merged_ranges=(merged,),
        cells=(
            CompetitionExcelCell(
                coordinate="C2",
                row_index=2,
                column_index=3,
                cell_type="text",
                text_value="截至当期",
                merged_range="C2:D2",
            ),
            CompetitionExcelCell(
                coordinate="D3",
                row_index=3,
                column_index=4,
                cell_type="text",
                text_value="账面余额",
            ),
            CompetitionExcelCell(
                coordinate="B4",
                row_index=4,
                column_index=2,
                cell_type="text",
                text_value="项目甲",
            ),
            CompetitionExcelCell(
                coordinate="D4",
                row_index=4,
                column_index=4,
                cell_type="number",
                text_value="100",
                numeric_value=100,
            ),
            CompetitionExcelCell(
                coordinate="B5",
                row_index=5,
                column_index=2,
                cell_type="text",
                text_value="项目乙",
            ),
            CompetitionExcelCell(
                coordinate="D5",
                row_index=5,
                column_index=4,
                cell_type="number",
                text_value="300",
                numeric_value=300,
            ),
            CompetitionExcelCell(
                coordinate="B6",
                row_index=6,
                column_index=2,
                cell_type="text",
                text_value="项目丙",
            ),
            CompetitionExcelCell(
                coordinate="D6",
                row_index=6,
                column_index=4,
                cell_type="number",
                text_value="200",
                numeric_value=200,
            ),
            CompetitionExcelCell(
                coordinate="B7",
                row_index=7,
                column_index=2,
                cell_type="text",
                text_value="项目丁",
            ),
            CompetitionExcelCell(
                coordinate="D7",
                row_index=7,
                column_index=4,
                cell_type="number",
                text_value="50",
                numeric_value=50,
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
        qa_type="表格比较",
        question=(
            "在“截至当期-账面余额”"
            "口径下，以下哪一项"
            "数值最高？"
        ),
        option_a="项目甲",
        option_b="项目乙",
        option_c="项目丙",
        option_d="项目丁",
        source_title="测试",
        file_label="test.xlsx",
    )

    result = (
        solve_excel_table_compare(
            question=question,
            workbook=workbook,
        )
    )

    assert (
        result.answer_option
        == "B"
    )

    assert (
        result.winning_value
        == 300
    )

    assert (
        "截至当期"
        in result.items[1]
        .context_labels
    )

    assert (
        "账面余额"
        in result.items[1]
        .context_labels
    )

def test_compare_resolves_structural_option_prefix(
) -> None:
    sheet = CompetitionExcelSheet(
        name="人身保险公司（月度）",
        sheet_index=0,
        visible=True,
        declared_max_row=8,
        declared_max_column=3,
        cells=(
            CompetitionExcelCell(
                coordinate="C4",
                row_index=4,
                column_index=3,
                cell_type="text",
                text_value="本年累计/截至当期",
            ),
            CompetitionExcelCell(
                coordinate="B5",
                row_index=5,
                column_index=2,
                cell_type="text",
                text_value="新增保险金额",
            ),
            CompetitionExcelCell(
                coordinate="C5",
                row_index=5,
                column_index=3,
                cell_type="number",
                text_value="100",
                numeric_value=100,
            ),
            CompetitionExcelCell(
                coordinate="B6",
                row_index=6,
                column_index=2,
                cell_type="text",
                text_value="其中：寿险",
            ),
            CompetitionExcelCell(
                coordinate="C6",
                row_index=6,
                column_index=3,
                cell_type="number",
                text_value="200",
                numeric_value=200,
            ),
            CompetitionExcelCell(
                coordinate="B7",
                row_index=7,
                column_index=2,
                cell_type="text",
                text_value="原保险保费收入",
            ),
            CompetitionExcelCell(
                coordinate="C7",
                row_index=7,
                column_index=3,
                cell_type="number",
                text_value="300",
                numeric_value=300,
            ),
            CompetitionExcelCell(
                coordinate="B8",
                row_index=8,
                column_index=2,
                cell_type="text",
                text_value="意外险",
            ),
            CompetitionExcelCell(
                coordinate="C8",
                row_index=8,
                column_index=3,
                cell_type="number",
                text_value="50",
                numeric_value=50,
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
        qa_type="表格比较",
        question=(
            "在“本年累计/截至当期”"
            "口径下，以下哪一项"
            "数值最高？"
        ),
        option_a="新增保险金额",
        option_b="寿险",
        option_c="原保险保费收入",
        option_d="意外险",
        source_title="测试",
        file_label="test.xlsx",
    )

    result = (
        solve_excel_table_compare(
            question=question,
            workbook=workbook,
        )
    )

    assert (
        result.answer_option
        == "C"
    )

    item_b = next(
        item
        for item in result.items
        if item.option == "B"
    )

    assert (
        item_b.label_text
        == "其中：寿险"
    )

def test_compare_excludes_option_from_other_section(
) -> None:
    sheet = CompetitionExcelSheet(
        name="Sheet1",
        sheet_index=0,
        visible=True,
        declared_max_row=13,
        declared_max_column=3,
        cells=(
            CompetitionExcelCell(
                coordinate="C4",
                row_index=4,
                column_index=3,
                cell_type="text",
                text_value="本年累计/截至当期",
            ),
            CompetitionExcelCell(
                coordinate="B6",
                row_index=6,
                column_index=2,
                cell_type="text",
                text_value="原保险保费收入",
            ),
            CompetitionExcelCell(
                coordinate="C6",
                row_index=6,
                column_index=3,
                cell_type="number",
                text_value="16590.18",
                numeric_value=16590.18,
            ),
            CompetitionExcelCell(
                coordinate="B7",
                row_index=7,
                column_index=2,
                cell_type="text",
                text_value="其中：寿险",
            ),
            CompetitionExcelCell(
                coordinate="C7",
                row_index=7,
                column_index=3,
                cell_type="number",
                text_value="13832.14",
                numeric_value=13832.14,
            ),
            CompetitionExcelCell(
                coordinate="B8",
                row_index=8,
                column_index=2,
                cell_type="text",
                text_value="意外险",
            ),
            CompetitionExcelCell(
                coordinate="C8",
                row_index=8,
                column_index=3,
                cell_type="number",
                text_value="117.14",
                numeric_value=117.14,
            ),
            CompetitionExcelCell(
                coordinate="B13",
                row_index=13,
                column_index=2,
                cell_type="text",
                text_value="新增保险金额",
            ),
            CompetitionExcelCell(
                coordinate="C13",
                row_index=13,
                column_index=3,
                cell_type="number",
                text_value="3816438.23",
                numeric_value=3816438.23,
            ),
        ),
    )

    workbook = CompetitionExcelWorkbook(
        source_id="src_0123456789abcdef",
        relative_path="test.xlsx",
        excel_format="xlsx",
        sheets=(sheet,),
    )

    question = CompetitionQuestion(
        case_id="Q001",
        source_type="excel",
        qa_type="表格比较",
        question=(
            "在本年累计/截至当期口径下，"
            "以下哪一项数值最高？"
        ),
        option_a="新增保险金额",
        option_b="寿险",
        option_c="原保险保费收入",
        option_d="意外险",
        source_title="test",
        file_label="test.xlsx",
    )

    result = solve_excel_table_compare(
        question=question,
        workbook=workbook,
    )

    assert (
        result.answer_option
        == "C"
    )

    assert {
        item.option
        for item in result.items
    } == {
        "B",
        "C",
        "D",
    }