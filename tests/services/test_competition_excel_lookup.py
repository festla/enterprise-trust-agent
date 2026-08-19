from app.schemas.competition import (
    CompetitionQuestion,
)
from app.services.competition_excel_lookup import (
    _resolve_answer_option,
)


def test_option_resolver_uses_display_precision(
) -> None:
    question = CompetitionQuestion(
        case_id="Q001",
        source_type="excel",
        qa_type="表格取数",
        question="测试",
        option_a="259039.98",
        option_b="259040.10",
        option_c="259039.99",
        option_d="259039.50",
        source_title="测试",
        file_label="测试.xlsx",
    )

    option = _resolve_answer_option(
        question=question,
        numeric_value=(
            259039.987564867
        ),
    )

    assert option == "C"


def test_option_resolver_refuses_ambiguous_rounding(
) -> None:
    question = CompetitionQuestion(
        case_id="Q001",
        source_type="excel",
        qa_type="表格取数",
        question="测试",
        option_a="10.0",
        option_b="10.04",
        option_c="20",
        option_d="30",
        source_title="测试",
        file_label="测试.xlsx",
    )

    option = _resolve_answer_option(
        question=question,
        numeric_value=10.04,
    )

    assert option is None