from app.schemas.competition import (
    CompetitionQaCase,
    CompetitionSourceResolution,
)
from app.services.competition_dataset import (
    build_competition_gold,
    build_competition_solver_input,
)


def _build_case() -> CompetitionQaCase:
    return CompetitionQaCase(
        case_id="Q001",
        source_type="excel",
        difficulty="easy",
        difficulty_cn="简单",
        qa_type="表格取数",
        question="某指标是多少？",
        option_a="10",
        option_b="20",
        option_c="30",
        option_d="40",
        answer="B",
        answer_text="20",
        evidence=(
            "文件.xlsx；"
            "工作表：Sheet1；"
            "单元格：C5"
        ),
        source_title="测试文件",
        file_label="测试文件.xlsx",
    )


def _build_resolution(
) -> CompetitionSourceResolution:
    return CompetitionSourceResolution(
        case_id="Q001",
        source_id=(
            "src_0123456789abcdef"
        ),
        source_type="excel",
        relative_path=(
            "001_测试文件_测试文件.xlsx"
        ),
        strategy="exact_tail",
    )


def test_solver_input_does_not_contain_gold(
) -> None:
    solver_input = (
        build_competition_solver_input(
            case=_build_case(),
            resolution=(
                _build_resolution()
            ),
        )
    )

    payload = solver_input.model_dump()

    serialized = str(payload)

    assert "answer" not in payload[
        "question"
    ]

    assert "answer_text" not in payload[
        "question"
    ]

    assert "evidence" not in payload[
        "question"
    ]

    assert "difficulty" not in payload[
        "question"
    ]

    # Gold 内容本身也不能意外进入
    # Solver Input。
    assert "工作表：Sheet1" not in (
        serialized
    )


def test_gold_is_kept_separate(
) -> None:
    gold = build_competition_gold(
        _build_case()
    )

    assert gold.answer == "B"
    assert gold.answer_text == "20"

    assert (
        "C5"
        in gold.evidence
    )