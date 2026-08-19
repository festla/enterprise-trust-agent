from __future__ import annotations

from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


CompetitionSourceType = Literal[
    "excel",
    "word",
    "pdf",
]

CompetitionDifficulty = Literal[
    "easy",
    "medium",
    "hard",
]

CompetitionDifficultyCn = Literal[
    "简单",
    "中等",
    "困难",
]

CompetitionQaType = Literal[
    "表格取数",
    "表格比较",
    "表格计算",
    "单事实检索",
    "多事实检索",
]

CompetitionAnswerOption = Literal[
    "A",
    "B",
    "C",
    "D",
]


class CompetitionQaCase(BaseModel):
    """比赛原始 QA Excel 中的一条题目记录。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    case_id: str = Field(
        pattern=r"^Q[0-9]{3}$",
    )

    source_type: CompetitionSourceType

    difficulty: CompetitionDifficulty

    difficulty_cn: CompetitionDifficultyCn

    qa_type: CompetitionQaType

    question: str = Field(
        min_length=1,
        max_length=20_000,
    )

    option_a: str = Field(min_length=1)
    option_b: str = Field(min_length=1)
    option_c: str = Field(min_length=1)
    option_d: str = Field(min_length=1)

    answer: CompetitionAnswerOption

    answer_text: str = Field(
        min_length=1,
        max_length=20_000,
    )

    evidence: str = Field(
        min_length=1,
        max_length=50_000,
    )

    source_title: str = Field(
        min_length=1,
        max_length=1000,
    )

    file_label: str = Field(
        min_length=1,
        max_length=1000,
    )

    @property
    def options(
        self,
    ) -> dict[str, str]:
        return {
            "A": self.option_a,
            "B": self.option_b,
            "C": self.option_c,
            "D": self.option_d,
        }

    @model_validator(mode="after")
    def validate_source_and_qa_type(
        self,
    ) -> Self:
        excel_types = {
            "表格取数",
            "表格比较",
            "表格计算",
        }

        text_types = {
            "单事实检索",
            "多事实检索",
        }

        if (
            self.source_type == "excel"
            and self.qa_type not in excel_types
        ):
            raise ValueError(
                "excel 题目只能是表格取数、"
                "表格比较或表格计算"
            )

        if (
            self.source_type in {"word", "pdf"}
            and self.qa_type not in text_types
        ):
            raise ValueError(
                "word/pdf 题目只能是"
                "单事实检索或多事实检索"
            )

        return self


class CompetitionQuestion(BaseModel):
    """
    真正允许进入 Solver 的比赛问题。

    注意：
    这里禁止出现：
    - answer
    - answer_text
    - evidence
    - difficulty

    Solver 只能看到完成任务所需的信息。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    case_id: str = Field(
        pattern=r"^Q[0-9]{3}$",
    )

    source_type: CompetitionSourceType

    qa_type: CompetitionQaType

    question: str = Field(
        min_length=1,
        max_length=20_000,
    )

    option_a: str = Field(min_length=1)
    option_b: str = Field(min_length=1)
    option_c: str = Field(min_length=1)
    option_d: str = Field(min_length=1)

    source_title: str = Field(
        min_length=1,
        max_length=1000,
    )

    file_label: str = Field(
        min_length=1,
        max_length=1000,
    )

    @property
    def options(
        self,
    ) -> dict[str, str]:
        return {
            "A": self.option_a,
            "B": self.option_b,
            "C": self.option_c,
            "D": self.option_d,
        }


class CompetitionGold(BaseModel):
    """
    只允许 Evaluation 使用的 Gold 数据。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    case_id: str = Field(
        pattern=r"^Q[0-9]{3}$",
    )

    source_type: CompetitionSourceType

    qa_type: CompetitionQaType

    difficulty: CompetitionDifficulty

    difficulty_cn: CompetitionDifficultyCn

    answer: CompetitionAnswerOption

    answer_text: str = Field(
        min_length=1,
        max_length=20_000,
    )

    evidence: str = Field(
        min_length=1,
        max_length=50_000,
    )




class CompetitionSourceRecord(BaseModel):
    """本地比赛附件 Manifest 中的一份源文件。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    source_id: str = Field(
        pattern=r"^src_[0-9a-f]{16}$",
    )

    source_type: CompetitionSourceType

    actual_filename: str = Field(
        min_length=1,
        max_length=2000,
    )

    relative_path: str = Field(
        min_length=1,
        max_length=4000,
    )

    extension: str = Field(
        min_length=2,
        max_length=16,
    )

    size_bytes: int = Field(
        ge=0,
    )


CompetitionResolutionStrategy = Literal[
    "exact_tail",
    "file_label",
    "normalized_file_label",
    "title_and_label",
    "unique_title",
]


class CompetitionSourceResolution(BaseModel):
    """一条 QA 与本地附件之间的确定性映射结果。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    case_id: str = Field(
        pattern=r"^Q[0-9]{3}$",
    )

    source_id: str = Field(
        pattern=r"^src_[0-9a-f]{16}$",
    )

    source_type: CompetitionSourceType

    relative_path: str = Field(
        min_length=1,
    )

    strategy: CompetitionResolutionStrategy


class CompetitionSolverInput(BaseModel):
    """
    Solver 真正接收到的完整输入。

    Question:
        用户问题与选项。

    Source:
        已经过 SourceResolver 唯一定位的附件。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    question: CompetitionQuestion

    source: CompetitionSourceResolution