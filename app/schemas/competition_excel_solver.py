from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.schemas.competition import (
    CompetitionAnswerOption,
)


class CompetitionExcelEvidence(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    sheet_name: str = Field(
        min_length=1,
    )

    value_coordinate: str = Field(
        min_length=2,
    )

    value_text: str

    numeric_value: (
        int | float | None
    ) = None

    number_format: str | None = Field(
        default=None,
        max_length=1000,
    )

    row_label_coordinate: (
        str | None
    ) = None

    row_label: (
        str | None
    ) = None

    column_label_coordinate: (
        str | None
    ) = None

    column_label: (
        str | None
    ) = None


class CompetitionExcelLookupResult(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    answer_option: (
        CompetitionAnswerOption
        | None
    ) = None

    answer_text: str

    evidence: CompetitionExcelEvidence

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

CompetitionExcelCompareOperation = Literal[
    "max",
    "min",
]


class CompetitionExcelCompareItem(
    BaseModel
):
    """
    一个比较选项及其实际 Excel 数值证据。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    option: CompetitionAnswerOption

    label_text: str = Field(
        min_length=1,
    )

    label_coordinate: str = Field(
        min_length=2,
    )

    value_coordinate: str = Field(
        min_length=2,
    )

    value_text: str

    numeric_value: int | float

    context_labels: tuple[
        str,
        ...
    ] = ()


class CompetitionExcelCompareResult(
    BaseModel
):
    """
    表格比较 Solver 的输出。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    operation: (
        CompetitionExcelCompareOperation
    )

    answer_option: CompetitionAnswerOption

    answer_text: str = Field(
        min_length=1,
    )

    winning_value: int | float

    sheet_name: str = Field(
        min_length=1,
    )

    items: tuple[
        CompetitionExcelCompareItem,
        ...
    ] = Field(
        min_length=2,
        max_length=4,
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

CompetitionExcelCalculationOperation = Literal[
    "difference",
]

CompetitionExcelCalculationResolutionMode = Literal[
    "semantic_headers",
    "option_constrained_fallback",
]


class CompetitionExcelCalculationOperand(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    role: Literal[
        "start",
        "end",
    ]

    scope_text: str = Field(
        min_length=1,
    )

    coordinate: str = Field(
        min_length=2,
    )

    numeric_value: int | float

    header_labels: tuple[
        str,
        ...
    ] = ()


class CompetitionExcelCalculationResult(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    sheet_name: str = Field(
        min_length=1,
    )

    operation: (
        CompetitionExcelCalculationOperation
    )

    entity_text: str = Field(
        min_length=1,
    )

    entity_coordinate: str = Field(
        min_length=2,
    )

    start: CompetitionExcelCalculationOperand

    end: CompetitionExcelCalculationOperand

    formula: str = Field(
        min_length=1,
    )

    result: int | float

    answer_option: CompetitionAnswerOption

    answer_text: str = Field(
        min_length=1,
    )

    resolution_mode: (
        CompetitionExcelCalculationResolutionMode
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )