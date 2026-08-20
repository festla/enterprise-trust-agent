from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


CompetitionKnowledgeSourceType = Literal[
    "word",
    "pdf",
    "excel",
    "web",
]

CompetitionEvidenceType = Literal[
    "text",
    "table_cell",
    "table_range",
    "calculation",
]


class CompetitionKnowledgeSource(
    BaseModel
):
    """
    统一知识来源元数据。

    对应赛题要求中的：
    doc_id、标题、发文机关、发布日期、
    文件类型、来源 URL 等字段。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    # 项目内部稳定 ID。
    source_id: str = Field(
        min_length=1,
    )

    # 赛题要求的文档级 ID。
    #
    # 当前 Excel 阶段可以暂时与 source_id 一致，
    # 后续接 manifest 时再统一生成规则。
    doc_id: str = Field(
        min_length=1,
    )

    title: str = Field(
        min_length=1,
    )

    source_type: (
        CompetitionKnowledgeSourceType
    )

    relative_path: str = Field(
        min_length=1,
    )

    source_url: str | None = None

    issuing_authority: str | None = None

    published_date: date | None = None

    sha256: str | None = None


class CompetitionTextEvidenceLocation(
    BaseModel
):
    """
    Word / PDF 文本证据位置。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    location_type: Literal[
        "text"
    ] = "text"

    page: int | None = Field(
        default=None,
        ge=1,
    )

    section: str | None = None

    article: str | None = None

    paragraph_index: int | None = Field(
        default=None,
        ge=0,
    )


class CompetitionExcelEvidenceLocation(
    BaseModel
):
    """
    Excel 表格证据位置。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    location_type: Literal[
        "excel"
    ] = "excel"

    sheet_name: str = Field(
        min_length=1,
    )

    cell: str | None = None

    cell_range: str | None = None

    row_label: str | None = None

    column_label: str | None = None

    unit: str | None = None

    @model_validator(
        mode="after"
    )
    def validate_cell_or_range(
        self,
    ):
        if (
            self.cell is None
            and self.cell_range is None
        ):
            raise ValueError(
                "Excel evidence 必须提供 "
                "cell 或 cell_range"
            )

        return self


CompetitionEvidenceLocation = (
    CompetitionTextEvidenceLocation
    | CompetitionExcelEvidenceLocation
)


class CompetitionEvidence(
    BaseModel
):
    """
    一条可独立核验的最小证据。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    evidence_id: str = Field(
        min_length=1,
    )

    evidence_type: (
        CompetitionEvidenceType
    )

    source: CompetitionKnowledgeSource

    location: CompetitionEvidenceLocation

    # 原始证据内容。
    #
    # Word/PDF:
    #   原文段落或条款。
    #
    # Excel:
    #   指标/值的可读表达。
    raw_content: str = Field(
        min_length=1,
    )

    # 对结构化表格值可选保存。
    numeric_value: int | float | None = None

    # 原始显示值，例如：
    # "259039.99"
    display_value: str | None = None


class CompetitionCalculationTrace(
    BaseModel
):
    """
    确定性计算的审计记录。

    不把计算结果伪装成原始文档证据，
    而是明确记录：
    哪些 Evidence 作为输入，
    使用什么公式，
    得到了什么结果。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    operation: str = Field(
        min_length=1,
    )

    formula: str = Field(
        min_length=1,
    )

    input_evidence_ids: tuple[
        str,
        ...
    ] = Field(
        min_length=1,
    )

    result: int | float

    resolution_mode: str | None = None


class CompetitionEvidenceBundle(
    BaseModel
):
    """
    Solver / Retriever 最终交给 Answer Generator
    的统一证据包。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    evidences: tuple[
        CompetitionEvidence,
        ...
    ] = Field(
        min_length=1,
    )

    calculation: (
        CompetitionCalculationTrace
        | None
    ) = None

    @model_validator(
        mode="after"
    )
    def validate_calculation_inputs(
        self,
    ):
        if self.calculation is None:
            return self

        evidence_ids = {
            evidence.evidence_id
            for evidence in self.evidences
        }

        missing = [
            evidence_id
            for evidence_id
            in (
                self.calculation
                .input_evidence_ids
            )
            if evidence_id
            not in evidence_ids
        ]

        if missing:
            raise ValueError(
                "Calculation 引用了"
                "不存在的 Evidence: "
                f"{missing}"
            )

        return self