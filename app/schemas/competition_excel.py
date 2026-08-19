from __future__ import annotations

from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


CompetitionExcelFormat = Literal[
    "xls",
    "xlsx",
]


CompetitionExcelCellType = Literal[
    "text",
    "number",
    "boolean",
    "date",
    "formula",
    "error",
    "blank",
]


class CompetitionExcelMergedRange(
    BaseModel
):
    """
    一个 Excel 合并区域。

    例如：
        A1:C1

    anchor_coordinate:
        A1

    合并区域中的语义值通常只存储在
    左上角 anchor cell。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    range_ref: str = Field(
        min_length=3,
        max_length=64,
    )

    min_row: int = Field(
        ge=1,
    )

    max_row: int = Field(
        ge=1,
    )

    min_column: int = Field(
        ge=1,
    )

    max_column: int = Field(
        ge=1,
    )

    anchor_coordinate: str = Field(
        min_length=2,
        max_length=32,
    )

    @model_validator(mode="after")
    def validate_bounds(
        self,
    ) -> Self:
        if self.min_row > self.max_row:
            raise ValueError(
                "min_row 不能大于 max_row"
            )

        if (
            self.min_column
            > self.max_column
        ):
            raise ValueError(
                "min_column 不能大于 "
                "max_column"
            )

        return self


class CompetitionExcelCell(
    BaseModel
):
    """
    稀疏 Excel Cell。

    只为真正有意义的单元格建立对象，
    不根据 worksheet.max_row /
    max_column 构造完整二维矩阵。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    coordinate: str = Field(
        min_length=2,
        max_length=32,
    )

    row_index: int = Field(
        ge=1,
    )

    column_index: int = Field(
        ge=1,
    )

    cell_type: CompetitionExcelCellType

    # --------------------------------------------------------
    # text_value
    #
    # 统一给 Retrieval / Header Matching 使用。
    #
    # 无论原 Cell 是：
    # 中文文本、数字、日期，
    # 都可以转换成稳定字符串。
    # --------------------------------------------------------

    text_value: str = Field(
        default="",
        max_length=50_000,
    )

    # --------------------------------------------------------
    # numeric_value
    #
    # 真正参与比较 / 计算时使用。
    #
    # 例如：
    # 120.5
    # --------------------------------------------------------

    numeric_value: int | float | None = None

    # --------------------------------------------------------
    # number_format
    #
    # Excel 原始数字格式。
    #
    # 非常重要：
    # 0.123 在 Excel 中可能显示为 12.3%。
    #
    # 后续 Unit / Percentage Resolver 会使用它。
    # --------------------------------------------------------

    number_format: str | None = Field(
        default=None,
        max_length=1000,
    )

    # --------------------------------------------------------
    # formula
    #
    # XLSX:
    # 可保存 "=SUM(C1:C5)"
    #
    # XLS:
    # xlrd 通常只能取得公式计算后的值，
    # 所以这里允许为 None。
    # --------------------------------------------------------

    formula: str | None = Field(
        default=None,
        max_length=20_000,
    )

    # --------------------------------------------------------
    # formula_cached_value
    #
    # XLSX 用 data_only=True 再加载工作簿后，
    # 可以尝试取得 Excel 保存的公式计算结果。
    # --------------------------------------------------------

    formula_cached_value: (
        int | float | str | bool | None
    ) = None

    # --------------------------------------------------------
    # merged_range
    #
    # 如果当前 Cell 是某个 merged region 的
    # anchor，则记录其范围。
    #
    # 非 anchor Cell 不建立 Cell Object，
    # 所以不会重复保存整个 merged region。
    # --------------------------------------------------------

    merged_range: str | None = Field(
        default=None,
        max_length=64,
    )

    @model_validator(mode="after")
    def validate_formula_contract(
        self,
    ) -> Self:
        if (
            self.formula is not None
            and self.cell_type
            != "formula"
        ):
            raise ValueError(
                "存在 formula 时 "
                "cell_type 必须为 formula"
            )

        if (
            self.cell_type == "formula"
            and not self.formula
        ):
            raise ValueError(
                "formula Cell 必须提供 "
                "formula"
            )

        return self


class CompetitionExcelSheet(
    BaseModel
):
    """
    一个 Sheet 的稀疏结构表示。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    name: str = Field(
        min_length=1,
        max_length=1000,
    )

    sheet_index: int = Field(
        ge=0,
    )

    visible: bool = True

    # Excel 声明出的物理范围。
    #
    # 只用于诊断，
    # Parser 不应该据此创建整个矩阵。
    declared_max_row: int = Field(
        ge=0,
    )

    declared_max_column: int = Field(
        ge=0,
    )

    cells: tuple[
        CompetitionExcelCell,
        ...
    ] = ()

    merged_ranges: tuple[
        CompetitionExcelMergedRange,
        ...
    ] = ()

    @property
    def non_empty_cell_count(
        self,
    ) -> int:
        return len(self.cells)


class CompetitionExcelWorkbook(
    BaseModel
):
    """
    Competition Excel Parser 的统一输出。

    Solver 不需要知道底层究竟来自
    openpyxl 还是 xlrd。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    source_id: str = Field(
        pattern=r"^src_[0-9a-f]{16}$",
    )

    relative_path: str = Field(
        min_length=1,
        max_length=4000,
    )

    excel_format: CompetitionExcelFormat

    sheets: tuple[
        CompetitionExcelSheet,
        ...
    ]

    @property
    def sheet_names(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            sheet.name
            for sheet in self.sheets
        )

    @property
    def total_cell_count(
        self,
    ) -> int:
        return sum(
            sheet.non_empty_cell_count
            for sheet in self.sheets
        )