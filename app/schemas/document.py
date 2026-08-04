from datetime import datetime
from pathlib import PurePosixPath, PureWindowsPath
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .enums import (
    DocumentValidationStatus,
    PageCountStatus,
)


class DocumentManifest(BaseModel):
    """某个实际 PDF 文件版本的不可变身份记录。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    schema_version: int = Field(
        default=1,
        ge=1,
        le=1,
        description="DocumentManifest 数据结构版本",
    )

    document_id: str = Field(
        min_length=1,
        pattern=(
            r"^doc_[a-z0-9_]+_[0-9a-f]{24}$"
        ),
        description=(
            "实际 PDF 文件版本 ID，由 report_id "
            "和 SHA-256 前 24 位确定"
        ),
    )

    report_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
        description="该实际文件对应的业务报告 ID",
    )

    source_path: str = Field(
        min_length=1,
        description=(
            "相对于项目根目录的 POSIX 路径，"
            "不能保存本机绝对路径"
        ),
    )

    source_filename: str = Field(
        min_length=1,
        description="原始 PDF 文件名",
    )

    sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="实际文件完整 SHA-256",
    )

    file_size_bytes: int = Field(
        gt=0,
        description="实际文件大小，单位为字节",
    )

    pdf_page_count: int = Field(
        ge=1,
        description="解析器读取到的实际 PDF 总页数",
    )

    expected_pdf_page_count: int = Field(
        ge=1,
        description="Report Registry 中人工核验的预期页数",
    )

    page_count_status: PageCountStatus

    parser_name: str = Field(
        min_length=1,
        description="用于检查 PDF 的解析器名称",
    )

    parser_version: str = Field(
        min_length=1,
        description="用于检查 PDF 的解析器版本",
    )

    validation_status: DocumentValidationStatus

    created_at: datetime

    @field_validator("source_path")
    @classmethod
    def validate_source_path(cls, value: str) -> str:
        """来源路径必须是安全、可迁移的相对 POSIX 路径。"""

        if "\\" in value:
            raise ValueError(
                "source_path 必须使用正斜杠"
            )

        posix_path = PurePosixPath(value)
        windows_path = PureWindowsPath(value)

        if (
            posix_path.is_absolute()
            or windows_path.is_absolute()
        ):
            raise ValueError(
                "source_path 不能保存绝对路径"
            )

        if ".." in posix_path.parts:
            raise ValueError(
                "source_path 不能包含 '..'"
            )

        return value

    @field_validator("created_at")
    @classmethod
    def validate_timezone(
        cls,
        value: datetime,
    ) -> datetime:
        """创建时间必须包含时区。"""

        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "datetime 必须包含时区信息"
            )

        return value

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        """检查文档 ID、文件名和页数状态。"""

        expected_document_id = (
            f"doc_{self.report_id}_"
            f"{self.sha256[:24]}"
        )

        if self.document_id != expected_document_id:
            raise ValueError(
                "document_id 必须由 report_id "
                "和 SHA-256 前 24 位生成"
            )

        if (
            PurePosixPath(self.source_path).name
            != self.source_filename
        ):
            raise ValueError(
                "source_filename 必须与 "
                "source_path 中的文件名一致"
            )

        page_count_matches = (
            self.pdf_page_count
            == self.expected_pdf_page_count
        )

        if (
            self.page_count_status
            is PageCountStatus.MATCHED
            and not page_count_matches
        ):
            raise ValueError(
                "page_count_status 为 matched 时，"
                "实际页数必须等于预期页数"
            )

        if (
            self.page_count_status
            is PageCountStatus.MISMATCHED
            and page_count_matches
        ):
            raise ValueError(
                "page_count_status 为 mismatched 时，"
                "实际页数不能等于预期页数"
            )

        if (
            self.validation_status
            is DocumentValidationStatus.VALID
            and self.page_count_status
            is not PageCountStatus.MATCHED
        ):
            raise ValueError(
                "valid 文档的页数检查必须通过"
            )

        if (
            self.validation_status
            is DocumentValidationStatus.BLOCKED
            and self.page_count_status
            is PageCountStatus.MATCHED
        ):
            raise ValueError(
                "当前版本中，页数检查通过的文档"
                "不能标记为 blocked"
            )

        return self