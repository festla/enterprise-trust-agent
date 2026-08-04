from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import (
    PackageNotFoundError,
    version,
)
from pathlib import Path

import pymupdf
from pydantic import ValidationError

from app.schemas.document import DocumentManifest
from app.schemas.enums import (
    DocumentValidationStatus,
    PageCountStatus,
)
from app.schemas.report import Report


HASH_READ_SIZE = 1024 * 1024


class DocumentIngestionError(ValueError):
    """文档接入过程的基础异常。"""


class DocumentFileNotFoundError(
    DocumentIngestionError
):
    """实际文档文件不存在。"""


class DocumentPathOutsideProjectError(
    DocumentIngestionError
):
    """实际文档不在项目根目录内。"""


class PdfOpenError(DocumentIngestionError):
    """实际文件不能作为有效 PDF 打开。"""


class EncryptedPdfError(DocumentIngestionError):
    """PDF 需要密码，当前管线不进行解密。"""


class MissingExpectedPageCountError(
    DocumentIngestionError
):
    """Report 没有配置预期 PDF 页数。"""


class DocumentManifestConflictError(
    DocumentIngestionError
):
    """已有 Manifest 与当前文件身份冲突。"""


@dataclass(frozen=True, slots=True)
class DocumentRegistrationResult:
    """一次文档登记的返回结果。"""

    manifest: DocumentManifest
    manifest_path: Path
    created: bool


def _resolve_source_path(
    *,
    pdf_path: Path,
    project_root: Path,
) -> tuple[Path, str]:
    """取得真实文件路径和项目相对路径。"""

    resolved_project_root = project_root.resolve()

    candidate_path = pdf_path

    if not candidate_path.is_absolute():
        candidate_path = (
            resolved_project_root / candidate_path
        )

    try:
        resolved_pdf_path = candidate_path.resolve(
            strict=True
        )
    except FileNotFoundError as exc:
        raise DocumentFileNotFoundError(
            f"PDF 文件不存在：{candidate_path}"
        ) from exc

    if not resolved_pdf_path.is_file():
        raise DocumentFileNotFoundError(
            f"目标不是普通文件：{resolved_pdf_path}"
        )

    try:
        relative_path = resolved_pdf_path.relative_to(
            resolved_project_root
        )
    except ValueError as exc:
        raise DocumentPathOutsideProjectError(
            "PDF 文件必须位于项目根目录内部："
            f"{resolved_pdf_path}"
        ) from exc

    return (
        resolved_pdf_path,
        relative_path.as_posix(),
    )


def calculate_file_sha256(path: Path) -> str:
    """分块计算完整文件 SHA-256。"""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(HASH_READ_SIZE)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def _get_parser_version() -> str:
    """读取当前安装的 PyMuPDF 包版本。"""

    try:
        return version("PyMuPDF")
    except PackageNotFoundError as exc:
        raise DocumentIngestionError(
            "无法读取 PyMuPDF 包版本"
        ) from exc


def _inspect_pdf(path: Path) -> int:
    """打开 PDF，并返回实际总页数。"""

    try:
        document = pymupdf.open(path)
    except Exception as exc:
        raise PdfOpenError(
            f"PDF 无法打开或文件已损坏：{path}"
        ) from exc

    try:
        if not document.is_pdf:
            raise PdfOpenError(
                f"文件不是有效 PDF：{path}"
            )

        if document.needs_pass:
            raise EncryptedPdfError(
                f"PDF 需要密码，当前不处理：{path}"
            )

        page_count = document.page_count

        if page_count < 1:
            raise PdfOpenError(
                f"PDF 页数必须大于 0：{path}"
            )

        return page_count
    finally:
        document.close()


def _build_document_id(
    *,
    report_id: str,
    sha256: str,
) -> str:
    """由报告 ID 和文件内容生成稳定文档 ID。"""

    return f"doc_{report_id}_{sha256[:24]}"


def _load_existing_manifest(
    path: Path,
) -> DocumentManifest:
    """读取并校验已有 Manifest。"""

    try:
        content = path.read_text(
            encoding="utf-8"
        )
    except OSError as exc:
        raise DocumentManifestConflictError(
            f"已有 Manifest 无法读取：{path}"
        ) from exc

    try:
        return DocumentManifest.model_validate_json(
            content
        )
    except ValidationError as exc:
        raise DocumentManifestConflictError(
            f"已有 Manifest 结构无效：{path}"
        ) from exc


def _write_manifest_once(
    *,
    manifest: DocumentManifest,
    manifest_path: Path,
) -> tuple[bool, DocumentManifest]:
    """只创建一次 Manifest，禁止覆盖已有文件。"""

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    content = (
        manifest.model_dump_json(indent=2)
        + "\n"
    )

    try:
        with manifest_path.open(
            "x",
            encoding="utf-8",
            newline="\n",
        ) as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())

        return True, manifest

    except FileExistsError:
        existing_manifest = (
            _load_existing_manifest(manifest_path)
        )

        same_identity = (
            existing_manifest.document_id
            == manifest.document_id
            and existing_manifest.report_id
            == manifest.report_id
            and existing_manifest.sha256
            == manifest.sha256
        )

        if not same_identity:
            raise DocumentManifestConflictError(
                "相同 Manifest 路径对应了不同文件身份："
                f"{manifest_path}"
            )

        return False, existing_manifest


def register_pdf_document(
    *,
    report: Report,
    pdf_path: Path,
    project_root: Path,
    output_root: Path,
    created_at: datetime | None = None,
) -> DocumentRegistrationResult:
    """检查并登记某个 Report 对应的实际 PDF。"""

    if report.expected_pdf_page_count is None:
        raise MissingExpectedPageCountError(
            f"Report '{report.report_id}' "
            "没有配置 expected_pdf_page_count"
        )

    (
        resolved_pdf_path,
        relative_source_path,
    ) = _resolve_source_path(
        pdf_path=pdf_path,
        project_root=project_root,
    )

    file_size_bytes = (
        resolved_pdf_path.stat().st_size
    )

    if file_size_bytes <= 0:
        raise PdfOpenError(
            f"PDF 文件不能为空：{resolved_pdf_path}"
        )

    file_sha256 = calculate_file_sha256(
        resolved_pdf_path
    )

    pdf_page_count = _inspect_pdf(
        resolved_pdf_path
    )

    page_count_matches = (
        pdf_page_count
        == report.expected_pdf_page_count
    )

    page_count_status = (
        PageCountStatus.MATCHED
        if page_count_matches
        else PageCountStatus.MISMATCHED
    )

    validation_status = (
        DocumentValidationStatus.VALID
        if page_count_matches
        else DocumentValidationStatus.BLOCKED
    )

    document_id = _build_document_id(
        report_id=report.report_id,
        sha256=file_sha256,
    )

    manifest = DocumentManifest(
        document_id=document_id,
        report_id=report.report_id,
        source_path=relative_source_path,
        source_filename=resolved_pdf_path.name,
        sha256=file_sha256,
        file_size_bytes=file_size_bytes,
        pdf_page_count=pdf_page_count,
        expected_pdf_page_count=(
            report.expected_pdf_page_count
        ),
        page_count_status=page_count_status,
        parser_name="pymupdf",
        parser_version=_get_parser_version(),
        validation_status=validation_status,
        created_at=(
            created_at
            if created_at is not None
            else datetime.now(timezone.utc)
        ),
    )

    manifest_path = (
        output_root
        / report.report_id
        / f"{document_id}.json"
    )

    created, stored_manifest = (
        _write_manifest_once(
            manifest=manifest,
            manifest_path=manifest_path,
        )
    )

    return DocumentRegistrationResult(
        manifest=stored_manifest,
        manifest_path=manifest_path,
        created=created,
    )