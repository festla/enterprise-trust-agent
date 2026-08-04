from datetime import date, datetime, timezone
from pathlib import Path

import pymupdf
import pytest

from app.schemas.report import Report
from app.services import document_ingestion
from app.services.document_ingestion import (
    DocumentFileNotFoundError,
    DocumentPathOutsideProjectError,
    EncryptedPdfError,
    PdfOpenError,
    register_pdf_document,
)


def build_report(
    *,
    expected_pdf_page_count: int = 1,
) -> Report:
    """创建用于文档接入测试的 Report。"""

    now = datetime.now(timezone.utc)

    return Report(
        report_id="midea_group_2024",
        company_id="midea_group",
        fiscal_year=2024,
        report_type="annual_report",
        title="美的集团：2024年年度报告",
        publication_date=date(2025, 3, 29),
        source_name="test",
        quality_grade="A",
        citation_risk="low",
        expected_pdf_page_count=(
            expected_pdf_page_count
        ),
        status="active",
        created_at=now,
        updated_at=now,
    )


def create_test_pdf(
    path: Path,
    *,
    page_count: int,
) -> None:
    """动态创建最小测试 PDF。"""

    document = pymupdf.open()

    try:
        for page_number in range(1, page_count + 1):
            page = document.new_page()
            page.insert_text(
                (72, 72),
                f"Test page {page_number}",
            )

        document.save(path)
    finally:
        document.close()


def test_register_valid_pdf(
    tmp_path: Path,
) -> None:
    """正常 PDF 应生成有效 Manifest。"""

    project_root = tmp_path / "project"
    pdf_path = (
        project_root
        / "data"
        / "raw_reports"
        / "midea"
        / "2024"
        / "report.pdf"
    )
    output_root = (
        project_root
        / "data"
        / "processed"
        / "documents"
    )

    pdf_path.parent.mkdir(parents=True)
    create_test_pdf(pdf_path, page_count=1)

    result = register_pdf_document(
        report=build_report(),
        pdf_path=pdf_path,
        project_root=project_root,
        output_root=output_root,
    )

    assert result.created is True
    assert result.manifest.pdf_page_count == 1
    assert result.manifest.page_count_status.value == (
        "matched"
    )
    assert result.manifest.validation_status.value == (
        "valid"
    )
    assert result.manifest_path.exists()

    assert result.manifest.source_path == (
        "data/raw_reports/midea/2024/report.pdf"
    )


def test_repeated_registration_is_idempotent(
    tmp_path: Path,
) -> None:
    """同一 PDF 重复登记不应创建重复记录。"""

    project_root = tmp_path / "project"
    project_root.mkdir()

    pdf_path = project_root / "report.pdf"
    output_root = project_root / "documents"

    create_test_pdf(pdf_path, page_count=1)

    first = register_pdf_document(
        report=build_report(),
        pdf_path=pdf_path,
        project_root=project_root,
        output_root=output_root,
    )

    second = register_pdf_document(
        report=build_report(),
        pdf_path=pdf_path,
        project_root=project_root,
        output_root=output_root,
    )

    assert first.created is True
    assert second.created is False

    assert (
        first.manifest.document_id
        == second.manifest.document_id
    )

    assert len(list(output_root.rglob("*.json"))) == 1


def test_same_filename_changed_content_creates_new_version(
    tmp_path: Path,
) -> None:
    """同名文件内容变化后应生成新的 document_id。"""

    project_root = tmp_path / "project"
    project_root.mkdir()

    pdf_path = project_root / "report.pdf"
    output_root = project_root / "documents"

    create_test_pdf(pdf_path, page_count=1)

    first = register_pdf_document(
        report=build_report(
            expected_pdf_page_count=1
        ),
        pdf_path=pdf_path,
        project_root=project_root,
        output_root=output_root,
    )

    create_test_pdf(pdf_path, page_count=2)

    second = register_pdf_document(
        report=build_report(
            expected_pdf_page_count=1
        ),
        pdf_path=pdf_path,
        project_root=project_root,
        output_root=output_root,
    )

    assert (
        first.manifest.document_id
        != second.manifest.document_id
    )

    assert first.manifest.sha256 != second.manifest.sha256

    assert second.manifest.validation_status.value == (
        "blocked"
    )

    assert len(list(output_root.rglob("*.json"))) == 2


def test_page_count_mismatch_is_blocked(
    tmp_path: Path,
) -> None:
    """实际页数不一致时应保留记录并阻止后续解析。"""

    project_root = tmp_path / "project"
    project_root.mkdir()

    pdf_path = project_root / "report.pdf"
    output_root = project_root / "documents"

    create_test_pdf(pdf_path, page_count=1)

    result = register_pdf_document(
        report=build_report(
            expected_pdf_page_count=2
        ),
        pdf_path=pdf_path,
        project_root=project_root,
        output_root=output_root,
    )

    assert result.manifest.pdf_page_count == 1

    assert result.manifest.page_count_status.value == (
        "mismatched"
    )

    assert result.manifest.validation_status.value == (
        "blocked"
    )


def test_reject_missing_pdf(
    tmp_path: Path,
) -> None:
    """文件不存在时应抛出明确异常。"""

    with pytest.raises(
        DocumentFileNotFoundError
    ):
        register_pdf_document(
            report=build_report(),
            pdf_path=Path("missing.pdf"),
            project_root=tmp_path,
            output_root=tmp_path / "documents",
        )


def test_reject_corrupted_pdf(
    tmp_path: Path,
) -> None:
    """损坏 PDF 不得生成 Manifest。"""

    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"%PDF-broken")

    with pytest.raises(PdfOpenError):
        register_pdf_document(
            report=build_report(),
            pdf_path=pdf_path,
            project_root=tmp_path,
            output_root=tmp_path / "documents",
        )


def test_reject_path_outside_project(
    tmp_path: Path,
) -> None:
    """项目外部的文件不能被登记。"""

    project_root = tmp_path / "project"
    project_root.mkdir()

    outside_pdf = tmp_path / "outside.pdf"
    create_test_pdf(outside_pdf, page_count=1)

    with pytest.raises(
        DocumentPathOutsideProjectError
    ):
        register_pdf_document(
            report=build_report(),
            pdf_path=outside_pdf,
            project_root=project_root,
            output_root=project_root / "documents",
        )


def test_reject_encrypted_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """需要密码的 PDF 应明确拒绝。"""

    pdf_path = tmp_path / "encrypted.pdf"
    pdf_path.write_bytes(b"placeholder")

    class FakeEncryptedDocument:
        is_pdf = True
        needs_pass = True
        page_count = 1

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        document_ingestion.pymupdf,
        "open",
        lambda path: FakeEncryptedDocument(),
    )

    with pytest.raises(EncryptedPdfError):
        register_pdf_document(
            report=build_report(),
            pdf_path=pdf_path,
            project_root=tmp_path,
            output_root=tmp_path / "documents",
        )