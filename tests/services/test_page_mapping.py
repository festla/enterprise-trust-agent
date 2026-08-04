from datetime import datetime, timezone

import pytest

from app.schemas.document import DocumentManifest
from app.schemas.report import PageMappingSegment
from app.services.page_mapping import (
    BlockedDocumentError,
    InvalidPageMappingConfigurationError,
    InvalidPdfPageError,
    MissingPageMappingError,
    OverlappingPageMappingError,
    PageMappingResolver,
)


def build_manifest(
    *,
    report_id: str,
    page_count: int,
    valid: bool = True,
) -> DocumentManifest:
    """创建测试用 DocumentManifest。"""

    sha256 = "a" * 64

    return DocumentManifest(
        document_id=(
            f"doc_{report_id}_{sha256[:24]}"
        ),
        report_id=report_id,
        source_path=f"data/{report_id}.pdf",
        source_filename=f"{report_id}.pdf",
        sha256=sha256,
        file_size_bytes=100,
        pdf_page_count=page_count,
        expected_pdf_page_count=(
            page_count
            if valid
            else page_count + 1
        ),
        page_count_status=(
            "matched"
            if valid
            else "mismatched"
        ),
        parser_name="pymupdf",
        parser_version="1.28.0",
        validation_status=(
            "valid"
            if valid
            else "blocked"
        ),
        created_at=datetime.now(timezone.utc),
    )


def build_midea_mapping() -> PageMappingSegment:
    """美的 2024 固定 offset 映射。"""

    return PageMappingSegment(
        mapping_id="midea_group_2024_main",
        report_id="midea_group_2024",
        printed_page_start=1,
        printed_page_end=294,
        pdf_page_start=2,
        pdf_page_end=295,
        offset=1,
        rule_type="offset",
        notes="PDF 第 1 页为封面",
        validation_status="verified",
    )


def build_hisense_mappings(
) -> list[PageMappingSegment]:
    """海信 2024 分段页码映射。"""

    return [
        PageMappingSegment(
            mapping_id=(
                "hisense_home_2024_before_duplicate_33"
            ),
            report_id="hisense_home_2024",
            printed_page_start=1,
            printed_page_end=33,
            pdf_page_start=2,
            pdf_page_end=34,
            offset=1,
            rule_type="offset",
            notes="重复页码前",
            validation_status="verified",
        ),
        PageMappingSegment(
            mapping_id=(
                "hisense_home_2024_duplicate_33"
            ),
            report_id="hisense_home_2024",
            printed_page_start=33,
            printed_page_end=33,
            pdf_page_start=35,
            pdf_page_end=35,
            rule_type="custom",
            notes="第二个印刷页码 33",
            validation_status="verified",
        ),
        PageMappingSegment(
            mapping_id=(
                "hisense_home_2024_after_duplicate_33"
            ),
            report_id="hisense_home_2024",
            printed_page_start=34,
            printed_page_end=241,
            pdf_page_start=36,
            pdf_page_end=243,
            offset=2,
            rule_type="offset",
            notes="重复页码后",
            validation_status="verified",
        ),
    ]


def test_resolve_midea_fixed_offset() -> None:
    """美的固定偏移页码应正确映射。"""

    resolver = PageMappingResolver(
        manifest=build_manifest(
            report_id="midea_group_2024",
            page_count=295,
        ),
        page_mappings=[
            build_midea_mapping()
        ],
    )

    cover = resolver.resolve(1)
    first_printed = resolver.resolve(2)
    evidence_page = resolver.resolve(158)
    last_page = resolver.resolve(295)

    assert cover.mapping_status.value == "unmapped"
    assert cover.printed_page is None

    assert first_printed.printed_page == 1
    assert evidence_page.printed_page == 157
    assert last_page.printed_page == 294

    assert evidence_page.page_id.endswith(
        "_page_0158"
    )


def test_resolve_hisense_duplicate_printed_page() -> None:
    """海信两个 PDF 页面应允许对应同一印刷页 33。"""

    resolver = PageMappingResolver(
        manifest=build_manifest(
            report_id="hisense_home_2024",
            page_count=243,
        ),
        page_mappings=(
            build_hisense_mappings()
        ),
    )

    assert resolver.resolve(34).printed_page == 33
    assert resolver.resolve(35).printed_page == 33
    assert resolver.resolve(36).printed_page == 34


def test_audit_midea_mapping() -> None:
    """美的应仅有封面未映射。"""

    resolver = PageMappingResolver(
        manifest=build_manifest(
            report_id="midea_group_2024",
            page_count=295,
        ),
        page_mappings=[
            build_midea_mapping()
        ],
    )

    audit = resolver.audit()

    assert audit.mapped_page_count == 294
    assert audit.unmapped_pdf_pages == (1,)
    assert audit.duplicate_printed_pages == {}


def test_audit_hisense_duplicate_page() -> None:
    """海信审计应记录重复印刷页 33。"""

    resolver = PageMappingResolver(
        manifest=build_manifest(
            report_id="hisense_home_2024",
            page_count=243,
        ),
        page_mappings=(
            build_hisense_mappings()
        ),
    )

    audit = resolver.audit()

    assert audit.mapped_page_count == 242
    assert audit.unmapped_pdf_pages == (1,)

    assert audit.duplicate_printed_pages == {
        33: (34, 35)
    }


@pytest.mark.parametrize(
    "pdf_page",
    [0, 296],
)
def test_reject_pdf_page_outside_document(
    pdf_page: int,
) -> None:
    """超出实际 PDF 范围的页码应失败。"""

    resolver = PageMappingResolver(
        manifest=build_manifest(
            report_id="midea_group_2024",
            page_count=295,
        ),
        page_mappings=[
            build_midea_mapping()
        ],
    )

    with pytest.raises(InvalidPdfPageError):
        resolver.resolve(pdf_page)


def test_reject_blocked_document() -> None:
    """未通过接入检查的文档不能映射。"""

    with pytest.raises(BlockedDocumentError):
        PageMappingResolver(
            manifest=build_manifest(
                report_id="midea_group_2024",
                page_count=295,
                valid=False,
            ),
            page_mappings=[
                build_midea_mapping()
            ],
        )


def test_reject_report_without_mapping() -> None:
    """目标报告没有映射规则时应失败。"""

    with pytest.raises(MissingPageMappingError):
        PageMappingResolver(
            manifest=build_manifest(
                report_id="midea_group_2024",
                page_count=295,
            ),
            page_mappings=(
                build_hisense_mappings()
            ),
        )


def test_reject_overlapping_pdf_ranges() -> None:
    """两个规则不能覆盖同一个 PDF 页面。"""

    first = PageMappingSegment(
        mapping_id="mapping_first",
        report_id="midea_group_2024",
        printed_page_start=1,
        printed_page_end=9,
        pdf_page_start=2,
        pdf_page_end=10,
        offset=1,
        rule_type="offset",
        notes="first",
        validation_status="verified",
    )

    second = PageMappingSegment(
        mapping_id="mapping_second",
        report_id="midea_group_2024",
        printed_page_start=9,
        printed_page_end=19,
        pdf_page_start=10,
        pdf_page_end=20,
        offset=1,
        rule_type="offset",
        notes="second",
        validation_status="verified",
    )

    with pytest.raises(
        OverlappingPageMappingError
    ):
        PageMappingResolver(
            manifest=build_manifest(
                report_id="midea_group_2024",
                page_count=295,
            ),
            page_mappings=[first, second],
        )


def test_reject_mapping_outside_document() -> None:
    """映射规则不能超过实际 PDF 总页数。"""

    mapping = PageMappingSegment(
        mapping_id="mapping_outside",
        report_id="midea_group_2024",
        printed_page_start=1,
        printed_page_end=295,
        pdf_page_start=2,
        pdf_page_end=296,
        offset=1,
        rule_type="offset",
        notes="超过文档范围",
        validation_status="verified",
    )

    with pytest.raises(
        InvalidPageMappingConfigurationError
    ):
        PageMappingResolver(
            manifest=build_manifest(
                report_id="midea_group_2024",
                page_count=295,
            ),
            page_mappings=[mapping],
        )


def test_reject_unverified_mapping() -> None:
    """未核验规则不得进入可信映射管线。"""

    mapping = PageMappingSegment(
        mapping_id="mapping_pending",
        report_id="midea_group_2024",
        printed_page_start=1,
        printed_page_end=294,
        pdf_page_start=2,
        pdf_page_end=295,
        offset=1,
        rule_type="offset",
        notes="尚未核验",
        validation_status="pending",
    )

    with pytest.raises(
        InvalidPageMappingConfigurationError
    ):
        PageMappingResolver(
            manifest=build_manifest(
                report_id="midea_group_2024",
                page_count=295,
            ),
            page_mappings=[mapping],
        )


def test_reject_non_linear_custom_mapping() -> None:
    """当前版本不接受无法确定计算的 custom 区间。"""

    mapping = PageMappingSegment(
        mapping_id="mapping_custom_invalid",
        report_id="midea_group_2024",
        printed_page_start=10,
        printed_page_end=11,
        pdf_page_start=20,
        pdf_page_end=22,
        rule_type="custom",
        notes="区间长度不一致",
        validation_status="verified",
    )

    with pytest.raises(
        InvalidPageMappingConfigurationError
    ):
        PageMappingResolver(
            manifest=build_manifest(
                report_id="midea_group_2024",
                page_count=295,
            ),
            page_mappings=[mapping],
        )