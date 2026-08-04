from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from app.schemas.document import DocumentManifest
from app.schemas.enums import (
    DocumentValidationStatus,
    PageMappingRuleType,
    PageMappingStatus,
    ValidationStatus,
)
from app.schemas.page import (
    PageMappingAudit,
    PageMappingResult,
)
from app.schemas.report import PageMappingSegment


class PageMappingError(ValueError):
    """页码映射基础异常。"""


class BlockedDocumentError(PageMappingError):
    """文档没有通过接入检查，不能进行页码映射。"""


class MissingPageMappingError(PageMappingError):
    """目标报告没有配置任何页码映射规则。"""


class InvalidPdfPageError(PageMappingError):
    """目标 PDF 页码超出实际文档范围。"""


class InvalidPageMappingConfigurationError(
    PageMappingError
):
    """页码映射规则本身无效或不受支持。"""


class OverlappingPageMappingError(
    PageMappingError
):
    """两个映射规则覆盖了同一个 PDF 页面。"""


class PageMappingResolver:
    """为一个具体 PDF 文档版本解析印刷页码。"""

    def __init__(
        self,
        *,
        manifest: DocumentManifest,
        page_mappings: Iterable[
            PageMappingSegment
        ],
    ) -> None:
        if (
            manifest.validation_status
            is not DocumentValidationStatus.VALID
        ):
            raise BlockedDocumentError(
                f"文档 '{manifest.document_id}' "
                "未通过接入检查，不能执行页码映射"
            )

        self._manifest = manifest

        selected_segments = [
            mapping
            for mapping in page_mappings
            if mapping.report_id
            == manifest.report_id
        ]

        if not selected_segments:
            raise MissingPageMappingError(
                f"Report '{manifest.report_id}' "
                "没有配置页码映射规则"
            )

        self._segments = tuple(
            sorted(
                selected_segments,
                key=lambda mapping: (
                    mapping.pdf_page_start,
                    mapping.pdf_page_end,
                    mapping.mapping_id,
                ),
            )
        )

        self._validate_segments()

    def _validate_segments(self) -> None:
        """检查当前报告的全部映射规则。"""

        seen_mapping_ids: set[str] = set()

        previous_segment: (
            PageMappingSegment | None
        ) = None

        for segment in self._segments:
            if segment.mapping_id in seen_mapping_ids:
                raise (
                    InvalidPageMappingConfigurationError(
                        "出现重复 mapping_id："
                        f"{segment.mapping_id}"
                    )
                )

            seen_mapping_ids.add(
                segment.mapping_id
            )

            if (
                segment.validation_status
                is not ValidationStatus.VERIFIED
            ):
                raise (
                    InvalidPageMappingConfigurationError(
                        "页码映射规则尚未核验："
                        f"{segment.mapping_id}"
                    )
                )

            if (
                segment.pdf_page_end
                > self._manifest.pdf_page_count
            ):
                raise (
                    InvalidPageMappingConfigurationError(
                        "页码映射超出实际 PDF 页数："
                        f"{segment.mapping_id}"
                    )
                )

            if (
                previous_segment is not None
                and segment.pdf_page_start
                <= previous_segment.pdf_page_end
            ):
                raise OverlappingPageMappingError(
                    "PDF 页码映射区间重叠："
                    f"{previous_segment.mapping_id} "
                    f"与 {segment.mapping_id}"
                )

            if (
                segment.rule_type
                is PageMappingRuleType.CUSTOM
            ):
                self._validate_custom_segment(
                    segment
                )

            previous_segment = segment

    @staticmethod
    def _validate_custom_segment(
        segment: PageMappingSegment,
    ) -> None:
        """当前版本只支持一对一连续 custom 映射。"""

        if (
            segment.printed_page_start is None
            or segment.printed_page_end is None
        ):
            raise (
                InvalidPageMappingConfigurationError(
                    "custom 映射必须包含完整印刷页区间："
                    f"{segment.mapping_id}"
                )
            )

        pdf_span = (
            segment.pdf_page_end
            - segment.pdf_page_start
        )

        printed_span = (
            segment.printed_page_end
            - segment.printed_page_start
        )

        if pdf_span != printed_span:
            raise (
                InvalidPageMappingConfigurationError(
                    "当前版本的 custom 映射必须是 "
                    "一对一连续映射："
                    f"{segment.mapping_id}"
                )
            )

    def resolve(
        self,
        pdf_page: int,
    ) -> PageMappingResult:
        """将一个 PDF 页码解析为印刷页码。"""

        if not (
            1
            <= pdf_page
            <= self._manifest.pdf_page_count
        ):
            raise InvalidPdfPageError(
                "PDF 页码超出实际文档范围："
                f"{pdf_page}，合法范围为 "
                f"1-{self._manifest.pdf_page_count}"
            )

        page_id = self._build_page_id(
            pdf_page
        )

        for segment in self._segments:
            if (
                segment.pdf_page_start
                <= pdf_page
                <= segment.pdf_page_end
            ):
                printed_page = (
                    self._calculate_printed_page(
                        segment=segment,
                        pdf_page=pdf_page,
                    )
                )

                return PageMappingResult(
                    page_id=page_id,
                    document_id=(
                        self._manifest.document_id
                    ),
                    report_id=(
                        self._manifest.report_id
                    ),
                    pdf_page=pdf_page,
                    printed_page=printed_page,
                    mapping_status=(
                        PageMappingStatus.MAPPED
                    ),
                    mapping_id=segment.mapping_id,
                )

        return PageMappingResult(
            page_id=page_id,
            document_id=self._manifest.document_id,
            report_id=self._manifest.report_id,
            pdf_page=pdf_page,
            printed_page=None,
            mapping_status=(
                PageMappingStatus.UNMAPPED
            ),
            mapping_id=None,
        )

    def resolve_all(
        self,
    ) -> tuple[PageMappingResult, ...]:
        """解析文档中的全部 PDF 页面。"""

        return tuple(
            self.resolve(pdf_page)
            for pdf_page in range(
                1,
                self._manifest.pdf_page_count + 1,
            )
        )

    def audit(self) -> PageMappingAudit:
        """生成整份文档的页码映射审计结果。"""

        results = self.resolve_all()

        unmapped_pdf_pages: list[int] = []

        printed_to_pdf_pages: dict[
            int,
            list[int],
        ] = defaultdict(list)

        for result in results:
            if (
                result.mapping_status
                is PageMappingStatus.UNMAPPED
            ):
                unmapped_pdf_pages.append(
                    result.pdf_page
                )
                continue

            if result.printed_page is None:
                raise PageMappingError(
                    "mapped 页面缺少 printed_page"
                )

            printed_to_pdf_pages[
                result.printed_page
            ].append(result.pdf_page)

        duplicate_printed_pages = {
            printed_page: tuple(pdf_pages)
            for printed_page, pdf_pages
            in sorted(
                printed_to_pdf_pages.items()
            )
            if len(pdf_pages) > 1
        }

        return PageMappingAudit(
            document_id=(
                self._manifest.document_id
            ),
            report_id=self._manifest.report_id,
            total_pdf_pages=(
                self._manifest.pdf_page_count
            ),
            mapped_page_count=(
                len(results)
                - len(unmapped_pdf_pages)
            ),
            unmapped_pdf_pages=tuple(
                unmapped_pdf_pages
            ),
            duplicate_printed_pages=(
                duplicate_printed_pages
            ),
        )

    def _build_page_id(
        self,
        pdf_page: int,
    ) -> str:
        """生成与文档版本绑定的稳定页面 ID。"""

        return (
            f"{self._manifest.document_id}"
            f"_page_{pdf_page:04d}"
        )

    @staticmethod
    def _calculate_printed_page(
        *,
        segment: PageMappingSegment,
        pdf_page: int,
    ) -> int:
        """按照映射规则计算印刷页码。"""

        if (
            segment.rule_type
            is PageMappingRuleType.IDENTITY
        ):
            printed_page = pdf_page

        elif (
            segment.rule_type
            is PageMappingRuleType.OFFSET
        ):
            if segment.offset is None:
                raise (
                    InvalidPageMappingConfigurationError(
                        "offset 映射缺少 offset："
                        f"{segment.mapping_id}"
                    )
                )

            printed_page = (
                pdf_page - segment.offset
            )

        elif (
            segment.rule_type
            is PageMappingRuleType.CUSTOM
        ):
            if (
                segment.printed_page_start
                is None
            ):
                raise (
                    InvalidPageMappingConfigurationError(
                        "custom 映射缺少 "
                        "printed_page_start："
                        f"{segment.mapping_id}"
                    )
                )

            relative_position = (
                pdf_page
                - segment.pdf_page_start
            )

            printed_page = (
                segment.printed_page_start
                + relative_position
            )

        else:
            raise (
                InvalidPageMappingConfigurationError(
                    "不支持的页码映射类型："
                    f"{segment.rule_type}"
                )
            )

        if (
            segment.printed_page_start is None
            or segment.printed_page_end is None
        ):
            raise (
                InvalidPageMappingConfigurationError(
                    "映射规则缺少印刷页码区间："
                    f"{segment.mapping_id}"
                )
            )

        if not (
            segment.printed_page_start
            <= printed_page
            <= segment.printed_page_end
        ):
            raise (
                InvalidPageMappingConfigurationError(
                    "计算出的印刷页码超出规则区间："
                    f"{segment.mapping_id}"
                )
            )

        return printed_page