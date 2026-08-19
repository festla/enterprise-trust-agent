from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

from app.schemas.competition import (
    CompetitionQaCase,
    CompetitionResolutionStrategy,
    CompetitionSourceRecord,
    CompetitionSourceResolution,
    CompetitionSourceType,
)


_EXCEL_SUFFIXES = {
    ".xlsx",
    ".xls",
    ".xlsm",
}

_WORD_SUFFIXES = {
    ".docx",
    ".doc",
}

_PDF_SUFFIXES = {
    ".pdf",
}


class CompetitionSourceResolverError(
    RuntimeError
):
    pass


def _normalize_filename_key(
    value: str,
) -> str:
    """
    用于比赛附件名称匹配。

    只保留 Unicode 字母和数字：
    - 去除空格
    - 去除 _
    - 去除 -
    - 去除中英文括号
    - 去除书名号
    - 去除其他标点

    例如：

    2023年经营情况表（10月）.xlsx
    2023年经营情况表_10月.xlsx

    会得到更加稳定的比较 Key。
    """

    normalized = unicodedata.normalize(
        "NFKC",
        value,
    ).casefold()

    return "".join(
        char
        for char in normalized
        if char.isalnum()
    )


def _normalize_stem_key(
    value: str,
) -> str:
    path = Path(value)

    return _normalize_filename_key(
        path.stem
    )

def _source_type_from_path(
    path: Path,
) -> CompetitionSourceType | None:
    suffix = path.suffix.casefold()

    if suffix in _EXCEL_SUFFIXES:
        return "excel"

    if suffix in _WORD_SUFFIXES:
        return "word"

    if suffix in _PDF_SUFFIXES:
        return "pdf"

    return None


def _build_source_id(
    relative_path: str,
) -> str:
    digest = hashlib.sha256(
        relative_path.encode("utf-8")
    ).hexdigest()[:16]

    return f"src_{digest}"


def build_competition_source_manifest(
    attachments_root: Path,
) -> tuple[
    CompetitionSourceRecord,
    ...,
]:
    if not attachments_root.exists():
        raise CompetitionSourceResolverError(
            "附件目录不存在: "
            f"{attachments_root}"
        )

    records: list[
        CompetitionSourceRecord
    ] = []

    for path in sorted(
        attachments_root.rglob("*"),
        key=lambda item: (
            item.as_posix()
        ),
    ):
        if not path.is_file():
            continue

        # Office 临时锁文件
        if path.name.startswith("~$"):
            continue

        source_type = (
            _source_type_from_path(path)
        )

        if source_type is None:
            continue

        relative_path = (
            path.relative_to(
                attachments_root
            )
            .as_posix()
        )

        records.append(
            CompetitionSourceRecord(
                source_id=_build_source_id(
                    relative_path
                ),
                source_type=source_type,
                actual_filename=path.name,
                relative_path=(
                    relative_path
                ),
                extension=(
                    path.suffix.casefold()
                ),
                size_bytes=(
                    path.stat().st_size
                ),
            )
        )

    if not records:
        raise CompetitionSourceResolverError(
            "附件目录中没有找到"
            " Excel / Word / PDF 文件"
        )

    return tuple(records)


class CompetitionSourceResolver:
    def __init__(
        self,
        manifest: tuple[
            CompetitionSourceRecord,
            ...,
        ],
    ) -> None:
        self._manifest = manifest

    def _build_resolution(
        self,
        *,
        case: CompetitionQaCase,
        record: CompetitionSourceRecord,
        strategy: (
            CompetitionResolutionStrategy
        ),
    ) -> CompetitionSourceResolution:
        return CompetitionSourceResolution(
            case_id=case.case_id,
            source_id=record.source_id,
            source_type=(
                record.source_type
            ),
            relative_path=(
                record.relative_path
            ),
            strategy=strategy,
        )

    def resolve(
        self,
        case: CompetitionQaCase,
    ) -> CompetitionSourceResolution:
        same_type_records = [
            record
            for record in self._manifest
            if (
                record.source_type
                == case.source_type
            )
        ]

        # ========================================================
        # Level 1
        # source_title + "_" + file_label
        # ========================================================

        expected_tail = (
            f"{case.source_title}_"
            f"{case.file_label}"
        )

        expected_tail_key = (
            _normalize_filename_key(
                expected_tail
            )
        )

        exact_matches = [
            record
            for record
            in same_type_records
            if (
                _normalize_filename_key(
                    record.actual_filename
                )
                .endswith(
                    expected_tail_key
                )
            )
        ]

        if len(exact_matches) == 1:
            return self._build_resolution(
                case=case,
                record=exact_matches[0],
                strategy="exact_tail",
            )

        if len(exact_matches) > 1:
            raise (
                CompetitionSourceResolverError(
                    f"{case.case_id}: "
                    "exact_tail 匹配不唯一"
                )
            )

        # ========================================================
        # Level 2
        # file_label 本身就可能对应真实文件名尾部
        #
        # 例如：
        #
        # 145_xxx_xxx.xlsx
        #
        # QA:
        # file_label = xxx.xlsx
        # ========================================================

        file_label_raw_key = (
            unicodedata.normalize(
                "NFKC",
                case.file_label,
            )
            .casefold()
            .replace("\\", "/")
            .strip()
        )

        raw_label_matches = [
            record
            for record
            in same_type_records
            if (
                unicodedata.normalize(
                    "NFKC",
                    record.actual_filename,
                )
                .casefold()
                .endswith(
                    file_label_raw_key
                )
            )
        ]

        if len(raw_label_matches) == 1:
            return self._build_resolution(
                case=case,
                record=raw_label_matches[0],
                strategy="file_label",
            )

        if len(raw_label_matches) > 1:
            # 不立即报错。
            # 后面的 source_title 可能帮助消歧。
            pass

        # ========================================================
        # Level 3
        # normalized file_label
        #
        # 忽略空格、下划线、中英文标点差异
        # ========================================================

        file_label_key = (
            _normalize_stem_key(
                case.file_label
            )
        )

        normalized_label_matches = [
            record
            for record
            in same_type_records
            if (
                file_label_key
                and (
                    _normalize_stem_key(
                        record.actual_filename
                    )
                    .endswith(
                        file_label_key
                    )
                )
            )
        ]

        if (
            len(normalized_label_matches)
            == 1
        ):
            return self._build_resolution(
                case=case,
                record=(
                    normalized_label_matches[
                        0
                    ]
                ),
                strategy=(
                    "normalized_file_label"
                ),
            )

        # ========================================================
        # Level 4
        # source_title + file_label 联合消歧
        # ========================================================

        title_key = (
            _normalize_filename_key(
                case.source_title
            )
        )

        candidate_pool = (
            normalized_label_matches
            if normalized_label_matches
            else (
                raw_label_matches
                if raw_label_matches
                else same_type_records
            )
        )

        title_and_label_matches = [
            record
            for record
            in candidate_pool
            if (
                title_key
                and (
                    title_key
                    in _normalize_filename_key(
                        record.actual_filename
                    )
                )
            )
        ]

        if (
            len(title_and_label_matches)
            == 1
        ):
            return self._build_resolution(
                case=case,
                record=(
                    title_and_label_matches[0]
                ),
                strategy="title_and_label",
            )

        # ========================================================
        # Level 5
        # source_title 单独唯一匹配
        #
        # 只有“唯一”时才允许放行。
        # ========================================================

        title_matches = [
            record
            for record
            in same_type_records
            if (
                title_key
                and (
                    title_key
                    in _normalize_filename_key(
                        record.actual_filename
                    )
                )
            )
        ]

        if len(title_matches) == 1:
            return self._build_resolution(
                case=case,
                record=title_matches[0],
                strategy="unique_title",
            )

        # ========================================================
        # Refuse ambiguous / missing match
        # ========================================================

        if (
            len(title_and_label_matches)
            > 1
            or len(title_matches) > 1
            or len(normalized_label_matches)
            > 1
            or len(raw_label_matches) > 1
        ):
            raise (
                CompetitionSourceResolverError(
                    f"{case.case_id}: "
                    "存在多个候选附件，"
                    "不能安全唯一定位"
                )
            )

        raise CompetitionSourceResolverError(
            f"{case.case_id}: "
            "无法定位对应附件"
        )