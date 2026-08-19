from pathlib import Path

import pytest

from app.schemas.competition import (
    CompetitionQaCase,
)
from app.services.competition_source_resolver import (
    CompetitionSourceResolver,
    CompetitionSourceResolverError,
    build_competition_source_manifest,
)


def _build_case() -> CompetitionQaCase:
    return CompetitionQaCase(
        case_id="Q001",
        source_type="excel",
        difficulty="easy",
        difficulty_cn="简单",
        qa_type="表格取数",
        question="测试问题",
        option_a="1",
        option_b="2",
        option_c="3",
        option_d="4",
        answer="A",
        answer_text="1",
        evidence="测试证据",
        source_title="测试经营情况表",
        file_label="测试经营情况表.xlsx",
    )


def test_source_resolver_matches_actual_file(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / (
            "145_测试经营情况表_"
            "测试经营情况表.xlsx"
        )
    )

    source.write_bytes(b"test")

    manifest = (
        build_competition_source_manifest(
            tmp_path
        )
    )

    resolver = CompetitionSourceResolver(
        manifest
    )

    resolution = resolver.resolve(
        _build_case()
    )

    assert (
        resolution.relative_path
        == source.name
    )

    assert (
        resolution.strategy
        == "exact_tail"
    )

    assert (
        resolution.source_type
        == "excel"
    )


def test_source_resolver_does_not_match_wrong_type(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / (
            "145_测试经营情况表_"
            "测试经营情况表.pdf"
        )
    )

    source.write_bytes(b"test")

    manifest = (
        build_competition_source_manifest(
            tmp_path
        )
    )

    resolver = CompetitionSourceResolver(
        manifest
    )

    with pytest.raises(
        CompetitionSourceResolverError
    ):
        resolver.resolve(
            _build_case()
        )

def test_source_resolver_normalizes_filename_punctuation(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "145_测试经营情况表（10月）.xlsx"
    )

    source.write_bytes(b"test")

    manifest = (
        build_competition_source_manifest(
            tmp_path
        )
    )

    case = CompetitionQaCase(
        case_id="Q001",
        source_type="excel",
        difficulty="easy",
        difficulty_cn="简单",
        qa_type="表格取数",
        question="测试问题",
        option_a="1",
        option_b="2",
        option_c="3",
        option_d="4",
        answer="A",
        answer_text="1",
        evidence="测试证据",
        source_title=(
            "测试经营情况表 10月"
        ),
        file_label=(
            "测试经营情况表_10月.xlsx"
        ),
    )

    resolution = (
        CompetitionSourceResolver(
            manifest
        ).resolve(case)
    )

    assert (
        resolution.relative_path
        == source.name
    )

def test_source_resolver_refuses_ambiguous_title(
    tmp_path: Path,
) -> None:
    (
        tmp_path
        / "001_测试报告_第一版.pdf"
    ).write_bytes(b"a")

    (
        tmp_path
        / "002_测试报告_第二版.pdf"
    ).write_bytes(b"b")

    manifest = (
        build_competition_source_manifest(
            tmp_path
        )
    )

    case = CompetitionQaCase(
        case_id="Q201",
        source_type="pdf",
        difficulty="medium",
        difficulty_cn="中等",
        qa_type="单事实检索",
        question="测试问题",
        option_a="A",
        option_b="B",
        option_c="C",
        option_d="D",
        answer="A",
        answer_text="A",
        evidence="证据",
        source_title="测试报告",
        file_label="不存在.pdf",
    )

    resolver = (
        CompetitionSourceResolver(
            manifest
        )
    )

    with pytest.raises(
        CompetitionSourceResolverError
    ):
        resolver.resolve(case)