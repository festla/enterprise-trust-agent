from __future__ import annotations

import hashlib

import pytest

from app.schemas.competition import (
    CompetitionQuestion,
    CompetitionSourceRecord,
)
from app.services.competition_source_catalog import (
    CompetitionSourceCatalogError,
    build_competition_knowledge_source,
)


def test_build_competition_knowledge_source(
    tmp_path,
) -> None:
    attachments = (
        tmp_path
        / "attachments"
    )

    attachments.mkdir()

    file_content = (
        b"competition-source-test"
    )

    source_path = (
        attachments
        / "test.xlsx"
    )

    source_path.write_bytes(
        file_content
    )

    source = CompetitionSourceRecord(
        source_id=(
            "src_0123456789abcdef"
        ),
        source_type="excel",
        actual_filename="test.xlsx",
        relative_path="test.xlsx",
        extension=".xlsx",
        size_bytes=len(
            file_content
        ),
    )

    question = CompetitionQuestion(
        case_id="Q001",
        source_type="excel",
        qa_type="表格取数",
        question="测试问题",
        option_a="1",
        option_b="2",
        option_c="3",
        option_d="4",
        source_title="测试监管报表",
        file_label="test.xlsx",
    )

    knowledge_source = (
        build_competition_knowledge_source(
            question=question,
            source=source,
            attachments_root=attachments,
        )
    )

    expected_sha256 = (
        hashlib.sha256(
            file_content
        ).hexdigest()
    )

    assert (
        knowledge_source.source_id
        == source.source_id
    )

    assert (
        knowledge_source.title
        == "测试监管报表"
    )

    assert (
        knowledge_source.relative_path
        == "test.xlsx"
    )

    assert (
        knowledge_source.sha256
        == expected_sha256
    )

    assert (
        knowledge_source.doc_id
        ==
        f"doc_{source.source_id}_"
        f"{expected_sha256[:24]}"
    )

    assert (
        knowledge_source.source_url
        is None
    )


def test_source_catalog_rejects_size_mismatch(
    tmp_path,
) -> None:
    attachments = (
        tmp_path
        / "attachments"
    )

    attachments.mkdir()

    (
        attachments
        / "test.pdf"
    ).write_bytes(
        b"real-content"
    )

    source = CompetitionSourceRecord(
        source_id=(
            "src_0123456789abcdef"
        ),
        source_type="pdf",
        actual_filename="test.pdf",
        relative_path="test.pdf",
        extension=".pdf",

        # 故意错误。
        size_bytes=999,
    )

    question = CompetitionQuestion(
        case_id="Q001",
        source_type="pdf",
        qa_type="单事实检索",
        question="测试问题",
        option_a="A",
        option_b="B",
        option_c="C",
        option_d="D",
        source_title="测试监管文件",
        file_label="test.pdf",
    )

    with pytest.raises(
        CompetitionSourceCatalogError
    ):
        build_competition_knowledge_source(
            question=question,
            source=source,
            attachments_root=attachments,
        )