from __future__ import annotations

from pathlib import Path

from app.schemas.competition import (
    CompetitionQuestion,
    CompetitionSourceRecord,
)
from app.schemas.competition_evidence import (
    CompetitionKnowledgeSource,
)
from app.services.document_ingestion import (
    calculate_file_sha256,
)


class CompetitionSourceCatalogError(
    RuntimeError
):
    pass


def resolve_competition_source_path(
    *,
    attachments_root: Path,
    source: CompetitionSourceRecord,
) -> Path:
    """
    将 CompetitionSourceRecord 的相对路径
    安全解析为实际附件路径。

    不允许 source.relative_path 逃出
    attachments_root。
    """

    root = attachments_root.resolve()

    candidate = (
        root
        / source.relative_path
    )

    try:
        resolved = candidate.resolve(
            strict=True
        )
    except FileNotFoundError as exc:
        raise CompetitionSourceCatalogError(
            "Competition source 文件不存在: "
            f"{candidate}"
        ) from exc

    if not resolved.is_file():
        raise CompetitionSourceCatalogError(
            "Competition source 不是普通文件: "
            f"{resolved}"
        )

    try:
        resolved.relative_to(
            root
        )
    except ValueError as exc:
        raise CompetitionSourceCatalogError(
            "Competition source 位于 "
            "attachments_root 外部: "
            f"{resolved}"
        ) from exc

    if (
        resolved.name
        != source.actual_filename
    ):
        raise CompetitionSourceCatalogError(
            "Source filename 与实际文件名不一致: "
            f"record={source.actual_filename}; "
            f"actual={resolved.name}"
        )

    actual_size = (
        resolved.stat().st_size
    )

    if (
        actual_size
        != source.size_bytes
    ):
        raise CompetitionSourceCatalogError(
            "Source 文件大小与 manifest record "
            "不一致: "
            f"record={source.size_bytes}; "
            f"actual={actual_size}"
        )

    return resolved


def build_competition_document_id(
    *,
    source_id: str,
    sha256: str,
) -> str:
    """
    将“逻辑 Source”和“实际文件版本”区分开。

    source_id:
        由附件相对路径得到，表示逻辑来源。

    doc_id:
        source_id + 文件内容 Hash，
        表示实际文件版本。
    """

    return (
        f"doc_{source_id}_"
        f"{sha256[:24]}"
    )


def build_competition_knowledge_source(
    *,
    question: CompetitionQuestion,
    source: CompetitionSourceRecord,
    attachments_root: Path,
) -> CompetitionKnowledgeSource:
    """
    将当前 Competition Source
    转换为统一 KnowledgeSource。

    只填写真实能够获得的字段，
    不推测 source_url / issuing_authority /
    published_date。
    """

    if (
        question.source_type
        != source.source_type
    ):
        raise CompetitionSourceCatalogError(
            "Question 与 Source 的 "
            "source_type 不一致: "
            f"question={question.source_type}; "
            f"source={source.source_type}"
        )

    source_path = (
        resolve_competition_source_path(
            attachments_root=(
                attachments_root
            ),
            source=source,
        )
    )

    # ========================================================
    # 直接复用 Week1 已有的 SHA-256 实现。
    # 不在 Competition 代码里重新写一遍。
    # ========================================================

    sha256 = calculate_file_sha256(
        source_path
    )

    document_id = (
        build_competition_document_id(
            source_id=source.source_id,
            sha256=sha256,
        )
    )

    return CompetitionKnowledgeSource(
        source_id=source.source_id,
        doc_id=document_id,
        title=question.source_title,
        source_type=source.source_type,
        relative_path=source.relative_path,
        source_url=None,
        issuing_authority=None,
        published_date=None,
        sha256=sha256,
    )