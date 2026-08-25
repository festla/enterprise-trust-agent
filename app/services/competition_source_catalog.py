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

def _is_deployment_storage_path(
    source: CompetitionSourceRecord,
) -> bool:
    """
    判断 Source 是否使用 Linux-safe 部署物理文件名。

    部署包允许：

        actual_filename:
            原始逻辑文件名，用于 Source Resolver 匹配

        relative_path:
            files/src_<source_id>.<ext>
            实际 Linux-safe 物理路径

    只允许我们约定的确定性命名规则，
    不允许任意文件名绕过 Catalog 校验。
    """

    relative_path = Path(
        source.relative_path
    )

    expected_filename = (
        f"{source.source_id}"
        f"{source.extension}"
    )

    return (
        relative_path.parent.as_posix()
        == "files"
        and relative_path.name
        == expected_filename
    )

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

    # ========================================================
    # Filename invariant
    #
    # 开发数据：
    #
    #   relative_path == actual_filename
    #
    # 部署数据：
    #
    #   actual_filename
    #       保留原始逻辑文件名，用于 Source Resolver；
    #
    #   relative_path
    #       使用 Linux-safe 的短物理文件名：
    #       files/src_<id>.<ext>
    #
    # 因此部署模式不能继续要求：
    #
    #   resolved.name == actual_filename
    #
    # 但仍然必须严格校验物理文件名符合我们定义的
    # source_id + extension 规则。
    # ========================================================

    relative_filename = (
        Path(
            source.relative_path
        ).name
    )

    if (
        resolved.name
        != relative_filename
    ):
        raise CompetitionSourceCatalogError(
            "Source relative_path 与实际文件名不一致: "
            f"relative={relative_filename}; "
            f"actual={resolved.name}"
        )

    logical_filename_matches = (
        resolved.name
        == source.actual_filename
    )

    deployment_filename_matches = (
        _is_deployment_storage_path(
            source
        )
    )

    if (
        not logical_filename_matches
        and not deployment_filename_matches
    ):
        raise CompetitionSourceCatalogError(
            "Source filename 不符合开发目录"
            "或部署目录约束: "
            f"logical={source.actual_filename}; "
            f"physical={resolved.name}; "
            f"relative={source.relative_path}"
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