from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import zipfile


DEFAULT_CONVERSION_TIMEOUT_SECONDS = 120

LIBREOFFICE_PATH_ENV = (
    "LIBREOFFICE_SOFFICE_PATH"
)


class CompetitionLegacyDocConversionError(
    RuntimeError
):
    """Legacy DOC 转换失败。"""


class CompetitionLegacyDocConverterUnavailableError(
    CompetitionLegacyDocConversionError
):
    """系统中找不到可用的 LibreOffice。"""


def _validate_executable(
    path: Path,
    *,
    source: str,
) -> Path:
    candidate = path.expanduser()

    try:
        resolved = candidate.resolve(
            strict=True
        )
    except FileNotFoundError as exc:
        raise (
            CompetitionLegacyDocConverterUnavailableError(
                "LibreOffice executable 不存在: "
                f"source={source}; "
                f"path={candidate}"
            )
        ) from exc

    if not resolved.is_file():
        raise (
            CompetitionLegacyDocConverterUnavailableError(
                "LibreOffice executable "
                "不是普通文件: "
                f"source={source}; "
                f"path={resolved}"
            )
        )

    return resolved


def find_libreoffice_executable(
    *,
    explicit_path: Path | None = None,
) -> Path:
    """
    查找 LibreOffice soffice executable。

    优先级：

    1. 调用方显式传入；
    2. LIBREOFFICE_SOFFICE_PATH；
    3. 系统 PATH；
    4. 常见安装目录。
    """

    if explicit_path is not None:
        return _validate_executable(
            explicit_path,
            source="explicit_path",
        )

    configured_path = os.environ.get(
        LIBREOFFICE_PATH_ENV
    )

    if configured_path:
        return _validate_executable(
            Path(configured_path),
            source=LIBREOFFICE_PATH_ENV,
        )

    for command_name in (
        "soffice",
        "soffice.exe",
        "libreoffice",
        "libreoffice.exe",
    ):
        located = shutil.which(
            command_name
        )

        if located:
            return _validate_executable(
                Path(located),
                source="PATH",
            )

    candidates: list[Path] = []

    for environment_name in (
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
    ):
        root = os.environ.get(
            environment_name
        )

        if not root:
            continue

        candidates.append(
            Path(root)
            / "LibreOffice"
            / "program"
            / "soffice.exe"
        )

    candidates.extend(
        (
            Path("/usr/bin/soffice"),
            Path("/usr/bin/libreoffice"),
            Path("/usr/local/bin/soffice"),
            Path(
                "/opt/libreoffice/program/soffice"
            ),
        )
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise (
        CompetitionLegacyDocConverterUnavailableError(
            "找不到 LibreOffice soffice；"
            f"可设置 {LIBREOFFICE_PATH_ENV}"
        )
    )


def convert_legacy_doc_to_docx(
    *,
    source_path: Path,
    output_directory: Path,
    soffice_path: Path | None = None,
    timeout_seconds: int = (
        DEFAULT_CONVERSION_TIMEOUT_SECONDS
    ),
) -> Path:
    """
    使用 LibreOffice headless 将旧版 DOC 转换为 DOCX。

    原始文件保持只读，不会被覆盖。
    """

    if timeout_seconds <= 0:
        raise ValueError(
            "timeout_seconds 必须大于 0"
        )

    try:
        source = source_path.resolve(
            strict=True
        )
    except FileNotFoundError as exc:
        raise CompetitionLegacyDocConversionError(
            "Legacy DOC 文件不存在: "
            f"{source_path}"
        ) from exc

    if not source.is_file():
        raise CompetitionLegacyDocConversionError(
            "Legacy DOC 不是普通文件: "
            f"{source}"
        )

    if source.suffix.casefold() != ".doc":
        raise CompetitionLegacyDocConversionError(
            "Legacy DOC 转换器只接受 .doc: "
            f"{source}"
        )

    executable = (
        find_libreoffice_executable(
            explicit_path=soffice_path,
        )
    )

    output_root = (
        output_directory.resolve()
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile_directory = (
        output_root
        / "libreoffice-profile"
    )

    profile_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_root
        / f"{source.stem}.docx"
    )

    command = [
        str(executable),
        "--headless",
        "--nologo",
        "--nodefault",
        "--nolockcheck",
        (
            "-env:UserInstallation="
            f"{profile_directory.resolve().as_uri()}"
        ),
        "--convert-to",
        "docx",
        "--outdir",
        str(output_root),
        str(source),
    ]

    creation_flags = getattr(
        subprocess,
        "CREATE_NO_WINDOW",
        0,
    )

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            creationflags=creation_flags,
        )

    except subprocess.TimeoutExpired as exc:
        raise CompetitionLegacyDocConversionError(
            "LibreOffice 转换超时: "
            f"timeout_seconds={timeout_seconds}; "
            f"source={source}"
        ) from exc

    except OSError as exc:
        raise CompetitionLegacyDocConversionError(
            "无法启动 LibreOffice: "
            f"{executable}"
        ) from exc

    if result.returncode != 0:
        raise CompetitionLegacyDocConversionError(
            "LibreOffice 转换失败: "
            f"returncode={result.returncode}; "
            f"stdout={result.stdout.strip()!r}; "
            f"stderr={result.stderr.strip()!r}"
        )

    if not output_path.is_file():
        raise CompetitionLegacyDocConversionError(
            "LibreOffice 未生成预期 DOCX: "
            f"{output_path}; "
            f"stdout={result.stdout.strip()!r}; "
            f"stderr={result.stderr.strip()!r}"
        )

    if output_path.stat().st_size <= 0:
        raise CompetitionLegacyDocConversionError(
            "LibreOffice 生成了空 DOCX: "
            f"{output_path}"
        )

    if not zipfile.is_zipfile(
        output_path
    ):
        raise CompetitionLegacyDocConversionError(
            "转换结果不是有效 OOXML ZIP: "
            f"{output_path}"
        )

    return output_path