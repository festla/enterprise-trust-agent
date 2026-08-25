from __future__ import annotations

import json
import shutil

from pathlib import Path

from app.schemas.competition import (
    CompetitionSourceRecord,
)
from app.services.competition_source_resolver import (
    build_competition_source_manifest,
)


SOURCE_ROOT = Path(
    "data/competition/private/attachments"
)

OUTPUT_ROOT = Path(
    "data/competition/deploy/attachments"
)

FILES_ROOT = (
    OUTPUT_ROOT
    / "files"
)

MANIFEST_PATH = (
    OUTPUT_ROOT
    / "source_manifest.json"
)


def main() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(
            OUTPUT_ROOT
        )

    FILES_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_records = (
        build_competition_source_manifest(
            SOURCE_ROOT
        )
    )

    deployment_records: list[
        CompetitionSourceRecord
    ] = []

    total_bytes = 0

    for record in original_records:
        source_path = (
            SOURCE_ROOT
            / record.relative_path
        )

        short_filename = (
            f"{record.source_id}"
            f"{record.extension}"
        )

        deployment_relative_path = (
            Path("files")
            / short_filename
        )

        target_path = (
            OUTPUT_ROOT
            / deployment_relative_path
        )

        shutil.copy2(
            source_path,
            target_path,
        )

        copied_size = (
            target_path
            .stat()
            .st_size
        )

        if (
            copied_size
            != record.size_bytes
        ):
            raise RuntimeError(
                "复制后文件大小不一致: "
                f"{record.relative_path}"
            )

        deployment_records.append(
            CompetitionSourceRecord(
                source_id=(
                    record.source_id
                ),
                source_type=(
                    record.source_type
                ),

                # 关键：
                # Resolver 仍看到原始文件名
                actual_filename=(
                    record.actual_filename
                ),

                # 物理文件使用短路径
                relative_path=(
                    deployment_relative_path
                    .as_posix()
                ),

                extension=(
                    record.extension
                ),
                size_bytes=(
                    record.size_bytes
                ),
            )
        )

        total_bytes += (
            record.size_bytes
        )

    payload = {
        "schema_version": 1,
        "sources": [
            record.model_dump(
                mode="json"
            )
            for record
            in deployment_records
        ],
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "===== Competition "
        "Deployment Attachments ====="
    )

    print(
        "Sources:",
        len(
            deployment_records
        ),
    )

    print(
        "Total MB:",
        round(
            total_bytes
            / 1024
            / 1024,
            2,
        ),
    )

    print(
        "Output:",
        OUTPUT_ROOT,
    )

    print(
        "Manifest:",
        MANIFEST_PATH,
    )

    longest_name_bytes = max(
        len(
            (
                Path(
                    record.relative_path
                ).name
            ).encode(
                "utf-8"
            )
        )
        for record
        in deployment_records
    )

    print(
        "Longest deployment "
        "filename bytes:",
        longest_name_bytes,
    )

    if (
        longest_name_bytes
        > 255
    ):
        raise RuntimeError(
            "部署文件名仍超过 "
            "Linux 255-byte 限制"
        )

    print()
    print(
        "DEPLOYMENT ATTACHMENTS PASS"
    )


if __name__ == "__main__":
    main()