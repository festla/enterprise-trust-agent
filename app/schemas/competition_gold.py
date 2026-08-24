from __future__ import annotations

from typing import (
    Literal,
    Self,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


CompetitionGoldEvidenceMode = Literal[
    "single_chunk",
    "parent_child",
    "multi_chunk",
]

CompetitionGoldReviewStatus = Literal[
    "pending",
    "confirmed",
    "unresolved",
]

CompetitionGoldChunkRole = Literal[
    "direct",
    "parent_context",
    "supporting",
]


class CompetitionGoldChunkReference(
    BaseModel
):
    """
    一条人工确认的 Gold Chunk 引用。

    direct:
        Chunk 直接包含事实内容。

    parent_context:
        Chunk 提供父标题、章节分类等必要上下文。

    supporting:
        Chunk 提供补充证据。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    chunk_id: str = Field(
        min_length=1,
    )

    role: CompetitionGoldChunkRole = (
        "direct"
    )


class CompetitionGoldFactRecord(
    BaseModel
):
    """
    Frozen Dev 中一个事实对应的
    人工 Gold Chunk 标注。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    case_id: str = Field(
        min_length=1,
    )

    fact_index: int = Field(
        ge=0,
    )

    fact: str = Field(
        min_length=1,
    )

    expected_source_id: str = Field(
        min_length=1,
    )

    review_status: (
        CompetitionGoldReviewStatus
    ) = "pending"

    evidence_mode: (
        CompetitionGoldEvidenceMode
        | None
    ) = None

    gold_chunks: tuple[
        CompetitionGoldChunkReference,
        ...,
    ] = ()

    review_note: str | None = Field(
        default=None,
        min_length=1,
    )

    @model_validator(
        mode="after"
    )
    def validate_review(
        self,
    ) -> Self:
        chunk_ids = [
            reference.chunk_id
            for reference
            in self.gold_chunks
        ]

        if (
            len(chunk_ids)
            != len(set(chunk_ids))
        ):
            raise ValueError(
                "Gold Chunk 引用不能包含"
                "重复 chunk_id"
            )

        if self.review_status == "pending":
            if (
                self.evidence_mode is not None
                or self.gold_chunks
            ):
                raise ValueError(
                    "pending 记录不能提前设置 "
                    "evidence_mode 或 Gold Chunk"
                )

            return self

        if self.review_status == "unresolved":
            if (
                self.evidence_mode is not None
                or self.gold_chunks
            ):
                raise ValueError(
                    "unresolved 记录不能设置 "
                    "evidence_mode 或 Gold Chunk"
                )

            if self.review_note is None:
                raise ValueError(
                    "unresolved 记录必须提供 "
                    "review_note"
                )

            return self

        if self.evidence_mode is None:
            raise ValueError(
                "confirmed 记录必须提供 "
                "evidence_mode"
            )

        if not self.gold_chunks:
            raise ValueError(
                "confirmed 记录必须提供 "
                "Gold Chunk"
            )

        roles = [
            reference.role
            for reference
            in self.gold_chunks
        ]

        if (
            self.evidence_mode
            == "single_chunk"
        ):
            if (
                len(self.gold_chunks) != 1
                or roles != ["direct"]
            ):
                raise ValueError(
                    "single_chunk 必须且只能"
                    "包含一个 direct Chunk"
                )

        elif (
            self.evidence_mode
            == "parent_child"
        ):
            if len(self.gold_chunks) < 2:
                raise ValueError(
                    "parent_child 至少需要"
                    "两个 Chunk"
                )

            if (
                "direct" not in roles
                or "parent_context"
                not in roles
            ):
                raise ValueError(
                    "parent_child 必须同时包含 "
                    "direct 和 parent_context"
                )

        elif (
            self.evidence_mode
            == "multi_chunk"
        ):
            if len(self.gold_chunks) < 2:
                raise ValueError(
                    "multi_chunk 至少需要"
                    "两个 Chunk"
                )

            if "direct" not in roles:
                raise ValueError(
                    "multi_chunk 至少需要"
                    "一个 direct Chunk"
                )

        return self


class CompetitionGoldChunkDataset(
    BaseModel
):
    """
    与指定 Frozen Corpus 和候选报告绑定的
    Dev Gold Chunk 标注集。

    实际 JSON 属于赛题私有评测数据，
    不应提交到公开仓库。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    gold_version: Literal[
        "competition_dev_gold_chunks_v1"
    ] = "competition_dev_gold_chunks_v1"

    split_version: Literal[
        "competition_eval_split_v1"
    ] = "competition_eval_split_v1"

    corpus_id: str = Field(
        pattern=(
            r"^competition_corpus_"
            r"[0-9a-f]{16}$"
        ),
    )

    candidate_report_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    records: tuple[
        CompetitionGoldFactRecord,
        ...,
    ] = Field(
        min_length=1,
    )

    @model_validator(
        mode="after"
    )
    def validate_records(
        self,
    ) -> Self:
        record_keys = [
            (
                record.case_id,
                record.fact_index,
            )
            for record
            in self.records
        ]

        if (
            record_keys
            != sorted(record_keys)
        ):
            raise ValueError(
                "Gold records 必须按照 "
                "case_id、fact_index 排序"
            )

        if (
            len(record_keys)
            != len(set(record_keys))
        ):
            raise ValueError(
                "Gold records 包含重复的 "
                "case_id、fact_index"
            )

        return self