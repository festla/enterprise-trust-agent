from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.schemas.competition_context_expansion import (
    CompetitionContextExpandedHit,
)
from app.schemas.competition_corpus import (
    CompetitionCorpusSourceRecord,
)
from app.schemas.competition_evidence import (
    CompetitionEvidence,
    CompetitionEvidenceBundle,
    CompetitionKnowledgeSource,
    CompetitionTextEvidenceLocation,
)


@dataclass(
    frozen=True,
    slots=True,
)
class CompetitionEvidenceAssemblyConfig:
    """
    Competition Evidence Pack V1。

    Retrieval 负责高召回；
    Evidence Assembly 负责限制最终送给
    Answer Generator 的证据规模。
    """

    max_evidences: int = 12
    max_chars: int = 6000

    def __post_init__(
        self,
    ) -> None:
        if self.max_evidences < 1:
            raise ValueError(
                "max_evidences 必须 >= 1"
            )

        if self.max_chars < 1:
            raise ValueError(
                "max_chars 必须 >= 1"
            )


def _build_source(
    record: CompetitionCorpusSourceRecord,
) -> CompetitionKnowledgeSource:
    return CompetitionKnowledgeSource(
        source_id=record.source_id,
        doc_id=record.doc_id,
        title=Path(
            record.relative_path
        ).stem,
        source_type=record.source_type,
        relative_path=record.relative_path,
        sha256=record.source_sha256,
    )


def _build_section(
    hit: CompetitionContextExpandedHit,
) -> str | None:
    chunk = hit.chunk

    parts = (
        *chunk.section_path,
        *chunk.item_path,
    )

    if not parts:
        return None

    return " > ".join(
        parts
    )


def _build_evidence(
    *,
    hit: CompetitionContextExpandedHit,
    source: CompetitionKnowledgeSource,
) -> CompetitionEvidence:
    chunk = hit.chunk

    location = (
        CompetitionTextEvidenceLocation(
            page=(
                chunk.page_start
                if chunk.source_type == "pdf"
                else None
            ),
            section=_build_section(
                hit
            ),
            article=chunk.article,
            paragraph_index=(
                chunk.paragraph_start_index
                if chunk.source_type == "word"
                else None
            ),
        )
    )

    evidence_type = (
        "table_range"
        if chunk.chunk_type == "table"
        else "text"
    )

    # chunk_id 本身已经是 Corpus 内稳定唯一 ID，
    # 直接作为 evidence_id，
    # 后续 Citation 可以无损回溯 Retrieval Chunk。
    return CompetitionEvidence(
        evidence_id=chunk.chunk_id,
        evidence_type=evidence_type,
        source=source,
        location=location,
        raw_content=chunk.text,
    )

def _order_hits_for_evidence_pack(
    hits: Sequence[
        CompetitionContextExpandedHit
    ],
) -> tuple[
    CompetitionContextExpandedHit,
    ...,
]:
    """
    Evidence Pack 不直接按照 Expanded Global Rank 截断。

    Structured Expansion 中一个 Seed 可能产生很多
    adjacent / item_child Chunk。如果简单取前 N 条，
    高排名 Seed 的 Context 会占满预算，使后面的
    Retrieval Seed 完全失去代表性。

    V1 策略：
        1. 按 seed_rank 建立候选桶；
        2. 每个桶内部优先：
             seed
             adjacent
             item hierarchy
        3. 各 seed_rank round-robin；
        4. 同一个 Chunk 只选择一次。

    这样可以在有限 Evidence Budget 下保留
    Top-K Retrieval Seeds 的覆盖面。
    """

    relation_priority = {
        "seed": 0,
        "previous_neighbor": 1,
        "next_neighbor": 2,
        "item_parent": 3,
        "item_child": 4,
    }

    seed_ranks = sorted(
        {
            origin.seed_rank
            for hit in hits
            for origin in hit.origins
        }
    )

    buckets: dict[
        int,
        list[
            CompetitionContextExpandedHit
        ],
    ] = {}

    for seed_rank in seed_ranks:
        candidates = []

        for hit in hits:
            matching_origins = [
                origin
                for origin in hit.origins
                if (
                    origin.seed_rank
                    == seed_rank
                )
            ]

            if not matching_origins:
                continue

            candidates.append(
                (
                    min(
                        relation_priority[
                            origin.relation
                        ]
                        for origin
                        in matching_origins
                    ),
                    min(
                        origin.distance
                        for origin
                        in matching_origins
                    ),
                    hit.rank,
                    hit,
                )
            )

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[3].chunk_id,
            )
        )

        buckets[seed_rank] = [
            item[3]
            for item in candidates
        ]

    ordered: list[
        CompetitionContextExpandedHit
    ] = []

    seen: set[str] = set()

    round_index = 0

    while True:
        added = False

        for seed_rank in seed_ranks:
            bucket = buckets[
                seed_rank
            ]

            if (
                round_index
                >= len(bucket)
            ):
                continue

            hit = bucket[
                round_index
            ]

            if hit.chunk_id in seen:
                continue

            seen.add(
                hit.chunk_id
            )

            ordered.append(
                hit
            )

            added = True

        if not added:
            # 当前轮可能全部都是重复 Chunk，
            # 但后面仍可能存在候选，
            # 因此只有所有 bucket 都耗尽才结束。
            if all(
                round_index + 1
                >= len(bucket)
                for bucket
                in buckets.values()
            ):
                break

        round_index += 1

        if all(
            round_index
            >= len(bucket)
            for bucket
            in buckets.values()
        ):
            break

    # 防御性补齐：
    # 如果某些 Hit 因多 Seed 重复归属没有进入，
    # 按原 Expanded Rank 加到尾部。
    for hit in sorted(
        hits,
        key=lambda item: (
            item.rank,
            item.chunk_id,
        ),
    ):
        if hit.chunk_id in seen:
            continue

        seen.add(
            hit.chunk_id
        )

        ordered.append(
            hit
        )

    return tuple(
        ordered
    )

def assemble_competition_evidence_bundle(
    *,
    hits: Sequence[
        CompetitionContextExpandedHit
    ],
    source_records: Sequence[
        CompetitionCorpusSourceRecord
    ],
    config: (
        CompetitionEvidenceAssemblyConfig
        | None
    ) = None,
) -> CompetitionEvidenceBundle:
    if not hits:
        raise ValueError(
            "hits 不能为空"
        )

    active_config = (
        config
        if config is not None
        else CompetitionEvidenceAssemblyConfig()
    )

    source_by_id = {
        record.source_id: record
        for record in source_records
    }

    if (
        len(source_by_id)
        != len(source_records)
    ):
        raise ValueError(
            "source_records 包含重复 source_id"
        )

    ordered_hits = (
        _order_hits_for_evidence_pack(
            hits
        )
    )

    #
    # 防御性去重。
    # Context Expansion 正常情况下已经唯一，
    # 这里仍保证 Evidence Pack 不重复。
    #
    unique_hits: list[
        CompetitionContextExpandedHit
    ] = []

    seen_chunk_ids: set[str] = set()

    for hit in ordered_hits:
        if hit.chunk_id in seen_chunk_ids:
            continue

        seen_chunk_ids.add(
            hit.chunk_id
        )

        unique_hits.append(
            hit
        )

    selected: list[
        CompetitionContextExpandedHit
    ] = []

    used_chars = 0

    for hit in unique_hits:
        if (
            len(selected)
            >= active_config.max_evidences
        ):
            break

        chunk_chars = len(
            hit.chunk.text
        )

        if (
            used_chars + chunk_chars
            > active_config.max_chars
        ):
            continue

        selected.append(
            hit
        )

        used_chars += chunk_chars

    #
    # 避免预算小于第一条 Chunk 时
    # 最终生成空 EvidenceBundle。
    #
    if not selected:
        selected.append(
            unique_hits[0]
        )

    source_cache: dict[
        str,
        CompetitionKnowledgeSource,
    ] = {}

    evidences: list[
        CompetitionEvidence
    ] = []

    for hit in selected:
        source_record = (
            source_by_id.get(
                hit.source_id
            )
        )

        if source_record is None:
            raise ValueError(
                "Evidence Hit 来源不存在于 "
                "Corpus Manifest："
                f"{hit.source_id}"
            )

        source = (
            source_cache.get(
                hit.source_id
            )
        )

        if source is None:
            source = _build_source(
                source_record
            )

            source_cache[
                hit.source_id
            ] = source

        evidences.append(
            _build_evidence(
                hit=hit,
                source=source,
            )
        )

    return CompetitionEvidenceBundle(
        evidences=tuple(
            evidences
        )
    )