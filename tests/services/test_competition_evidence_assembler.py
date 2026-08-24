import hashlib

import pytest

from app.schemas.competition_chunk import (
    CompetitionChunkSourceSpan,
    CompetitionTextChunk,
)
from app.schemas.competition_context_expansion import (
    CompetitionContextExpandedHit,
    CompetitionContextExpansionOrigin,
)
from app.schemas.competition_corpus import (
    CompetitionCorpusSourceRecord,
)
from app.services.competition_evidence_assembler import (
    CompetitionEvidenceAssemblyConfig,
    _order_hits_for_evidence_pack,
    assemble_competition_evidence_bundle,
)

def _chunk(
    *,
    index: int,
    text: str,
) -> CompetitionTextChunk:
    return CompetitionTextChunk(
        chunk_id=f"chunk:test:{index:05d}",
        source_id="src_test",
        doc_id="doc_test",
        source_type="word",
        chunk_index=index,
        chunk_type="text",
        source_spans=(
            CompetitionChunkSourceSpan(
                block_id=f"block:{index}",
                block_index=index,
                start_char=0,
                end_char=len(text),
            ),
        ),
        text=text,
        char_count=len(text),
        text_sha256=hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest(),
        section_path=(
            "资本工具合格标准",
        ),
        item_path=(
            f"第{index + 1}项",
        ),
        paragraph_start_index=index,
        paragraph_end_index=index,
    )


def _hit(
    *,
    index: int,
    text: str,
) -> CompetitionContextExpandedHit:
    chunk = _chunk(
        index=index,
        text=text,
    )

    return CompetitionContextExpandedHit(
        rank=index + 1,
        effective_seed_rank=index + 1,
        origins=(
            CompetitionContextExpansionOrigin(
                seed_chunk_id=chunk.chunk_id,
                seed_rank=index + 1,
                trigger_chunk_id=chunk.chunk_id,
                relation="seed",
                distance=0,
            ),
        ),
        chunk=chunk,
    )


def _source_record(
) -> CompetitionCorpusSourceRecord:
    return CompetitionCorpusSourceRecord(
        source_id="src_test",
        doc_id="doc_test",
        source_type="word",
        relative_path="rules/test.docx",
        source_sha256="0" * 64,
        chunk_count=3,
        chunk_char_count=30,
        text_chunk_count=3,
        table_chunk_count=0,
    )


def test_build_evidence_bundle(
) -> None:
    hits = (
        _hit(
            index=0,
            text="第一条证据",
        ),
        _hit(
            index=1,
            text="第二条证据",
        ),
    )

    bundle = (
        assemble_competition_evidence_bundle(
            hits=hits,
            source_records=(
                _source_record(),
            ),
        )
    )

    assert len(
        bundle.evidences
    ) == 2

    first = bundle.evidences[0]

    assert (
        first.evidence_id
        == hits[0].chunk_id
    )

    assert (
        first.source.source_id
        == "src_test"
    )

    assert (
        first.raw_content
        == "第一条证据"
    )

    assert (
        first.location.section
        == (
            "资本工具合格标准"
            " > 第1项"
        )
    )


def test_respects_evidence_limit(
) -> None:
    hits = (
        _hit(
            index=0,
            text="证据一",
        ),
        _hit(
            index=1,
            text="证据二",
        ),
        _hit(
            index=2,
            text="证据三",
        ),
    )

    bundle = (
        assemble_competition_evidence_bundle(
            hits=hits,
            source_records=(
                _source_record(),
            ),
            config=(
                CompetitionEvidenceAssemblyConfig(
                    max_evidences=2,
                    max_chars=1000,
                )
            ),
        )
    )

    assert len(
        bundle.evidences
    ) == 2

    assert (
        bundle.evidences[0]
        .evidence_id
        == hits[0].chunk_id
    )

    assert (
        bundle.evidences[1]
        .evidence_id
        == hits[1].chunk_id
    )


def test_respects_character_budget(
) -> None:
    hits = (
        _hit(
            index=0,
            text="12345",
        ),
        _hit(
            index=1,
            text="67890",
        ),
    )

    bundle = (
        assemble_competition_evidence_bundle(
            hits=hits,
            source_records=(
                _source_record(),
            ),
            config=(
                CompetitionEvidenceAssemblyConfig(
                    max_evidences=10,
                    max_chars=5,
                )
            ),
        )
    )

    assert len(
        bundle.evidences
    ) == 1

    assert (
        bundle.evidences[0]
        .raw_content
        == "12345"
    )


def test_missing_source_rejected(
) -> None:
    hit = _hit(
        index=0,
        text="测试证据",
    )

    with pytest.raises(
        ValueError,
        match="Corpus Manifest",
    ):
        assemble_competition_evidence_bundle(
            hits=(hit,),
            source_records=(),
        )


def test_invalid_budget_rejected(
) -> None:
    with pytest.raises(
        ValueError
    ):
        CompetitionEvidenceAssemblyConfig(
            max_evidences=0,
        )

def test_stratified_order_covers_seed_ranks(
) -> None:
    hit_1 = _hit(
        index=0,
        text="seed1",
    )

    hit_2 = _hit(
        index=1,
        text="seed2",
    )

    ordered = (
        _order_hits_for_evidence_pack(
            (
                hit_1,
                hit_2,
            )
        )
    )

    assert {
        hit.chunk_id
        for hit in ordered
    } == {
        hit_1.chunk_id,
        hit_2.chunk_id,
    }


def test_stratified_order_is_deterministic(
) -> None:
    hits = (
        _hit(
            index=0,
            text="A",
        ),
        _hit(
            index=1,
            text="B",
        ),
        _hit(
            index=2,
            text="C",
        ),
    )

    first = (
        _order_hits_for_evidence_pack(
            hits
        )
    )

    second = (
        _order_hits_for_evidence_pack(
            hits
        )
    )

    assert [
        hit.chunk_id
        for hit in first
    ] == [
        hit.chunk_id
        for hit in second
    ]