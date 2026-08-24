from __future__ import annotations

from dataclasses import dataclass

from app.schemas.competition import (
    CompetitionQaCase,
)
from app.schemas.competition_chunk import (
    CompetitionTextChunk,
)
from app.services.competition_retrieval_eval import (
    normalize_competition_evidence_text,
    split_competition_evidence_facts,
)


@dataclass(frozen=True, slots=True)
class CompetitionEvidenceFactCoverage:
    fact: str
    normalized_fact: str
    single_chunk_ids: tuple[str, ...]
    two_chunk_window_ids: tuple[
        tuple[str, str],
        ...,
    ]
    document_covered: bool

    @property
    def single_chunk_covered(self) -> bool:
        return bool(self.single_chunk_ids)

    @property
    def two_chunk_window_covered(self) -> bool:
        return (
            self.single_chunk_covered
            or bool(self.two_chunk_window_ids)
        )

    @property
    def boundary_only(self) -> bool:
        return (
            not self.single_chunk_covered
            and bool(self.two_chunk_window_ids)
        )

    @property
    def document_only(self) -> bool:
        return (
            self.document_covered
            and not self.two_chunk_window_covered
        )

    @property
    def missing(self) -> bool:
        return not self.document_covered


@dataclass(frozen=True, slots=True)
class CompetitionEvidenceCaseCoverage:
    case_id: str
    source_type: str
    qa_type: str
    difficulty: str
    expected_source_id: str
    source_chunk_count: int
    fact_coverages: tuple[
        CompetitionEvidenceFactCoverage,
        ...,
    ]

    @property
    def fact_count(self) -> int:
        return len(self.fact_coverages)

    @property
    def single_chunk_covered_fact_count(self) -> int:
        return sum(
            fact.single_chunk_covered
            for fact in self.fact_coverages
        )

    @property
    def two_chunk_covered_fact_count(self) -> int:
        return sum(
            fact.two_chunk_window_covered
            for fact in self.fact_coverages
        )

    @property
    def document_covered_fact_count(self) -> int:
        return sum(
            fact.document_covered
            for fact in self.fact_coverages
        )

    @property
    def boundary_only_fact_count(self) -> int:
        return sum(
            fact.boundary_only
            for fact in self.fact_coverages
        )

    @property
    def document_only_fact_count(self) -> int:
        return sum(
            fact.document_only
            for fact in self.fact_coverages
        )

    @property
    def missing_fact_count(self) -> int:
        return sum(
            fact.missing
            for fact in self.fact_coverages
        )

    @property
    def all_facts_single_chunk_covered(self) -> bool:
        return (
            self.single_chunk_covered_fact_count
            == self.fact_count
        )

    @property
    def all_facts_two_chunk_covered(self) -> bool:
        return (
            self.two_chunk_covered_fact_count
            == self.fact_count
        )

    @property
    def all_facts_document_covered(self) -> bool:
        return (
            self.document_covered_fact_count
            == self.fact_count
        )


@dataclass(frozen=True, slots=True)
class CompetitionEvidenceCoverageSummary:
    case_count: int
    fact_count: int
    single_chunk_covered_fact_count: int
    two_chunk_covered_fact_count: int
    document_covered_fact_count: int
    boundary_only_fact_count: int
    document_only_fact_count: int
    missing_fact_count: int
    all_facts_single_chunk_case_count: int
    all_facts_two_chunk_case_count: int
    all_facts_document_case_count: int

    @property
    def single_chunk_fact_coverage(self) -> float:
        return (
            self.single_chunk_covered_fact_count
            / self.fact_count
        )

    @property
    def two_chunk_fact_coverage(self) -> float:
        return (
            self.two_chunk_covered_fact_count
            / self.fact_count
        )

    @property
    def document_fact_coverage(self) -> float:
        return (
            self.document_covered_fact_count
            / self.fact_count
        )

    @property
    def all_facts_single_chunk_case_rate(self) -> float:
        return (
            self.all_facts_single_chunk_case_count
            / self.case_count
        )

    @property
    def all_facts_two_chunk_case_rate(self) -> float:
        return (
            self.all_facts_two_chunk_case_count
            / self.case_count
        )

    @property
    def all_facts_document_case_rate(self) -> float:
        return (
            self.all_facts_document_case_count
            / self.case_count
        )


def audit_competition_evidence_coverage_case(
    *,
    case: CompetitionQaCase,
    expected_source_id: str,
    chunks: tuple[
        CompetitionTextChunk,
        ...,
    ],
) -> CompetitionEvidenceCaseCoverage:
    if case.source_type not in {
        "word",
        "pdf",
    }:
        raise ValueError(
            "证据覆盖审计只支持 Word/PDF QA"
        )

    source_chunks = tuple(
        sorted(
            (
                chunk
                for chunk in chunks
                if chunk.source_id == expected_source_id
            ),
            key=lambda chunk: (
                chunk.chunk_index,
                chunk.chunk_id,
            ),
        )
    )

    if not source_chunks:
        raise ValueError(
            "Corpus 中没有正确来源的 Chunk: "
            f"{expected_source_id}"
        )

    chunk_indexes = tuple(
        chunk.chunk_index
        for chunk in source_chunks
    )

    if len(chunk_indexes) != len(set(chunk_indexes)):
        raise ValueError(
            "正确来源包含重复 chunk_index"
        )

    if any(
        chunk.source_type != case.source_type
        for chunk in source_chunks
    ):
        raise ValueError(
            "QA source_type 与 Corpus 来源不一致"
        )

    normalized_chunks = tuple(
        (
            chunk,
            normalize_competition_evidence_text(
                chunk.text
            ),
        )
        for chunk in source_chunks
    )

    two_chunk_windows = tuple(
        (
            (
                left_chunk.chunk_id,
                right_chunk.chunk_id,
            ),
            left_text + right_text,
        )
        for (
            left_chunk,
            left_text,
        ), (
            right_chunk,
            right_text,
        ) in zip(
            normalized_chunks,
            normalized_chunks[1:],
        )
    )

    normalized_document = "".join(
        normalized_text
        for _, normalized_text in normalized_chunks
    )

    fact_coverages = []

    for fact in split_competition_evidence_facts(
        case.evidence
    ):
        normalized_fact = (
            normalize_competition_evidence_text(
                fact
            )
        )

        if not normalized_fact:
            raise ValueError(
                "Evidence Fact 归一化后为空"
            )

        single_chunk_ids = tuple(
            chunk.chunk_id
            for chunk, normalized_text in normalized_chunks
            if normalized_fact in normalized_text
        )

        two_chunk_window_ids = tuple(
            window_ids
            for window_ids, normalized_text
            in two_chunk_windows
            if normalized_fact in normalized_text
        )

        fact_coverages.append(
            CompetitionEvidenceFactCoverage(
                fact=fact,
                normalized_fact=normalized_fact,
                single_chunk_ids=single_chunk_ids,
                two_chunk_window_ids=(
                    two_chunk_window_ids
                ),
                document_covered=(
                    normalized_fact
                    in normalized_document
                ),
            )
        )

    return CompetitionEvidenceCaseCoverage(
        case_id=case.case_id,
        source_type=case.source_type,
        qa_type=case.qa_type,
        difficulty=case.difficulty,
        expected_source_id=expected_source_id,
        source_chunk_count=len(source_chunks),
        fact_coverages=tuple(fact_coverages),
    )


def summarize_competition_evidence_coverage(
    results: tuple[
        CompetitionEvidenceCaseCoverage,
        ...,
    ],
) -> CompetitionEvidenceCoverageSummary:
    if not results:
        raise ValueError(
            "证据覆盖审计结果不能为空"
        )

    fact_count = sum(
        result.fact_count
        for result in results
    )

    if fact_count == 0:
        raise ValueError(
            "证据覆盖审计没有可统计事实"
        )

    return CompetitionEvidenceCoverageSummary(
        case_count=len(results),
        fact_count=fact_count,
        single_chunk_covered_fact_count=sum(
            result.single_chunk_covered_fact_count
            for result in results
        ),
        two_chunk_covered_fact_count=sum(
            result.two_chunk_covered_fact_count
            for result in results
        ),
        document_covered_fact_count=sum(
            result.document_covered_fact_count
            for result in results
        ),
        boundary_only_fact_count=sum(
            result.boundary_only_fact_count
            for result in results
        ),
        document_only_fact_count=sum(
            result.document_only_fact_count
            for result in results
        ),
        missing_fact_count=sum(
            result.missing_fact_count
            for result in results
        ),
        all_facts_single_chunk_case_count=sum(
            result.all_facts_single_chunk_covered
            for result in results
        ),
        all_facts_two_chunk_case_count=sum(
            result.all_facts_two_chunk_covered
            for result in results
        ),
        all_facts_document_case_count=sum(
            result.all_facts_document_covered
            for result in results
        ),
    )