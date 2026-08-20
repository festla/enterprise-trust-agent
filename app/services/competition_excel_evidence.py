from __future__ import annotations

from pathlib import Path

from app.services.competition_source_catalog import (
    build_competition_knowledge_source,
)

from app.schemas.competition import (
    CompetitionQuestion,
    CompetitionSourceRecord,
)
from app.schemas.competition_evidence import (
    CompetitionCalculationTrace,
    CompetitionEvidence,
    CompetitionEvidenceBundle,
    CompetitionExcelEvidenceLocation,
    CompetitionKnowledgeSource,
)
from app.schemas.competition_excel_solver import (
    CompetitionExcelCalculationResult,
    CompetitionExcelCompareResult,
    CompetitionExcelLookupResult,
)



def _evidence_id(
    *,
    source_id: str,
    sheet_name: str,
    coordinate: str,
) -> str:
    """
    同一个 Cell 应稳定生成同一个 Evidence ID。
    """

    return (
        f"ev:{source_id}:"
        f"{sheet_name}:"
        f"{coordinate}"
    )


def build_excel_lookup_evidence_bundle(
    *,
    question: CompetitionQuestion,
    source: CompetitionSourceRecord,
    result: CompetitionExcelLookupResult,
    attachments_root: Path,
) -> CompetitionEvidenceBundle:
    knowledge_source = (
        build_competition_knowledge_source(
            question=question,
            source=source,
            attachments_root=attachments_root,
        )
    )

    evidence = CompetitionEvidence(
        evidence_id=_evidence_id(
            source_id=source.source_id,
            sheet_name=result.evidence.sheet_name,
            coordinate=result.evidence.value_coordinate,
        ),
        evidence_type="table_cell",
        source=knowledge_source,
        location=(
            CompetitionExcelEvidenceLocation(
                sheet_name=result.evidence.sheet_name,
                cell=result.evidence.value_coordinate,
                row_label=result.evidence.row_label,
                column_label=result.evidence.column_label,
                unit=None,
            )
        ),
        raw_content=(
            f"{result.evidence.row_label or ''}"
            f"｜"
            f"{result.evidence.column_label or ''}"
            f"="
            f"{result.answer_text}"
        ),
        numeric_value=result.evidence.numeric_value,
        display_value=result.answer_text,
    )

    return CompetitionEvidenceBundle(
        evidences=(evidence,),
    )


def build_excel_compare_evidence_bundle(
    *,
    question: CompetitionQuestion,
    source: CompetitionSourceRecord,
    result: CompetitionExcelCompareResult,
    attachments_root: Path,
) -> CompetitionEvidenceBundle:
    knowledge_source = (
        build_competition_knowledge_source(
            question=question,
            source=source,
            attachments_root=attachments_root,
        )
    )

    evidences = []

    for item in result.items:
        evidence_id = _evidence_id(
            source_id=source.source_id,
            sheet_name=result.sheet_name,
            coordinate=item.value_coordinate,
        )

        context = " / ".join(
            item.context_labels
        )

        raw_content = (
            f"{item.label_text}"
            f"={item.value_text}"
        )

        if context:
            raw_content += (
                f"｜context={context}"
            )

        evidences.append(
            CompetitionEvidence(
                evidence_id=evidence_id,
                evidence_type="table_cell",
                source=knowledge_source,
                location=(
                    CompetitionExcelEvidenceLocation(
                        sheet_name=result.sheet_name,
                        cell=item.value_coordinate,
                        row_label=item.label_text,
                        column_label=(
                            context
                            if context
                            else None
                        ),
                        unit=None,
                    )
                ),
                raw_content=raw_content,
                numeric_value=item.numeric_value,
                display_value=item.value_text,
            )
        )

    return CompetitionEvidenceBundle(
        evidences=tuple(evidences),
    )


def build_excel_calculation_evidence_bundle(
    *,
    question: CompetitionQuestion,
    source: CompetitionSourceRecord,
    result: CompetitionExcelCalculationResult,
    attachments_root: Path,
) -> CompetitionEvidenceBundle:
    knowledge_source = (
        build_competition_knowledge_source(
            question=question,
            source=source,
            attachments_root=attachments_root,
        )
    )

    start_id = _evidence_id(
        source_id=source.source_id,
        sheet_name=result.sheet_name,
        coordinate=result.start.coordinate,
    )

    end_id = _evidence_id(
        source_id=source.source_id,
        sheet_name=result.sheet_name,
        coordinate=result.end.coordinate,
    )

    start = CompetitionEvidence(
        evidence_id=start_id,
        evidence_type="table_cell",
        source=knowledge_source,
        location=(
            CompetitionExcelEvidenceLocation(
                sheet_name=result.sheet_name,
                cell=result.start.coordinate,
                row_label=result.entity_text,
                column_label=result.start.scope_text,
                unit=None,
            )
        ),
        raw_content=(
            f"{result.entity_text}"
            f"｜{result.start.scope_text}"
            f"={result.start.numeric_value}"
        ),
        numeric_value=result.start.numeric_value,
        display_value=str(
            result.start.numeric_value
        ),
    )

    end = CompetitionEvidence(
        evidence_id=end_id,
        evidence_type="table_cell",
        source=knowledge_source,
        location=(
            CompetitionExcelEvidenceLocation(
                sheet_name=result.sheet_name,
                cell=result.end.coordinate,
                row_label=result.entity_text,
                column_label=result.end.scope_text,
                unit=None,
            )
        ),
        raw_content=(
            f"{result.entity_text}"
            f"｜{result.end.scope_text}"
            f"={result.end.numeric_value}"
        ),
        numeric_value=result.end.numeric_value,
        display_value=str(
            result.end.numeric_value
        ),
    )

    calculation = CompetitionCalculationTrace(
        operation=result.operation,
        formula=result.formula,
        input_evidence_ids=(
            start_id,
            end_id,
        ),
        result=result.result,
        resolution_mode=(
            result.resolution_mode
        ),
    )

    return CompetitionEvidenceBundle(
        evidences=(
            start,
            end,
        ),
        calculation=calculation,
    )