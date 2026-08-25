from __future__ import annotations

from pathlib import Path

from app.services.competition_source_catalog import (
    build_competition_knowledge_source,
)

from app.schemas.competition import (
    CompetitionQuestion,
    CompetitionSourceRecord,
)
from app.schemas.competition_answer import (
    CompetitionFinalAnswer,
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

CompetitionExcelSolverResult = (
    CompetitionExcelLookupResult
    | CompetitionExcelCompareResult
    | CompetitionExcelCalculationResult
)


class CompetitionExcelEvidenceBuildError(
    RuntimeError
):
    pass


def build_competition_excel_evidence_bundle(
    *,
    question: CompetitionQuestion,
    source: CompetitionSourceRecord,
    result: CompetitionExcelSolverResult,
    attachments_root: Path,
) -> CompetitionEvidenceBundle:
    """
    Excel Solver Result
        ↓
    统一 CompetitionEvidenceBundle

    底层仍复用已有：
        Lookup Evidence Builder
        Compare Evidence Builder
        Calculation Evidence Builder
    """

    if isinstance(
        result,
        CompetitionExcelLookupResult,
    ):
        return (
            build_excel_lookup_evidence_bundle(
                question=question,
                source=source,
                result=result,
                attachments_root=(
                    attachments_root
                ),
            )
        )

    if isinstance(
        result,
        CompetitionExcelCompareResult,
    ):
        return (
            build_excel_compare_evidence_bundle(
                question=question,
                source=source,
                result=result,
                attachments_root=(
                    attachments_root
                ),
            )
        )

    if isinstance(
        result,
        CompetitionExcelCalculationResult,
    ):
        return (
            build_excel_calculation_evidence_bundle(
                question=question,
                source=source,
                result=result,
                attachments_root=(
                    attachments_root
                ),
            )
        )

    raise CompetitionExcelEvidenceBuildError(
        "未知 Excel Solver Result"
    )

def build_competition_excel_final_answer(
    *,
    question: CompetitionQuestion,
    result: CompetitionExcelSolverResult,
    evidence_bundle: CompetitionEvidenceBundle,
    solver_type: str,
) -> CompetitionFinalAnswer:
    """
    把确定性 Excel Solver 的输出
    转成统一 CompetitionFinalAnswer。

    Excel Solver 本身已经确定答案，
    因此这里不再调用 LLM。
    """

    answer_option = (
        result.answer_option
    )

    if answer_option is None:
        raise CompetitionExcelEvidenceBuildError(
            "Excel Solver "
            "无法唯一确定答案选项"
        )

    evidence_ids = tuple(
        evidence.evidence_id
        for evidence
        in evidence_bundle.evidences
    )

    if not evidence_ids:
        raise CompetitionExcelEvidenceBuildError(
            "Excel Answer "
            "缺少 Evidence"
        )

    citation_ids = tuple(
        f"E{index}"
        for index in range(
            1,
            len(evidence_ids) + 1,
        )
    )

    answer_text = (
        result.answer_text.strip()
    )

    if not answer_text:
        answer_text = (
            question.options[
                answer_option
            ]
        )

    return CompetitionFinalAnswer(
        case_id=(
            question.case_id
        ),
        answer=(
            answer_option
        ),
        answer_text=(
            answer_text
        ),
        citation_ids=(
            citation_ids
        ),
        evidence_ids=(
            evidence_ids
        ),
        generator_id=(
            "competition_excel_solver:"
            f"{solver_type}"
        ),
    )