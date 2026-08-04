from __future__ import annotations

from datetime import (
    date,
    datetime,
    timezone,
)

import pytest

from app.rag.evidence_context import (
    EvidenceSourceMismatchError,
    assess_financial_fact_evidence,
    build_evidence_context,
)
from app.rag.query_planning import (
    build_financial_fact_query_plan,
)
from app.schemas.enums import (
    ChunkStrategy,
    DocumentQualityGrade,
    PageMappingStatus,
    ReportType,
    Severity,
    StatementScope,
    StatementType,
)
from app.schemas.report import Report
from app.schemas.retrieval import RetrievalHit
from app.rag.answer_control import (
    AnswerControlSourceMismatchError,
    build_financial_fact_answer_packet,
)

from app.rag.answer_generation import (
    MissingInlineCitationError,
    UnauthorizedGeneratedCitationError,
    generate_financial_fact_answer,
)
from app.schemas.answer_generation import (
    GeneratedFinancialFactAnswer,
)

REPORT_ID = "midea_group_2024"

DOCUMENT_ID = (
    f"doc_{REPORT_ID}_{'a' * 24}"
)

CHUNK_DATASET_ID = (
    f"chunk_dataset_{REPORT_ID}_"
    f"{'b' * 24}"
)


def build_report() -> Report:
    timestamp = datetime(
        2025,
        4,
        1,
        tzinfo=timezone.utc,
    )

    return Report(
        report_id=REPORT_ID,
        company_id="midea_group",
        fiscal_year=2024,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        title=(
            "美的集团股份有限公司"
            "2024年年度报告"
        ),
        publication_date=date(
            2025,
            4,
            1,
        ),
        source_name="巨潮资讯网",
        quality_grade=(
            DocumentQualityGrade.A
        ),
        citation_risk=Severity.LOW,
        active_document_id=DOCUMENT_ID,
        created_at=timestamp,
        updated_at=timestamp,
    )


def build_plan():
    return build_financial_fact_query_plan(
        original_query=(
            "美的集团2024年营业收入是多少？"
        ),
        metric_name="营业收入",
        fiscal_year=2024,
        company_id="midea_group",
        report_id=REPORT_ID,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        statement_type=(
            StatementType.INCOME_STATEMENT
        ),
        statement_scope=(
            StatementScope.CONSOLIDATED
        ),
    )


def build_hit(
    *,
    rank: int,
    pdf_page: int,
    text: str,
    chunk_suffix: int | None = None,
) -> RetrievalHit:
    suffix = (
        rank
        if chunk_suffix is None
        else chunk_suffix
    )

    return RetrievalHit(
        rank=rank,
        score=max(
            0.1,
            0.9 - rank * 0.05,
        ),
        chunk_id=(
            f"chunk_{REPORT_ID}_"
            f"{suffix:024x}"
        ),
        chunk_dataset_id=(
            CHUNK_DATASET_ID
        ),
        company_id="midea_group",
        report_id=REPORT_ID,
        fiscal_year=2024,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        document_id=DOCUMENT_ID,
        page_id=(
            f"{DOCUMENT_ID}_page_"
            f"{pdf_page:04d}"
        ),
        pdf_page=pdf_page,
        printed_page=pdf_page - 1,
        mapping_status=(
            PageMappingStatus.MAPPED
        ),
        chunk_index=rank - 1,
        strategy=(
            ChunkStrategy.FIXED_LENGTH
        ),
        source_start_char=0,
        source_end_char=len(text),
        section_path=(),
        text=text,
    )


def test_build_traceable_context() -> None:
    context = build_evidence_context(
        report=build_report(),
        plan=build_plan(),
        hits=(
            build_hit(
                rank=1,
                pdf_page=158,
                text=(
                    "合并及公司利润表 "
                    "营业收入 407,149,600"
                ),
            ),
            build_hit(
                rank=2,
                pdf_page=247,
                text="营业收入附注",
            ),
        ),
        max_hits=2,
        max_chars=2000,
    )

    assert len(context.items) == 2
    assert (
        context.items[0]
        .citation.citation_id
        == "E1"
    )
    assert (
        context.items[0]
        .citation.pdf_page
        == 158
    )
    assert (
        context.used_chunk_ids[0]
        == context.items[0]
        .citation.chunk_id
    )
    assert "[E1]" in context.context_text
    assert "PDF第158页" in (
        context.context_text
    )


def test_deduplicate_chunk_ids() -> None:
    first = build_hit(
        rank=1,
        pdf_page=158,
        text=(
            "合并利润表 "
            "营业收入 407,149,600"
        ),
        chunk_suffix=1,
    )

    duplicate = build_hit(
        rank=2,
        pdf_page=158,
        text=first.text,
        chunk_suffix=1,
    )

    context = build_evidence_context(
        report=build_report(),
        plan=build_plan(),
        hits=(first, duplicate),
    )

    assert len(context.items) == 1
    assert context.duplicate_hit_count == 1


def test_reject_mixed_report_source(
) -> None:
    hit = build_hit(
        rank=1,
        pdf_page=158,
        text="营业收入 407,149,600",
    )

    mixed_hit = hit.model_copy(
        update={
            "report_id": "other_company_2024",
        }
    )

    with pytest.raises(
        EvidenceSourceMismatchError,
        match="report_id",
    ):
        build_evidence_context(
            report=build_report(),
            plan=build_plan(),
            hits=(mixed_hit,),
        )


def test_context_respects_char_budget(
) -> None:
    long_text = (
        "合并利润表 营业收入 407,149,600 "
        * 100
    )

    context = build_evidence_context(
        report=build_report(),
        plan=build_plan(),
        hits=(
            build_hit(
                rank=1,
                pdf_page=158,
                text=long_text,
            ),
        ),
        max_chars=300,
    )

    assert context.used_chars <= 300
    assert context.truncated is True
    assert (
        context.items[0].text_truncated
        is True
    )


def test_ready_for_generation() -> None:
    context = build_evidence_context(
        report=build_report(),
        plan=build_plan(),
        hits=(
            build_hit(
                rank=1,
                pdf_page=158,
                text=(
                    "2024年度合并及公司利润表 "
                    "营业收入 407,149,600 千元"
                ),
            ),
        ),
    )

    decision = (
        assess_financial_fact_evidence(
            plan=build_plan(),
            context=context,
        )
    )

    assert (
        decision.status
        == "ready_for_generation"
    )

    assert (
        decision.supporting_citation_ids
        == ("E1",)
    )


def test_reject_split_evidence() -> None:
    context = build_evidence_context(
        report=build_report(),
        plan=build_plan(),
        hits=(
            build_hit(
                rank=1,
                pdf_page=158,
                text=(
                    "2024年度合并利润表"
                ),
            ),
            build_hit(
                rank=2,
                pdf_page=247,
                text=(
                    "营业收入 "
                    "407,149,600 千元"
                ),
            ),
        ),
    )

    decision = (
        assess_financial_fact_evidence(
            plan=build_plan(),
            context=context,
        )
    )

    assert (
        decision.status
        == "insufficient_evidence"
    )

    assert (
        decision.supporting_citation_ids
        == ()
    )


def test_no_retrieval_hits() -> None:
    context = build_evidence_context(
        report=build_report(),
        plan=build_plan(),
        hits=(),
    )

    decision = (
        assess_financial_fact_evidence(
            plan=build_plan(),
            context=context,
        )
    )

    assert (
        decision.status
        == "no_retrieval_hits"
    )

    assert context.context_text == ""


def build_inventory_plan():
    return build_financial_fact_query_plan(
        original_query=(
            "海信家电2024年末合并口径的"
            "存货是多少？"
        ),
        metric_name="存货",
        fiscal_year=2024,
        company_id="midea_group",
        report_id=REPORT_ID,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        statement_type=(
            StatementType.BALANCE_SHEET
        ),
        statement_scope=(
            StatementScope.CONSOLIDATED
        ),
    )


def test_reject_inventory_policy_as_value(
) -> None:
    plan = build_inventory_plan()

    context = build_evidence_context(
        report=build_report(),
        plan=plan,
        hits=(
            build_hit(
                rank=1,
                pdf_page=138,
                text=(
                    "财务报表附注 "
                    "资产负债表日，存货按照成本与"
                    "可变现净值孰低计量。"
                    "13. 合同资产与合同负债 "
                    "14. 与合同成本有关的资产"
                ),
            ),
        ),
    )

    decision = (
        assess_financial_fact_evidence(
            plan=plan,
            context=context,
        )
    )

    assert (
        decision.status
        == "insufficient_evidence"
    )


def test_inventory_balance_sheet_row_ready(
) -> None:
    plan = build_inventory_plan()

    context = build_evidence_context(
        report=build_report(),
        plan=plan,
        hits=(
            build_hit(
                rank=1,
                pdf_page=112,
                text=(
                    "1、合并资产负债表 "
                    "2024年12月31日 单位：元 "
                    "流动资产： "
                    "存货 7,566,932,954.39 "
                    "6,774,603,438.00"
                ),
            ),
        ),
    )

    decision = (
        assess_financial_fact_evidence(
            plan=plan,
            context=context,
        )
    )

    assert (
        decision.status
        == "ready_for_generation"
    )

    assert (
        decision.supporting_citation_ids
        == ("E1",)
    )



def test_reject_value_far_from_metric(
) -> None:
    plan = build_inventory_plan()

    text = (
        "1、合并资产负债表 "
        "存货按照有关会计政策进行确认和计量。"
        + "无关说明" * 80
        + "其他应付款 5,300,124,294.55"
    )

    context = build_evidence_context(
        report=build_report(),
        plan=plan,
        hits=(
            build_hit(
                rank=1,
                pdf_page=112,
                text=text,
            ),
        ),
    )

    decision = (
        assess_financial_fact_evidence(
            plan=plan,
            context=context,
        )
    )

    assert (
        decision.status
        == "insufficient_evidence"
    )


def test_answer_packet_uses_only_supporting_items(
) -> None:
    plan = build_plan()

    context = build_evidence_context(
        report=build_report(),
        plan=plan,
        hits=(
            build_hit(
                rank=1,
                pdf_page=158,
                text=(
                    "2024年度合并及公司利润表 "
                    "营业收入 407,149,600 千元"
                ),
            ),
            build_hit(
                rank=2,
                pdf_page=160,
                text=(
                    "2024年度合并现金流量表 "
                    "经营活动现金流入小计 "
                    "412,775,133 千元"
                ),
            ),
        ),
    )

    readiness = (
        assess_financial_fact_evidence(
            plan=plan,
            context=context,
        )
    )

    packet = (
        build_financial_fact_answer_packet(
            plan=plan,
            readiness=readiness,
        )
    )

    assert (
        packet.status
        == "ready_for_generation"
    )

    assert packet.action == "call_model"

    assert (
        packet.supporting_citation_ids
        == ("E1",)
    )

    assert len(packet.supporting_items) == 1

    assert "407,149,600" in (
        packet.generation_context
    )

    assert "412,775,133" not in (
        packet.generation_context
    )


def test_answer_packet_refuses_insufficient_evidence(
) -> None:
    plan = build_inventory_plan()

    context = build_evidence_context(
        report=build_report(),
        plan=plan,
        hits=(
            build_hit(
                rank=1,
                pdf_page=138,
                text=(
                    "财务报表附注。"
                    "资产负债表日，存货按照成本与"
                    "可变现净值孰低计量。"
                ),
            ),
        ),
    )

    readiness = (
        assess_financial_fact_evidence(
            plan=plan,
            context=context,
        )
    )

    packet = (
        build_financial_fact_answer_packet(
            plan=plan,
            readiness=readiness,
        )
    )

    assert packet.status == "refused"

    assert (
        packet.action
        == "return_refusal"
    )

    assert packet.supporting_items == ()

    assert packet.generation_context == ""

    assert packet.refusal_reason is not None


def test_answer_packet_refuses_no_hits(
) -> None:
    plan = build_plan()

    context = build_evidence_context(
        report=build_report(),
        plan=plan,
        hits=(),
    )

    readiness = (
        assess_financial_fact_evidence(
            plan=plan,
            context=context,
        )
    )

    packet = (
        build_financial_fact_answer_packet(
            plan=plan,
            readiness=readiness,
        )
    )

    assert packet.status == "refused"

    assert (
        packet.action
        == "return_refusal"
    )

    assert "未检索到" in packet.message

    assert packet.generation_context == ""


def test_answer_packet_rejects_plan_mismatch(
) -> None:
    original_plan = build_plan()

    context = build_evidence_context(
        report=build_report(),
        plan=original_plan,
        hits=(
            build_hit(
                rank=1,
                pdf_page=158,
                text=(
                    "2024年度合并利润表 "
                    "营业收入 407,149,600"
                ),
            ),
        ),
    )

    readiness = (
        assess_financial_fact_evidence(
            plan=original_plan,
            context=context,
        )
    )

    mismatched_plan = (
        original_plan.model_copy(
            update={
                "original_query": (
                    "另一个不相干的问题"
                ),
            }
        )
    )

    with pytest.raises(
        AnswerControlSourceMismatchError,
        match="original_query",
    ):
        build_financial_fact_answer_packet(
            plan=mismatched_plan,
            readiness=readiness,
        )

class FakeAnswerProvider:
    def __init__(
        self,
        *,
        answer_text: str,
        citation_ids: tuple[str, ...],
    ) -> None:
        self._answer_text = answer_text
        self._citation_ids = citation_ids

        self.call_count = 0
        self.last_generation_context = None

    @property
    def provider_id(self) -> str:
        return "fake_answer_provider_v1"

    def generate(
        self,
        *,
        question: str,
        metric_name: str,
        generation_context: str,
        allowed_citation_ids: tuple[
            str,
            ...,
        ],
    ) -> GeneratedFinancialFactAnswer:
        self.call_count += 1

        self.last_generation_context = (
            generation_context
        )

        return GeneratedFinancialFactAnswer(
            answer_text=self._answer_text,
            citation_ids=(
                self._citation_ids
            ),
        )

def build_ready_answer_packet():
    plan = build_plan()

    context = build_evidence_context(
        report=build_report(),
        plan=plan,
        hits=(
            build_hit(
                rank=1,
                pdf_page=158,
                text=(
                    "2024年度合并及公司利润表 "
                    "营业收入 407,149,600 千元"
                ),
            ),
            build_hit(
                rank=2,
                pdf_page=160,
                text=(
                    "2024年度合并现金流量表 "
                    "经营活动现金流入小计 "
                    "412,775,133 千元"
                ),
            ),
        ),
    )

    readiness = (
        assess_financial_fact_evidence(
            plan=plan,
            context=context,
        )
    )

    packet = (
        build_financial_fact_answer_packet(
            plan=plan,
            readiness=readiness,
        )
    )

    return packet

def test_generate_answer_with_allowed_citation(
) -> None:
    packet = build_ready_answer_packet()

    provider = FakeAnswerProvider(
        answer_text=(
            "美的集团2024年度营业收入为"
            "407,149,600千元。[E1]"
        ),
        citation_ids=("E1",),
    )

    result = generate_financial_fact_answer(
        packet=packet,
        provider=provider,
    )

    assert result.status == "answered"

    assert result.citation_ids == (
        "E1",
    )

    assert provider.call_count == 1

    assert "407,149,600" in (
        provider.last_generation_context
    )

    assert "412,775,133" not in (
        provider.last_generation_context
    )

def test_refusal_does_not_call_provider(
) -> None:
    plan = build_inventory_plan()

    context = build_evidence_context(
        report=build_report(),
        plan=plan,
        hits=(
            build_hit(
                rank=1,
                pdf_page=138,
                text=(
                    "财务报表附注。"
                    "资产负债表日，存货按照"
                    "成本与可变现净值孰低计量。"
                ),
            ),
        ),
    )

    readiness = (
        assess_financial_fact_evidence(
            plan=plan,
            context=context,
        )
    )

    packet = (
        build_financial_fact_answer_packet(
            plan=plan,
            readiness=readiness,
        )
    )

    provider = FakeAnswerProvider(
        answer_text="不应被调用[E1]",
        citation_ids=("E1",),
    )

    result = generate_financial_fact_answer(
        packet=packet,
        provider=provider,
    )

    assert result.status == "refused"

    assert provider.call_count == 0

    assert result.citations == ()

def test_reject_unauthorized_generated_citation(
) -> None:
    packet = build_ready_answer_packet()

    provider = FakeAnswerProvider(
        answer_text="答案来自其他证据。[E2]",
        citation_ids=("E2",),
    )

    with pytest.raises(
        UnauthorizedGeneratedCitationError,
        match="E2",
    ):
        generate_financial_fact_answer(
            packet=packet,
            provider=provider,
        )

def test_reject_missing_inline_citation(
) -> None:
    packet = build_ready_answer_packet()

    provider = FakeAnswerProvider(
        answer_text=(
            "美的集团2024年度营业收入为"
            "407,149,600千元。"
        ),
        citation_ids=("E1",),
    )

    with pytest.raises(
        MissingInlineCitationError,
        match="正文",
    ):
        generate_financial_fact_answer(
            packet=packet,
            provider=provider,
        )