from __future__ import annotations

from decimal import Decimal

import pytest

from app.schemas.agent_runtime import (
    AgentState,
    CitationRecord,
)
from app.schemas.company import (
    Company,
)
from app.schemas.complex_plan_eval_result import (
    ComplexCalculationTrace,
)
from app.schemas.enums import (
    StatementScope,
    StatementType,
    UnitCode,
    ValidationStatus,
)
from app.schemas.evidence import (
    SourceEvidence,
)
from app.schemas.financial_fact import (
    FinancialFact,
)
from app.schemas.metric import (
    FinancialMetric,
)
from app.schemas.tool_registry import (
    RetrievedDocument,
)
from app.schemas.trust import (
    AnswerDraft,
    Claim,
    ClaimSupport,
)
from app.services.registry import (
    RegistryBundle,
)
from app.services.runtime_trust_verifier import (
    RuntimeTrustVerificationError,
    RuntimeTrustVerifier,
)


# ============================================================
# Test Fixtures / Builders
# ============================================================


def _build_bundle() -> RegistryBundle:
    """构造 RuntimeTrustVerifier 测试所需的最小 Registry。"""

    bundle = RegistryBundle()

    # --------------------------------------------------------
    # Company
    # --------------------------------------------------------

    bundle.companies.add(
        Company.model_construct(
            company_id="midea_group",
            short_name_cn="美的集团",
        )
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    bundle.metrics.add(
        FinancialMetric.model_construct(
            metric_id="revenue",
            display_name_cn="营业收入",
        )
    )

    bundle.metrics.add(
        FinancialMetric.model_construct(
            metric_id="operating_cost",
            display_name_cn="营业成本",
        )
    )

    bundle.metrics.add(
        FinancialMetric.model_construct(
            metric_id=(
                "gross_profit_margin"
            ),
            display_name_cn="毛利率",
        )
    )

    # --------------------------------------------------------
    # Financial Facts + Evidences
    # --------------------------------------------------------

    _add_fact(
        bundle,
        year=2024,
        metric_id="revenue",
        value="407149600000",
    )

    _add_fact(
        bundle,
        year=2024,
        metric_id="operating_cost",
        value="322560000000",
    )

    return bundle


def _add_fact(
    bundle: RegistryBundle,
    *,
    year: int,
    metric_id: str,
    value: str,
) -> None:
    """向 Registry 同时添加 Fact 和其 Primary Evidence。"""

    fact_id = (
        f"fact_midea_group_"
        f"{year}_{metric_id}"
    )

    evidence_id = (
        f"evidence_midea_group_"
        f"{year}_{metric_id}"
    )

    report_id = (
        f"midea_group_{year}"
    )

    chunk_id = (
        f"chunk_midea_{year}_"
        f"{metric_id}"
    )

    fact = FinancialFact.model_construct(
        fact_id=fact_id,
        company_id="midea_group",
        report_id=report_id,
        metric_id=metric_id,
        fiscal_year=year,
        statement_type=(
            StatementType
            .INCOME_STATEMENT
        ),
        statement_scope=(
            StatementScope
            .CONSOLIDATED
        ),
        normalized_value=(
            Decimal(value)
        ),
        normalized_unit=(
            UnitCode.CNY
        ),
        primary_evidence_id=(
            evidence_id
        ),
        validation_status=(
            ValidationStatus
            .VERIFIED
        ),
    )

    evidence = (
        SourceEvidence.model_construct(
            evidence_id=evidence_id,
            report_id=report_id,
            document_id=(
                f"document_midea_{year}"
            ),
            page_id=(
                f"page_midea_{year}_158"
            ),
            chunk_id=chunk_id,
            statement_type=(
                StatementType
                .INCOME_STATEMENT
            ),
            statement_scope=(
                StatementScope
                .CONSOLIDATED
            ),
            pdf_page=158,
            printed_page=157,
            evidence_text=(
                f"{year}年"
                f"{metric_id}"
                f"为{value}元。"
            ),
            validation_status=(
                ValidationStatus
                .VERIFIED
            ),
        )
    )

    bundle.financial_facts.add(
        fact
    )

    bundle.evidences.add(
        evidence
    )


def _citation_for_fact(
    bundle: RegistryBundle,
    *,
    fact_id: str,
    citation_id: str,
) -> CitationRecord:
    """根据 FinancialFact 构造正确 Citation。"""

    fact = (
        bundle
        .financial_facts
        .require(
            fact_id
        )
    )

    evidence = (
        bundle
        .evidences
        .require(
            fact.primary_evidence_id
        )
    )

    return CitationRecord(
        citation_id=citation_id,
        report_id=(
            evidence.report_id
        ),
        pdf_page=(
            evidence.pdf_page
        ),
        printed_page=157,
        evidence_id=(
            evidence.evidence_id
        ),
        chunk_id=(
            evidence.chunk_id
        ),
        text_excerpt=(
            evidence
            .evidence_text
            .strip()
        ),
    )


# ============================================================
# Financial Fact State
# ============================================================


def _build_financial_fact_state(
    bundle: RegistryBundle,
) -> AgentState:
    """构造一条完全正确的营业收入 AnswerDraft。"""

    fact_id = (
        "fact_midea_group_"
        "2024_revenue"
    )

    citation = _citation_for_fact(
        bundle,
        fact_id=fact_id,
        citation_id="citation_1",
    )

    claim = Claim(
        claim_id=(
            "claim_fact_midea_group_"
            "2024_revenue"
        ),
        claim_type="financial_fact",
        claim_text=(
            "美的集团2024年"
            "营业收入为"
            "407149600000元"
        ),
        support=ClaimSupport(
            fact_ids=(
                fact_id,
            ),
            citation_ids=(
                citation.citation_id,
            ),
        ),
        confidence=1.0,
    )

    draft = AnswerDraft(
        draft_id="draft_fact_test",
        draft_type="financial",
        claims=(
            claim,
        ),
    )

    # --------------------------------------------------------
    # RuntimeTrustVerifier 实际只依赖：
    #
    # answer_draft
    # citations
    # calculation_traces
    # retrieved_documents
    #
    # 所以这里使用 model_construct 构造最小 State，
    # 避免单元测试被 Runtime 其他字段干扰。
    # --------------------------------------------------------

    return AgentState.model_construct(
        answer_draft=draft,
        citations=(
            citation,
        ),
        calculation_traces=(),
        retrieved_documents=(),
    )


# ============================================================
# Financial Calculation State
# ============================================================


def _build_calculation_state(
    bundle: RegistryBundle,
) -> AgentState:
    """构造一条正确的毛利率 Calculation Claim。"""

    revenue_fact_id = (
        "fact_midea_group_"
        "2024_revenue"
    )

    cost_fact_id = (
        "fact_midea_group_"
        "2024_operating_cost"
    )

    revenue_citation = (
        _citation_for_fact(
            bundle,
            fact_id=(
                revenue_fact_id
            ),
            citation_id=(
                "citation_1"
            ),
        )
    )

    cost_citation = (
        _citation_for_fact(
            bundle,
            fact_id=(
                cost_fact_id
            ),
            citation_id=(
                "citation_2"
            ),
        )
    )

    calculation_id = (
        "calculation_"
        "midea_2024_"
        "gross_profit_margin"
    )

    trace = ComplexCalculationTrace(
        calculation_id=(
            calculation_id
        ),
        metric_id=(
            "gross_profit_margin"
        ),
        formula_id=(
            "gross_profit_margin_formula"
        ),
        input_fact_ids=(
            revenue_fact_id,
            cost_fact_id,
        ),
        status="completed",
        result_value=(
            Decimal("20.7768")
        ),
        result_unit="percent",
        latency_ms=1.0,
        error_message=None,
    )

    claim = Claim(
        claim_id=(
            "claim_calculation_"
            "midea_2024_"
            "gross_profit_margin"
        ),
        claim_type=(
            "financial_calculation"
        ),
        claim_text=(
            "美的集团2024年"
            "毛利率为20.7768%"
        ),
        support=ClaimSupport(
            fact_ids=(
                revenue_fact_id,
                cost_fact_id,
            ),
            calculation_ids=(
                calculation_id,
            ),
            citation_ids=(
                revenue_citation
                .citation_id,
                cost_citation
                .citation_id,
            ),
        ),
        confidence=1.0,
    )

    draft = AnswerDraft(
        draft_id=(
            "draft_calculation_test"
        ),
        draft_type="financial",
        claims=(
            claim,
        ),
    )

    return AgentState.model_construct(
        answer_draft=draft,
        citations=(
            revenue_citation,
            cost_citation,
        ),
        calculation_traces=(
            trace,
        ),
        retrieved_documents=(),
    )


# ============================================================
# Document State
# ============================================================


def _build_document_state() -> AgentState:
    """构造一条正确的 Document Analysis Claim。"""

    document_text = (
        "公司面临原材料价格波动、"
        "行业竞争加剧等经营风险。"
    )

    document = RetrievedDocument(
        query_id="q1",
        rank=1,
        chunk_id=(
            "chunk_midea_2024_risk"
        ),
        document_id=(
            "document_midea_2024"
        ),
        page_id=(
            "page_midea_2024_100"
        ),
        company_id="midea_group",
        report_id=(
            "midea_group_2024"
        ),
        fiscal_year=2024,
        pdf_page=100,
        printed_page=99,
        score=0.95,
        section_path=(
            "经营情况讨论与分析",
            "风险因素",
        ),
        text=document_text,
    )

    citation = CitationRecord(
        citation_id="citation_1",
        report_id=(
            document.report_id
        ),
        pdf_page=(
            document.pdf_page
        ),
        printed_page=(
            document.printed_page
        ),
        evidence_id=None,
        chunk_id=(
            document.chunk_id
        ),
        text_excerpt=(
            document.text.strip()
        ),
    )

    claim = Claim(
        claim_id="claim_document_1",
        claim_type=(
            "document_analysis"
        ),
        claim_text=document_text,
        support=ClaimSupport(
            citation_ids=(
                citation.citation_id,
            ),
        ),
        confidence=1.0,
    )

    draft = AnswerDraft(
        draft_id="draft_document_test",
        draft_type="document",
        claims=(
            claim,
        ),
    )

    return AgentState.model_construct(
        answer_draft=draft,
        citations=(
            citation,
        ),
        calculation_traces=(),
        retrieved_documents=(
            document,
        ),
    )


# ============================================================
# Tests
# ============================================================


def test_verify_requires_answer_draft(
) -> None:
    """没有 AnswerDraft 时，校验器本身无法执行。"""

    verifier = RuntimeTrustVerifier(
        registry_bundle=(
            RegistryBundle()
        )
    )

    state = AgentState.model_construct(
        answer_draft=None,
        citations=(),
        calculation_traces=(),
        retrieved_documents=(),
    )

    with pytest.raises(
        RuntimeTrustVerificationError,
        match="answer_draft",
    ):
        verifier.verify(
            state
        )


def test_valid_financial_fact_passes(
) -> None:
    """正确 Fact + Evidence + Citation 应全部通过。"""

    bundle = _build_bundle()

    state = (
        _build_financial_fact_state(
            bundle
        )
    )

    verifier = RuntimeTrustVerifier(
        registry_bundle=bundle
    )

    report = verifier.verify(
        state
    )

    assert report.passed is True

    assert (
        report.numeric_verified
        is True
    )

    assert (
        report.evidence_verified
        is True
    )

    assert (
        report.citation_verified
        is True
    )

    assert (
        report.evidence_sufficient
        is True
    )

    assert report.issues == ()


def test_tampered_financial_value_is_rejected(
) -> None:
    """Claim 数值被篡改后必须被发现。"""

    bundle = _build_bundle()

    state = (
        _build_financial_fact_state(
            bundle
        )
    )

    original_draft = (
        state.answer_draft
    )

    assert original_draft is not None

    original_claim = (
        original_draft.claims[0]
    )

    tampered_claim = (
        original_claim.model_copy(
            update={
                "claim_text": (
                    "美的集团2024年"
                    "营业收入为1元"
                )
            }
        )
    )

    tampered_draft = (
        original_draft.model_copy(
            update={
                "claims": (
                    tampered_claim,
                )
            }
        )
    )

    tampered_state = (
        state.model_copy(
            update={
                "answer_draft": (
                    tampered_draft
                )
            }
        )
    )

    verifier = RuntimeTrustVerifier(
        registry_bundle=bundle
    )

    report = verifier.verify(
        tampered_state
    )

    assert report.passed is False

    assert (
        report.numeric_verified
        is False
    )

    issue_types = {
        issue.issue_type
        for issue in report.issues
    }

    assert (
        "unsupported_claim"
        in issue_types
    )

    issue = next(
        issue
        for issue in report.issues
        if (
            issue.issue_type
            == "unsupported_claim"
        )
    )

    assert (
        issue.expected_value
        ==
        "美的集团2024年"
        "营业收入为"
        "407149600000元"
    )

    assert (
        issue.actual_value
        ==
        "美的集团2024年"
        "营业收入为1元"
    )


def test_tampered_financial_citation_is_rejected(
) -> None:
    """答案数字正确，但 Citation 页码被篡改，也必须失败。"""

    bundle = _build_bundle()

    state = (
        _build_financial_fact_state(
            bundle
        )
    )

    original_citation = (
        state.citations[0]
    )

    tampered_citation = (
        original_citation.model_copy(
            update={
                "pdf_page": 999,
            }
        )
    )

    tampered_state = (
        state.model_copy(
            update={
                "citations": (
                    tampered_citation,
                )
            }
        )
    )

    verifier = RuntimeTrustVerifier(
        registry_bundle=bundle
    )

    report = verifier.verify(
        tampered_state
    )

    assert report.passed is False

    # 数值本身仍然正确。
    assert (
        report.numeric_verified
        is True
    )

    # 但 Citation 已不可信。
    assert (
        report.citation_verified
        is False
    )

    issue_types = {
        issue.issue_type
        for issue in report.issues
    }

    assert (
        "citation_mismatch"
        in issue_types
    )


def test_valid_calculation_passes(
) -> None:
    """正确 CalculationTrace + Input Facts 应通过。"""

    bundle = _build_bundle()

    state = (
        _build_calculation_state(
            bundle
        )
    )

    verifier = RuntimeTrustVerifier(
        registry_bundle=bundle
    )

    report = verifier.verify(
        state
    )

    assert report.passed is True

    assert (
        report.numeric_verified
        is True
    )

    assert (
        report.evidence_verified
        is True
    )

    assert (
        report.citation_verified
        is True
    )

    assert (
        report.evidence_sufficient
        is True
    )

    assert report.issues == ()


def test_calculation_input_mismatch_is_rejected(
) -> None:
    """Claim 声称的输入 Fact 与真实 Trace 不一致时必须失败。"""

    bundle = _build_bundle()

    state = (
        _build_calculation_state(
            bundle
        )
    )

    original_draft = (
        state.answer_draft
    )

    assert original_draft is not None

    original_claim = (
        original_draft.claims[0]
    )

    tampered_support = (
        original_claim
        .support
        .model_copy(
            update={
                # 故意删除营业成本 Fact。
                "fact_ids": (
                    "fact_midea_group_"
                    "2024_revenue",
                ),
            }
        )
    )

    tampered_claim = (
        original_claim.model_copy(
            update={
                "support": (
                    tampered_support
                )
            }
        )
    )

    tampered_draft = (
        original_draft.model_copy(
            update={
                "claims": (
                    tampered_claim,
                )
            }
        )
    )

    tampered_state = (
        state.model_copy(
            update={
                "answer_draft": (
                    tampered_draft
                )
            }
        )
    )

    verifier = RuntimeTrustVerifier(
        registry_bundle=bundle
    )

    report = verifier.verify(
        tampered_state
    )

    assert report.passed is False

    assert (
        report.numeric_verified
        is False
    )

    issue_types = {
        issue.issue_type
        for issue in report.issues
    }

    assert (
        "calculation_input_mismatch"
        in issue_types
    )


def test_valid_document_claim_passes(
) -> None:
    """Document Claim 与 RetrievedDocument 一致时应通过。"""

    state = (
        _build_document_state()
    )

    verifier = RuntimeTrustVerifier(
        registry_bundle=(
            RegistryBundle()
        )
    )

    report = verifier.verify(
        state
    )

    assert report.passed is True

    assert (
        report.evidence_verified
        is True
    )

    assert (
        report.citation_verified
        is True
    )

    assert (
        report.evidence_sufficient
        is True
    )

    assert report.issues == ()


def test_tampered_document_claim_is_rejected(
) -> None:
    """Document Claim 编造了检索结果中不存在的内容时必须失败。"""

    state = (
        _build_document_state()
    )

    original_draft = (
        state.answer_draft
    )

    assert original_draft is not None

    original_claim = (
        original_draft.claims[0]
    )

    tampered_claim = (
        original_claim.model_copy(
            update={
                "claim_text": (
                    "公司不存在任何经营风险。"
                )
            }
        )
    )

    tampered_draft = (
        original_draft.model_copy(
            update={
                "claims": (
                    tampered_claim,
                )
            }
        )
    )

    tampered_state = (
        state.model_copy(
            update={
                "answer_draft": (
                    tampered_draft
                )
            }
        )
    )

    verifier = RuntimeTrustVerifier(
        registry_bundle=(
            RegistryBundle()
        )
    )

    report = verifier.verify(
        tampered_state
    )

    assert report.passed is False

    issue_types = {
        issue.issue_type
        for issue in report.issues
    }

    assert (
        "unsupported_claim"
        in issue_types
    )