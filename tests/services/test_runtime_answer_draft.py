from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.schemas.agent_runtime import (
    CitationRecord,
)
from app.services.runtime_answer_draft import (
    RuntimeAnswerDraftBuilder,
    RuntimeAnswerDraftError,
)


def _citation(
    *,
    citation_id: str,
    evidence_id: str | None = None,
    chunk_id: str | None = None,
) -> CitationRecord:
    return CitationRecord(
        citation_id=citation_id,
        report_id="midea_2024",
        pdf_page=100,
        evidence_id=evidence_id,
        chunk_id=chunk_id,
        text_excerpt="测试证据",
    )


def _financial_registry() -> MagicMock:
    bundle = MagicMock()

    fact = SimpleNamespace(
        fact_id="fact_midea_2024_revenue",
        metric_id="revenue",
        company_id="midea_group",
        fiscal_year=2024,
        normalized_value=Decimal(
            "407149600000"
        ),
        normalized_unit="cny",
        primary_evidence_id=(
            "evidence_midea_2024_revenue"
        ),
    )

    bundle.financial_facts.require.return_value = (
        fact
    )

    bundle.metrics.require.return_value = (
        SimpleNamespace(
            display_name_cn="营业收入",
        )
    )

    bundle.companies.require.return_value = (
        SimpleNamespace(
            short_name_cn="美的集团",
        )
    )

    return bundle


def _financial_state() -> SimpleNamespace:
    final_step = SimpleNamespace(
        output_ref="final_output",
    )

    return SimpleNamespace(
        status="verifying",
        intent="financial_fact",
        run_id="run_week7_001",
        confidence=0.95,
        runtime_plan=SimpleNamespace(
            plan=SimpleNamespace(
                steps=(
                    final_step,
                )
            )
        ),
        runtime_refs={
            "final_output": (
                "fact_midea_2024_revenue",
            )
        },
        citations=(
            _citation(
                citation_id="citation_1",
                evidence_id=(
                    "evidence_midea_2024_revenue"
                ),
            ),
        ),
        calculation_traces=(),
        retrieved_documents=(),
    )


def test_builder_builds_financial_fact_claim() -> None:
    builder = RuntimeAnswerDraftBuilder(
        registry_bundle=(
            _financial_registry()
        )
    )

    draft = builder.build(
        _financial_state()
    )

    assert draft.draft_type == "financial"
    assert len(draft.claims) == 1

    claim = draft.claims[0]

    assert (
        claim.claim_id
        == "claim_fact_midea_2024_revenue"
    )

    assert (
        claim.claim_type
        == "financial_fact"
    )

    assert (
        claim.support.fact_ids
        == (
            "fact_midea_2024_revenue",
        )
    )

    assert (
        claim.support.citation_ids
        == (
            "citation_1",
        )
    )


def test_builder_rejects_state_before_verification() -> None:
    state = _financial_state()
    state.status = "executing"

    builder = RuntimeAnswerDraftBuilder(
        registry_bundle=(
            _financial_registry()
        )
    )

    with pytest.raises(
        RuntimeAnswerDraftError
    ):
        builder.build(state)


def test_builder_rejects_missing_fact_citation() -> None:
    state = _financial_state()

    state.citations = (
        _citation(
            citation_id="citation_1",
            evidence_id="evidence_other",
        ),
    )

    builder = RuntimeAnswerDraftBuilder(
        registry_bundle=(
            _financial_registry()
        )
    )

    with pytest.raises(
        RuntimeAnswerDraftError
    ):
        builder.build(state)


def test_builder_builds_calculation_claim() -> None:
    bundle = MagicMock()

    facts = {
        "fact_revenue": SimpleNamespace(
            fact_id="fact_revenue",
            metric_id="revenue",
            company_id="midea_group",
            fiscal_year=2024,
            normalized_value=Decimal("100"),
            normalized_unit="cny",
            primary_evidence_id="evidence_revenue",
        ),
        "fact_cost": SimpleNamespace(
            fact_id="fact_cost",
            metric_id="cost",
            company_id="midea_group",
            fiscal_year=2024,
            normalized_value=Decimal("60"),
            normalized_unit="cny",
            primary_evidence_id="evidence_cost",
        ),
    }

    bundle.financial_facts.require.side_effect = (
        lambda fact_id: facts[fact_id]
    )

    bundle.metrics.require.return_value = (
        SimpleNamespace(
            display_name_cn="毛利率",
        )
    )

    bundle.companies.require.return_value = (
        SimpleNamespace(
            short_name_cn="美的集团",
        )
    )

    final_step = SimpleNamespace(
        output_ref="final_output",
    )

    state = SimpleNamespace(
        status="verifying",
        intent="financial_calculation",
        run_id="run_week7_calculation",
        confidence=0.9,
        runtime_plan=SimpleNamespace(
            plan=SimpleNamespace(
                steps=(final_step,)
            )
        ),
        runtime_refs={
            "final_output": (
                "calculation_gross_margin",
            )
        },
        calculation_traces=(
            SimpleNamespace(
                calculation_id=(
                    "calculation_gross_margin"
                ),
                metric_id="gross_margin",
                input_fact_ids=(
                    "fact_revenue",
                    "fact_cost",
                ),
                result_value=Decimal("40"),
                result_unit="percent",
            ),
        ),
        citations=(
            _citation(
                citation_id="citation_1",
                evidence_id="evidence_revenue",
            ),
            _citation(
                citation_id="citation_2",
                evidence_id="evidence_cost",
            ),
        ),
        retrieved_documents=(),
    )

    builder = RuntimeAnswerDraftBuilder(
        registry_bundle=bundle
    )

    draft = builder.build(state)

    claim = draft.claims[0]

    assert (
        claim.claim_type
        == "financial_calculation"
    )

    assert claim.support.fact_ids == (
        "fact_revenue",
        "fact_cost",
    )

    assert (
        claim.support.calculation_ids
        == (
            "calculation_gross_margin",
        )
    )

    assert claim.support.citation_ids == (
        "citation_1",
        "citation_2",
    )


def test_builder_builds_document_claim() -> None:
    state = SimpleNamespace(
        status="verifying",
        intent="document_evidence",
        run_id="run_week7_document",
        confidence=0.8,
        citations=(
            _citation(
                citation_id="citation_1",
                chunk_id="chunk_001",
            ),
        ),
        retrieved_documents=(
            SimpleNamespace(
                chunk_id="chunk_001",
                text=(
                    "公司在年报中披露了"
                    "原材料价格波动风险。"
                ),
            ),
        ),
    )

    builder = RuntimeAnswerDraftBuilder(
        registry_bundle=MagicMock()
    )

    draft = builder.build(state)

    assert draft.draft_type == "document"
    assert len(draft.claims) == 1

    claim = draft.claims[0]

    assert (
        claim.claim_type
        == "document_analysis"
    )

    assert claim.support.citation_ids == (
        "citation_1",
    )


def test_builder_rejects_missing_final_runtime_ref() -> None:
    state = _financial_state()

    state.runtime_refs = {}

    builder = RuntimeAnswerDraftBuilder(
        registry_bundle=(
            _financial_registry()
        )
    )

    with pytest.raises(
        RuntimeAnswerDraftError
    ):
        builder.build(state)