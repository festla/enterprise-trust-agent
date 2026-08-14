from __future__ import annotations

import re

from dataclasses import dataclass
from decimal import Decimal

from app.schemas.agent_runtime import (
    AgentState,
)
from app.schemas.enums import UnitCode
from app.schemas.trust import (
    AnswerDraft,
    Claim,
    ClaimSupport,
)
from app.services.registry import (
    RegistryBundle,
)


class RuntimeAnswerDraftError(
    ValueError
):
    """Runtime 无法构造可信回答草稿。"""


@dataclass(
    frozen=True,
    slots=True,
)
class RuntimeAnswerDraftBuilder:
    """把 Runtime 产物转换成可逐条验证的 Claim。"""

    registry_bundle: RegistryBundle

    def build(
        self,
        state: AgentState,
    ) -> AnswerDraft:
        # 当前 Week6 流程中，
        # Citation 是 verify_evidence 阶段生成的。
        #
        # Step2.2 暂时基于这个边界构造 Draft。
        # Step2.3 / Step4 再调整 Runtime 节点顺序。
        if state.status != "verifying":
            raise RuntimeAnswerDraftError(
                "只有完成基础 evidence verification "
                "的状态才能构造 AnswerDraft"
            )

        if (
            state.intent is None
            or state.intent == "unsupported"
        ):
            raise RuntimeAnswerDraftError(
                "AgentState 缺少可支持的 intent"
            )

        if not state.citations:
            raise RuntimeAnswerDraftError(
                "AnswerDraft 缺少 citation"
            )

        if (
            state.intent
            == "document_evidence"
        ):
            claims = (
                self._build_document_claims(
                    state
                )
            )

            draft_type = "document"

        else:
            claims = (
                self._build_financial_claims(
                    state
                )
            )

            draft_type = "financial"

        return AnswerDraft(
            draft_id=self._build_draft_id(
                state.run_id
            ),
            draft_type=draft_type,
            claims=claims,
        )

    def _build_financial_claims(
        self,
        state: AgentState,
    ) -> tuple[
        Claim,
        ...
    ]:
        artifact_ids = (
            self._final_artifact_ids(
                state
            )
        )

        claims: list[Claim] = []

        for artifact_id in artifact_ids:
            if artifact_id.startswith(
                "fact_"
            ):
                claims.append(
                    self._build_fact_claim(
                        state=state,
                        fact_id=artifact_id,
                    )
                )

                continue

            if artifact_id.startswith(
                "calculation_"
            ):
                claims.append(
                    self._build_calculation_claim(
                        state=state,
                        calculation_id=(
                            artifact_id
                        ),
                    )
                )

                continue

            raise RuntimeAnswerDraftError(
                "遇到未知 Runtime Artifact："
                f"{artifact_id}"
            )

        if not claims:
            raise RuntimeAnswerDraftError(
                "Financial AnswerDraft "
                "没有可用 Claim"
            )

        return tuple(claims)

    def _build_fact_claim(
        self,
        *,
        state: AgentState,
        fact_id: str,
    ) -> Claim:
        fact = (
            self.registry_bundle
            .financial_facts
            .require(fact_id)
        )

        metric = (
            self.registry_bundle
            .metrics
            .require(
                fact.metric_id
            )
        )

        company = (
            self.registry_bundle
            .companies
            .require(
                fact.company_id
            )
        )

        citation_ids = (
            self._citation_ids_for_fact(
                state=state,
                fact_id=fact_id,
            )
        )

        value_text = self._format_decimal(
            fact.normalized_value
        )

        unit_text = self._format_unit(
            fact.normalized_unit
        )

        claim_text = (
            f"{company.short_name_cn}"
            f"{fact.fiscal_year}年"
            f"{metric.display_name_cn}"
            f"为"
            f"{value_text}"
            f"{unit_text}"
        )

        return Claim(
            claim_id=(
                f"claim_{fact_id}"
            ),
            claim_type=(
                "financial_fact"
            ),
            claim_text=claim_text,
            support=ClaimSupport(
                fact_ids=(
                    fact_id,
                ),
                citation_ids=(
                    citation_ids
                ),
            ),
            confidence=state.confidence,
        )

    def _build_calculation_claim(
        self,
        *,
        state: AgentState,
        calculation_id: str,
    ) -> Claim:
        trace = next(
            (
                candidate
                for candidate
                in state.calculation_traces
                if (
                    candidate.calculation_id
                    == calculation_id
                )
            ),
            None,
        )

        if trace is None:
            raise RuntimeAnswerDraftError(
                "找不到 Calculation Trace："
                f"{calculation_id}"
            )

        if (
            trace.result_value is None
            or trace.result_unit is None
        ):
            raise RuntimeAnswerDraftError(
                "Calculation 缺少结果："
                f"{calculation_id}"
            )

        metric = (
            self.registry_bundle
            .metrics
            .require(
                trace.metric_id
            )
        )

        first_fact = (
            self.registry_bundle
            .financial_facts
            .require(
                trace.input_fact_ids[0]
            )
        )

        company = (
            self.registry_bundle
            .companies
            .require(
                first_fact.company_id
            )
        )

        citation_ids: list[str] = []

        for fact_id in (
            trace.input_fact_ids
        ):
            citation_ids.extend(
                self._citation_ids_for_fact(
                    state=state,
                    fact_id=fact_id,
                )
            )

        unique_citation_ids = tuple(
            dict.fromkeys(
                citation_ids
            )
        )

        value_text = self._format_decimal(
            trace.result_value
        )

        unit_text = self._format_unit(
            trace.result_unit
        )

        claim_text = (
            f"{company.short_name_cn}"
            f"{first_fact.fiscal_year}年"
            f"{metric.display_name_cn}"
            f"为"
            f"{value_text}"
            f"{unit_text}"
        )

        return Claim(
            claim_id=(
                f"claim_{calculation_id}"
            ),
            claim_type=(
                "financial_calculation"
            ),
            claim_text=claim_text,
            support=ClaimSupport(
                fact_ids=(
                    trace.input_fact_ids
                ),
                calculation_ids=(
                    calculation_id,
                ),
                citation_ids=(
                    unique_citation_ids
                ),
            ),
            confidence=state.confidence,
        )

    def _build_document_claims(
        self,
        state: AgentState,
    ) -> tuple[
        Claim,
        ...
    ]:
        document_by_chunk_id = {
            document.chunk_id: document
            for document
            in state.retrieved_documents
        }

        claims: list[Claim] = []

        for index, citation in enumerate(
            state.citations,
            start=1,
        ):
            if citation.chunk_id is None:
                raise RuntimeAnswerDraftError(
                    "Document Citation "
                    "缺少 chunk_id"
                )

            document = (
                document_by_chunk_id.get(
                    citation.chunk_id
                )
            )

            if document is None:
                raise RuntimeAnswerDraftError(
                    "Citation 引用了不存在的 "
                    "RetrievedDocument："
                    f"{citation.chunk_id}"
                )

            text = (
                document.text
                .strip()
                .replace(
                    "\n",
                    " ",
                )
            )

            if not text:
                raise RuntimeAnswerDraftError(
                    "RetrievedDocument "
                    "正文不能为空"
                )

            claims.append(
                Claim(
                    claim_id=(
                        f"claim_document_{index}"
                    ),
                    claim_type=(
                        "document_analysis"
                    ),
                    claim_text=text[:500],
                    support=ClaimSupport(
                        citation_ids=(
                            citation.citation_id,
                        ),
                    ),
                    confidence=(
                        state.confidence
                    ),
                )
            )

        if not claims:
            raise RuntimeAnswerDraftError(
                "Document AnswerDraft "
                "没有可用 Claim"
            )

        return tuple(claims)

    def _citation_ids_for_fact(
        self,
        *,
        state: AgentState,
        fact_id: str,
    ) -> tuple[str, ...]:
        fact = (
            self.registry_bundle
            .financial_facts
            .require(fact_id)
        )

        citation_ids = tuple(
            citation.citation_id
            for citation
            in state.citations
            if (
                citation.evidence_id
                == fact.primary_evidence_id
            )
        )

        if not citation_ids:
            raise RuntimeAnswerDraftError(
                "FinancialFact 缺少对应 Citation："
                f"{fact_id}"
            )

        return citation_ids

    @staticmethod
    def _final_artifact_ids(
        state: AgentState,
    ) -> tuple[str, ...]:
        runtime_plan = state.runtime_plan

        if runtime_plan is None:
            raise RuntimeAnswerDraftError(
                "缺少 runtime_plan"
            )

        final_step = (
            runtime_plan.plan.steps[-1]
        )

        artifact_ids = (
            state.runtime_refs.get(
                final_step.output_ref
            )
        )

        if artifact_ids is None:
            raise RuntimeAnswerDraftError(
                "Final Step 没有对应 "
                "Runtime Reference："
                f"{final_step.output_ref}"
            )

        return artifact_ids

    @staticmethod
    def _build_draft_id(
        run_id: str,
    ) -> str:
        normalized = re.sub(
            r"[^a-z0-9_]+",
            "_",
            run_id.lower(),
        ).strip("_")

        if not normalized:
            normalized = "runtime"

        return (
            f"draft_{normalized}"
        )

    @staticmethod
    def _format_decimal(
        value: Decimal,
    ) -> str:
        text = format(
            value,
            "f",
        )

        if "." in text:
            text = (
                text.rstrip("0")
                .rstrip(".")
            )

        return text

    @staticmethod
    def _format_unit(
        value: object,
    ) -> str:
        raw_value = getattr(
            value,
            "value",
            value,
        )

        unit_map = {
            UnitCode.CNY.value: "元",
            UnitCode.PERCENT.value: "%",
            UnitCode.PERCENTAGE_POINT.value:
                "个百分点",
            UnitCode.RATIO.value: "",
            UnitCode.CNY_PER_SHARE.value:
                "元/股",
            UnitCode.COUNT.value: "",
        }

        return unit_map.get(
            str(raw_value),
            str(raw_value),
        )