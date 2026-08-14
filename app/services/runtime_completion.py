from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from datetime import (
    datetime,
    timezone,
)
from decimal import Decimal
from time import perf_counter
from typing import Protocol
from uuid import uuid4

from app.schemas.agent_runtime import (
    AgentAnswer,
    AgentState,
    CitationRecord,
    NodeSpan,
)
from app.schemas.enums import (
    UnitCode,
    ValidationStatus,
)
from app.services.registry import (
    RegistryBundle,
)


class RuntimeEvidenceError(
    ValueError
):
    """Runtime 证据校验失败。"""


class RuntimeAnswerGenerationError(
    ValueError
):
    """Runtime 回答生成失败。"""


class CompletionClock(
    Protocol
):
    def now(
        self,
    ) -> datetime:
        """返回当前时间。"""


class CompletionIdFactory(
    Protocol
):
    def new_id(
        self,
        prefix: str,
    ) -> str:
        """生成 Runtime ID。"""


@dataclass(
    frozen=True,
    slots=True,
)
class UTCCompletionClock:
    def now(
        self,
    ) -> datetime:
        return datetime.now(
            timezone.utc
        )


@dataclass(
    frozen=True,
    slots=True,
)
class UUIDCompletionIdFactory:
    def new_id(
        self,
        prefix: str,
    ) -> str:
        return (
            f"{prefix}_{uuid4().hex}"
        )


@dataclass(
    frozen=True,
    slots=True,
)
class RuntimeEvidenceVerifier:
    registry_bundle: RegistryBundle

    clock: CompletionClock = field(
        default_factory=(
            UTCCompletionClock
        )
    )

    id_factory: CompletionIdFactory = (
        field(
            default_factory=(
                UUIDCompletionIdFactory
            )
        )
    )

    def verify(
        self,
        state: AgentState,
    ) -> AgentState:
        if state.runtime_plan is None:
            raise RuntimeEvidenceError(
                "verify_evidence 缺少 runtime_plan"
            )

        if (
            state.current_step
            != len(
                state.runtime_plan
                .plan.steps
            )
        ):
            raise RuntimeEvidenceError(
                "RuntimePlan 尚未执行完成"
            )

        expected_step_ids = tuple(
            step.step_id
            for step
            in state.runtime_plan
            .plan.steps
        )

        if (
            state.completed_step_ids
            != expected_step_ids
        ):
            raise RuntimeEvidenceError(
                "completed_step_ids "
                "与 RuntimePlan 不一致"
            )

        if (
            state.step_count
            >= state.max_steps
        ):
            raise RuntimeEvidenceError(
                "Runtime 已达到 max_steps"
            )

        started_at = self.clock.now()
        timer_start = perf_counter()

        if (
            state.intent
            == "document_evidence"
        ):
            citations = (
                self
                ._verify_document_evidence(
                    state
                )
            )

        else:
            citations = (
                self
                ._verify_financial_evidence(
                    state
                )
            )

        completed_at = self.clock.now()

        span = NodeSpan(
            span_id=(
                self.id_factory.new_id(
                    "span"
                )
            ),
            node_name="verify_evidence",
            attempt=1,
            status="completed",
            input_summary={
                "fact_count": len(
                    state.resolved_fact_ids
                ),
                "calculation_count": len(
                    state.calculation_ids
                ),
                "document_count": len(
                    state.retrieved_documents
                ),
            },
            output_summary={
                "citation_count": len(
                    citations
                ),
                "verified": True,
            },
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=max(
                (
                    perf_counter()
                    - timer_start
                )
                * 1000.0,
                0.0,
            ),
            checkpoint_revision=(
                state.checkpoint_revision
            ),
            error_type=None,
            error_message=None,
        )

        return self._replace_state(
            state,
            status="verifying",
            citations=citations,
            node_spans=(
                state.node_spans
                + (
                    span,
                )
            ),
            step_count=(
                state.step_count + 1
            ),
            current_node=(
                "verify_evidence"
            ),
            next_node=(
                "prepare_answer"
            ),
            updated_at=completed_at,
        )

    def _verify_financial_evidence(
        self,
        state: AgentState,
    ) -> tuple[
        CitationRecord,
        ...
    ]:
        if not state.resolved_fact_ids:
            raise RuntimeEvidenceError(
                "结构化财务回答缺少 fact_id"
            )

        if not state.evidence_ids:
            raise RuntimeEvidenceError(
                "结构化财务回答缺少 evidence_id"
            )

        resolved_fact_set = set(
            state.resolved_fact_ids
        )

        evidence_set = set(
            state.evidence_ids
        )

        for fact_id in (
            state.resolved_fact_ids
        ):
            fact = (
                self.registry_bundle
                .financial_facts
                .require(fact_id)
            )

            if (
                fact.validation_status
                is not ValidationStatus.VERIFIED
            ):
                raise RuntimeEvidenceError(
                    "FinancialFact 尚未核验："
                    f"{fact_id}"
                )

            if (
                fact.primary_evidence_id
                not in evidence_set
            ):
                raise RuntimeEvidenceError(
                    "FinancialFact 的主要证据"
                    "没有进入 Runtime："
                    f"{fact_id}"
                )

        for trace in (
            state.calculation_traces
        ):
            if trace.status != "completed":
                raise RuntimeEvidenceError(
                    "存在未完成 Calculation："
                    f"{trace.calculation_id}"
                )

            unknown_inputs = (
                set(trace.input_fact_ids)
                - resolved_fact_set
            )

            if unknown_inputs:
                raise RuntimeEvidenceError(
                    "Calculation 使用了"
                    "未解析的 FinancialFact："
                    f"{sorted(unknown_inputs)}"
                )

        citations: list[
            CitationRecord
        ] = []

        for index, evidence_id in enumerate(
            state.evidence_ids,
            start=1,
        ):
            evidence = (
                self.registry_bundle
                .evidences
                .require(evidence_id)
            )

            if (
                evidence.validation_status
                is not ValidationStatus.VERIFIED
            ):
                raise RuntimeEvidenceError(
                    "SourceEvidence 尚未核验："
                    f"{evidence_id}"
                )

            printed_page = (
                self._normalize_printed_page(
                    evidence.printed_page
                )
            )

            excerpt = (
                evidence.evidence_text
                .strip()
            )

            if not excerpt:
                excerpt = (
                    "结构化财务事实来源证据"
                )

            citations.append(
                CitationRecord(
                    citation_id=(
                        f"citation_{index}"
                    ),
                    report_id=(
                        evidence.report_id
                    ),
                    pdf_page=(
                        evidence.pdf_page
                    ),
                    printed_page=(
                        printed_page
                    ),
                    evidence_id=(
                        evidence.evidence_id
                    ),
                    chunk_id=(
                        evidence.chunk_id
                    ),
                    text_excerpt=(
                        excerpt[:1000]
                    ),
                )
            )

        return tuple(citations)

    def _verify_document_evidence(
        self,
        state: AgentState,
    ) -> tuple[
        CitationRecord,
        ...
    ]:
        if not state.retrieved_documents:
            raise RuntimeEvidenceError(
                "文档问题没有可用检索证据"
            )

        citations: list[
            CitationRecord
        ] = []

        for index, document in enumerate(
            state.retrieved_documents,
            start=1,
        ):
            excerpt = (
                document.text.strip()
            )

            if not excerpt:
                raise RuntimeEvidenceError(
                    "RetrievedDocument "
                    "正文不能为空"
                )

            citations.append(
                CitationRecord(
                    citation_id=(
                        f"citation_{index}"
                    ),
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
                        excerpt[:1000]
                    ),
                )
            )

        return tuple(citations)

    @staticmethod
    def _normalize_printed_page(
        value: object,
    ) -> int | None:
        if isinstance(value, int):
            return value

        if (
            isinstance(value, str)
            and value.isdigit()
        ):
            return int(value)

        return None

    @staticmethod
    def _replace_state(
        state: AgentState,
        **updates: object,
    ) -> AgentState:
        payload = state.model_dump(
            mode="python"
        )

        payload.update(updates)

        return AgentState.model_validate(
            payload
        )


@dataclass(
    frozen=True,
    slots=True,
)
class RuntimeAnswerGenerator:
    registry_bundle: RegistryBundle

    generator_version: str = (
        "deterministic_runtime_answer_v1"
    )

    clock: CompletionClock = field(
        default_factory=(
            UTCCompletionClock
        )
    )

    id_factory: CompletionIdFactory = (
        field(
            default_factory=(
                UUIDCompletionIdFactory
            )
        )
    )

    def generate(
        self,
        state: AgentState,
    ) -> AgentState:
        if state.status != "verifying":
            raise RuntimeAnswerGenerationError(
                "只有通过 verify_evidence "
                "的状态才能生成答案"
            )

        if (
            state.step_count
            >= state.max_steps
        ):
            raise RuntimeAnswerGenerationError(
                "Runtime 已达到 max_steps"
            )

        if state.intent is None:
            raise RuntimeAnswerGenerationError(
                "AgentState 缺少 intent"
            )

        if state.answer_draft is None:
            raise RuntimeAnswerGenerationError(
                "generate_answer 缺少 answer_draft"
            )

        started_at = self.clock.now()
        timer_start = perf_counter()

        if (
            state.intent
            == "document_evidence"
        ):
            answer = (
                self._build_document_answer(
                    state
                )
            )

        else:
            answer = (
                self._build_financial_answer(
                    state
                )
            )

        completed_at = self.clock.now()

        span = NodeSpan(
            span_id=(
                self.id_factory.new_id(
                    "span"
                )
            ),
            node_name="generate_answer",
            attempt=1,
            status="completed",
            input_summary={
                "intent": state.intent,
                "citation_count": len(
                    state.citations
                ),
            },
            output_summary={
                "answer_type": (
                    answer.answer_type
                ),
                "answer_length": len(
                    answer.answer_text
                ),
                "generator_version": (
                    self.generator_version
                ),
            },
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=max(
                (
                    perf_counter()
                    - timer_start
                )
                * 1000.0,
                0.0,
            ),
            checkpoint_revision=(
                state.checkpoint_revision
            ),
            error_type=None,
            error_message=None,
        )

        return self._replace_state(
            state,
            answer=answer,
            status="completed",
            stop_reason="completed",
            generator_version=(
                self.generator_version
            ),
            node_spans=(
                state.node_spans
                + (
                    span,
                )
            ),
            step_count=(
                state.step_count + 1
            ),
            current_node=(
                "generate_answer"
            ),
            next_node="finish",
            updated_at=completed_at,
            completed_at=completed_at,
        )

    def _build_financial_answer(
        self,
        state: AgentState,
    ) -> AgentAnswer:
        draft = state.answer_draft

        if draft is None:
            raise RuntimeAnswerGenerationError(
                "Financial Answer 缺少 AnswerDraft"
            )

        if draft.draft_type != "financial":
            raise RuntimeAnswerGenerationError(
                "Financial Answer 收到了"
                "非 financial AnswerDraft"
            )

        rendered_items = tuple(
            claim.claim_text
            for claim in draft.claims
        )

        if not rendered_items:
            raise RuntimeAnswerGenerationError(
                "Financial AnswerDraft "
                "没有可生成的 Claim"
            )

        runtime_plan = state.runtime_plan

        if runtime_plan is None:
            raise RuntimeAnswerGenerationError(
                "缺少 runtime_plan"
            )

        final_step = (
            runtime_plan.plan.steps[-1]
        )

        if final_step.action == "rank":
            answer_text = (
                "排序结果：\n"
                + "\n".join(
                    (
                        f"{index}. {text}"
                    )
                    for index, text
                    in enumerate(
                        rendered_items,
                        start=1,
                    )
                )
            )

        elif final_step.action == "compare":
            answer_text = (
                "比较结果："
                + "；".join(
                    rendered_items
                )
                + "。"
            )

        else:
            answer_text = (
                "；".join(
                    rendered_items
                )
                + "。"
            )

        return AgentAnswer(
            answer_type="financial",
            answer_text=answer_text,

            # 暂时保留 Week6 的全局审计字段，
            # 避免破坏 AgentState 既有契约。
            supporting_fact_ids=(
                state.resolved_fact_ids
            ),

            supporting_calculation_ids=(
                state.calculation_ids
            ),

            citation_evidence_ids=(
                state.evidence_ids
            ),

            document_citation_ids=(),

            confidence=(
                state.confidence
            ),
        )

    def _build_document_answer(
        self,
        state: AgentState,
    ) -> AgentAnswer:
        draft = state.answer_draft

        if draft is None:
            raise RuntimeAnswerGenerationError(
                "Document Answer 缺少 AnswerDraft"
            )

        if draft.draft_type != "document":
            raise RuntimeAnswerGenerationError(
                "Document Answer 收到了"
                "非 document AnswerDraft"
            )

        lines: list[str] = []

        citation_ids: list[str] = []

        for claim in draft.claims:
            claim_citation_ids = (
                claim.support.citation_ids
            )

            if not claim_citation_ids:
                raise RuntimeAnswerGenerationError(
                    "Document Claim 缺少 citation"
                )

            citation_ids.extend(
                claim_citation_ids
            )

            citation_prefix = " ".join(
                (
                    f"[{citation_id}]"
                    for citation_id
                    in claim_citation_ids
                )
            )

            lines.append(
                (
                    f"{citation_prefix} "
                    f"{claim.claim_text}"
                )
            )

        if not lines:
            raise RuntimeAnswerGenerationError(
                "Document AnswerDraft "
                "没有可用 Claim"
            )

        unique_citation_ids = tuple(
            dict.fromkeys(
                citation_ids
            )
        )

        answer_text = (
            "根据检索到的财报证据：\n"
            + "\n".join(
                lines
            )
        )

        return AgentAnswer(
            answer_type="document",
            answer_text=answer_text,
            supporting_fact_ids=(),
            supporting_calculation_ids=(),
            citation_evidence_ids=(),
            document_citation_ids=(
                unique_citation_ids
            ),
            confidence=(
                state.confidence
            ),
        )

    def _final_artifact_ids(
        self,
        state: AgentState,
    ) -> tuple[str, ...]:
        runtime_plan = state.runtime_plan

        if runtime_plan is None:
            raise RuntimeAnswerGenerationError(
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
            raise RuntimeAnswerGenerationError(
                "Final Step 没有"
                "对应 Runtime Reference："
                f"{final_step.output_ref}"
            )

        return artifact_ids

    def _render_financial_artifact(
        self,
        *,
        state: AgentState,
        artifact_id: str,
    ) -> str:
        if artifact_id.startswith(
            "fact_"
        ):
            fact = (
                self.registry_bundle
                .financial_facts
                .require(artifact_id)
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

            return (
                f"{company.short_name_cn}"
                f"{fact.fiscal_year}年"
                f"{metric.display_name_cn}"
                f"为"
                f"{self._format_decimal(fact.normalized_value)}"
                f"{self._format_unit(fact.normalized_unit)}"
            )

        if artifact_id.startswith(
            "calculation_"
        ):
            trace = None

            for candidate in (
                state.calculation_traces
            ):
                if (
                    candidate.calculation_id
                    == artifact_id
                ):
                    trace = candidate
                    break

            if trace is None:
                raise RuntimeAnswerGenerationError(
                    "找不到 Calculation："
                    f"{artifact_id}"
                )

            if (
                trace.result_value is None
                or trace.result_unit is None
            ):
                raise RuntimeAnswerGenerationError(
                    "Calculation 缺少结果："
                    f"{artifact_id}"
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

            return (
                f"{company.short_name_cn}"
                f"{first_fact.fiscal_year}年"
                f"{metric.display_name_cn}"
                f"为"
                f"{self._format_decimal(trace.result_value)}"
                f"{self._format_unit(trace.result_unit)}"
            )

        raise RuntimeAnswerGenerationError(
            "Financial Answer "
            "遇到未知 Runtime Artifact："
            f"{artifact_id}"
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
            UnitCode.PERCENTAGE_POINT.value: "个百分点",
            UnitCode.RATIO.value: "",
            UnitCode.CNY_PER_SHARE.value: "元/股",
            UnitCode.COUNT.value: "",
        }

        return unit_map.get(
            str(raw_value),
            str(raw_value),
        )

    @staticmethod
    def _replace_state(
        state: AgentState,
        **updates: object,
    ) -> AgentState:
        payload = state.model_dump(
            mode="python"
        )

        payload.update(updates)

        return AgentState.model_validate(
            payload
        )