from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.schemas.agent_runtime import (
    AgentState,
    CitationRecord,
)
from app.schemas.complex_plan_eval_result import (
    ComplexCalculationTrace,
)
from app.schemas.enums import (
    UnitCode,
    ValidationStatus,
)
from app.schemas.evidence import (
    SourceEvidence,
)
from app.schemas.financial_fact import (
    FinancialFact,
)
from app.schemas.tool_registry import (
    RetrievedDocument,
)
from app.schemas.trust import (
    Claim,
    VerificationIssue,
    VerificationIssueType,
    VerificationReport,
)
from app.services.registry import (
    RegistryBundle,
)


class RuntimeTrustVerificationError(
    ValueError
):
    """回答级可信校验本身无法执行。"""


@dataclass(
    frozen=True,
    slots=True,
)
class RuntimeTrustVerifier:
    """对 AnswerDraft 中的 Claim 做确定性可信校验。

    该服务只负责：

    AgentState
        ↓
    AnswerDraft
        ↓
    Claim
        ↓
    Fact / Calculation / Document
        ↓
    Evidence / Citation
        ↓
    VerificationReport

    它不负责 Runtime 状态迁移，
    也不负责 allow / refuse / HITL。
    """

    registry_bundle: RegistryBundle

    # ============================================================
    # Public API
    # ============================================================

    def verify(
        self,
        state: AgentState,
    ) -> VerificationReport:
        """验证 AnswerDraft，并返回结构化校验报告。"""

        draft = state.answer_draft

        if draft is None:
            raise RuntimeTrustVerificationError(
                "verify_answer 缺少 answer_draft"
            )

        issues: list[
            VerificationIssue
        ] = []

        # --------------------------------------------------------
        # Citation 通过 citation_id 建索引。
        #
        # ClaimSupport 中保存的是 citation_id，
        # 所以后续需要快速反查 CitationRecord。
        # --------------------------------------------------------

        citation_map = {
            citation.citation_id: citation
            for citation
            in state.citations
        }

        # --------------------------------------------------------
        # Document 通过 chunk_id 建索引。
        #
        # Document Claim 的 Citation 使用 chunk_id
        # 指向 RetrievedDocument。
        # --------------------------------------------------------

        document_map = {
            document.chunk_id: document
            for document
            in state.retrieved_documents
        }

        # --------------------------------------------------------
        # Claim 是最小可信验证单元。
        # --------------------------------------------------------

        for claim in draft.claims:
            citations = (
                self._resolve_citations(
                    claim=claim,
                    citation_map=citation_map,
                    issues=issues,
                )
            )

            if (
                claim.claim_type
                == "financial_fact"
            ):
                self._verify_fact_claim(
                    claim=claim,
                    citations=citations,
                    issues=issues,
                )

                continue

            if (
                claim.claim_type
                == "financial_calculation"
            ):
                self._verify_calculation_claim(
                    state=state,
                    claim=claim,
                    citations=citations,
                    issues=issues,
                )

                continue

            if (
                claim.claim_type
                == "document_analysis"
            ):
                self._verify_document_claim(
                    claim=claim,
                    citations=citations,
                    document_map=document_map,
                    issues=issues,
                )

                continue

            if (
                claim.claim_type
                == "limitation"
            ):
                # limitation 本身不是事实声明，
                # 当前 Step3.1 不要求 Fact / Calculation 支持。
                continue

            self._issue(
                issues=issues,
                claim=claim,
                issue_type="unsupported_claim",
                message=(
                    "遇到未知 Claim 类型："
                    f"{claim.claim_type}"
                ),
            )

        return self._build_report(
            issues=issues
        )

    # ============================================================
    # Citation Resolution
    # ============================================================

    def _resolve_citations(
        self,
        *,
        claim: Claim,
        citation_map: dict[
            str,
            CitationRecord,
        ],
        issues: list[
            VerificationIssue
        ],
    ) -> tuple[
        CitationRecord,
        ...
    ]:
        """把 ClaimSupport 中的 citation_id 解析成 CitationRecord。"""

        citation_ids = (
            claim.support.citation_ids
        )

        if not citation_ids:
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "missing_evidence"
                ),
                message=(
                    "Claim 没有 Citation 支持"
                ),
            )

            return ()

        resolved: list[
            CitationRecord
        ] = []

        for citation_id in citation_ids:
            citation = citation_map.get(
                citation_id
            )

            if citation is None:
                self._issue(
                    issues=issues,
                    claim=claim,
                    issue_type=(
                        "citation_mismatch"
                    ),
                    message=(
                        "Claim 引用了不存在的 "
                        "Citation："
                        f"{citation_id}"
                    ),
                    actual_value=citation_id,
                )

                continue

            resolved.append(
                citation
            )

        return tuple(
            resolved
        )

    # ============================================================
    # Financial Fact Claim
    # ============================================================

    def _verify_fact_claim(
        self,
        *,
        claim: Claim,
        citations: tuple[
            CitationRecord,
            ...
        ],
        issues: list[
            VerificationIssue
        ],
    ) -> None:
        """验证直接来源于 FinancialFact 的 Claim。"""

        fact_ids = (
            claim.support.fact_ids
        )

        # 一个 financial_fact Claim
        # 应该明确对应一个 FinancialFact。
        if len(fact_ids) != 1:
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "unsupported_claim"
                ),
                message=(
                    "financial_fact Claim "
                    "必须且只能绑定一个 fact_id"
                ),
                actual_value=str(
                    fact_ids
                ),
            )

            return

        fact_id = fact_ids[0]

        fact = (
            self.registry_bundle
            .financial_facts
            .get(
                fact_id
            )
        )

        if fact is None:
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "missing_evidence"
                ),
                message=(
                    "FinancialFact 不存在："
                    f"{fact_id}"
                ),
                actual_value=fact_id,
            )

            return

        # --------------------------------------------------------
        # Fact 必须已经通过基础数据核验。
        # --------------------------------------------------------

        if (
            fact.validation_status
            is not ValidationStatus.VERIFIED
        ):
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "missing_evidence"
                ),
                message=(
                    "FinancialFact 尚未核验："
                    f"{fact.fact_id}"
                ),
                expected_value=(
                    ValidationStatus
                    .VERIFIED
                    .value
                ),
                actual_value=(
                    fact
                    .validation_status
                    .value
                ),
            )

        # --------------------------------------------------------
        # 检查 Fact -> Evidence -> Citation。
        # --------------------------------------------------------

        self._verify_fact_source(
            claim=claim,
            fact=fact,
            citations=citations,
            issues=issues,
        )

        self._verify_citation_coverage(
            claim=claim,
            citations=citations,
            expected_evidence_ids={
                fact.primary_evidence_id
            },
            issues=issues,
        )

        # --------------------------------------------------------
        # 重新从 Registry 构造“正确 Claim”。
        #
        # 不相信 AnswerDraft 自己写出的数字。
        # --------------------------------------------------------

        company = (
            self.registry_bundle
            .companies
            .get(
                fact.company_id
            )
        )

        metric = (
            self.registry_bundle
            .metrics
            .get(
                fact.metric_id
            )
        )

        if (
            company is None
            or metric is None
        ):
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "unsupported_claim"
                ),
                message=(
                    "FinancialFact 对应的 "
                    "Company 或 Metric 未注册"
                ),
            )

            return

        unit_text = self._format_unit(
            fact.normalized_unit
        )

        value_text = (
            self._format_decimal(
                fact.normalized_value
            )
        )

        expected_text = (
            f"{company.short_name_cn}"
            f"{fact.fiscal_year}年"
            f"{metric.display_name_cn}"
            f"为"
            f"{value_text}"
            f"{unit_text}"
        )

        self._verify_claim_text(
            claim=claim,
            expected=expected_text,
            expected_year=(
                fact.fiscal_year
            ),
            expected_unit=unit_text,
            issues=issues,
        )

    # ============================================================
    # Financial Calculation Claim
    # ============================================================

    def _verify_calculation_claim(
        self,
        *,
        state: AgentState,
        claim: Claim,
        citations: tuple[
            CitationRecord,
            ...
        ],
        issues: list[
            VerificationIssue
        ],
    ) -> None:
        """验证由 CalculationTrace 产生的 Claim。"""

        calculation_ids = (
            claim
            .support
            .calculation_ids
        )

        # 一个 calculation Claim
        # 必须唯一绑定一个 CalculationTrace。
        if len(calculation_ids) != 1:
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "unsupported_claim"
                ),
                message=(
                    "financial_calculation Claim "
                    "必须且只能绑定一个 calculation_id"
                ),
                actual_value=str(
                    calculation_ids
                ),
            )

            return

        calculation_id = (
            calculation_ids[0]
        )

        trace = self._find_calculation_trace(
            state=state,
            calculation_id=(
                calculation_id
            ),
        )

        if trace is None:
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "unsupported_claim"
                ),
                message=(
                    "找不到 CalculationTrace："
                    f"{calculation_id}"
                ),
                actual_value=(
                    calculation_id
                ),
            )

            return

        # --------------------------------------------------------
        # Calculation 必须已经成功完成。
        # --------------------------------------------------------

        if trace.status != "completed":
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "unsupported_claim"
                ),
                message=(
                    "CalculationTrace "
                    "尚未成功完成"
                ),
                expected_value=(
                    "completed"
                ),
                actual_value=(
                    str(trace.status)
                ),
            )

            return

        if (
            trace.result_value is None
            or trace.result_unit is None
        ):
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "unsupported_claim"
                ),
                message=(
                    "CalculationTrace "
                    "缺少 result_value "
                    "或 result_unit"
                ),
            )

            return

        # --------------------------------------------------------
        # Claim 中声明的 fact_ids
        # 必须与 CalculationTrace 输入完全一致。
        #
        # 防止：
        #
        # Trace 实际使用 2024 Revenue + 2024 Cost，
        # Claim 却声称自己由其他 Fact 支持。
        # --------------------------------------------------------

        if (
            claim.support.fact_ids
            != trace.input_fact_ids
        ):
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "calculation_input_mismatch"
                ),
                message=(
                    "Claim 的 fact_ids 与 "
                    "CalculationTrace.input_fact_ids "
                    "不一致"
                ),
                expected_value=str(
                    trace.input_fact_ids
                ),
                actual_value=str(
                    claim.support.fact_ids
                ),
            )

        if not trace.input_fact_ids:
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "calculation_input_mismatch"
                ),
                message=(
                    "CalculationTrace "
                    "没有输入 FinancialFact"
                ),
            )

            return

        input_facts: list[
            FinancialFact
        ] = []

        expected_evidence_ids: set[
            str
        ] = set()

        # --------------------------------------------------------
        # 每一个计算输入 Fact 都必须可信。
        # --------------------------------------------------------

        for fact_id in (
            trace.input_fact_ids
        ):
            fact = (
                self.registry_bundle
                .financial_facts
                .get(
                    fact_id
                )
            )

            if fact is None:
                self._issue(
                    issues=issues,
                    claim=claim,
                    issue_type=(
                        "missing_evidence"
                    ),
                    message=(
                        "Calculation 输入 "
                        "FinancialFact 不存在："
                        f"{fact_id}"
                    ),
                    actual_value=fact_id,
                )

                continue

            input_facts.append(
                fact
            )

            expected_evidence_ids.add(
                fact.primary_evidence_id
            )

            if (
                fact.validation_status
                is not
                ValidationStatus.VERIFIED
            ):
                self._issue(
                    issues=issues,
                    claim=claim,
                    issue_type=(
                        "missing_evidence"
                    ),
                    message=(
                        "Calculation 输入 Fact "
                        "尚未核验："
                        f"{fact.fact_id}"
                    ),
                )

            self._verify_fact_source(
                claim=claim,
                fact=fact,
                citations=citations,
                issues=issues,
            )

        if not input_facts:
            return

        # --------------------------------------------------------
        # 检查计算输入是否来自同公司、同年份。
        #
        # 当前 RuntimeAnswerDraftBuilder
        # 使用 first_fact 决定 Claim 的公司和年份，
        # 所以输入之间不能互相冲突。
        # --------------------------------------------------------

        first_fact = input_facts[0]

        for fact in input_facts[1:]:
            if (
                fact.company_id
                != first_fact.company_id
            ):
                self._issue(
                    issues=issues,
                    claim=claim,
                    issue_type=(
                        "calculation_input_mismatch"
                    ),
                    message=(
                        "Calculation 输入 Fact "
                        "来自不同公司"
                    ),
                    expected_value=(
                        first_fact.company_id
                    ),
                    actual_value=(
                        fact.company_id
                    ),
                )

            if (
                fact.fiscal_year
                != first_fact.fiscal_year
            ):
                self._issue(
                    issues=issues,
                    claim=claim,
                    issue_type=(
                        "calculation_input_mismatch"
                    ),
                    message=(
                        "Calculation 输入 Fact "
                        "来自不同 fiscal_year"
                    ),
                    expected_value=str(
                        first_fact.fiscal_year
                    ),
                    actual_value=str(
                        fact.fiscal_year
                    ),
                )

            if (
                fact.statement_scope
                != first_fact.statement_scope
            ):
                self._issue(
                    issues=issues,
                    claim=claim,
                    issue_type=(
                        "statement_scope_mismatch"
                    ),
                    message=(
                        "Calculation 输入 Fact "
                        "使用了不同 statement_scope"
                    ),
                    expected_value=str(
                        first_fact.statement_scope
                    ),
                    actual_value=str(
                        fact.statement_scope
                    ),
                )

        self._verify_citation_coverage(
            claim=claim,
            citations=citations,
            expected_evidence_ids=(
                expected_evidence_ids
            ),
            issues=issues,
        )

        # --------------------------------------------------------
        # 根据 CalculationTrace 的真实 result
        # 重新构造预期 Claim。
        # --------------------------------------------------------

        company = (
            self.registry_bundle
            .companies
            .get(
                first_fact.company_id
            )
        )

        metric = (
            self.registry_bundle
            .metrics
            .get(
                trace.metric_id
            )
        )

        if (
            company is None
            or metric is None
        ):
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "unsupported_claim"
                ),
                message=(
                    "Calculation 对应的 "
                    "Company 或 Metric 未注册"
                ),
            )

            return

        unit_text = self._format_unit(
            trace.result_unit
        )

        value_text = (
            self._format_decimal(
                trace.result_value
            )
        )

        expected_text = (
            f"{company.short_name_cn}"
            f"{first_fact.fiscal_year}年"
            f"{metric.display_name_cn}"
            f"为"
            f"{value_text}"
            f"{unit_text}"
        )

        self._verify_claim_text(
            claim=claim,
            expected=expected_text,
            expected_year=(
                first_fact.fiscal_year
            ),
            expected_unit=unit_text,
            issues=issues,
        )

    # ============================================================
    # Document Analysis Claim
    # ============================================================

    def _verify_document_claim(
        self,
        *,
        claim: Claim,
        citations: tuple[
            CitationRecord,
            ...
        ],
        document_map: dict[
            str,
            RetrievedDocument,
        ],
        issues: list[
            VerificationIssue
        ],
    ) -> None:
        """验证 Document Claim 是否真的来源于检索文档。"""

        # 当前 AnswerDraftBuilder 的约定：
        # 一个 document Claim 对应一个 Citation。
        if len(citations) != 1:
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "citation_mismatch"
                ),
                message=(
                    "document_analysis Claim "
                    "必须且只能对应一个 Citation"
                ),
                actual_value=str(
                    tuple(
                        citation.citation_id
                        for citation
                        in citations
                    )
                ),
            )

            return

        citation = citations[0]

        if citation.chunk_id is None:
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "citation_mismatch"
                ),
                message=(
                    "Document Citation "
                    "缺少 chunk_id"
                ),
            )

            return

        document = document_map.get(
            citation.chunk_id
        )

        if document is None:
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "missing_evidence"
                ),
                message=(
                    "Citation 指向的 "
                    "RetrievedDocument 不存在："
                    f"{citation.chunk_id}"
                ),
                actual_value=(
                    citation.chunk_id
                ),
            )

            return

        # --------------------------------------------------------
        # Citation 与 RetrievedDocument 的元数据必须一致。
        # --------------------------------------------------------

        if (
            citation.report_id
            != document.report_id
        ):
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "citation_mismatch"
                ),
                message=(
                    "Document Citation.report_id "
                    "与 RetrievedDocument 不一致"
                ),
                expected_value=(
                    document.report_id
                ),
                actual_value=(
                    citation.report_id
                ),
            )

        if (
            citation.pdf_page
            != document.pdf_page
        ):
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "citation_mismatch"
                ),
                message=(
                    "Document Citation.pdf_page "
                    "与 RetrievedDocument 不一致"
                ),
                expected_value=str(
                    document.pdf_page
                ),
                actual_value=str(
                    citation.pdf_page
                ),
            )

        if (
            citation.printed_page
            != document.printed_page
        ):
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "citation_mismatch"
                ),
                message=(
                    "Document Citation.printed_page "
                    "与 RetrievedDocument 不一致"
                ),
                expected_value=str(
                    document.printed_page
                ),
                actual_value=str(
                    citation.printed_page
                ),
            )

        expected_excerpt = (
            document
            .text
            .strip()[:1000]
        )

        if (
            citation.text_excerpt
            != expected_excerpt
        ):
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "citation_mismatch"
                ),
                message=(
                    "Document Citation.text_excerpt "
                    "与 RetrievedDocument 正文不一致"
                ),
                expected_value=(
                    expected_excerpt
                ),
                actual_value=(
                    citation.text_excerpt
                ),
            )

        # --------------------------------------------------------
        # AnswerDraftBuilder 当前使用：
        #
        # document.text.strip()
        #              .replace("\\n", " ")
        #              [:500]
        #
        # 所以 TrustVerifier 使用完全相同的确定性规则
        # 重建 expected Claim。
        # --------------------------------------------------------

        expected_claim_text = (
            document
            .text
            .strip()
            .replace(
                "\n",
                " ",
            )[:500]
        )

        actual_claim_text = (
            self._normalize(
                claim.claim_text
            )
        )

        normalized_expected = (
            self._normalize(
                expected_claim_text
            )
        )

        if (
            actual_claim_text
            != normalized_expected
        ):
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "unsupported_claim"
                ),
                message=(
                    "Document Claim 内容与 "
                    "RetrievedDocument 不一致"
                ),
                expected_value=(
                    normalized_expected
                ),
                actual_value=(
                    actual_claim_text
                ),
            )

    # ============================================================
    # Fact -> Evidence -> Citation
    # ============================================================

    def _verify_fact_source(
        self,
        *,
        claim: Claim,
        fact: FinancialFact,
        citations: tuple[
            CitationRecord,
            ...
        ],
        issues: list[
            VerificationIssue
        ],
    ) -> None:
        """验证单个 FinancialFact 的证据链。"""

        evidence = (
            self.registry_bundle
            .evidences
            .get(
                fact.primary_evidence_id
            )
        )

        if evidence is None:
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "missing_evidence"
                ),
                message=(
                    "FinancialFact 的 "
                    "SourceEvidence 不存在："
                    f"{fact.primary_evidence_id}"
                ),
                actual_value=(
                    fact.primary_evidence_id
                ),
            )

            return

        # --------------------------------------------------------
        # Evidence 必须已核验。
        # --------------------------------------------------------

        if (
            evidence.validation_status
            is not ValidationStatus.VERIFIED
        ):
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "missing_evidence"
                ),
                message=(
                    "SourceEvidence 尚未核验："
                    f"{evidence.evidence_id}"
                ),
                expected_value=(
                    ValidationStatus
                    .VERIFIED
                    .value
                ),
                actual_value=(
                    evidence
                    .validation_status
                    .value
                ),
            )

        # --------------------------------------------------------
        # Fact / Evidence 必须属于同一个 Report。
        # --------------------------------------------------------

        if (
            evidence.report_id
            != fact.report_id
        ):
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "evidence_conflict"
                ),
                message=(
                    "FinancialFact 与 "
                    "SourceEvidence.report_id "
                    "不一致"
                ),
                expected_value=(
                    fact.report_id
                ),
                actual_value=(
                    evidence.report_id
                ),
            )

        # --------------------------------------------------------
        # 财务口径必须一致。
        # --------------------------------------------------------

        if (
            evidence.statement_scope
            is not None
            and (
                evidence.statement_scope
                != fact.statement_scope
            )
        ):
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "statement_scope_mismatch"
                ),
                message=(
                    "FinancialFact 与 "
                    "SourceEvidence.statement_scope "
                    "不一致"
                ),
                expected_value=str(
                    fact.statement_scope
                ),
                actual_value=str(
                    evidence.statement_scope
                ),
            )

        # statement_type 没有独立 IssueType，
        # 因此归入 evidence_conflict。
        if (
            evidence.statement_type
            is not None
            and (
                evidence.statement_type
                != fact.statement_type
            )
        ):
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "evidence_conflict"
                ),
                message=(
                    "FinancialFact 与 "
                    "SourceEvidence.statement_type "
                    "不一致"
                ),
                expected_value=str(
                    fact.statement_type
                ),
                actual_value=str(
                    evidence.statement_type
                ),
            )

        # --------------------------------------------------------
        # 找出真正指向当前 Evidence 的 Citation。
        # --------------------------------------------------------

        matching_citations = tuple(
            citation
            for citation
            in citations
            if (
                citation.evidence_id
                == evidence.evidence_id
            )
        )

        if not matching_citations:
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "citation_mismatch"
                ),
                message=(
                    "Claim Citation 没有指向 "
                    "FinancialFact.primary_evidence_id："
                    f"{evidence.evidence_id}"
                ),
                expected_value=(
                    evidence.evidence_id
                ),
            )

            return

        # --------------------------------------------------------
        # Citation 不仅 ID 要对，
        # report / page / chunk / excerpt 也必须对。
        # --------------------------------------------------------

        for citation in matching_citations:
            self._verify_evidence_citation(
                claim=claim,
                evidence=evidence,
                citation=citation,
                issues=issues,
            )

    def _verify_evidence_citation(
        self,
        *,
        claim: Claim,
        evidence: SourceEvidence,
        citation: CitationRecord,
        issues: list[
            VerificationIssue
        ],
    ) -> None:
        """检查 CitationRecord 是否忠实表示 SourceEvidence。"""

        if (
            citation.report_id
            != evidence.report_id
        ):
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "citation_mismatch"
                ),
                message=(
                    "Citation.report_id 与 "
                    "SourceEvidence 不一致"
                ),
                expected_value=(
                    evidence.report_id
                ),
                actual_value=(
                    citation.report_id
                ),
            )

        if (
            citation.pdf_page
            != evidence.pdf_page
        ):
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "citation_mismatch"
                ),
                message=(
                    "Citation.pdf_page 与 "
                    "SourceEvidence 不一致"
                ),
                expected_value=str(
                    evidence.pdf_page
                ),
                actual_value=str(
                    citation.pdf_page
                ),
            )

        expected_printed_page = (
            self._normalize_printed_page(
                evidence.printed_page
            )
        )

        if (
            citation.printed_page
            != expected_printed_page
        ):
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "citation_mismatch"
                ),
                message=(
                    "Citation.printed_page 与 "
                    "SourceEvidence 不一致"
                ),
                expected_value=str(
                    expected_printed_page
                ),
                actual_value=str(
                    citation.printed_page
                ),
            )

        if (
            evidence.chunk_id
            is not None
            and (
                citation.chunk_id
                != evidence.chunk_id
            )
        ):
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "citation_mismatch"
                ),
                message=(
                    "Citation.chunk_id 与 "
                    "SourceEvidence 不一致"
                ),
                expected_value=(
                    evidence.chunk_id
                ),
                actual_value=(
                    citation.chunk_id
                ),
            )

        expected_excerpt = (
            evidence
            .evidence_text
            .strip()[:1000]
        )

        if (
            citation.text_excerpt
            != expected_excerpt
        ):
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "citation_mismatch"
                ),
                message=(
                    "Citation.text_excerpt 与 "
                    "SourceEvidence.evidence_text "
                    "不一致"
                ),
                expected_value=(
                    expected_excerpt
                ),
                actual_value=(
                    citation.text_excerpt
                ),
            )

    # ============================================================
    # Citation Coverage
    # ============================================================

    def _verify_citation_coverage(
        self,
        *,
        claim: Claim,
        citations: tuple[
            CitationRecord,
            ...
        ],
        expected_evidence_ids: set[
            str
        ],
        issues: list[
            VerificationIssue
        ],
    ) -> None:
        """检查 Claim Citation 是否准确覆盖所需 Evidence。"""

        actual_evidence_ids = {
            citation.evidence_id
            for citation in citations
            if (
                citation.evidence_id
                is not None
            )
        }

        missing = (
            expected_evidence_ids
            - actual_evidence_ids
        )

        unexpected = (
            actual_evidence_ids
            - expected_evidence_ids
        )

        if missing:
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "citation_mismatch"
                ),
                message=(
                    "Claim 缺少必要 Evidence 的 Citation"
                ),
                expected_value=str(
                    sorted(
                        expected_evidence_ids
                    )
                ),
                actual_value=str(
                    sorted(
                        actual_evidence_ids
                    )
                ),
            )

        if unexpected:
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "citation_mismatch"
                ),
                message=(
                    "Claim 包含无法归属于当前 "
                    "Fact/Calculation 的 Citation"
                ),
                expected_value=str(
                    sorted(
                        expected_evidence_ids
                    )
                ),
                actual_value=str(
                    sorted(
                        actual_evidence_ids
                    )
                ),
            )

        # 财务 Claim 的 Citation
        # 不应该是 document-only Citation。
        has_document_only_citation = any(
            citation.evidence_id is None
            for citation in citations
        )

        if has_document_only_citation:
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "citation_mismatch"
                ),
                message=(
                    "Financial Claim 包含 "
                    "没有 evidence_id 的 Citation"
                ),
            )

    # ============================================================
    # Claim Text Verification
    # ============================================================

    def _verify_claim_text(
        self,
        *,
        claim: Claim,
        expected: str,
        expected_year: int,
        expected_unit: str,
        issues: list[
            VerificationIssue
        ],
    ) -> None:
        """检查 Claim 最终说出的内容是否与确定性结果一致。"""

        actual = self._normalize(
            claim.claim_text
        )

        normalized_expected = (
            self._normalize(
                expected
            )
        )

        # --------------------------------------------------------
        # 年份检查
        # --------------------------------------------------------

        expected_year_text = (
            f"{expected_year}年"
        )

        if (
            expected_year_text
            not in actual
        ):
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "year_mismatch"
                ),
                message=(
                    "Claim fiscal_year 与 "
                    "确定性结果不一致"
                ),
                expected_value=(
                    expected_year_text
                ),
                actual_value=actual,
            )

        # --------------------------------------------------------
        # 单位检查
        # --------------------------------------------------------

        if (
            expected_unit
            and not actual.endswith(
                expected_unit
            )
        ):
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "unit_mismatch"
                ),
                message=(
                    "Claim 单位与 "
                    "确定性结果不一致"
                ),
                expected_value=(
                    expected_unit
                ),
                actual_value=actual,
            )

        # --------------------------------------------------------
        # 最终完整内容一致性检查。
        #
        # 这一层负责兜底捕获：
        #
        # - 公司错误
        # - 指标错误
        # - 数值错误
        # - 其他内容篡改
        # --------------------------------------------------------

        if (
            actual
            != normalized_expected
        ):
            self._issue(
                issues=issues,
                claim=claim,
                issue_type=(
                    "unsupported_claim"
                ),
                message=(
                    "Claim 文本与 "
                    "确定性结构化结果不一致"
                ),
                expected_value=(
                    normalized_expected
                ),
                actual_value=actual,
            )

    # ============================================================
    # Calculation Helpers
    # ============================================================

    @staticmethod
    def _find_calculation_trace(
        *,
        state: AgentState,
        calculation_id: str,
    ) -> (
        ComplexCalculationTrace
        | None
    ):
        """按照 calculation_id 查找 CalculationTrace。"""

        return next(
            (
                trace
                for trace
                in state.calculation_traces
                if (
                    trace.calculation_id
                    == calculation_id
                )
            ),
            None,
        )

    # ============================================================
    # VerificationReport
    # ============================================================

    @staticmethod
    def _build_report(
        *,
        issues: list[
            VerificationIssue
        ],
    ) -> VerificationReport:
        """根据结构化 Issue 汇总不同可信维度。"""

        issue_types = {
            issue.issue_type
            for issue in issues
        }

        # --------------------------------------------------------
        # 数值/Claim 内容正确性
        # --------------------------------------------------------

        numeric_blockers = {
            "unsupported_claim",
            "year_mismatch",
            "unit_mismatch",
            "calculation_input_mismatch",
        }

        # --------------------------------------------------------
        # 原始事实与 Evidence 是否可信
        # --------------------------------------------------------

        evidence_blockers = {
            "missing_evidence",
            "statement_scope_mismatch",
            "evidence_conflict",
        }

        # --------------------------------------------------------
        # Citation 是否正确归属于 Evidence
        # --------------------------------------------------------

        citation_blockers = {
            "citation_mismatch",
        }

        # --------------------------------------------------------
        # 是否拥有足够信息支持最终结论
        # --------------------------------------------------------

        sufficiency_blockers = {
            "missing_evidence",
            "unsupported_claim",
            "statement_scope_mismatch",
            "calculation_input_mismatch",
            "evidence_conflict",
        }

        numeric_verified = not bool(
            issue_types
            & numeric_blockers
        )

        evidence_verified = not bool(
            issue_types
            & evidence_blockers
        )

        citation_verified = not bool(
            issue_types
            & citation_blockers
        )

        evidence_sufficient = not bool(
            issue_types
            & sufficiency_blockers
        )

        passed = (
            numeric_verified
            and evidence_verified
            and citation_verified
            and evidence_sufficient
        )

        return VerificationReport(
            passed=passed,
            numeric_verified=(
                numeric_verified
            ),
            evidence_verified=(
                evidence_verified
            ),
            citation_verified=(
                citation_verified
            ),
            evidence_sufficient=(
                evidence_sufficient
            ),
            issues=tuple(
                issues
            ),
        )

    # ============================================================
    # Issue Helper
    # ============================================================

    @staticmethod
    def _issue(
        *,
        issues: list[
            VerificationIssue
        ],
        claim: Claim,
        issue_type: (
            VerificationIssueType
        ),
        message: str,
        expected_value: (
            str | None
        ) = None,
        actual_value: (
            str | None
        ) = None,
    ) -> None:
        """记录一条结构化校验问题。"""

        issues.append(
            VerificationIssue(
                issue_type=(
                    issue_type
                ),
                severity="error",
                message=message,
                claim_id=(
                    claim.claim_id
                ),
                expected_value=(
                    expected_value
                ),
                actual_value=(
                    actual_value
                ),
            )
        )

    # ============================================================
    # Normalization / Formatting
    # ============================================================

    @staticmethod
    def _normalize(
        value: str,
    ) -> str:
        """规范化空白字符，避免换行/多空格造成误判。"""

        return " ".join(
            value
            .strip()
            .split()
        )

    @staticmethod
    def _normalize_printed_page(
        value: object,
    ) -> int | None:
        """与 RuntimeEvidenceVerifier 的页码规则保持一致。"""

        if isinstance(
            value,
            int,
        ):
            return value

        if (
            isinstance(
                value,
                str,
            )
            and value.isdigit()
        ):
            return int(
                value
            )

        return None

    @staticmethod
    def _format_decimal(
        value: Decimal,
    ) -> str:
        """Decimal 转换成稳定、不带无意义尾零的文本。"""

        text = format(
            value,
            "f",
        )

        if "." in text:
            text = (
                text
                .rstrip("0")
                .rstrip(".")
            )

        return text

    @staticmethod
    def _format_unit(
        value: object,
    ) -> str:
        """把内部 UnitCode 转换成 AnswerDraft 使用的显示单位。"""

        raw_value = getattr(
            value,
            "value",
            value,
        )

        unit_map = {
            UnitCode.CNY.value:
                "元",

            UnitCode.PERCENT.value:
                "%",

            UnitCode
            .PERCENTAGE_POINT
            .value:
                "个百分点",

            UnitCode.RATIO.value:
                "",

            UnitCode
            .CNY_PER_SHARE
            .value:
                "元/股",

            UnitCode.COUNT.value:
                "",
        }

        return unit_map.get(
            str(raw_value),
            str(raw_value),
        )