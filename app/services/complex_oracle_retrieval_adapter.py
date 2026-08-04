from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from app.schemas.complex_plan_eval_result import (
    ComplexRetrievalQueryOutput,
    ComplexRetrievalTrace,
)
from app.schemas.enums import ValidationStatus
from app.schemas.evidence import SourceEvidence
from app.schemas.reranker import (
    RerankedRetrievalHit,
)
from app.services.registry import RegistryBundle


class ComplexOracleRetrievalAdapterError(
    ValueError
):
    """复杂问题真实检索适配失败。"""


class ComplexRerankedHitProvider(Protocol):
    """Hybrid+Reranker Hit 提供器接口。"""

    @property
    def provider_id(self) -> str:
        """返回检索及重排配置标识。"""

    def search(
        self,
        *,
        query: ComplexRetrievalQueryOutput,
        top_k: int,
    ) -> Sequence[
        RerankedRetrievalHit
    ]:
        """返回经过 Reranker 的有序 Hit。"""


@dataclass(slots=True)
class ComplexOracleRetrievalAdapter:
    """把 Reranked Chunk 转换为 Fact/Evidence Trace。"""

    registry_bundle: RegistryBundle

    hit_provider: ComplexRerankedHitProvider

    resolver_version: str = (
        "registry_fact_resolver_v1"
    )

    @property
    def retriever_id(self) -> str:
        provider_id = (
            self.hit_provider
            .provider_id
            .strip()
        )

        if not provider_id:
            return ""

        return (
            f"{provider_id}_"
            f"{self.resolver_version}"
        )

    def retrieve(
        self,
        *,
        query: ComplexRetrievalQueryOutput,
        top_k: int,
    ) -> ComplexRetrievalTrace:
        """执行检索并将 Chunk 解析为 Fact/Evidence。"""

        if top_k <= 0:
            raise (
                ComplexOracleRetrievalAdapterError(
                    "top_k 必须大于 0"
                )
            )

        timer_start = perf_counter()

        try:
            self._validate_provider_id()

            self._validate_query_report(
                query
            )

            raw_hits = (
                self.hit_provider.search(
                    query=query,
                    top_k=top_k,
                )
            )

            hits = tuple(
                RerankedRetrievalHit
                .model_validate(hit)
                for hit in raw_hits
            )

            self._validate_hits(
                query=query,
                hits=hits,
                top_k=top_k,
            )

            (
                fact_ids,
                evidence_ids,
            ) = self._resolve_facts(
                query=query,
                hits=hits,
            )

            latency_ms = (
                perf_counter()
                - timer_start
            ) * 1000

            return ComplexRetrievalTrace(
                query_id=query.query_id,
                status="completed",
                retrieved_fact_ids=(
                    fact_ids
                ),
                retrieved_evidence_ids=(
                    evidence_ids
                ),
                retrieved_chunk_ids=tuple(
                    hit.chunk_id
                    for hit in hits
                ),
                top_k=top_k,
                latency_ms=latency_ms,
                error_message=None,
            )

        except Exception as exc:
            latency_ms = (
                perf_counter()
                - timer_start
            ) * 1000

            return ComplexRetrievalTrace(
                query_id=query.query_id,
                status="failed",
                retrieved_fact_ids=(),
                retrieved_evidence_ids=(),
                retrieved_chunk_ids=(),
                top_k=top_k,
                latency_ms=latency_ms,
                error_message=str(exc),
            )

    def _validate_provider_id(self) -> None:
        if not (
            self.hit_provider
            .provider_id
            .strip()
        ):
            raise (
                ComplexOracleRetrievalAdapterError(
                    "Hit Provider 的 "
                    "provider_id 不能为空"
                )
            )

    def _validate_query_report(
        self,
        query: ComplexRetrievalQueryOutput,
    ) -> None:
        report = (
            self.registry_bundle
            .reports
            .get(query.report_id)
        )

        if report is None:
            raise (
                ComplexOracleRetrievalAdapterError(
                    "Query 引用了不存在的 Report："
                    f"{query.report_id}"
                )
            )

        if (
            report.company_id
            != query.company_id
        ):
            raise (
                ComplexOracleRetrievalAdapterError(
                    "Query 的 company_id "
                    "与 Report 不一致"
                )
            )

        if (
            report.fiscal_year
            != query.fiscal_year
        ):
            raise (
                ComplexOracleRetrievalAdapterError(
                    "Query 的 fiscal_year "
                    "与 Report 不一致"
                )
            )

        if (
            report.report_type
            is not query.report_type
        ):
            raise (
                ComplexOracleRetrievalAdapterError(
                    "Query 的 report_type "
                    "与 Report 不一致"
                )
            )

    def _validate_hits(
        self,
        *,
        query: ComplexRetrievalQueryOutput,
        hits: tuple[
            RerankedRetrievalHit,
            ...,
        ],
        top_k: int,
    ) -> None:
        if len(hits) > top_k:
            raise (
                ComplexOracleRetrievalAdapterError(
                    "Hit 数量超过 top_k："
                    f"hit_count={len(hits)}, "
                    f"top_k={top_k}"
                )
            )

        actual_ranks = tuple(
            hit.rank
            for hit in hits
        )

        expected_ranks = tuple(
            range(1, len(hits) + 1)
        )

        if actual_ranks != expected_ranks:
            raise (
                ComplexOracleRetrievalAdapterError(
                    "Reranked Hit 的 rank "
                    "必须从 1 连续递增"
                )
            )

        chunk_ids = tuple(
            hit.chunk_id
            for hit in hits
        )

        if len(chunk_ids) != len(
            set(chunk_ids)
        ):
            raise (
                ComplexOracleRetrievalAdapterError(
                    "Reranked Hit 包含重复 "
                    "chunk_id"
                )
            )

        for hit in hits:
            if (
                hit.company_id
                != query.company_id
                or hit.report_id
                != query.report_id
                or hit.fiscal_year
                != query.fiscal_year
                or hit.report_type
                is not query.report_type
            ):
                raise (
                    ComplexOracleRetrievalAdapterError(
                        "Reranked Hit 与 Query "
                        "的报告身份不一致："
                        f"{hit.chunk_id}"
                    )
                )

    def _resolve_facts(
        self,
        *,
        query: ComplexRetrievalQueryOutput,
        hits: tuple[
            RerankedRetrievalHit,
            ...,
        ],
    ) -> tuple[
        tuple[str, ...],
        tuple[str, ...],
    ]:
        """根据结构化 Query 和命中页面解析事实。

        这里只使用 Query 的公司、报告、指标、年份、
        报表类型和口径，不使用 Gold Fact ID。
        """

        candidate_facts = (
            self.registry_bundle
            .financial_facts
            .find(
                company_id=query.company_id,
                report_id=query.report_id,
                metric_id=query.metric_id,
                fiscal_year=(
                    query.fiscal_year
                ),
                statement_scope=(
                    query
                    .statement_scope
                    .value
                ),
            )
        )

        verified_candidates = tuple(
            fact
            for fact in candidate_facts
            if (
                fact.statement_type
                is query.statement_type
                and fact.validation_status
                is ValidationStatus.VERIFIED
            )
        )

        facts_by_evidence: list[
            tuple[object, SourceEvidence]
        ] = []

        for fact in verified_candidates:
            evidence = (
                self.registry_bundle
                .evidences
                .get(
                    fact.primary_evidence_id
                )
            )

            if evidence is None:
                continue

            if (
                evidence.validation_status
                is not ValidationStatus.VERIFIED
            ):
                continue

            facts_by_evidence.append(
                (fact, evidence)
            )

        resolved_fact_ids: list[str] = []
        resolved_evidence_ids: list[str] = []

        seen_fact_ids: set[str] = set()
        seen_evidence_ids: set[str] = set()

        # 按实际 Hit 排名决定 Fact 顺序。
        for hit in hits:
            for fact, evidence in (
                facts_by_evidence
            ):
                if fact.fact_id in seen_fact_ids:
                    continue

                if not _hit_supports_evidence(
                    hit=hit,
                    evidence=evidence,
                ):
                    continue

                seen_fact_ids.add(
                    fact.fact_id
                )

                resolved_fact_ids.append(
                    fact.fact_id
                )

                if (
                    evidence.evidence_id
                    not in seen_evidence_ids
                ):
                    seen_evidence_ids.add(
                        evidence.evidence_id
                    )

                    resolved_evidence_ids.append(
                        evidence.evidence_id
                    )

        return (
            tuple(resolved_fact_ids),
            tuple(resolved_evidence_ids),
        )


def _hit_supports_evidence(
    *,
    hit: RerankedRetrievalHit,
    evidence: SourceEvidence,
) -> bool:
    """判断一个 Hit 是否覆盖目标 Evidence。"""

    if hit.report_id != evidence.report_id:
        return False

    if evidence.chunk_id is not None:
        return (
            hit.chunk_id
            == evidence.chunk_id
        )

    # 当前人工 Evidence 主要记录到 PDF 页级，
    # 因此 chunk_id 为空时采用报告 + PDF 页匹配。
    return hit.pdf_page == evidence.pdf_page