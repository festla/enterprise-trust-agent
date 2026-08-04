from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Iterable, TypeVar

from app.schemas import (
    Company,
    FinancialFact,
    FinancialMetric,
    Report,
    SourceEvidence,
)


T = TypeVar("T")


class RegistryError(ValueError):
    """Registry 基础异常"""


class DuplicateRegistryKeyError(RegistryError):
    """注册表中出现重复主键。"""


class RegistryItemNotFoundError(RegistryError):
    """注册表中不存在目标对象。"""


class RegistryIntegrityError(RegistryError):
    """多个 Registry 之间的引用关系不完整。"""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors

        message = "Registry 跨对象完整性检查失败：\n- "
        message += "\n- ".join(errors)

        super().__init__(message)


class BaseRegistry(Generic[T]):
    """按照指定字段保存和查询 Pydantic 对象。"""

    def __init__(self, key_field: str) -> None:
        self._key_field = key_field
        self._items: dict[str, T] = {}

    def _get_item_key(self, item: T) -> str:
        """从对象中取得 Registry 主键。"""

        key = getattr(item, self._key_field, None)

        if not isinstance(key, str) or not key:
            raise RegistryError(
                f"对象缺少合法的字符串主键字段："
                f"{self._key_field}"
            )

        return key

    def add(self, item: T) -> None:
        """向 Registry 添加一条记录"""

        key = self._get_item_key(item)

        if key in self._items:
            raise DuplicateRegistryKeyError(
                f"{self._key_field} 已存在：{key}"
            )

        self._items[key] = item

    def add_many(self, items: Iterable[T]) -> None:
        """批量添加记录"""

        for item in items:
            self.add(item)

    def get(self, key: str) -> T | None:
        """查询对象：不存在时返回 None"""

        return self._items.get(key)

    def require(self, key: str) -> T:
        """查询对象；不存在时抛出明确异常"""

        item = self.get(key)

        if item is None:
            raise RegistryItemNotFoundError(
                f"{self._key_field} 不存在：{key}"
            )

        return item

    def contains(self, key: str) -> bool:
        """判断主键是否已经存在。"""

        return key in self._items

    def values(self) -> tuple[T, ...]:
        """以只读元组形式返回全部对象。"""

        return tuple(self._items.values())

    def keys(self) -> tuple[str, ...]:
        """返回 Registry 的全部主键。"""

        return tuple(self._items.keys())

    def __len__(self) -> int:
        """返回记录数量。"""

        return len(self._items)


class CompanyRegistry(BaseRegistry[Company]):
    """公司注册表。"""

    def __init__(self) -> None:
        super().__init__(key_field="company_id")


class ReportRegistry(BaseRegistry[Report]):
    """报告注册表。"""

    def __init__(self) -> None:
        super().__init__(key_field="report_id")


class MetricRegistry(BaseRegistry[FinancialMetric]):
    """财务指标注册表。"""

    def __init__(self) -> None:
        super().__init__(key_field="metric_id")


class EvidenceRegistry(BaseRegistry[SourceEvidence]):
    """来源证据注册表。"""

    def __init__(self) -> None:
        super().__init__(key_field="evidence_id")


class FinancialFactRegistry(BaseRegistry[FinancialFact]):
    """财务事实注册表。"""

    def __init__(self) -> None:
        super().__init__(key_field="fact_id")

    def find(
        self,
        *,
        company_id: str | None = None,
        report_id: str | None = None,
        metric_id: str | None = None,
        fiscal_year: int | None = None,
        statement_scope: str | None = None,
    ) -> list[FinancialFact]:
        """按照常用业务条件查询财务事实。"""

        results: list[FinancialFact] = []

        for fact in self.values():
            if (
                company_id is not None
                and fact.company_id != company_id
            ):
                continue

            if (
                report_id is not None
                and fact.report_id != report_id
            ):
                continue

            if (
                metric_id is not None
                and fact.metric_id != metric_id
            ):
                continue

            if (
                fiscal_year is not None
                and fact.fiscal_year != fiscal_year
            ):
                continue

            if (
                statement_scope is not None
                and fact.statement_scope.value
                != statement_scope
            ):
                continue

            results.append(fact)

        return results


@dataclass(slots=True) #限制对象只能拥有预先定义的属性
class RegistryBundle:
    """聚合项目当前使用的所有 Registry。"""

    companies: CompanyRegistry = field(
        default_factory=CompanyRegistry
    )

    reports: ReportRegistry = field(
        default_factory=ReportRegistry
    )

    metrics: MetricRegistry = field(
        default_factory=MetricRegistry
    )

    evidences: EvidenceRegistry = field(
        default_factory=EvidenceRegistry
    )

    financial_facts: FinancialFactRegistry = field(
        default_factory=FinancialFactRegistry
    )

    def validate_relationships(self) -> None:
        """检查不同 Registry 之间的引用完整性。"""

        errors: list[str] = []

        self._validate_reports(errors)
        self._validate_evidences(errors)
        self._validate_financial_facts(errors)

        if errors:
            raise RegistryIntegrityError(errors)

    def _validate_reports(
        self,
        errors: list[str],
    ) -> None:
        """检查 Report 到 Company 的引用。"""

        for report in self.reports.values():
            if not self.companies.contains(
                report.company_id
            ):
                errors.append(
                    f"Report '{report.report_id}' "
                    f"引用了不存在的 Company "
                    f"'{report.company_id}'"
                )

    def _validate_evidences(
        self,
        errors: list[str],
    ) -> None:
        """检查 Evidence 到 Report 的引用。"""

        for evidence in self.evidences.values():
            if not self.reports.contains(
                evidence.report_id
            ):
                errors.append(
                    f"Evidence '{evidence.evidence_id}' "
                    f"引用了不存在的 Report "
                    f"'{evidence.report_id}'"
                )

    def _validate_financial_facts(
        self,
        errors: list[str],
    ) -> None:
        """检查 FinancialFact 的全部主要引用。"""

        for fact in self.financial_facts.values():
            self._validate_fact_company(fact, errors)
            self._validate_fact_report(fact, errors)
            self._validate_fact_metric(fact, errors)
            self._validate_fact_evidence(fact, errors)

    def _validate_fact_company(
        self,
        fact: FinancialFact,
        errors: list[str],
    ) -> None:
        if not self.companies.contains(fact.company_id):
            errors.append(
                f"FinancialFact '{fact.fact_id}' "
                f"引用了不存在的 Company "
                f"'{fact.company_id}'"
            )

    def _validate_fact_report(
        self,
        fact: FinancialFact,
        errors: list[str],
    ) -> None:
        report = self.reports.get(fact.report_id)

        if report is None:
            errors.append(
                f"FinancialFact '{fact.fact_id}' "
                f"引用了不存在的 Report "
                f"'{fact.report_id}'"
            )
            return

        if report.company_id != fact.company_id:
            errors.append(
                f"FinancialFact '{fact.fact_id}' "
                "与来源 Report 的 company_id 不一致"
            )

    def _validate_fact_metric(
        self,
        fact: FinancialFact,
        errors: list[str],
    ) -> None:
        metric = self.metrics.get(fact.metric_id)

        if metric is None:
            errors.append(
                f"FinancialFact '{fact.fact_id}' "
                f"引用了不存在的 FinancialMetric "
                f"'{fact.metric_id}'"
            )
            return

        if metric.period_type is not fact.period_type:
            errors.append(
                f"FinancialFact '{fact.fact_id}' "
                "的 period_type 与 FinancialMetric 不一致"
            )

        if (
            fact.statement_scope
            not in metric.allowed_scopes
        ):
            errors.append(
                f"FinancialFact '{fact.fact_id}' "
                "使用了 FinancialMetric 不允许的 "
                "statement_scope"
            )

        if (
            metric.default_unit
            is not fact.normalized_unit
        ):
            errors.append(
                f"FinancialFact '{fact.fact_id}' "
                "的 normalized_unit 与 "
                "FinancialMetric.default_unit 不一致"
            )

    def _validate_fact_evidence(
        self,
        fact: FinancialFact,
        errors: list[str],
    ) -> None:
        evidence = self.evidences.get(
            fact.primary_evidence_id
        )

        if evidence is None:
            errors.append(
                f"FinancialFact '{fact.fact_id}' "
                f"引用了不存在的 SourceEvidence "
                f"'{fact.primary_evidence_id}'"
            )
            return

        if evidence.report_id != fact.report_id:
            errors.append(
                f"FinancialFact '{fact.fact_id}' "
                "与主要 Evidence 的 report_id 不一致"
            )

        if (
            evidence.statement_scope is not None
            and evidence.statement_scope
            is not fact.statement_scope
        ):
            errors.append(
                f"FinancialFact '{fact.fact_id}' "
                "与主要 Evidence 的 statement_scope "
                "不一致"
            )