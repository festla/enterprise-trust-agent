from __future__ import annotations

import re

from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import (
    dataclass,
    field,
)

from app.schemas.agent_runtime import (
    ParsedFinancialQuery,
)
from app.schemas.enums import (
    MetricOrigin,
    StatementScope,
)
from app.services.registry import (
    RegistryBundle,
)


class RuntimeQueryParserError(
    ValueError
):
    """Runtime 用户问题解析基础异常。"""


# ============================================================
# 这些是“业务别名”，不是 Gold 数据。
#
# 它们解决的是：
#
#   用户说“营收”
#   Registry 标准名称叫“营业收入”
#
#   用户说“美的”
#   Registry 标准名称叫“美的集团”
#
# 这属于实体标准化，而不是评测作弊。
#
# 更完整的 Alias Registry 后续可以替换这里，
# Parser 本身不用改变。
# ============================================================


_DEFAULT_COMPANY_ALIASES = {
    "美的": "midea_group",
    "格力": "gree_electric",
    "海尔": "haier_smart_home",
    "海信": "hisense_home",
    "老板": "robam",
    "苏泊尔": "supor",
}


_DEFAULT_METRIC_ALIASES = {
    "营收": "revenue",
    "营业额": "revenue",
    "归母净利润": (
        "net_profit_attributable_to_parent"
    ),
    "经营现金流": (
        "net_cash_flow_from_operating_activities"
    ),
    "经营活动现金流净额": (
        "net_cash_flow_from_operating_activities"
    ),
    "总资产": "total_assets",
    "总负债": "total_liabilities",
    "研发费用": (
        "research_and_development_expenses"
    ),
    "销售费用": "selling_expenses",
    "毛利率": "gross_profit_margin",
}


_YEAR_PATTERN = re.compile(
    r"(?<!\d)(20\d{2}|2100)(?:\s*年)?"
)


_COMPARISON_KEYWORDS = (
    "比较",
    "对比",
    "相比",
    "同比",
    "环比",
    "差异",
)


_RANKING_KEYWORDS = (
    "排名",
    "排行",
    "最高",
    "最低",
    "最大",
    "最小",
    "哪家更高",
    "谁更高",
)


_EXPLANATION_KEYWORDS = (
    "为什么",
    "为何",
    "原因",
    "解释",
    "驱动因素",
    "变动原因",
    "增长原因",
    "下降原因",
)


# ============================================================
# 我们做 Phrase Matching 时不能简单：
#
#     if "净利润" in question
#
# 因为：
#
# “归属于母公司所有者的净利润”
#
# 同时也包含：
#
# “净利润”
#
# 如果两个都匹配，
# Agent 就不知道用户究竟问哪个指标。
#
# 所以这里采用：
#
#       最长短语优先
#       +
#       重叠区间不重复匹配
#
# ============================================================


def _extract_non_overlapping_ids(
    *,
    text: str, # 用户问题
    candidates: Sequence[  # 候选词 + 标准ID
        tuple[str, str]
    ],
) -> tuple[str, ...]:
    normalized_text = (
        text.casefold()
    )

    raw_matches: list[
        tuple[
            int,
            int,
            int,
            str,
        ]
    ] = []

    for phrase, canonical_id in candidates:
        normalized_phrase = (
            phrase.strip().casefold()
        )

        if not normalized_phrase:
            continue

        search_start = 0

        while True:
            start = normalized_text.find(
                normalized_phrase,
                search_start,
            )

            if start < 0:
                break

            end = (
                start
                + len(normalized_phrase)
            )

            raw_matches.append(
                (
                    start,
                    end,
                    len(normalized_phrase),
                    canonical_id,
                )
            )

            search_start = start + 1

    # ========================================================
    # 长词先处理。
    #
    # 例如：
    #
    # 归属于母公司所有者的净利润
    #              ↓
    # 比“净利润”优先。
    # ========================================================

    raw_matches.sort(
        key=lambda item: (
            -item[2],
            item[0],
            item[1],
            item[3],
        )
    )

    accepted_matches: list[
        tuple[
            int,
            int,
            str,
        ]
    ] = []

    for (
        start,
        end,
        _,
        canonical_id,
    ) in raw_matches:
        overlaps = any(
            not (
                end <= accepted_start
                or start >= accepted_end
            )
            for (
                accepted_start,
                accepted_end,
                _,
            )
            in accepted_matches
        )

        if overlaps:
            continue

        accepted_matches.append(
            (
                start,
                end,
                canonical_id,
            )
        )

    # 最终还是按照用户原问题中出现的顺序返回。
    accepted_matches.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    result: list[str] = []
    seen: set[str] = set()

    for _, _, canonical_id in (
        accepted_matches
    ):
        if canonical_id in seen:
            continue

        seen.add(canonical_id)
        result.append(canonical_id)

    return tuple(result)


def _normalize_question(
    question: str,
) -> str:
    # 把连续空白统一成一个空格。
    return " ".join(
        question.strip().split()
    )


def _contains_any(
    text: str,
    keywords: Sequence[str],
) -> bool:
    return any(
        keyword in text
        for keyword in keywords
    )


# ============================================================
# Parser 只做：
#
# 用户自然语言
#       ↓
# ParsedFinancialQuery
#
# 它绝对不负责：
#
# - 查 FinancialFact
# - 检索 Chunk
# - 执行公式
# - 生成答案
#
# 后面分别由 Router / Planner / ToolExecutor 完成。
# ============================================================


@dataclass(
    frozen=True, #对象创建后字段不允许修改
    slots=True, #限制对象能够拥有的属性
)
class RuntimeQueryParser:
    registry_bundle: RegistryBundle

    company_aliases: Mapping[
        str,
        str,
    ] = field(
        default_factory=lambda: dict(
            _DEFAULT_COMPANY_ALIASES
        )
    )

    metric_aliases: Mapping[
        str,
        str,
    ] = field(
        default_factory=lambda: dict(
            _DEFAULT_METRIC_ALIASES
        )
    )

    parser_version: str = (
        "deterministic_runtime_query_parser_v1"
    )

    def parse(
        self,
        question: str,
    ) -> ParsedFinancialQuery:
        normalized_question = (
            _normalize_question(
                question
            )
        )

        if not normalized_question:
            raise RuntimeQueryParserError(
                "用户问题不能为空"
            )

        company_ids = (
            _extract_non_overlapping_ids(
                text=normalized_question,
                candidates=(
                    self
                    ._build_company_candidates()
                ),
            )
        )

        years = self._extract_years(
            normalized_question
        )

        matched_metric_ids = (
            _extract_non_overlapping_ids(
                text=normalized_question,
                candidates=(
                    self
                    ._build_metric_candidates()
                ),
            )
        )

        (
            metric_ids,
            calculation_metric_ids,
        ) = self._split_metric_types(
            matched_metric_ids
        )

        # 报表口径问题
        (
            statement_scope,
            scope_notes,
        ) = self._parse_statement_scope(
            normalized_question
        )

        (
            report_ids,
            report_notes,
        ) = self._resolve_report_ids(
            company_ids=company_ids,
            years=years,
        )

        comparison_requested = (
            _contains_any(
                normalized_question,
                _COMPARISON_KEYWORDS,
            )
            or len(years) > 1
            or len(company_ids) > 1
        )

        ranking_requested = (
            _contains_any(
                normalized_question,
                _RANKING_KEYWORDS,
            )
        )

        explanation_requested = (
            _contains_any(
                normalized_question,
                _EXPLANATION_KEYWORDS,
            )
        )

        ambiguity_notes = (
            scope_notes # 口径歧义
            + report_notes # 报告缺失问题
        )

        missing_fields = (
            self._build_missing_fields(
                company_ids=company_ids,
                years=years,
                report_ids=report_ids,
            )
        )

        confidence = (
            self._calculate_confidence(
                company_ids=company_ids,
                years=years,
                metric_ids=metric_ids,
                calculation_metric_ids=(
                    calculation_metric_ids
                ),
                explanation_requested=(
                    explanation_requested
                ),
                ranking_requested=(
                    ranking_requested
                ),
                missing_fields=(
                    missing_fields
                ),
                ambiguity_notes=(
                    ambiguity_notes
                ),
            )
        )

        return ParsedFinancialQuery(
            normalized_question=(
                normalized_question
            ),
            company_ids=company_ids,
            report_ids=report_ids,
            years=years,
            metric_ids=metric_ids,
            calculation_metric_ids=(
                calculation_metric_ids
            ),
            statement_scope=(
                statement_scope
            ),
            comparison_requested=(
                comparison_requested
            ),
            ranking_requested=(
                ranking_requested
            ),
            explanation_requested=(
                explanation_requested
            ),

            # =================================================
            # Parser 不判断 supported / unsupported。
            #
            # “这个问题系统能不能处理”
            # 是下一步 IntentRouter 的职责。
            # =================================================
            unsupported_reason=None,

            missing_fields=(
                missing_fields
            ),
            assumptions=(),
            ambiguity_notes=(
                ambiguity_notes
            ),
            confidence=confidence,
        )

    # ========================================================
    # 公司不是写死成：
    #
    # if "美的集团" ...
    #
    # 而是优先从 CompanyRegistry 动态建立词典。
    #
    # 因此以后 Registry 新增公司，
    # Parser 不需要改核心代码。
    # ========================================================

    def _build_company_candidates(
        self,
    ) -> tuple[
        tuple[str, str],
        ...
    ]:
        candidates: list[
            tuple[str, str]
        ] = []

        known_company_ids = set(
            self.registry_bundle
            .companies
            .keys()
        )

        for company in (
            self.registry_bundle
            .companies
            .values()
        ):
            candidates.extend(
                (
                    (
                        company.legal_name_cn,
                        company.company_id,
                    ),
                    (
                        company.short_name_cn,
                        company.company_id,
                    ),
                    (
                        company.stock_code,
                        company.company_id,
                    ),
                )
            )

        # 常用简称只是标准 Registry 的补充。
        for (
            alias,
            company_id,
        ) in self.company_aliases.items():
            if (
                company_id
                not in known_company_ids
            ):
                continue

            candidates.append(
                (
                    alias,
                    company_id,
                )
            )

        return tuple(candidates)

    def _build_metric_candidates(
        self,
    ) -> tuple[
        tuple[str, str],
        ...
    ]:
        candidates: list[
            tuple[str, str]
        ] = []

        known_metric_ids = set(
            self.registry_bundle
            .metrics
            .keys()
        )

        for metric in (
            self.registry_bundle
            .metrics
            .values()
        ):
            candidates.append(
                (
                    metric.display_name_cn,
                    metric.metric_id,
                )
            )

            # 支持用户直接输入内部 metric_id。
            candidates.append(
                (
                    metric.metric_id,
                    metric.metric_id,
                )
            )

            if metric.display_name_en:
                candidates.append(
                    (
                        metric.display_name_en,
                        metric.metric_id,
                    )
                )

        for (
            alias,
            metric_id,
        ) in self.metric_aliases.items():
            if (
                metric_id
                not in known_metric_ids
            ):
                continue

            candidates.append(
                (
                    alias,
                    metric_id,
                )
            )

        return tuple(candidates)

    @staticmethod
    def _extract_years(
        text: str,
    ) -> tuple[int, ...]:
        result: list[int] = []
        seen: set[int] = set()

        for match in (
            _YEAR_PATTERN.finditer(
                text
            )
        ):
            year = int(
                match.group(1)
            )

            if year in seen:
                continue

            seen.add(year)
            result.append(year)

        return tuple(result)

    # ========================================================
    # reported metric：
    #
    #   revenue
    #   operating_cost
    #
    # 可以直接查询 FinancialFact。
    #
    # derived metric：
    #
    #   gross_profit_margin
    #
    # 后面 Planner 要先查询输入 Fact，
    # 再调用 execute_calculation。
    #
    # 所以 Parser 必须在这里分开。
    # ========================================================

    def _split_metric_types(
        self,
        metric_ids: tuple[str, ...],
    ) -> tuple[
        tuple[str, ...],
        tuple[str, ...],
    ]:
        reported: list[str] = []
        derived: list[str] = []

        for metric_id in metric_ids:
            metric = (
                self.registry_bundle
                .metrics
                .get(metric_id)
            )

            if metric is None:
                continue

            if (
                metric.metric_origin
                is MetricOrigin.DERIVED
            ):
                derived.append(
                    metric_id
                )
            else:
                reported.append(
                    metric_id
                )

        return (
            tuple(reported),
            tuple(derived),
        )

    @staticmethod
    def _parse_statement_scope(
        text: str,
    ) -> tuple[
        StatementScope | None,
        tuple[str, ...],
    ]:
        has_consolidated = (
            "合并口径" in text
            or "合并报表" in text
            or "合并财务报表" in text
        )

        has_parent = (
            "母公司口径" in text
            or "母公司报表" in text
            or "母公司财务报表" in text
        )

        if (
            has_consolidated
            and has_parent
        ):
            return (
                None,
                (
                    "问题同时出现合并口径"
                    "和母公司口径，无法唯一确定"
                    " statement_scope",
                ),
            )

        if has_consolidated:
            return (
                StatementScope.CONSOLIDATED,
                (),
            )

        if has_parent:
            return (
                StatementScope.PARENT_COMPANY,
                (),
            )

        return None, ()

    # ========================================================
    # 不能直接：
    #
    # report_id = f"{company_id}_{year}"
    #
    # 虽然你现在的命名规范确实如此。
    #
    # Production Runtime 应该确认这个 Report
    # 真的存在于 Registry。
    #
    # 不能“猜出一个不存在的 report_id”。
    # ========================================================

    def _resolve_report_ids(
        self,
        *,
        company_ids: tuple[str, ...],
        years: tuple[int, ...],
    ) -> tuple[
        tuple[str, ...],
        tuple[str, ...],
    ]:
        if (
            not company_ids
            or not years
        ):
            return (), ()

        report_ids: list[str] = []
        notes: list[str] = []

        reports = (
            self.registry_bundle
            .reports
            .values()
        )

        for company_id in company_ids:
            for year in years:
                matching_reports = [
                    report
                    for report in reports
                    if (
                        report.company_id
                        == company_id
                        and report.fiscal_year
                        == year
                    )
                ]

                if not matching_reports:
                    notes.append(
                        "Registry 中未找到报告："
                        f"company_id={company_id}, "
                        f"year={year}"
                    )
                    continue

                matching_reports.sort(
                    key=lambda report: (
                        report.report_id
                    )
                )

                for report in (
                    matching_reports
                ):
                    if (
                        report.report_id
                        not in report_ids
                    ):
                        report_ids.append(
                            report.report_id
                        )

        return (
            tuple(report_ids),
            tuple(notes),
        )

    @staticmethod
    def _build_missing_fields(
        *,
        company_ids: tuple[str, ...],
        years: tuple[int, ...],
        report_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        missing: list[str] = []

        if not company_ids:
            missing.append(
                "company_ids"
            )

        if not years:
            missing.append(
                "years"
            )

        if company_ids and years:
            expected_report_count = (
                len(company_ids)
                * len(years)
            )

            if (
                len(report_ids)
                < expected_report_count
            ):
                missing.append(
                    "report_ids"
                )

        # ====================================================
        # 这里故意不写：
        #
        # if not metric_ids:
        #     missing.append("metric_ids")
        #
        # 因为：
        #
        # “美的2024年经营风险有哪些？”
        #
        # 是合法 document_evidence Query，
        # 它本来就可能没有财务 metric。
        #
        # 是否必须存在 metric，
        # 要等 7B Router 判断 Intent 后才能决定。
        # ====================================================

        return tuple(missing)

    @staticmethod
    def _calculate_confidence(
        *,
        company_ids: tuple[str, ...],
        years: tuple[int, ...],
        metric_ids: tuple[str, ...],
        calculation_metric_ids: tuple[
            str,
            ...
        ],
        explanation_requested: bool,
        ranking_requested: bool,
        missing_fields: tuple[str, ...],
        ambiguity_notes: tuple[str, ...],
    ) -> float:
        # 三类主要解析信号：
        #
        # 公司 / 年份 / 业务目标
        signal_count = 0

        if company_ids:
            signal_count += 1

        if years:
            signal_count += 1

        has_business_target = (
            bool(metric_ids)
            or bool(
                calculation_metric_ids
            )
            or explanation_requested
            or ranking_requested
        )

        if has_business_target:
            signal_count += 1

        confidence = (
            signal_count / 3.0
        )

        if (
            "report_ids"
            in missing_fields
        ):
            confidence -= 0.1

        if ambiguity_notes:
            confidence -= 0.1

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        return round(
            confidence,
            4,
        )