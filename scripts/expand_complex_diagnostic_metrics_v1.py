from __future__ import annotations

import shutil
from collections import Counter
from copy import deepcopy
from pathlib import Path

import yaml

from app.schemas.metric import (
    FinancialMetric,
    MetricAlias,
)
from app.services.registry import MetricRegistry
from app.services.registry_loader import (
    load_metrics,
    validate_metric_alias_relationships,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "registries"
    / "metrics.yaml"
)

BACKUP_PATH = METRICS_PATH.with_name(
    "metrics.before_complex_diagnostic_dev_v1.yaml"
)

TEMP_PATH = METRICS_PATH.with_name(
    "metrics.complex_diagnostic_dev_v1.tmp.yaml"
)

TIMESTAMP = "2026-08-03T00:00:00+08:00"


NEW_METRICS = [
    {
        "metric_id": "non_current_assets",
        "display_name_cn": "非流动资产合计",
        "display_name_en": "Total Non-current Assets",
        "description": (
            "资产负债表日被分类为非流动资产的资产合计，"
            "不等同于流动资产合计或资产总计。"
        ),
        "metric_origin": "reported",
        "statement_type": "balance_sheet",
        "period_type": "instant",
        "default_unit": "CNY",
        "allowed_scopes": [
            "consolidated",
            "parent_company",
        ],
        "value_type": "decimal",
        "is_core_metric": True,
        "confusable_metric_ids": [
            "current_assets",
            "total_assets",
        ],
        "status": "active",
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
    },
    {
        "metric_id": (
            "cash_outflows_from_investing_"
            "activities_subtotal"
        ),
        "display_name_cn": "投资活动现金流出小计",
        "display_name_en": (
            "Subtotal of Cash Outflows "
            "from Investing Activities"
        ),
        "description": (
            "现金流量表中投资活动各项现金流出的合计，"
            "不等同于投资活动产生的现金流量净额。"
        ),
        "metric_origin": "reported",
        "statement_type": "cash_flow_statement",
        "period_type": "duration",
        "default_unit": "CNY",
        "allowed_scopes": [
            "consolidated",
            "parent_company",
        ],
        "value_type": "decimal",
        "is_core_metric": True,
        "confusable_metric_ids": [
            "net_cash_flow_from_investing_activities",
        ],
        "status": "active",
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
    },
    {
        "metric_id": "taxes_and_surcharges",
        "display_name_cn": "税金及附加",
        "display_name_en": "Taxes and Surcharges",
        "description": (
            "利润表中计入当期损益的税金及附加，"
            "不等同于所得税费用。"
        ),
        "metric_origin": "reported",
        "statement_type": "income_statement",
        "period_type": "duration",
        "default_unit": "CNY",
        "allowed_scopes": [
            "consolidated",
            "parent_company",
        ],
        "value_type": "decimal",
        "is_core_metric": True,
        "confusable_metric_ids": [
            "income_tax_expense",
        ],
        "status": "active",
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
    },
    {
        "metric_id": (
            "cash_received_from_sales_of_goods_"
            "and_rendering_of_services"
        ),
        "display_name_cn": (
            "销售商品、提供劳务收到的现金"
        ),
        "display_name_en": (
            "Cash Received from Sales of Goods "
            "and Rendering of Services"
        ),
        "description": (
            "现金流量表中因销售商品和提供劳务收到的现金，"
            "不等同于营业收入或经营活动现金流量净额。"
        ),
        "metric_origin": "reported",
        "statement_type": "cash_flow_statement",
        "period_type": "duration",
        "default_unit": "CNY",
        "allowed_scopes": [
            "consolidated",
            "parent_company",
        ],
        "value_type": "decimal",
        "is_core_metric": True,
        "confusable_metric_ids": [
            "net_cash_flow_from_operating_activities",
        ],
        "status": "active",
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
    },
    {
        "metric_id": "tax_refunds_received",
        "display_name_cn": "收到的税费返还",
        "display_name_en": "Tax Refunds Received",
        "description": (
            "现金流量表中企业收到的税费返还现金，"
            "不等同于利润表中的所得税费用。"
        ),
        "metric_origin": "reported",
        "statement_type": "cash_flow_statement",
        "period_type": "duration",
        "default_unit": "CNY",
        "allowed_scopes": [
            "consolidated",
            "parent_company",
        ],
        "value_type": "decimal",
        "is_core_metric": True,
        "confusable_metric_ids": [
            "income_tax_expense",
        ],
        "status": "active",
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
    },
    {
        "metric_id": (
            "other_comprehensive_income_net_of_tax"
        ),
        "display_name_cn": "其他综合收益的税后净额",
        "display_name_en": (
            "Other Comprehensive Income, "
            "Net of Tax"
        ),
        "description": (
            "利润表中扣除相关所得税影响后的其他综合收益，"
            "不等同于净利润或综合收益总额。"
        ),
        "metric_origin": "reported",
        "statement_type": "income_statement",
        "period_type": "duration",
        "default_unit": "CNY",
        "allowed_scopes": [
            "consolidated",
            "parent_company",
        ],
        "value_type": "decimal",
        "is_core_metric": True,
        "confusable_metric_ids": [
            "net_profit",
            "total_comprehensive_income",
        ],
        "status": "active",
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
    },
    {
        "metric_id": "total_comprehensive_income",
        "display_name_cn": "综合收益总额",
        "display_name_en": (
            "Total Comprehensive Income"
        ),
        "description": (
            "报告期净利润与其他综合收益税后净额形成的"
            "综合收益总额，不等同于净利润。"
        ),
        "metric_origin": "reported",
        "statement_type": "income_statement",
        "period_type": "duration",
        "default_unit": "CNY",
        "allowed_scopes": [
            "consolidated",
            "parent_company",
        ],
        "value_type": "decimal",
        "is_core_metric": True,
        "confusable_metric_ids": [
            "net_profit",
            "other_comprehensive_income_net_of_tax",
        ],
        "status": "active",
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
    },
    {
        "metric_id": "non_current_liabilities",
        "display_name_cn": "非流动负债合计",
        "display_name_en": (
            "Total Non-current Liabilities"
        ),
        "description": (
            "资产负债表日被分类为非流动负债的负债合计，"
            "不等同于流动负债合计或负债合计。"
        ),
        "metric_origin": "reported",
        "statement_type": "balance_sheet",
        "period_type": "instant",
        "default_unit": "CNY",
        "allowed_scopes": [
            "consolidated",
            "parent_company",
        ],
        "value_type": "decimal",
        "is_core_metric": True,
        "confusable_metric_ids": [
            "current_liabilities",
            "total_liabilities",
        ],
        "status": "active",
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
    },
]


NEW_ALIASES = [
    {
        "alias_id": "non_current_assets_total_cn",
        "metric_id": "non_current_assets",
        "alias": "非流动资产合计",
        "statement_type": "balance_sheet",
        "match_type": "exact",
        "priority": 10,
        "notes": "资产负债表标准披露名称。",
        "status": "active",
    },
    {
        "alias_id": (
            "investing_cash_outflows_subtotal_cn"
        ),
        "metric_id": (
            "cash_outflows_from_investing_"
            "activities_subtotal"
        ),
        "alias": "投资活动现金流出小计",
        "statement_type": "cash_flow_statement",
        "match_type": "exact",
        "priority": 10,
        "notes": "现金流量表标准披露名称。",
        "status": "active",
    },
    {
        "alias_id": (
            "taxes_and_surcharges_standard_cn"
        ),
        "metric_id": "taxes_and_surcharges",
        "alias": "税金及附加",
        "statement_type": "income_statement",
        "match_type": "exact",
        "priority": 10,
        "notes": "利润表标准披露名称。",
        "status": "active",
    },
    {
        "alias_id": (
            "cash_received_from_sales_and_services_cn"
        ),
        "metric_id": (
            "cash_received_from_sales_of_goods_"
            "and_rendering_of_services"
        ),
        "alias": "销售商品、提供劳务收到的现金",
        "statement_type": "cash_flow_statement",
        "match_type": "exact",
        "priority": 10,
        "notes": "现金流量表标准披露名称。",
        "status": "active",
    },
    {
        "alias_id": (
            "tax_refunds_received_standard_cn"
        ),
        "metric_id": "tax_refunds_received",
        "alias": "收到的税费返还",
        "statement_type": "cash_flow_statement",
        "match_type": "exact",
        "priority": 10,
        "notes": "现金流量表标准披露名称。",
        "status": "active",
    },
    {
        "alias_id": (
            "other_comprehensive_income_net_of_tax_cn"
        ),
        "metric_id": (
            "other_comprehensive_income_net_of_tax"
        ),
        "alias": "其他综合收益的税后净额",
        "statement_type": "income_statement",
        "match_type": "exact",
        "priority": 10,
        "notes": "利润表标准披露名称。",
        "status": "active",
    },
    {
        "alias_id": (
            "total_comprehensive_income_standard_cn"
        ),
        "metric_id": "total_comprehensive_income",
        "alias": "综合收益总额",
        "statement_type": "income_statement",
        "match_type": "exact",
        "priority": 10,
        "notes": "利润表标准披露名称。",
        "status": "active",
    },
    {
        "alias_id": (
            "non_current_liabilities_total_cn"
        ),
        "metric_id": "non_current_liabilities",
        "alias": "非流动负债合计",
        "statement_type": "balance_sheet",
        "match_type": "exact",
        "priority": 10,
        "notes": "资产负债表标准披露名称。",
        "status": "active",
    },
]


def validate_candidate(
    raw_data: dict,
) -> tuple[MetricRegistry, list[MetricAlias]]:
    metrics = [
        FinancialMetric.model_validate(item)
        for item in raw_data["metrics"]
    ]

    aliases = [
        MetricAlias.model_validate(item)
        for item in raw_data["metric_aliases"]
    ]

    metric_ids = [
        metric.metric_id
        for metric in metrics
    ]

    alias_ids = [
        alias.alias_id
        for alias in aliases
    ]

    duplicate_metric_ids = sorted(
        metric_id
        for metric_id, count
        in Counter(metric_ids).items()
        if count > 1
    )

    duplicate_alias_ids = sorted(
        alias_id
        for alias_id, count
        in Counter(alias_ids).items()
        if count > 1
    )

    if duplicate_metric_ids:
        raise SystemExit(
            "duplicate_metric_ids="
            + ",".join(duplicate_metric_ids)
        )

    if duplicate_alias_ids:
        raise SystemExit(
            "duplicate_alias_ids="
            + ",".join(duplicate_alias_ids)
        )

    registry = MetricRegistry()
    registry.add_many(metrics)

    validate_metric_alias_relationships(
        registry,
        aliases,
    )

    known_metric_ids = set(metric_ids)

    unknown_confusable_ids = sorted({
        confusable_id
        for metric in metrics
        for confusable_id
        in metric.confusable_metric_ids
        if confusable_id not in known_metric_ids
    })

    if unknown_confusable_ids:
        raise SystemExit(
            "unknown_confusable_ids="
            + ",".join(unknown_confusable_ids)
        )

    return registry, aliases


def main() -> None:
    if not METRICS_PATH.is_file():
        raise SystemExit(
            f"missing_metrics_file={METRICS_PATH}"
        )

    with METRICS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw_data = yaml.safe_load(file)

    if not isinstance(raw_data, dict):
        raise SystemExit(
            "metrics_yaml_root_must_be_mapping"
        )

    old_metrics = deepcopy(
        raw_data.get("metrics", [])
    )
    old_aliases = deepcopy(
        raw_data.get("metric_aliases", [])
    )

    before_metric_ids = {
        item["metric_id"]
        for item in old_metrics
    }

    before_alias_ids = {
        item["alias_id"]
        for item in old_aliases
    }

    new_metric_ids = {
        item["metric_id"]
        for item in NEW_METRICS
    }

    new_alias_ids = {
        item["alias_id"]
        for item in NEW_ALIASES
    }

    existing_target_metrics = (
        before_metric_ids & new_metric_ids
    )

    existing_target_aliases = (
        before_alias_ids & new_alias_ids
    )

    if (
        existing_target_metrics == new_metric_ids
        and existing_target_aliases == new_alias_ids
    ):
        registry, aliases = load_metrics(
            METRICS_PATH
        )

        print("already_applied=true")
        print(f"metric_count={len(registry)}")
        print(
            f"metric_alias_count={len(aliases)}"
        )
        return

    if existing_target_metrics:
        raise SystemExit(
            "partial_existing_metric_ids="
            + ",".join(
                sorted(existing_target_metrics)
            )
        )

    if existing_target_aliases:
        raise SystemExit(
            "partial_existing_alias_ids="
            + ",".join(
                sorted(existing_target_aliases)
            )
        )

    if len(old_metrics) != 33:
        raise SystemExit(
            "unexpected_before_metric_count="
            f"{len(old_metrics)}"
        )

    if len(old_aliases) != 44:
        raise SystemExit(
            "unexpected_before_alias_count="
            f"{len(old_aliases)}"
        )

    candidate = deepcopy(raw_data)

    candidate["metrics"].extend(
        deepcopy(NEW_METRICS)
    )

    candidate["metric_aliases"].extend(
        deepcopy(NEW_ALIASES)
    )

    registry, aliases = validate_candidate(
        candidate
    )

    if len(registry) != 41:
        raise SystemExit(
            f"unexpected_after_metric_count="
            f"{len(registry)}"
        )

    if len(aliases) != 52:
        raise SystemExit(
            f"unexpected_after_alias_count="
            f"{len(aliases)}"
        )

    if candidate["metrics"][:33] != old_metrics:
        raise SystemExit(
            "old_metric_records_changed=true"
        )

    if (
        candidate["metric_aliases"][:44]
        != old_aliases
    ):
        raise SystemExit(
            "old_alias_records_changed=true"
        )

    if not BACKUP_PATH.exists():
        shutil.copy2(
            METRICS_PATH,
            BACKUP_PATH,
        )

    try:
        with TEMP_PATH.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as file:
            yaml.safe_dump(
                candidate,
                file,
                allow_unicode=True,
                sort_keys=False,
                width=1000,
            )

        temporary_registry, temporary_aliases = (
            load_metrics(TEMP_PATH)
        )

        validate_metric_alias_relationships(
            temporary_registry,
            temporary_aliases,
        )

        TEMP_PATH.replace(METRICS_PATH)

    finally:
        if TEMP_PATH.exists():
            TEMP_PATH.unlink()

    final_registry, final_aliases = load_metrics(
        METRICS_PATH
    )

    validate_metric_alias_relationships(
        final_registry,
        final_aliases,
    )

    missing_metric_ids = sorted(
        new_metric_ids
        - set(final_registry.keys())
    )

    final_alias_ids = {
        alias.alias_id
        for alias in final_aliases
    }

    missing_alias_ids = sorted(
        new_alias_ids - final_alias_ids
    )

    print(f"source_path={METRICS_PATH}")
    print(f"backup_path={BACKUP_PATH}")
    print(
        f"before_metric_count="
        f"{len(old_metrics)}"
    )
    print(
        f"after_metric_count="
        f"{len(final_registry)}"
    )
    print(
        f"new_metric_count="
        f"{len(NEW_METRICS)}"
    )
    print(
        f"before_alias_count="
        f"{len(old_aliases)}"
    )
    print(
        f"after_alias_count="
        f"{len(final_aliases)}"
    )
    print(
        f"new_alias_count="
        f"{len(NEW_ALIASES)}"
    )
    print(
        f"missing_metric_ids="
        f"{missing_metric_ids}"
    )
    print(
        f"missing_alias_ids="
        f"{missing_alias_ids}"
    )
    print("old_metric_records_preserved=true")
    print("old_alias_records_preserved=true")
    print("required_formula_count=0")
    print(
        "complex_diagnostic_metric_"
        "expansion_passed=true"
    )


if __name__ == "__main__":
    main()