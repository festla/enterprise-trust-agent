from pathlib import Path

import pytest

from app.services.registry_loader import (
    load_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "registries"
    / "metrics.yaml"
)


EXPECTED_CONTRACTS = {
    "non_current_assets": {
        "display_name_cn": "非流动资产合计",
        "statement_type": "balance_sheet",
        "period_type": "instant",
    },
    "cash_outflows_from_investing_activities_subtotal": {
        "display_name_cn": "投资活动现金流出小计",
        "statement_type": "cash_flow_statement",
        "period_type": "duration",
    },
    "taxes_and_surcharges": {
        "display_name_cn": "税金及附加",
        "statement_type": "income_statement",
        "period_type": "duration",
    },
    "cash_received_from_sales_of_goods_and_rendering_of_services": {
        "display_name_cn": (
            "销售商品、提供劳务收到的现金"
        ),
        "statement_type": "cash_flow_statement",
        "period_type": "duration",
    },
    "tax_refunds_received": {
        "display_name_cn": "收到的税费返还",
        "statement_type": "cash_flow_statement",
        "period_type": "duration",
    },
    "other_comprehensive_income_net_of_tax": {
        "display_name_cn": (
            "其他综合收益的税后净额"
        ),
        "statement_type": "income_statement",
        "period_type": "duration",
    },
    "total_comprehensive_income": {
        "display_name_cn": "综合收益总额",
        "statement_type": "income_statement",
        "period_type": "duration",
    },
    "non_current_liabilities": {
        "display_name_cn": "非流动负债合计",
        "statement_type": "balance_sheet",
        "period_type": "instant",
    },
}


EXPECTED_ALIASES = {
    "non_current_assets": {
        "alias_id": (
            "non_current_assets_total_cn"
        ),
        "alias": "非流动资产合计",
        "statement_type": "balance_sheet",
    },
    "cash_outflows_from_investing_activities_subtotal": {
        "alias_id": (
            "investing_cash_outflows_subtotal_cn"
        ),
        "alias": "投资活动现金流出小计",
        "statement_type": "cash_flow_statement",
    },
    "taxes_and_surcharges": {
        "alias_id": (
            "taxes_and_surcharges_standard_cn"
        ),
        "alias": "税金及附加",
        "statement_type": "income_statement",
    },
    "cash_received_from_sales_of_goods_and_rendering_of_services": {
        "alias_id": (
            "cash_received_from_sales_and_services_cn"
        ),
        "alias": (
            "销售商品、提供劳务收到的现金"
        ),
        "statement_type": "cash_flow_statement",
    },
    "tax_refunds_received": {
        "alias_id": (
            "tax_refunds_received_standard_cn"
        ),
        "alias": "收到的税费返还",
        "statement_type": "cash_flow_statement",
    },
    "other_comprehensive_income_net_of_tax": {
        "alias_id": (
            "other_comprehensive_income_net_of_tax_cn"
        ),
        "alias": "其他综合收益的税后净额",
        "statement_type": "income_statement",
    },
    "total_comprehensive_income": {
        "alias_id": (
            "total_comprehensive_income_standard_cn"
        ),
        "alias": "综合收益总额",
        "statement_type": "income_statement",
    },
    "non_current_liabilities": {
        "alias_id": (
            "non_current_liabilities_total_cn"
        ),
        "alias": "非流动负债合计",
        "statement_type": "balance_sheet",
    },
}


EXPECTED_CONFUSABLES = {
    "non_current_assets": {
        "current_assets",
        "total_assets",
    },
    "cash_outflows_from_investing_activities_subtotal": {
        "net_cash_flow_from_investing_activities",
    },
    "taxes_and_surcharges": {
        "income_tax_expense",
    },
    "cash_received_from_sales_of_goods_and_rendering_of_services": {
        "net_cash_flow_from_operating_activities",
    },
    "tax_refunds_received": {
        "income_tax_expense",
    },
    "other_comprehensive_income_net_of_tax": {
        "net_profit",
        "total_comprehensive_income",
    },
    "total_comprehensive_income": {
        "net_profit",
        "other_comprehensive_income_net_of_tax",
    },
    "non_current_liabilities": {
        "current_liabilities",
        "total_liabilities",
    },
}


NEW_METRIC_IDS = set(EXPECTED_CONTRACTS)


@pytest.fixture(scope="module")
def loaded_metrics():
    return load_metrics(METRICS_PATH)


def test_diagnostic_metric_registry_sizes(
    loaded_metrics,
) -> None:
    registry, aliases = loaded_metrics

    assert len(registry) == 41
    assert len(aliases) == 52


def test_diagnostic_metric_contracts(
    loaded_metrics,
) -> None:
    registry, _ = loaded_metrics

    missing_metric_ids = (
        NEW_METRIC_IDS - set(registry.keys())
    )

    assert missing_metric_ids == set()

    for metric_id, expected in (
        EXPECTED_CONTRACTS.items()
    ):
        metric = registry.require(metric_id)

        assert metric.display_name_cn == (
            expected["display_name_cn"]
        )

        assert metric.metric_origin.value == (
            "reported"
        )

        assert metric.statement_type.value == (
            expected["statement_type"]
        )

        assert metric.period_type.value == (
            expected["period_type"]
        )

        assert metric.default_unit.value == "CNY"

        assert metric.value_type.value == (
            "decimal"
        )

        assert metric.formula_id is None
        assert metric.is_core_metric is True

        actual_scopes = {
            scope.value
            for scope in metric.allowed_scopes
        }

        assert actual_scopes == {
            "consolidated",
            "parent_company",
        }


def test_diagnostic_metric_aliases(
    loaded_metrics,
) -> None:
    _, aliases = loaded_metrics

    diagnostic_aliases = [
        alias
        for alias in aliases
        if alias.metric_id in NEW_METRIC_IDS
    ]

    assert len(diagnostic_aliases) == 8

    aliases_by_metric = {
        alias.metric_id: alias
        for alias in diagnostic_aliases
    }

    assert set(aliases_by_metric) == (
        NEW_METRIC_IDS
    )

    for metric_id, expected in (
        EXPECTED_ALIASES.items()
    ):
        alias = aliases_by_metric[metric_id]

        assert alias.alias_id == (
            expected["alias_id"]
        )

        assert alias.alias == expected["alias"]

        assert alias.statement_type is not None

        assert alias.statement_type.value == (
            expected["statement_type"]
        )

        assert alias.statement_scope is None
        assert alias.match_type.value == "exact"
        assert alias.priority == 10
        assert alias.status.value == "active"


def test_diagnostic_confusable_boundaries(
    loaded_metrics,
) -> None:
    registry, _ = loaded_metrics

    for metric_id, expected_ids in (
        EXPECTED_CONFUSABLES.items()
    ):
        metric = registry.require(metric_id)

        assert (
            set(metric.confusable_metric_ids)
            == expected_ids
        )

        for confusable_id in expected_ids:
            assert registry.contains(
                confusable_id
            )