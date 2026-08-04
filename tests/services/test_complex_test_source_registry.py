from collections import Counter
from decimal import Decimal
from pathlib import Path

import pytest

from app.services.registry_loader import (
    load_evidences,
    load_financial_facts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REGISTRY_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "registries"
)

EVIDENCES_PATH = REGISTRY_ROOT / "evidences.yaml"

FINANCIAL_FACTS_PATH = (
    REGISTRY_ROOT / "financial_facts.yaml"
)


EXPECTED_SOURCE_FACTS = {
    "fact_gree_electric_2024_current_assets": (
        "224802908651.88", "CNY", 111, 111,
        "224,802,908,651.88",
    ),
    "fact_gree_electric_2024_current_liabilities": (
        "201125441381.86", "CNY", 112, 112,
        "201,125,441,381.86",
    ),
    "fact_gree_electric_2024_income_tax_expense": (
        "4524926560.95", "CNY", 113, 113,
        "4,524,926,560.95",
    ),
    "fact_gree_electric_2024_management_expenses": (
        "6057608713.94", "CNY", 113, 113,
        "6,057,608,713.94",
    ),
    "fact_gree_electric_2024_monetary_funds": (
        "113900461797.94", "CNY", 111, 111,
        "113,900,461,797.94",
    ),
    "fact_gree_electric_2024_total_equity": (
        "141513694947.97", "CNY", 112, 112,
        "141,513,694,947.97",
    ),
    "fact_gree_electric_2024_total_profit": (
        "36895995848.30", "CNY", 113, 113,
        "36,895,995,848.30",
    ),
    "fact_haier_smart_home_2024_current_assets": (
        "151689536276.72", "CNY", 118, 118,
        "151,689,536,276.72",
    ),
    "fact_haier_smart_home_2024_current_liabilities": (
        "149571374519.76", "CNY", 119, 119,
        "149,571,374,519.76",
    ),
    "fact_haier_smart_home_2024_fixed_assets": (
        "37518645325.08", "CNY", 118, 118,
        "37,518,645,325.08",
    ),
    "fact_haier_smart_home_2024_goodwill": (
        "27384007599.06", "CNY", 118, 118,
        "27,384,007,599.06",
    ),
    "fact_haier_smart_home_2024_intangible_assets": (
        "14034674912.54", "CNY", 118, 118,
        "14,034,674,912.54",
    ),
    "fact_haier_smart_home_2024_management_expenses": (
        "12110235915.35", "CNY", 122, 122,
        "12,110,235,915.35",
    ),
    "fact_haier_smart_home_2024_net_cash_flow_from_financing_activities": (
        "-7913904092.66", "CNY", 126, 126,
        "-7,913,904,092.66",
    ),
    "fact_hisense_home_2024_current_assets": (
        "52507910456.50", "CNY", 112, 110,
        "52,507,910,456.50",
    ),
    "fact_hisense_home_2024_current_liabilities": (
        "47919366943.34", "CNY", 113, 111,
        "47,919,366,943.34",
    ),
    "fact_hisense_home_2024_management_expenses": (
        "2499492962.69", "CNY", 117, 115,
        "2,499,492,962.69",
    ),
    "fact_hisense_home_2024_monetary_funds": (
        "4397693443.73", "CNY", 112, 110,
        "4,397,693,443.73",
    ),
    "fact_hisense_home_2024_net_cash_flow_from_financing_activities": (
        "-5121320126.09", "CNY", 121, 119,
        "-5,121,320,126.09",
    ),
    "fact_hisense_home_2024_net_profit_attributable_to_parent": (
        "3347881773.89", "CNY", 117, 115,
        "3,347,881,773.89",
    ),
    "fact_hisense_home_2024_operating_profit": (
        "5679142269.33", "CNY", 117, 115,
        "5,679,142,269.33",
    ),
    "fact_hisense_home_2024_total_equity": (
        "19374524711.13", "CNY", 114, 112,
        "19,374,524,711.13",
    ),
    "fact_hisense_home_2024_total_liabilities": (
        "50327415106.20", "CNY", 114, 112,
        "50,327,415,106.20",
    ),
    "fact_hisense_home_2024_total_profit": (
        "5966389023.89", "CNY", 117, 115,
        "5,966,389,023.89",
    ),
    "fact_midea_group_2024_bonds_payable": (
        "3266775", "CNY_thousand", 157, 156,
        "3,266,775",
    ),
    "fact_midea_group_2024_current_assets": (
        "389063786", "CNY_thousand", 156, 155,
        "389,063,786",
    ),
    "fact_midea_group_2024_current_liabilities": (
        "351819806", "CNY_thousand", 157, 156,
        "351,819,806",
    ),
    "fact_midea_group_2024_income_tax_expense": (
        "7932532", "CNY_thousand", 158, 157,
        "(7,932,532)",
    ),
    "fact_midea_group_2024_long_term_borrowings": (
        "10491757", "CNY_thousand", 157, 156,
        "10,491,757",
    ),
    "fact_midea_group_2024_management_expenses": (
        "14505864", "CNY_thousand", 158, 157,
        "(14,505,864)",
    ),
    "fact_midea_group_2024_monetary_funds": (
        "140410308", "CNY_thousand", 156, 155,
        "140,410,308",
    ),
    "fact_midea_group_2024_short_term_borrowings": (
        "31008549", "CNY_thousand", 157, 156,
        "31,008,549",
    ),
    "fact_midea_group_2024_total_equity": (
        "227667391", "CNY_thousand", 157, 156,
        "227,667,391",
    ),
    "fact_midea_group_2024_total_profit": (
        "46689746", "CNY_thousand", 158, 157,
        "46,689,746",
    ),
}


EXPECTED_COMPANY_COUNTS = {
    "gree_electric": 7,
    "haier_smart_home": 7,
    "hisense_home": 10,
    "midea_group": 10,
}


@pytest.fixture(scope="module")
def source_registry_data():
    evidences = load_evidences(
        EVIDENCES_PATH
    )

    financial_facts, links = (
        load_financial_facts(
            FINANCIAL_FACTS_PATH
        )
    )

    return evidences, financial_facts, links


def test_complex_test_source_batch_has_exact_identity(
    source_registry_data,
) -> None:
    evidences, financial_facts, _ = (
        source_registry_data
    )

    assert len(EXPECTED_SOURCE_FACTS) == 34

    target_facts = [
        financial_facts.require(fact_id)
        for fact_id in EXPECTED_SOURCE_FACTS
    ]

    target_evidence_ids = {
        fact.primary_evidence_id
        for fact in target_facts
    }

    assert len(target_evidence_ids) == 34

    for evidence_id in target_evidence_ids:
        assert evidences.contains(
            evidence_id
        )

    company_counts = Counter(
        fact.company_id
        for fact in target_facts
    )

    assert dict(
        sorted(company_counts.items())
    ) == EXPECTED_COMPANY_COUNTS


def test_complex_test_source_batch_values_and_pages(
    source_registry_data,
) -> None:
    evidences, financial_facts, _ = (
        source_registry_data
    )

    for fact_id, expected in (
        EXPECTED_SOURCE_FACTS.items()
    ):
        (
            expected_raw_value,
            expected_raw_unit,
            expected_pdf_page,
            expected_printed_page,
            expected_cell_value,
        ) = expected

        fact = financial_facts.require(
            fact_id
        )

        evidence = evidences.require(
            fact.primary_evidence_id
        )

        assert fact.raw_value == Decimal(
            expected_raw_value
        )

        assert fact.raw_unit.value == (
            expected_raw_unit
        )

        expected_multiplier = (
            Decimal("1000")
            if expected_raw_unit
            == "CNY_thousand"
            else Decimal("1")
        )

        assert fact.unit_multiplier == (
            expected_multiplier
        )

        assert fact.normalized_value == (
            Decimal(expected_raw_value)
            * expected_multiplier
        )

        assert fact.normalized_unit.value == (
            "CNY"
        )

        assert fact.currency == "CNY"

        assert evidence.pdf_page == (
            expected_pdf_page
        )

        assert evidence.printed_page == (
            expected_printed_page
        )

        assert evidence.cell_value == (
            expected_cell_value
        )


def test_complex_test_source_batch_is_verified_and_linked(
    source_registry_data,
) -> None:
    evidences, financial_facts, links = (
        source_registry_data
    )

    link_keys = {
        (
            link.fact_id,
            link.evidence_id,
            link.support_type.value,
        )
        for link in links
    }

    for fact_id in EXPECTED_SOURCE_FACTS:
        fact = financial_facts.require(
            fact_id
        )

        evidence = evidences.require(
            fact.primary_evidence_id
        )

        assert fact.fiscal_year == 2024

        assert fact.statement_scope.value == (
            "consolidated"
        )

        assert fact.validation_status.value == (
            "verified"
        )

        assert fact.validated_by == (
            "manual_review"
        )

        assert fact.validated_at is not None

        assert evidence.validation_status.value == (
            "verified"
        )

        assert evidence.validated_by == (
            "manual_review"
        )

        assert evidence.report_id == (
            fact.report_id
        )

        assert evidence.statement_type == (
            fact.statement_type
        )

        assert evidence.statement_scope == (
            fact.statement_scope
        )

        assert evidence.chunk_id is not None

        assert len(evidence.source_hash) == 64

        assert (
            fact.fact_id,
            evidence.evidence_id,
            "primary",
        ) in link_keys


def test_complex_test_source_batch_period_contract(
    source_registry_data,
) -> None:
    _, financial_facts, _ = (
        source_registry_data
    )

    for fact_id in EXPECTED_SOURCE_FACTS:
        fact = financial_facts.require(
            fact_id
        )

        if (
            fact.statement_type.value
            == "balance_sheet"
        ):
            assert fact.period_type.value == (
                "instant"
            )

            assert fact.period_start is None
            assert fact.period_end is None

        else:
            assert fact.statement_type.value in {
                "income_statement",
                "cash_flow_statement",
            }

            assert fact.period_type.value == (
                "duration"
            )

            assert (
                fact.period_start.isoformat()
                == "2024-01-01"
            )

            assert (
                fact.period_end.isoformat()
                == "2024-12-31"
            )
