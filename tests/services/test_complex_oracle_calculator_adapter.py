from datetime import (
    datetime,
    timedelta,
    timezone,
)
from pathlib import Path
from decimal import Decimal
import pytest
from app.schemas.calculation import (
    DerivedCalculation,
)
from app.schemas.complex_plan_eval_result import (
    ComplexFinalAnswerOutput,
    ComplexRetrievalTrace,
)
from app.services.complex_oracle_calculator_adapter import (
    ComplexOracleCalculatorAdapter,
    ComplexOracleCalculatorAdapterError,
)
from app.services.complex_plan_eval_dataset import (
    load_complex_financial_eval_cases,
)
from app.services.complex_plan_oracle import (
    execute_gold_oracle_case,
)
from app.services.registry_loader import (
    load_registry_bundle,
)
import app.services.complex_oracle_calculator_adapter \
    as calculator_adapter_module

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REGISTRY_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "registries"
)

CASES_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "complex_plan"
    / "complex_plan_pilot_v1.jsonl"
)

TEST_TIME = datetime(
    2026,
    8,
    2,
    17,
    0,
    tzinfo=timezone(
        timedelta(hours=8)
    ),
)


@pytest.fixture(scope="module")
def bundle():
    registry_bundle, _, _, _ = (
        load_registry_bundle(
            companies_path=(
                REGISTRY_ROOT
                / "companies.yaml"
            ),
            reports_path=(
                REGISTRY_ROOT
                / "reports.yaml"
            ),
            metrics_path=(
                REGISTRY_ROOT
                / "metrics.yaml"
            ),
            evidences_path=(
                REGISTRY_ROOT
                / "evidences.yaml"
            ),
            financial_facts_path=(
                REGISTRY_ROOT
                / "financial_facts.yaml"
            ),
        )
    )

    return registry_bundle


@pytest.fixture(scope="module")
def complex_002():
    cases = load_complex_financial_eval_cases(
        CASES_PATH
    )

    return next(
        case
        for case in cases
        if case.case_id == "complex_002"
    )


def test_calculates_real_hisense_margin(
    bundle,
) -> None:
    adapter = (
        ComplexOracleCalculatorAdapter(
            registry_bundle=bundle,
            clock=lambda: TEST_TIME,
        )
    )

    trace = adapter.calculate(
        calculation_id=(
            "calculation_hisense_home_"
            "2024_gross_profit_margin"
        ),
        formula_id=(
            "gross_profit_margin_formula"
        ),
        input_fact_ids=(
            "fact_hisense_home_2024_revenue",
            "fact_hisense_home_2024_operating_cost",
        ),
    )

    assert trace.status == "completed"

    assert str(trace.result_value) == (
        "20.7768"
    )

    assert trace.result_unit == "percent"

    assert trace.metric_id == (
        "gross_profit_margin"
    )

    assert trace.input_fact_ids == (
        "fact_hisense_home_2024_revenue",
        "fact_hisense_home_2024_operating_cost",
    )


def test_reversed_fact_order_is_rejected(
    bundle,
) -> None:
    adapter = (
        ComplexOracleCalculatorAdapter(
            registry_bundle=bundle,
            clock=lambda: TEST_TIME,
        )
    )

    trace = adapter.calculate(
        calculation_id=(
            "calculation_hisense_home_"
            "2024_gross_profit_margin"
        ),
        formula_id=(
            "gross_profit_margin_formula"
        ),
        input_fact_ids=(
            "fact_hisense_home_2024_operating_cost",
            "fact_hisense_home_2024_revenue",
        ),
    )

    assert trace.status == "failed"

    assert "必须为 revenue" in (
        trace.error_message or ""
    )


def test_missing_fact_is_rejected(
    bundle,
) -> None:
    adapter = (
        ComplexOracleCalculatorAdapter(
            registry_bundle=bundle,
            clock=lambda: TEST_TIME,
        )
    )

    trace = adapter.calculate(
        calculation_id=(
            "calculation_hisense_home_"
            "2024_gross_profit_margin"
        ),
        formula_id=(
            "gross_profit_margin_formula"
        ),
        input_fact_ids=(
            "fact_missing_revenue",
            "fact_hisense_home_2024_operating_cost",
        ),
    )

    assert trace.status == "failed"

    assert "不存在" in (
        trace.error_message or ""
    )


def test_unsupported_formula_is_rejected(
    bundle,
) -> None:
    adapter = (
        ComplexOracleCalculatorAdapter(
            registry_bundle=bundle,
            clock=lambda: TEST_TIME,
        )
    )

    trace = adapter.calculate(
        calculation_id=(
            "calculation_test_unknown"
        ),
        formula_id=(
            "unsupported_formula"
        ),
        input_fact_ids=(
            "fact_hisense_home_2024_revenue",
            "fact_hisense_home_2024_operating_cost",
        ),
    )

    assert trace.status == "failed"
    assert trace.metric_id == (
        "unknown_metric"
    )

    assert "不支持" in (
        trace.error_message or ""
    )


def test_wrong_calculation_id_is_rejected(
    bundle,
) -> None:
    adapter = (
        ComplexOracleCalculatorAdapter(
            registry_bundle=bundle,
            clock=lambda: TEST_TIME,
        )
    )

    trace = adapter.calculate(
        calculation_id=(
            "calculation_wrong_result"
        ),
        formula_id=(
            "gross_profit_margin_formula"
        ),
        input_fact_ids=(
            "fact_hisense_home_2024_revenue",
            "fact_hisense_home_2024_operating_cost",
        ),
    )

    assert trace.status == "failed"

    assert "calculation_id" in (
        trace.error_message or ""
    )


def test_empty_input_is_rejected(
    bundle,
) -> None:
    adapter = (
        ComplexOracleCalculatorAdapter(
            registry_bundle=bundle
        )
    )

    with pytest.raises(
        ComplexOracleCalculatorAdapterError,
        match="input_fact_ids 不能为空",
    ):
        adapter.calculate(
            calculation_id=(
                "calculation_test"
            ),
            formula_id=(
                "gross_profit_margin_formula"
            ),
            input_fact_ids=(),
        )

@pytest.mark.parametrize(
    (
        "formula_id",
        "input_fact_ids",
        "expected_metric_id",
        "expected_input_count",
    ),
    [
        (
            (
                "selling_and_r_and_d_"
                "expense_ratio_formula"
            ),
            (
                "fact_midea_group_2024_revenue",
                (
                    "fact_midea_group_2024_"
                    "net_profit_attributable_to_parent"
                ),
            ),
            (
                "selling_and_r_and_d_"
                "expense_ratio"
            ),
            3,
        ),
        (
            (
                "operating_cash_flow_to_"
                "net_profit_ratio_formula"
            ),
            (
                (
                    "fact_midea_group_2024_"
                    "net_cash_flow_from_"
                    "operating_activities"
                ),
                (
                    "fact_midea_group_2024_"
                    "net_profit_attributable_to_parent"
                ),
                "fact_midea_group_2024_revenue",
            ),
            (
                "operating_cash_flow_to_"
                "net_profit_ratio"
            ),
            2,
        ),
    ],
)
def test_supported_formula_input_count_is_validated(
    bundle,
    formula_id: str,
    input_fact_ids: tuple[str, ...],
    expected_metric_id: str,
    expected_input_count: int,
) -> None:
    """已支持公式应校验各自的输入Fact数量。"""

    adapter = (
        ComplexOracleCalculatorAdapter(
            registry_bundle=bundle,
            clock=lambda: TEST_TIME,
        )
    )

    trace = adapter.calculate(
        calculation_id=(
            "calculation_test_input_count"
        ),
        formula_id=formula_id,
        input_fact_ids=input_fact_ids,
    )

    assert trace.status == "failed"
    assert trace.metric_id == expected_metric_id

    assert (
        f"必须接收 {expected_input_count} 个输入 Fact"
        in (trace.error_message or "")
    )


def test_routes_selling_and_r_and_d_expense_ratio_formula(
    bundle,
    monkeypatch,
) -> None:
    """Adapter应按固定顺序路由三个费用率输入。"""

    input_fact_ids = (
        "fact_midea_group_2024_revenue",
        (
            "fact_midea_group_2024_"
            "net_profit_attributable_to_parent"
        ),
        (
            "fact_midea_group_2024_"
            "net_cash_flow_from_operating_activities"
        ),
    )

    calculation_id = (
        "calculation_midea_group_2024_"
        "selling_and_r_and_d_expense_ratio"
    )

    def fake_builder(
        *,
        revenue_fact,
        selling_expenses_fact,
        research_and_development_expenses_fact,
        created_at,
    ):
        assert revenue_fact.fact_id == (
            input_fact_ids[0]
        )
        assert selling_expenses_fact.fact_id == (
            input_fact_ids[1]
        )
        assert (
            research_and_development_expenses_fact
            .fact_id
            == input_fact_ids[2]
        )
        assert created_at == TEST_TIME

        return DerivedCalculation(
            calculation_id=calculation_id,
            metric_id=(
                "selling_and_r_and_d_expense_ratio"
            ),
            formula_id=(
                "selling_and_r_and_d_expense_ratio_formula"
            ),
            result_value=Decimal("13.5052"),
            result_unit="percent",
            input_fact_ids=list(input_fact_ids),
            calculation_version="v1",
            validation_status="verified",
            validated_by=(
                "deterministic_calculator_v1"
            ),
            created_at=TEST_TIME,
        )

    monkeypatch.setattr(
        calculator_adapter_module,
        (
            "build_selling_and_r_and_d_"
            "expense_ratio_calculation"
        ),
        fake_builder,
    )

    adapter = (
        ComplexOracleCalculatorAdapter(
            registry_bundle=bundle,
            clock=lambda: TEST_TIME,
        )
    )

    trace = adapter.calculate(
        calculation_id=calculation_id,
        formula_id=(
            "selling_and_r_and_d_expense_ratio_formula"
        ),
        input_fact_ids=input_fact_ids,
    )

    assert trace.status == "completed"
    assert trace.metric_id == (
        "selling_and_r_and_d_expense_ratio"
    )
    assert trace.result_value == Decimal(
        "13.5052"
    )
    assert trace.result_unit == "percent"


def test_routes_operating_cash_flow_to_net_profit_ratio_formula(
    bundle,
    monkeypatch,
) -> None:
    """Adapter应按固定顺序路由现金流与净利润。"""

    input_fact_ids = (
        (
            "fact_midea_group_2024_"
            "net_cash_flow_from_operating_activities"
        ),
        (
            "fact_midea_group_2024_"
            "net_profit_attributable_to_parent"
        ),
    )

    calculation_id = (
        "calculation_midea_group_2024_"
        "operating_cash_flow_to_net_profit_ratio"
    )

    def fake_builder(
        *,
        operating_cash_flow_fact,
        net_profit_fact,
        created_at,
    ):
        assert (
            operating_cash_flow_fact.fact_id
            == input_fact_ids[0]
        )
        assert net_profit_fact.fact_id == (
            input_fact_ids[1]
        )
        assert created_at == TEST_TIME

        return DerivedCalculation(
            calculation_id=calculation_id,
            metric_id=(
                "operating_cash_flow_to_"
                "net_profit_ratio"
            ),
            formula_id=(
                "operating_cash_flow_to_"
                "net_profit_ratio_formula"
            ),
            result_value=Decimal("1.3559"),
            result_unit="ratio",
            input_fact_ids=list(input_fact_ids),
            calculation_version="v1",
            validation_status="verified",
            validated_by=(
                "deterministic_calculator_v1"
            ),
            created_at=TEST_TIME,
        )

    monkeypatch.setattr(
        calculator_adapter_module,
        (
            "build_operating_cash_flow_to_"
            "net_profit_ratio_calculation"
        ),
        fake_builder,
    )

    adapter = (
        ComplexOracleCalculatorAdapter(
            registry_bundle=bundle,
            clock=lambda: TEST_TIME,
        )
    )

    trace = adapter.calculate(
        calculation_id=calculation_id,
        formula_id=(
            "operating_cash_flow_to_net_profit_ratio_formula"
        ),
        input_fact_ids=input_fact_ids,
    )

    assert trace.status == "completed"
    assert trace.metric_id == (
        "operating_cash_flow_to_net_profit_ratio"
    )
    assert trace.result_value == Decimal(
        "1.3559"
    )
    assert trace.result_unit == "ratio"


class FakeRetriever:
    @property
    def retriever_id(self) -> str:
        return "fake_retriever_v1"

    def retrieve(
        self,
        *,
        query,
        top_k,
    ):
        fact_ids = {
            "revenue": (
                "fact_hisense_home_2024_revenue"
            ),
            "operating_cost": (
                "fact_hisense_home_2024_operating_cost"
            ),
        }

        fact_id = fact_ids[
            query.metric_id
        ]

        evidence_id = (
            "evidence_"
            + fact_id.removeprefix(
                "fact_"
            )
        )

        return ComplexRetrievalTrace(
            query_id=query.query_id,
            status="completed",
            retrieved_fact_ids=(
                fact_id,
            ),
            retrieved_evidence_ids=(
                evidence_id,
            ),
            retrieved_chunk_ids=(
                f"{query.report_id}_"
                f"chunk_{query.query_id}",
            ),
            top_k=top_k,
            latency_ms=1.0,
        )


class FakeGenerator:
    @property
    def generator_id(self) -> str:
        return "fake_generator_v1"

    def generate(
        self,
        *,
        question,
        rewrite,
        plan,
        retrieval_traces,
        calculation_traces,
    ):
        return ComplexFinalAnswerOutput(
            answer_text=(
                "海信家电2024年毛利率"
                "约为20.78%。"
            ),
            supporting_fact_ids=tuple(
                trace.retrieved_fact_ids[0]
                for trace
                in retrieval_traces
            ),
            supporting_calculation_ids=tuple(
                trace.calculation_id
                for trace
                in calculation_traces
            ),
            citation_evidence_ids=tuple(
                trace.retrieved_evidence_ids[0]
                for trace
                in retrieval_traces
            ),
        )


def test_adapter_integrates_with_oracle_executor(
    bundle,
    complex_002,
) -> None:
    calculator = (
        ComplexOracleCalculatorAdapter(
            registry_bundle=bundle,
            clock=lambda: TEST_TIME,
        )
    )

    result = execute_gold_oracle_case(
        run_id=(
            "complex_run_calculator_test"
        ),
        case=complex_002,
        retriever=FakeRetriever(),
        calculator=calculator,
        generator=FakeGenerator(),
        top_k=5,
    )

    assert result.status == "completed"

    assert len(
        result.calculation_traces
    ) == 1

    assert str(
        result
        .calculation_traces[0]
        .result_value
    ) == "20.7768"

    assert result.calculator_id == (
        "deterministic_calculator_v1"
    )

@pytest.mark.parametrize(
    (
        "formula_id",
        "expected_metric_id",
    ),
    [
        (
            "current_ratio_formula",
            "current_ratio",
        ),
        (
            "debt_to_equity_ratio_formula",
            "debt_to_equity_ratio",
        ),
        (
            "effective_income_tax_rate_formula",
            "effective_income_tax_rate",
        ),
    ],
)
def test_new_formula_input_count_is_validated(
    bundle,
    formula_id: str,
    expected_metric_id: str,
) -> None:
    adapter = ComplexOracleCalculatorAdapter(
        registry_bundle=bundle,
        clock=lambda: TEST_TIME,
    )

    trace = adapter.calculate(
        calculation_id=(
            "calculation_test_new_formula_input_count"
        ),
        formula_id=formula_id,
        input_fact_ids=(
            "fact_hisense_home_2024_revenue",
        ),
    )

    assert trace.status == "failed"
    assert trace.metric_id == expected_metric_id

    error_message = trace.error_message or ""

    assert formula_id in error_message
    assert "2" in error_message
    assert "Fact" in error_message


@pytest.mark.parametrize(
    (
        "formula_id",
        "metric_id",
        "builder_name",
        "fact_argument_names",
        "result_unit",
    ),
    [
        (
            "current_ratio_formula",
            "current_ratio",
            "build_current_ratio_calculation",
            (
                "current_assets_fact",
                "current_liabilities_fact",
            ),
            "ratio",
        ),
        (
            "debt_to_equity_ratio_formula",
            "debt_to_equity_ratio",
            "build_debt_to_equity_ratio_calculation",
            (
                "total_liabilities_fact",
                "total_equity_fact",
            ),
            "ratio",
        ),
        (
            "effective_income_tax_rate_formula",
            "effective_income_tax_rate",
            (
                "build_effective_income_tax_rate_"
                "calculation"
            ),
            (
                "income_tax_expense_fact",
                "total_profit_fact",
            ),
            "percent",
        ),
    ],
)
def test_routes_new_formula_to_expected_builder(
    bundle,
    monkeypatch,
    formula_id: str,
    metric_id: str,
    builder_name: str,
    fact_argument_names: tuple[str, str],
    result_unit: str,
) -> None:
    input_fact_ids = (
        "fact_hisense_home_2024_revenue",
        "fact_hisense_home_2024_operating_cost",
    )

    calculation_id = (
        "calculation_adapter_test_"
        + metric_id
    )

    def fake_builder(**kwargs):
        assert kwargs["created_at"] == TEST_TIME

        actual_fact_ids = tuple(
            kwargs[argument_name].fact_id
            for argument_name
            in fact_argument_names
        )

        assert actual_fact_ids == input_fact_ids

        assert set(kwargs) == {
            *fact_argument_names,
            "created_at",
        }

        return DerivedCalculation(
            calculation_id=calculation_id,
            metric_id=metric_id,
            formula_id=formula_id,
            result_value=Decimal("1.2500"),
            result_unit=result_unit,
            input_fact_ids=list(input_fact_ids),
            calculation_version="v1",
            validation_status="verified",
            validated_by=(
                "deterministic_calculator_v1"
            ),
            created_at=TEST_TIME,
        )

    monkeypatch.setattr(
        calculator_adapter_module,
        builder_name,
        fake_builder,
    )

    adapter = ComplexOracleCalculatorAdapter(
        registry_bundle=bundle,
        clock=lambda: TEST_TIME,
    )

    trace = adapter.calculate(
        calculation_id=calculation_id,
        formula_id=formula_id,
        input_fact_ids=input_fact_ids,
    )

    assert trace.status == "completed"
    assert trace.metric_id == metric_id
    assert trace.formula_id == formula_id
    assert trace.result_value == Decimal(
        "1.2500"
    )
    assert trace.result_unit == result_unit
    assert trace.input_fact_ids == input_fact_ids