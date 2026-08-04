import json
from pathlib import Path

import pytest

from app.services.derived_calculation_dataset import (
    DerivedCalculationDatasetNotFoundError,
    InvalidDerivedCalculationDatasetError,
    load_derived_calculations,
)


def build_valid_calculation(
    calculation_id: str = (
        "calculation_hisense_home_2024_"
        "gross_profit_margin"
    ),
) -> dict:
    """构造合法派生计算记录。"""

    return {
        "calculation_id": calculation_id,
        "metric_id": "gross_profit_margin",
        "formula_id": (
            "gross_profit_margin_formula"
        ),
        "result_value": "20.7768",
        "result_unit": "percent",
        "input_fact_ids": [
            "fact_hisense_home_2024_revenue",
            (
                "fact_hisense_home_2024_"
                "operating_cost"
            ),
        ],
        "calculation_version": "v1",
        "validation_status": "verified",
        "validated_by": (
            "deterministic_calculator_v1"
        ),
        "created_at": (
            "2026-08-02T16:18:27.200992"
            "+08:00"
        ),
    }


def serialize_calculation(
    data: dict,
) -> str:
    return json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def write_calculations(
    path: Path,
    *calculations: dict,
) -> None:
    text = "\n".join(
        serialize_calculation(calculation)
        for calculation in calculations
    )

    path.write_text(
        text + "\n",
        encoding="utf-8",
    )


def test_load_valid_calculation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calculations.jsonl"

    write_calculations(
        path,
        build_valid_calculation(),
    )

    calculations = (
        load_derived_calculations(path)
    )

    assert isinstance(calculations, tuple)
    assert len(calculations) == 1

    assert calculations[0].calculation_id == (
        "calculation_hisense_home_2024_"
        "gross_profit_margin"
    )

    assert str(
        calculations[0].result_value
    ) == "20.7768"


def test_load_multiple_calculations(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calculations.jsonl"

    write_calculations(
        path,
        build_valid_calculation(
            "calculation_company_a_2024_margin"
        ),
        build_valid_calculation(
            "calculation_company_b_2024_margin"
        ),
    )

    calculations = (
        load_derived_calculations(path)
    )

    assert [
        calculation.calculation_id
        for calculation in calculations
    ] == [
        "calculation_company_a_2024_margin",
        "calculation_company_b_2024_margin",
    ]


def test_reject_missing_dataset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing.jsonl"

    with pytest.raises(
        DerivedCalculationDatasetNotFoundError,
    ) as exc_info:
        load_derived_calculations(path)

    assert str(path) in str(exc_info.value)


def test_reject_empty_dataset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.jsonl"

    path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        InvalidDerivedCalculationDatasetError,
    ) as exc_info:
        load_derived_calculations(path)

    assert "不能为空" in str(exc_info.value)


def test_reject_blank_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "blank.jsonl"

    first_line = serialize_calculation(
        build_valid_calculation()
    )

    second_line = serialize_calculation(
        build_valid_calculation(
            "calculation_second"
        )
    )

    path.write_text(
        first_line
        + "\n\n"
        + second_line
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        InvalidDerivedCalculationDatasetError,
    ) as exc_info:
        load_derived_calculations(path)

    assert "第 2 行" in str(exc_info.value)


def test_reject_non_utf8_dataset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid_encoding.jsonl"

    path.write_bytes(
        b"\xff\xfe\x00\x00"
    )

    with pytest.raises(
        InvalidDerivedCalculationDatasetError,
    ) as exc_info:
        load_derived_calculations(path)

    assert "UTF-8" in str(exc_info.value)


def test_reject_invalid_json_with_line_number(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid_json.jsonl"

    path.write_text(
        serialize_calculation(
            build_valid_calculation()
        )
        + "\n"
        + "{not-valid-json}"
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        InvalidDerivedCalculationDatasetError,
    ) as exc_info:
        load_derived_calculations(path)

    assert "第 2 行" in str(exc_info.value)


def test_reject_schema_error_with_line_number(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid_schema.jsonl"

    invalid = build_valid_calculation(
        "calculation_invalid"
    )

    del invalid["formula_id"]

    path.write_text(
        serialize_calculation(
            build_valid_calculation()
        )
        + "\n"
        + serialize_calculation(invalid)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        InvalidDerivedCalculationDatasetError,
    ) as exc_info:
        load_derived_calculations(path)

    assert "第 2 行" in str(exc_info.value)


def test_reject_duplicate_calculation_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate.jsonl"

    write_calculations(
        path,
        build_valid_calculation(),
        build_valid_calculation(),
    )

    with pytest.raises(
        InvalidDerivedCalculationDatasetError,
    ) as exc_info:
        load_derived_calculations(path)

    message = str(exc_info.value)

    assert "重复 calculation_id" in message
    assert "第 1 行" in message
    assert "第 2 行" in message