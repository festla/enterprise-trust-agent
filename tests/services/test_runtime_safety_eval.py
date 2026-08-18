from collections import Counter

from app.services.runtime_safety_eval import (
    build_week7_safety_cases,
)


def test_week7_safety_eval_has_exactly_40_cases(
) -> None:
    cases = (
        build_week7_safety_cases()
    )

    assert len(cases) == 40


def test_week7_safety_eval_category_distribution(
) -> None:
    cases = (
        build_week7_safety_cases()
    )

    counts = Counter(
        case.category
        for case in cases
    )

    assert counts == {
        "evidence_citation": 6,
        "numeric_scope": 6,
        "rbac": 5,
        "prompt_injection": 6,
        "unsupported_boundary": 5,
        "risk_hitl": 6,
        "normal_safe": 6,
    }


def test_week7_safety_eval_case_ids_are_unique(
) -> None:
    cases = (
        build_week7_safety_cases()
    )

    case_ids = tuple(
        case.case_id
        for case in cases
    )

    assert (
        len(case_ids)
        == len(set(case_ids))
    )


def test_week7_safety_eval_case_ids_are_stable(
) -> None:
    cases = (
        build_week7_safety_cases()
    )

    assert tuple(
        case.case_id
        for case in cases
    ) == tuple(
        f"safety_{index:03d}"
        for index in range(
            1,
            41,
        )
    )


def test_normal_safe_cases_are_not_adversarial(
) -> None:
    cases = (
        build_week7_safety_cases()
    )

    safe_cases = tuple(
        case
        for case in cases
        if (
            case.category
            == "normal_safe"
        )
    )

    assert len(safe_cases) == 6

    assert all(
        not case.adversarial
        for case in safe_cases
    )

    assert all(
        case.expected_outcome
        == "allow"
        for case in safe_cases
    )


def test_prompt_injection_cases_have_rule_expectations(
) -> None:
    cases = (
        build_week7_safety_cases()
    )

    prompt_cases = tuple(
        case
        for case in cases
        if (
            case.category
            == "prompt_injection"
        )
    )

    assert len(prompt_cases) == 6

    assert all(
        case.document_text
        for case in prompt_cases
    )

    assert all(
        case.expected_rule_ids
        for case in prompt_cases
    )


def test_safety_eval_contains_attack_and_safe_cases(
) -> None:
    cases = (
        build_week7_safety_cases()
    )

    assert any(
        case.adversarial
        for case in cases
    )

    assert any(
        not case.adversarial
        for case in cases
    )