from __future__ import annotations

import json
from pathlib import Path

from app.rag.bailian_answer_provider import (
    BailianAnswerProviderError,
)
from app.rag.competition_bailian_sufficiency_provider import (
    CompetitionBailianSufficiencyProvider,
)
from app.rag.competition_sufficiency import (
    CompetitionEvidenceSufficiencyError,
    assess_competition_semantic_sufficiency,
)
from app.schemas.competition_evidence import (
    CompetitionEvidenceBundle,
)
from app.services.competition_corpus import (
    load_competition_chunk_corpus,
)
from app.services.competition_dataset import (
    load_competition_qa_excel,
)
from app.services.competition_evidence_assembler import (
    CompetitionEvidenceAssemblyConfig,
    assemble_competition_evidence_bundle,
)
from scripts.evaluate_competition_answer_generation_dev import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_QA_FILE,
    DEFAULT_RETRIEVAL_FILE,
    DEFAULT_SPLIT_FILE,
    _build_question,
    _load_json,
)
from scripts.evaluate_competition_bm25_dev import (
    load_dev_case_ids,
)
from scripts.evaluate_competition_evidence_assembly_dev import (
    _rebuild_hits,
)


ANSWER_FILE = Path(
    "data/competition/processed/eval/"
    "competition_answer_generation_dev_v1.json"
)

OUTPUT_FILE = Path(
    "data/competition/processed/eval/"
    "competition_sufficiency_negative_dev_v1.json"
)

EVIDENCE_CONFIG = (
    CompetitionEvidenceAssemblyConfig(
        max_evidences=48,
        max_chars=4000,
    )
)

SINGLE_CASE_COUNT = 3
MULTI_CASE_COUNT = 3


def _mask_used_evidence(
    *,
    bundle: CompetitionEvidenceBundle,
    used_evidence_ids: set[str],
) -> CompetitionEvidenceBundle:
    """
    只遮掉最终正确回答实际引用过的 Evidence。

    其他 Evidence 保留，因此这是较难负例。
    """

    evidences = tuple(
        evidence.model_copy(
            update={
                "raw_content": (
                    "【该 Evidence 的关键正文"
                    "已在负例测试中移除。】"
                )
            }
        )
        if evidence.evidence_id
        in used_evidence_ids
        else evidence
        for evidence
        in bundle.evidences
    )

    return bundle.model_copy(
        update={
            "evidences": evidences
        }
    )


def _redact_all_evidence(
    bundle: CompetitionEvidenceBundle,
) -> CompetitionEvidenceBundle:
    """
    删除所有 Evidence 的事实内容。

    这是强负例：
    Judge 必须拒答，不能凭模型记忆猜测。
    """

    evidences = tuple(
        evidence.model_copy(
            update={
                "raw_content": (
                    "【正文已移除。"
                    "当前 Evidence 不包含"
                    "可用于判断选项的事实。】"
                )
            }
        )
        for evidence
        in bundle.evidences
    )

    return bundle.model_copy(
        update={
            "evidences": evidences
        }
    )


def _run_assessment(
    *,
    case,
    bundle: CompetitionEvidenceBundle,
    provider: CompetitionBailianSufficiencyProvider,
) -> dict[str, object]:
    question = _build_question(
        case
    )

    try:
        assessment = (
            assess_competition_semantic_sufficiency(
                question=question,
                evidence_bundle=bundle,
                provider=provider,
            )
        )

    except (
        BailianAnswerProviderError,
        CompetitionEvidenceSufficiencyError,
    ) as exc:
        return {
            "status": "error",
            "supported_answer": None,
            "reason": None,
            "citation_ids": [],
            "error_type": (
                type(exc).__name__
            ),
            "error_message": str(exc),
        }

    return {
        "status": assessment.status,
        "supported_answer": (
            assessment.supported_answer
        ),
        "reason": assessment.reason,
        "citation_ids": list(
            assessment.citation_ids
        ),
        "error_type": None,
        "error_message": None,
    }


def main() -> None:
    all_cases = tuple(
        load_competition_qa_excel(
            DEFAULT_QA_FILE
        )
    )

    case_by_id = {
        case.case_id: case
        for case in all_cases
    }

    dev_case_ids = (
        load_dev_case_ids(
            DEFAULT_SPLIT_FILE
        )
    )

    text_cases = [
        case_by_id[case_id]
        for case_id in dev_case_ids
        if (
            case_by_id[
                case_id
            ].source_type
            in {
                "word",
                "pdf",
            }
        )
    ]

    text_cases.sort(
        key=lambda case: (
            case.case_id
        )
    )

    single_cases = [
        case
        for case in text_cases
        if (
            case.qa_type
            == "单事实检索"
        )
    ][
        :SINGLE_CASE_COUNT
    ]

    multi_cases = [
        case
        for case in text_cases
        if (
            case.qa_type
            == "多事实检索"
        )
    ][
        :MULTI_CASE_COUNT
    ]

    selected_cases = (
        single_cases
        + multi_cases
    )

    if len(
        selected_cases
    ) != (
        SINGLE_CASE_COUNT
        + MULTI_CASE_COUNT
    ):
        raise RuntimeError(
            "无法构造预期的"
            "单事实/多事实测试集"
        )

    retrieval = _load_json(
        DEFAULT_RETRIEVAL_FILE
    )

    raw_retrieval_cases = (
        retrieval.get(
            "cases"
        )
    )

    if not isinstance(
        raw_retrieval_cases,
        list,
    ):
        raise RuntimeError(
            "Retrieval Artifact "
            "缺少 cases"
        )

    retrieval_by_case = {
        raw_case["case_id"]:
            raw_case
        for raw_case
        in raw_retrieval_cases
    }

    answer_payload = _load_json(
        ANSWER_FILE
    )

    raw_answer_cases = (
        answer_payload.get(
            "cases"
        )
    )

    if not isinstance(
        raw_answer_cases,
        list,
    ):
        raise RuntimeError(
            "Answer Artifact "
            "缺少 cases"
        )

    answer_by_case = {
        raw_case["case_id"]:
            raw_case
        for raw_case
        in raw_answer_cases
    }

    corpus = (
        load_competition_chunk_corpus(
            DEFAULT_CORPUS_DIR
        )
    )

    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunk in corpus.chunks
    }

    provider = (
        CompetitionBailianSufficiencyProvider
        .from_environment()
    )

    results: list[
        dict[str, object]
    ] = []

    print()
    print(
        "===== Competition "
        "Sufficiency Negative Eval ====="
    )
    print(
        "Provider:",
        provider.provider_id,
    )
    print(
        "Cases:",
        len(selected_cases),
    )
    print()

    for (
        index,
        case,
    ) in enumerate(
        selected_cases,
        start=1,
    ):
        retrieval_case = (
            retrieval_by_case.get(
                case.case_id
            )
        )

        if retrieval_case is None:
            raise RuntimeError(
                "缺少 Retrieval Case："
                f"{case.case_id}"
            )

        answer_case = (
            answer_by_case.get(
                case.case_id
            )
        )

        if answer_case is None:
            raise RuntimeError(
                "缺少 Answer Case："
                f"{case.case_id}"
            )

        if (
            answer_case.get(
                "generation_status"
            )
            != "generated"
        ):
            raise RuntimeError(
                "测试 Case 没有 Frozen "
                "成功回答："
                f"{case.case_id}"
            )

        expanded_hits = (
            _rebuild_hits(
                case_payload=(
                    retrieval_case
                ),
                chunks_by_id=(
                    chunks_by_id
                ),
            )
        )

        full_bundle = (
            assemble_competition_evidence_bundle(
                hits=expanded_hits,
                source_records=(
                    corpus.manifest
                    .source_records
                ),
                config=(
                    EVIDENCE_CONFIG
                ),
            )
        )

        used_evidence_ids = set(
            answer_case.get(
                "evidence_ids",
                [],
            )
        )

        if not used_evidence_ids:
            raise RuntimeError(
                "Frozen Answer "
                "缺少 evidence_ids："
                f"{case.case_id}"
            )

        masked_bundle = (
            _mask_used_evidence(
                bundle=full_bundle,
                used_evidence_ids=(
                    used_evidence_ids
                ),
            )
        )

        redacted_bundle = (
            _redact_all_evidence(
                full_bundle
            )
        )

        full = _run_assessment(
            case=case,
            bundle=full_bundle,
            provider=provider,
        )

        masked = _run_assessment(
            case=case,
            bundle=masked_bundle,
            provider=provider,
        )

        redacted = _run_assessment(
            case=case,
            bundle=redacted_bundle,
            provider=provider,
        )

        positive_pass = (
            full["status"]
            == "sufficient"
            and
            full[
                "supported_answer"
            ]
            == case.answer
        )

        masked_refused = (
            masked["status"]
            == "insufficient"
        )

        redacted_pass = (
            redacted["status"]
            == "insufficient"
        )

        results.append(
            {
                "case_id": (
                    case.case_id
                ),
                "qa_type": (
                    case.qa_type
                ),
                "source_type": (
                    case.source_type
                ),
                "gold_answer": (
                    case.answer
                ),
                "used_evidence_ids": (
                    sorted(
                        used_evidence_ids
                    )
                ),
                "full": full,
                "masked_cited": (
                    masked
                ),
                "redacted_all": (
                    redacted
                ),
                "positive_pass": (
                    positive_pass
                ),
                "masked_cited_refused": (
                    masked_refused
                ),
                "redacted_pass": (
                    redacted_pass
                ),
            }
        )

        print(
            f"[{index}/"
            f"{len(selected_cases)}] "
            f"{case.case_id} "
            f"{case.qa_type}"
        )

        print(
            "    full:"
            f" {full['status']} "
            f"answer="
            f"{full['supported_answer']} "
            f"pass={positive_pass}"
        )

        print(
            "    mask:"
            f" {masked['status']} "
            f"answer="
            f"{masked['supported_answer']} "
            f"refused="
            f"{masked_refused}"
        )

        print(
            "    redact:"
            f" {redacted['status']} "
            f"answer="
            f"{redacted['supported_answer']} "
            f"pass={redacted_pass}"
        )

    case_count = len(
        results
    )

    positive_pass_count = sum(
        int(
            result[
                "positive_pass"
            ]
        )
        for result in results
    )

    masked_refusal_count = sum(
        int(
            result[
                "masked_cited_refused"
            ]
        )
        for result in results
    )

    redacted_pass_count = sum(
        int(
            result[
                "redacted_pass"
            ]
        )
        for result in results
    )

    acceptance = (
        positive_pass_count
        == case_count
        and
        redacted_pass_count
        == case_count
    )

    summary = {
        "case_count": case_count,
        "request_count": (
            provider.request_count
        ),
        "positive_pass_count": (
            positive_pass_count
        ),
        "masked_cited_refusal_count": (
            masked_refusal_count
        ),
        "redacted_pass_count": (
            redacted_pass_count
        ),
        "acceptance": (
            acceptance
        ),
    }

    print()
    print(
        "===== Summary ====="
    )

    print(
        "Positive full:",
        f"{positive_pass_count}/"
        f"{case_count}",
    )

    print(
        "Masked cited refusal:",
        f"{masked_refusal_count}/"
        f"{case_count}",
        "(diagnostic)",
    )

    print(
        "Redacted negative:",
        f"{redacted_pass_count}/"
        f"{case_count}",
    )

    print(
        "Requests:",
        provider.request_count,
    )

    print(
        "Acceptance:",
        "PASS"
        if acceptance
        else "FAIL",
    )

    payload = {
        "schema_version": 1,
        "evaluation_version": (
            "competition_sufficiency_"
            "negative_dev_v1"
        ),
        "provider_id": (
            provider.provider_id
        ),
        "summary": summary,
        "cases": results,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "Saved:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()