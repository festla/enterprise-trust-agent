from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from app.rag.bailian_answer_provider import (
    BailianAnswerProviderError,
)
from app.rag.competition_answer_generation import (
    CompetitionAnswerGenerationError,
    generate_competition_answer,
)
from app.rag.competition_bailian_answer_provider import (
    CompetitionBailianOpenAIAnswerProvider,
)
from app.schemas.competition import (
    CompetitionQaCase,
    CompetitionQuestion,
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
from scripts.evaluate_competition_bm25_dev import (
    load_dev_case_ids,
)
from scripts.evaluate_competition_evidence_assembly_dev import (
    _rebuild_hits,
)


DEFAULT_QA_FILE = Path(
    "data/competition/private/qa/QA数据.xlsx"
)

DEFAULT_SPLIT_FILE = Path(
    "data/competition/processed/"
    "competition_eval_split_v1.json"
)

DEFAULT_RETRIEVAL_FILE = Path(
    "data/competition/processed/eval/"
    "competition_bm25_structured_context_"
    "source_filtered_dev_v1.json"
)

DEFAULT_CORPUS_DIR = Path(
    "data/competition/processed/corpora/"
    "competition_corpus_8cbec682dfce4962"
)

DEFAULT_OUTPUT_FILE = Path(
    "data/competition/processed/eval/"
    "competition_answer_generation_dev_v1.json"
)

EXPECTED_TEXT_DEV_CASES = 69

EVIDENCE_MAX_COUNT = 48
EVIDENCE_MAX_CHARS = 4000


@dataclass(
    frozen=True,
    slots=True,
)
class CompetitionAnswerCaseEval:
    case_id: str
    source_type: str
    qa_type: str
    difficulty: str

    gold_answer: str
    predicted_answer: str | None

    correct: bool

    generation_status: str

    answer_text: str | None

    citation_ids: tuple[
        str,
        ...
    ]

    evidence_ids: tuple[
        str,
        ...
    ]

    evidence_count: int
    evidence_chars: int

    generator_id: str | None

    error_type: str | None
    error_message: str | None


def _load_json(
    path: Path,
) -> dict[str, object]:
    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            f"JSON 根节点必须是 Object：{path}"
        )

    return payload


def _build_question(
    case: CompetitionQaCase,
) -> CompetitionQuestion:
    """
    显式从 CompetitionQaCase 构造 Solver 输入。

    不传入：
        answer
        answer_text
        evidence
        difficulty

    防止 Gold 泄漏。
    """

    return CompetitionQuestion(
        case_id=case.case_id,
        source_type=case.source_type,
        qa_type=case.qa_type,
        question=case.question,
        option_a=case.option_a,
        option_b=case.option_b,
        option_c=case.option_c,
        option_d=case.option_d,
        source_title=case.source_title,
        file_label=case.file_label,
    )


def _build_group_summary(
    results: list[
        CompetitionAnswerCaseEval
    ],
    *,
    field_name: str,
) -> dict[str, object]:
    groups: dict[
        str,
        list[
            CompetitionAnswerCaseEval
        ],
    ] = {}

    for result in results:
        value = getattr(
            result,
            field_name,
        )

        groups.setdefault(
            value,
            [],
        ).append(
            result
        )

    payload = {}

    for (
        value,
        group,
    ) in sorted(
        groups.items()
    ):
        correct_count = sum(
            int(result.correct)
            for result in group
        )

        generated_count = sum(
            int(
                result.generation_status
                == "generated"
            )
            for result in group
        )

        payload[value] = {
            "case_count": len(group),
            "generated_count": (
                generated_count
            ),
            "correct_count": (
                correct_count
            ),
            "accuracy": (
                correct_count
                / len(group)
            ),
        }

    return payload


def _build_summary(
    results: list[
        CompetitionAnswerCaseEval
    ],
) -> dict[str, object]:
    case_count = len(
        results
    )

    generated_count = sum(
        int(
            result.generation_status
            == "generated"
        )
        for result in results
    )

    correct_count = sum(
        int(result.correct)
        for result in results
    )

    error_count = (
        case_count
        - generated_count
    )

    citation_count = sum(
        len(
            result.citation_ids
        )
        for result in results
    )

    return {
        "case_count": case_count,
        "generated_count": (
            generated_count
        ),
        "generation_error_count": (
            error_count
        ),
        "generation_success_rate": (
            generated_count
            / case_count
            if case_count
            else 0.0
        ),
        "correct_count": correct_count,
        # Generation Error 也按错误答案计入。
        "accuracy": (
            correct_count
            / case_count
            if case_count
            else 0.0
        ),
        "citation_count": (
            citation_count
        ),
        "mean_citations_per_case": (
            citation_count
            / case_count
            if case_count
            else 0.0
        ),
        "error_types": dict(
            sorted(
                Counter(
                    result.error_type
                    for result in results
                    if (
                        result.error_type
                        is not None
                    )
                ).items()
            )
        ),
        "by_qa_type": (
            _build_group_summary(
                results,
                field_name="qa_type",
            )
        ),
        "by_source_type": (
            _build_group_summary(
                results,
                field_name=(
                    "source_type"
                ),
            )
        ),
        "by_difficulty": (
            _build_group_summary(
                results,
                field_name=(
                    "difficulty"
                ),
            )
        ),
    }


def run(
    *,
    qa_file: Path,
    split_file: Path,
    retrieval_file: Path,
    corpus_dir: Path,
    output_file: Path,
    limit: int | None,
    case_ids: tuple[
        str,
        ...
    ],
) -> None:
    all_cases = tuple(
        load_competition_qa_excel(
            qa_file
        )
    )

    case_by_id = {
        case.case_id: case
        for case in all_cases
    }

    dev_case_ids = (
        load_dev_case_ids(
            split_file
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

    if (
        not case_ids
        and len(text_cases)
        != EXPECTED_TEXT_DEV_CASES
    ):
        raise RuntimeError(
            "Frozen Dev 文本 Case "
            "数量改变："
            f"{len(text_cases)}"
        )

    if case_ids:
        requested = set(
            case_ids
        )

        unknown = (
            requested
            - {
                case.case_id
                for case in text_cases
            }
        )

        if unknown:
            raise RuntimeError(
                "请求了不存在于文本 Dev "
                "中的 Case："
                f"{tuple(sorted(unknown))}"
            )

        text_cases = [
            case
            for case in text_cases
            if (
                case.case_id
                in requested
            )
        ]

    if limit is not None:
        if limit < 1:
            raise ValueError(
                "limit 必须 >= 1"
            )

        text_cases = (
            text_cases[:limit]
        )

    retrieval = _load_json(
        retrieval_file
    )

    corpus = (
        load_competition_chunk_corpus(
            corpus_dir
        )
    )

    if (
        retrieval.get(
            "corpus_id"
        )
        != corpus.manifest.corpus_id
    ):
        raise RuntimeError(
            "Retrieval 与 Corpus "
            "身份不一致"
        )

    raw_cases = retrieval.get(
        "cases"
    )

    if not isinstance(
        raw_cases,
        list,
    ):
        raise RuntimeError(
            "Retrieval Artifact "
            "缺少 cases"
        )

    retrieval_by_case: dict[
        str,
        dict[str, object],
    ] = {}

    for raw_case in raw_cases:
        if not isinstance(
            raw_case,
            dict,
        ):
            raise RuntimeError(
                "Retrieval Case 类型无效"
            )

        case_id = raw_case.get(
            "case_id"
        )

        if not isinstance(
            case_id,
            str,
        ):
            raise RuntimeError(
                "Retrieval Case "
                "缺少 case_id"
            )

        retrieval_by_case[
            case_id
        ] = raw_case

    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunk in corpus.chunks
    }

    evidence_config = (
        CompetitionEvidenceAssemblyConfig(
            max_evidences=(
                EVIDENCE_MAX_COUNT
            ),
            max_chars=(
                EVIDENCE_MAX_CHARS
            ),
        )
    )

    provider = (
        CompetitionBailianOpenAIAnswerProvider
        .from_environment()
    )

    results: list[
        CompetitionAnswerCaseEval
    ] = []

    print()
    print(
        "===== Competition Frozen Dev "
        "Answer Evaluation ====="
    )
    print(
        "Cases:",
        len(text_cases),
    )
    print(
        "Provider:",
        provider.provider_id,
    )
    print(
        "Evidence budget:",
        f"{EVIDENCE_MAX_COUNT} / "
        f"{EVIDENCE_MAX_CHARS}",
    )
    print()

    for (
        index,
        case,
    ) in enumerate(
        text_cases,
        start=1,
    ):
        raw_case = (
            retrieval_by_case.get(
                case.case_id
            )
        )

        if raw_case is None:
            raise RuntimeError(
                "缺少 Retrieval Case："
                f"{case.case_id}"
            )

        expanded_hits = (
            _rebuild_hits(
                case_payload=raw_case,
                chunks_by_id=(
                    chunks_by_id
                ),
            )
        )

        evidence_bundle = (
            assemble_competition_evidence_bundle(
                hits=expanded_hits,
                source_records=(
                    corpus.manifest
                    .source_records
                ),
                config=evidence_config,
            )
        )

        evidence_count = len(
            evidence_bundle.evidences
        )

        evidence_chars = sum(
            len(
                evidence.raw_content
            )
            for evidence
            in evidence_bundle.evidences
        )

        question = _build_question(
            case
        )

        try:
            answer = (
                generate_competition_answer(
                    question=question,
                    evidence_bundle=(
                        evidence_bundle
                    ),
                    provider=provider,
                )
            )

        except (
            BailianAnswerProviderError,
            CompetitionAnswerGenerationError,
        ) as exc:
            result = (
                CompetitionAnswerCaseEval(
                    case_id=case.case_id,
                    source_type=(
                        case.source_type
                    ),
                    qa_type=case.qa_type,
                    difficulty=(
                        case.difficulty
                    ),
                    gold_answer=(
                        case.answer
                    ),
                    predicted_answer=None,
                    correct=False,
                    generation_status=(
                        "error"
                    ),
                    answer_text=None,
                    citation_ids=(),
                    evidence_ids=(),
                    evidence_count=(
                        evidence_count
                    ),
                    evidence_chars=(
                        evidence_chars
                    ),
                    generator_id=None,
                    error_type=(
                        type(exc).__name__
                    ),
                    error_message=str(exc),
                )
            )

            results.append(
                result
            )

            print(
                f"[{index:02d}/"
                f"{len(text_cases):02d}] "
                f"{case.case_id} "
                "ERROR "
                f"{type(exc).__name__}"
            )

            continue

        correct = (
            answer.answer
            == case.answer
        )

        result = (
            CompetitionAnswerCaseEval(
                case_id=case.case_id,
                source_type=(
                    case.source_type
                ),
                qa_type=case.qa_type,
                difficulty=(
                    case.difficulty
                ),
                gold_answer=(
                    case.answer
                ),
                predicted_answer=(
                    answer.answer
                ),
                correct=correct,
                generation_status=(
                    "generated"
                ),
                answer_text=(
                    answer.answer_text
                ),
                citation_ids=(
                    answer.citation_ids
                ),
                evidence_ids=(
                    answer.evidence_ids
                ),
                evidence_count=(
                    evidence_count
                ),
                evidence_chars=(
                    evidence_chars
                ),
                generator_id=(
                    answer.generator_id
                ),
                error_type=None,
                error_message=None,
            )
        )

        results.append(
            result
        )

        print(
            f"[{index:02d}/"
            f"{len(text_cases):02d}] "
            f"{case.case_id} "
            f"pred={answer.answer} "
            f"gold={case.answer} "
            f"correct={correct} "
            f"citations="
            f"{len(answer.citation_ids)}"
        )

    summary = (
        _build_summary(
            results
        )
    )

    print()
    print(
        "===== Summary ====="
    )

    print(
        "Generated:",
        f"{summary['generated_count']}/"
        f"{summary['case_count']}",
    )

    print(
        "Correct:",
        f"{summary['correct_count']}/"
        f"{summary['case_count']}",
    )

    print(
        "Accuracy:",
        f"{summary['accuracy']:.4f}",
    )

    print(
        "Mean citations:",
        f"{summary['mean_citations_per_case']:.2f}",
    )

    print()
    print(
        "By QA type:"
    )

    for (
        qa_type,
        group,
    ) in summary[
        "by_qa_type"
    ].items():
        print(
            f"  {qa_type}: "
            f"{group['correct_count']}/"
            f"{group['case_count']} "
            f"({group['accuracy']:.4f})"
        )

    incorrect = [
        result
        for result in results
        if not result.correct
    ]

    if incorrect:
        print()
        print(
            "Incorrect / Error Cases:"
        )

        for result in incorrect:
            print(
                " ",
                result.case_id,
                "gold=",
                result.gold_answer,
                "pred=",
                result.predicted_answer,
                "status=",
                result.generation_status,
            )

    payload = {
        "schema_version": 1,
        "evaluation_version": (
            "competition_answer_"
            "generation_dev_v1"
        ),
        "corpus_id": (
            corpus.manifest.corpus_id
        ),
        "retrieval_file": str(
            retrieval_file
        ),
        "provider_id": (
            provider.provider_id
        ),
        "provider_request_count": (
            provider.request_count
        ),
        "evidence_config": {
            "max_evidences": (
                EVIDENCE_MAX_COUNT
            ),
            "max_chars": (
                EVIDENCE_MAX_CHARS
            ),
        },
        "summary": summary,
        "cases": [
            asdict(result)
            for result in results
        ],
    }

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file.write_text(
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

    print()
    print(
        "Requests:",
        provider.request_count,
    )

    print(
        "Saved:",
        output_file,
    )


def build_argument_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "评测 Competition Frozen Dev "
            "文本 QA Answer Generation"
        )
    )

    parser.add_argument(
        "--qa",
        type=Path,
        default=DEFAULT_QA_FILE,
    )

    parser.add_argument(
        "--split",
        type=Path,
        default=DEFAULT_SPLIT_FILE,
    )

    parser.add_argument(
        "--retrieval",
        type=Path,
        default=(
            DEFAULT_RETRIEVAL_FILE
        ),
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
    )

    return parser


def main() -> None:
    arguments = (
        build_argument_parser()
        .parse_args()
    )

    run(
        qa_file=arguments.qa,
        split_file=arguments.split,
        retrieval_file=(
            arguments.retrieval
        ),
        corpus_dir=arguments.corpus,
        output_file=arguments.output,
        limit=arguments.limit,
        case_ids=tuple(
            arguments.case_id
        ),
    )


if __name__ == "__main__":
    main()