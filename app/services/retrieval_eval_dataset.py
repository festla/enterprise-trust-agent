from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from app.schemas.retrieval_eval import (
    FinancialFactRetrievalEvalCase,
)


class RetrievalEvalDatasetError(
    ValueError
):
    """检索评测集读取基础异常。"""


class RetrievalEvalDatasetNotFoundError(
    RetrievalEvalDatasetError
):
    """评测集文件不存在。"""


class InvalidRetrievalEvalDatasetError(
    RetrievalEvalDatasetError
):
    """评测集内容无效。"""


def load_financial_fact_retrieval_cases(
    path: Path,
) -> tuple[
    FinancialFactRetrievalEvalCase,
    ...,
]:
    """从 JSONL 加载并校验财务事实检索题。"""

    if not path.is_file():
        raise RetrievalEvalDatasetNotFoundError(
            f"评测集文件不存在：{path}"
        )

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise RetrievalEvalDatasetError(
            "无法读取检索评测集"
        ) from exc

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidRetrievalEvalDatasetError(
            "检索评测集必须使用 UTF-8 编码"
        ) from exc

    lines = text.splitlines()

    if not lines:
        raise InvalidRetrievalEvalDatasetError(
            "检索评测集不能为空"
        )

    if any(not line.strip() for line in lines):
        raise InvalidRetrievalEvalDatasetError(
            "JSONL 不能包含空记录"
        )

    try:
        cases = tuple(
            FinancialFactRetrievalEvalCase
            .model_validate_json(line)
            for line in lines
        )
    except ValidationError as exc:
        raise InvalidRetrievalEvalDatasetError(
            "检索评测集包含无效题目"
        ) from exc

    case_ids = [
        case.case_id
        for case in cases
    ]

    if len(case_ids) != len(set(case_ids)):
        raise InvalidRetrievalEvalDatasetError(
            "检索评测集包含重复 case_id"
        )

    return cases