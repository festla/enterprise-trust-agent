from __future__ import annotations

import json
from pathlib import Path

TARGET = Path("data/evaluation/retrieval/financial_fact_retrieval_dev_v1.jsonl")
APPEND = Path("data/evaluation/retrieval/hisense_retrieval_cases_append_v1.jsonl")

def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL: {path}, line={line_number}") from exc
    return rows

existing = load_rows(TARGET)
new_rows = load_rows(APPEND)

existing_ids = {row["case_id"] for row in existing}
new_ids = [row["case_id"] for row in new_rows]

duplicates = sorted(existing_ids.intersection(new_ids))
if duplicates:
    raise ValueError(f"case_id already exists: {duplicates}")

if len(new_ids) != len(set(new_ids)):
    raise ValueError("append file contains duplicate case_id")

merged = existing + new_rows
TARGET.write_text(
    "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in merged) + "\n",
    encoding="utf-8",
)

print(f"target={TARGET}")
print(f"previous_count={len(existing)}")
print(f"appended_count={len(new_rows)}")
print(f"final_count={len(merged)}")
