from pathlib import Path

from app.services.runtime_eval import (
    build_runtime_control_dev_v1_cases,
    write_runtime_eval_cases,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "runtime"
    / "runtime_control_dev_v1.jsonl"
)


def main(
) -> None:
    cases = (
        build_runtime_control_dev_v1_cases()
    )

    write_runtime_eval_cases(
        path=OUTPUT_PATH,
        cases=cases,
    )

    print(
        "runtime_control_dev_v1 "
        f"已生成：{len(cases)} cases"
    )

    print(
        f"path={OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()