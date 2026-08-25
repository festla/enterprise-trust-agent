from __future__ import annotations

from pathlib import Path

import httpx

from app.services.competition_dataset import (
    build_competition_question,
    load_competition_qa_excel,
)


QA_FILE = Path(
    "data/competition/private/qa/QA数据.xlsx"
)

BASE_URL = "http://127.0.0.1:8000"

CASE_IDS = (
    "Q105",  # Text / PDF
    "Q003",  # Excel / Lookup
)


def main() -> None:
    cases = tuple(
        load_competition_qa_excel(
            QA_FILE
        )
    )

    case_by_id = {
        case.case_id: case
        for case in cases
    }

    print()
    print(
        "===== Competition API Smoke ====="
    )

    with httpx.Client(
        base_url=BASE_URL,
        timeout=300.0,
    ) as client:

        # ==========================================
        # Health
        # ==========================================

        health_response = client.get(
            "/health"
        )

        health_response.raise_for_status()

        health = health_response.json()

        print()
        print(
            "Health:",
            health,
        )

        if (
            health.get("status")
            != "ok"
        ):
            raise RuntimeError(
                "API health status "
                "不是 ok"
            )

        if (
            health.get("ready")
            is not True
        ):
            raise RuntimeError(
                "Competition Runtime "
                "尚未 ready"
            )

        # ==========================================
        # Cases
        # ==========================================

        for case_id in CASE_IDS:
            case = case_by_id[
                case_id
            ]

            question = (
                build_competition_question(
                    case
                )
            )

            # 注意：
            # 这里只发送 CompetitionQuestion。
            #
            # answer / evidence / difficulty
            # 都不会进入 API。
            payload = (
                question.model_dump(
                    mode="json"
                )
            )

            response = client.post(
                "/api/competition/answer",
                json=payload,
            )

            print()
            print(
                "Case:",
                case_id,
            )

            print(
                "HTTP:",
                response.status_code,
            )

            if (
                response.status_code
                != 200
            ):
                print(
                    "Body:",
                    response.text,
                )

                raise RuntimeError(
                    f"{case_id} API "
                    "请求失败"
                )

            body = response.json()

            thread_id = (
                body.get(
                    "thread_id"
                )
            )

            result = (
                body.get(
                    "result",
                    {},
                )
            )

            status = (
                result.get(
                    "status"
                )
            )

            print(
                "Source type:",
                case.source_type,
            )

            print(
                "QA type:",
                case.qa_type,
            )

            print(
                "Thread:",
                thread_id,
            )

            print(
                "Status:",
                status,
            )

            if status != "answered":
                print(
                    "Result:",
                    result,
                )

                raise RuntimeError(
                    f"{case_id} "
                    "没有 answered"
                )

            answer_payload = (
                result.get(
                    "answer",
                    {}
                )
            )

            prediction = (
                answer_payload.get(
                    "answer"
                )
            )

            print(
                "Agent answer:",
                prediction,
            )

            print(
                "Gold answer:",
                case.answer,
            )

            if (
                prediction
                != case.answer
            ):
                raise RuntimeError(
                    f"{case_id} "
                    "答案错误："
                    f"{prediction} "
                    f"!= {case.answer}"
                )

            citations = (
                result.get(
                    "citations"
                )
            )

            if citations is None:
                raise RuntimeError(
                    f"{case_id} "
                    "缺少 citations"
                )

            print(
                "Citation:",
                "PASS",
            )

    print()
    print(
        "COMPETITION API SMOKE PASS"
    )


if __name__ == "__main__":
    main()