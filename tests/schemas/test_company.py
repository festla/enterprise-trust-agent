from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.schemas.company import Company
from app.schemas.enums import Exchange, RecordStatus


def build_valid_company_data() -> dict:
    """生成一份合法的公司测试数据。"""

    now = datetime.now(timezone.utc)

    return {
        "company_id": "midea",
        "legal_name_cn": "美的集团股份有限公司",
        "short_name_cn": "美的集团",
        "stock_code": "000333",
        "exchange": "SZSE",
        "industry": "家电制造业",
        "status": "active",
        "created_at": now,
        "updated_at": now,
    }


def test_create_valid_company() -> None:
    """合法数据应成功创建 Company。"""

    company = Company(**build_valid_company_data())

    assert company.company_id == "midea"
    assert company.stock_code == "000333"
    assert company.exchange is Exchange.SZSE
    assert company.status is RecordStatus.ACTIVE


def test_reject_unknown_field() -> None:
    """Schema 未声明的字段应被拒绝。"""

    data = build_valid_company_data()
    data["unknown_field"] = "unexpected"

    with pytest.raises(ValidationError):
        Company(**data)


def test_reject_invalid_company_id() -> None:
    """company_id 只能包含小写字母、数字和下划线。"""

    data = build_valid_company_data()
    data["company_id"] = "Midea Group"

    with pytest.raises(ValidationError):
        Company(**data)


def test_reject_invalid_stock_code() -> None:
    """股票代码必须是六位数字字符串。"""

    data = build_valid_company_data()
    data["stock_code"] = "333"

    with pytest.raises(ValidationError):
        Company(**data)


def test_reject_timezone_naive_datetime() -> None:
    """没有时区信息的 datetime 应被拒绝。"""

    data = build_valid_company_data()
    data["created_at"] = datetime.now()

    with pytest.raises(ValidationError):
        Company(**data)


def test_reject_updated_at_before_created_at() -> None:
    """更新时间早于创建时间时应被拒绝。"""

    now = datetime.now(timezone.utc)

    data = build_valid_company_data()
    data["created_at"] = now
    data["updated_at"] = now - timedelta(minutes=1)

    with pytest.raises(ValidationError):
        Company(**data)