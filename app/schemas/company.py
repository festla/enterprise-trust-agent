from datetime import datetime
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .enums import Exchange, RecordStatus


class Company(BaseModel):
    """企业标准身份数据。"""

    # 给 Company 这个 Pydantic 数据模型设置全局校验规则
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    company_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_]+$",
        description="系统内部使用的标准公司 ID",
    )

    legal_name_cn: str = Field(
        min_length=1,
        description="公司正式中文名称",
    )

    short_name_cn: str = Field(
        min_length=1,
        description="公司常用中文简称",
    )

    stock_code: str = Field(
        pattern=r"^\d{6}$",
        description="六位股票代码，必须保留前导0",
    )

    exchange: Exchange

    industry: str = Field(
        min_length=1,
        description="所属行业",
    )

    status: RecordStatus = RecordStatus.ACTIVE

    created_at: datetime
    updated_at: datetime

    # 装饰器：告诉 Pydantic：下面这个函数不是普通函数，而是用于校验指定字段的函数。
    @field_validator("created_at", "updated_at")
    # cls == Company 表示这个方法属于当前这个类，而不是某一个已经创建好的对象
    @classmethod
    def validate_timezone(cls, value: datetime) -> datetime:
        """时间必须包含时区信息"""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("datetime 必须包含时区信息")

        return value

    # 装饰器，但它不是校验某一个字段，而是校验整个模型。
    # mode="after" 表示：等所有字段都完成解析和单字段校验，
    # 并且模型对象已经基本创建完成后，再执行下面的校验。
    @model_validator(mode="after")
    def validate_time_order(self) -> Self:
        """更新时间不能早于创建时间。"""

        if self.updated_at < self.created_at:
            raise ValueError("updated_at 不能早于 created_at")

        return self

    # 字段自己的规则用 field_validator，字段之间的关系用 model_validator