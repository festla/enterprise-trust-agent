from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
)


class BailianAnswerProviderConfig(
    BaseModel
):
    """百炼 OpenAI 兼容回答 Provider 配置。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    api_key: SecretStr

    api_base: str = Field(
        min_length=1,
    )

    model_name: str = Field(
        min_length=1,
    )

    timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        le=120,
    )

    max_retries: int = Field(
        default=1,
        ge=0,
        le=3,
    )

    enable_thinking: bool = False

    @field_validator("api_key")
    @classmethod
    def validate_api_key(
        cls,
        value: SecretStr,
    ) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError(
                "api_key 不能为空"
            )

        return value

    @field_validator("api_base")
    @classmethod
    def validate_api_base(
        cls,
        value: str,
    ) -> str:
        normalized = value.strip().rstrip("/")

        if not normalized.startswith(
            "https://"
        ):
            raise ValueError(
                "api_base 必须使用 HTTPS"
            )

        return normalized

    @field_validator("model_name")
    @classmethod
    def validate_model_name(
        cls,
        value: str,
    ) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "model_name 不能为空"
            )

        return normalized