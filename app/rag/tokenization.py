from __future__ import annotations

import unicodedata
from dataclasses import (
    dataclass,
    field,
)
from typing import Protocol

from app.schemas.bm25 import (
    BM25TokenizerSpec,
)


_CJK_CODE_POINT_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0x2CEB0, 0x2EBEF),
    (0x30000, 0x3134F),
)


class BM25Tokenizer(Protocol):
    """BM25 构建和查询共同使用的分词接口。"""

    @property
    def spec(self) -> BM25TokenizerSpec:
        """返回可审计的分词配置。"""

    def tokenize(
        self,
        text: str,
    ) -> tuple[str, ...]:
        """将文本转换为有序 Token 序列。"""


def _is_cjk_character(
    value: str,
) -> bool:
    """判断一个字符是否属于支持的 CJK 表意文字范围。"""

    code_point = ord(value)

    return any(
        range_start
        <= code_point
        <= range_end
        for range_start, range_end
        in _CJK_CODE_POINT_RANGES
    )


@dataclass(frozen=True, slots=True)
class DeterministicChineseBigramTokenizer:
    """无外部词典依赖的确定性中文 Bigram 分词器。"""

    spec: BM25TokenizerSpec = field(
        default_factory=BM25TokenizerSpec
    )

    def tokenize(
        self,
        text: str,
    ) -> tuple[str, ...]:
        """按照显示规则生成中文 Bigram 和 ASCII Token。"""

        normalized_text = unicodedata.normalize(
            self.spec.unicode_normalization,
            text,
        )

        if self.spec.lowercase_ascii:
            normalized_text = (
                normalized_text.lower()
            )

        tokens: list[str] = []
        cjk_buffer: list[str] = []
        ascii_buffer: list[str] = []

        def flush_cjk_buffer() -> None:
            if not cjk_buffer:
                return

            cjk_run = "".join(cjk_buffer)
            ngram_size = (
                self.spec.cjk_ngram_size
            )

            if len(cjk_run) < ngram_size:
                tokens.append(cjk_run)
            else:
                tokens.extend(
                    cjk_run[
                        start:
                        start + ngram_size
                    ]
                    for start in range(
                        len(cjk_run)
                        - ngram_size
                        + 1
                    )
                )

            cjk_buffer.clear()

        def flush_ascii_buffer() -> None:
            if not ascii_buffer:
                return

            tokens.append(
                "".join(ascii_buffer)
            )

            ascii_buffer.clear()

        for character in normalized_text:
            if _is_cjk_character(character):
                flush_ascii_buffer()
                cjk_buffer.append(character)

            elif (
                character.isascii()
                and character.isalnum()
            ):
                flush_cjk_buffer()
                ascii_buffer.append(character)

            else:
                flush_cjk_buffer()
                flush_ascii_buffer()

        flush_cjk_buffer()
        flush_ascii_buffer()

        return tuple(tokens)