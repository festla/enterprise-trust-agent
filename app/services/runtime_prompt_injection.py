from __future__ import annotations

import re
import unicodedata

from dataclasses import (
    dataclass,
)

from typing import (
    Literal,
)


# ============================================================
# Week7 - Step 5.1
#
# Deterministic Prompt Injection Detector
#
# 这一层只负责：
#
# Untrusted Text
#      ↓
# Normalization
#      ↓
# Rule Matching
#      ↓
# InjectionDetectionResult
#
# 暂时不负责：
#
# - AgentState
# - Runtime Node
# - Controlled Refusal
# - Checkpoint
# - Trajectory
# - HITL
# - Risk Policy
#
# 这些在 Step5.2 / Step5.3 接入。
# ============================================================


InjectionSeverity = Literal[
    "none",
    "medium",
    "high",
    "critical",
]


class RuntimePromptInjectionError(
    ValueError
):
    """Prompt Injection 检测基础异常。"""


@dataclass(
    frozen=True,
    slots=True,
)
class InjectionDetectionResult:
    """一次确定性的 Prompt Injection 检测结果。"""

    detected: bool

    severity: InjectionSeverity

    matched_rule_ids: tuple[
        str,
        ...
    ]

    reason: str


@dataclass(
    frozen=True,
    slots=True,
)
class InjectionRule:
    """一条 Prompt Injection 检测规则。"""

    rule_id: str

    severity: InjectionSeverity

    patterns: tuple[
        str,
        ...
    ]

    description: str


# ============================================================
# Severity Ranking
#
# 多条规则同时命中时，
# 返回其中最高的风险等级。
# ============================================================

_SEVERITY_RANK: dict[
    InjectionSeverity,
    int,
] = {
    "none": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}


# ============================================================
# Prompt Injection Rule Set
#
# 第一版故意采用确定性规则，而不是再调用 LLM。
#
# 主要覆盖：
#
# 1. Instruction Override
# 2. System Prompt Extraction
# 3. Authority Hijacking
# 4. Tool Manipulation
# 5. Security Bypass
#
# 规则需要尽量识别“指令性攻击”，
# 而不是简单看到某一个普通词就拦截。
# ============================================================

_INJECTION_RULES: tuple[
    InjectionRule,
    ...
] = (

    # ========================================================
    # A. Instruction Override
    #
    # 试图要求模型忽略已有的上层指令。
    # ========================================================

    InjectionRule(
        rule_id=(
            "instruction_override"
        ),
        severity="critical",
        patterns=(
            (
                r"\bignore\s+"
                r"(?:all\s+)?"
                r"(?:previous|prior|above)"
                r"\s+instructions?\b"
            ),
            (
                r"\bdisregard\s+"
                r"(?:all\s+)?"
                r"(?:previous|prior|above)"
                r"\s+instructions?\b"
            ),
            (
                r"\bforget\s+"
                r"(?:all\s+)?"
                r"(?:previous|prior)"
                r"\s+instructions?\b"
            ),
            (
                r"(?:忽略|无视|不要遵循|不必遵循)"
                r"(?:之前|先前|此前|上述|前面)"
                r"(?:的)?"
                r"(?:所有|全部)?"
                r"(?:指令|要求|规则|提示)"
            ),
        ),
        description=(
            "检测到试图覆盖或忽略已有指令"
        ),
    ),

    # ========================================================
    # B. System Prompt Extraction
    #
    # 试图获取 System / Developer Prompt。
    # ========================================================

    InjectionRule(
        rule_id=(
            "system_prompt_extraction"
        ),
        severity="critical",
        patterns=(
            (
                r"\b(?:reveal|show|print|display|"
                r"expose|output)\s+"
                r"(?:the\s+)?"
                r"(?:system|developer|hidden)"
                r"\s+(?:prompt|instructions?)\b"
            ),
            (
                r"\bwhat\s+(?:is|are)\s+"
                r"(?:your\s+)?"
                r"(?:system|developer)"
                r"\s+(?:prompt|instructions?)\b"
            ),
            (
                r"(?:输出|显示|展示|泄露|透露|打印)"
                r"(?:你的|当前的|隐藏的|内部的)?"
                r"(?:系统提示词|系统指令|"
                r"开发者提示词|开发者指令|隐藏提示词)"
            ),
        ),
        description=(
            "检测到试图获取系统或开发者指令"
        ),
    ),

    # ========================================================
    # C. Authority Hijacking
    #
    # 试图把普通文档内容提升为：
    #
    # system / developer / administrator
    # ========================================================

    InjectionRule(
        rule_id=(
            "authority_hijacking"
        ),
        severity="high",
        patterns=(
            (
                r"\byou\s+are\s+now\s+"
                r"(?:the\s+)?"
                r"(?:system|developer|"
                r"administrator|admin)\b"
            ),
            (
                r"\bact\s+as\s+"
                r"(?:the\s+)?"
                r"(?:system|developer|"
                r"administrator|admin)\b"
            ),
            (
                r"\bpretend\s+to\s+be\s+"
                r"(?:the\s+)?"
                r"(?:system|developer|"
                r"administrator|admin)\b"
            ),
            (
                r"(?:你现在是|现在你是|"
                r"请充当|假装你是)"
                r"(?:系统|开发者|管理员|超级管理员)"
            ),
        ),
        description=(
            "检测到试图劫持模型角色或权限身份"
        ),
    ),

    # ========================================================
    # D. Tool Manipulation
    #
    # Retrieved Document 只能是 DATA。
    #
    # 文档不能命令 Agent 调 Tool / Shell / Command。
    # ========================================================

    InjectionRule(
        rule_id=(
            "tool_manipulation"
        ),
        severity="high",
        patterns=(
            (
                r"\b(?:call|invoke|use)\s+"
                r"(?:the\s+)?"
                r"(?:tool|function)\b"
            ),
            (
                r"\b(?:execute|run)\s+"
                r"(?:the\s+)?"
                r"(?:shell\s+)?command\b"
            ),
            (
                r"\brun\s+(?:a\s+)?shell\b"
            ),
            (
                r"(?:请|立即|必须)?"
                r"(?:调用|使用)"
                r"(?:这个|该|以下|指定的)?"
                r"(?:工具|函数)"
            ),
            (
                r"(?:请|立即|必须)?"
                r"(?:执行|运行)"
                r"(?:以下|这个|该)?"
                r"(?:命令|shell命令|系统命令)"
            ),
        ),
        description=(
            "检测到文档试图控制 Tool 或命令执行"
        ),
    ),

    # ========================================================
    # E. Security Bypass
    #
    # 试图关闭、忽略或绕过安全边界。
    # ========================================================

    InjectionRule(
        rule_id=(
            "security_bypass"
        ),
        severity="critical",
        patterns=(
            (
                r"\b(?:bypass|disable|ignore|"
                r"circumvent)\s+"
                r"(?:the\s+)?"
                r"(?:safety|security|policy|"
                r"permissions?|guardrails?)\b"
            ),
            (
                r"(?:绕过|关闭|禁用|跳过|无视)"
                r"(?:所有|当前|现有的)?"
                r"(?:安全策略|安全规则|"
                r"权限检查|权限控制|安全限制)"
            ),
        ),
        description=(
            "检测到试图绕过系统安全或权限边界"
        ),
    ),
)


@dataclass(
    frozen=True,
    slots=True,
)
class RuntimePromptInjectionDetector:
    """确定性的 Prompt Injection 检测器。"""

    def detect(
        self,
        text: str,
    ) -> InjectionDetectionResult:
        """检测一段不可信文本中的 Prompt Injection。"""

        if not isinstance(
            text,
            str,
        ):
            raise RuntimePromptInjectionError(
                "Prompt Injection Detector "
                "只接受字符串输入"
            )

        normalized_text = (
            self._normalize_text(
                text
            )
        )

        # 空文本本身不构成 Prompt Injection。
        if not normalized_text:
            return (
                self._clean_result()
            )

        matched_rules: list[
            InjectionRule
        ] = []

        for rule in _INJECTION_RULES:
            if self._rule_matches(
                rule=rule,
                text=normalized_text,
            ):
                matched_rules.append(
                    rule
                )

        if not matched_rules:
            return (
                self._clean_result()
            )

        severity = max(
            (
                rule.severity
                for rule
                in matched_rules
            ),
            key=lambda value: (
                _SEVERITY_RANK[
                    value
                ]
            ),
        )

        matched_rule_ids = tuple(
            rule.rule_id
            for rule
            in matched_rules
        )

        descriptions = tuple(
            rule.description
            for rule
            in matched_rules
        )

        reason = (
            "；".join(
                descriptions
            )
        )

        return InjectionDetectionResult(
            detected=True,
            severity=severity,
            matched_rule_ids=(
                matched_rule_ids
            ),
            reason=reason,
        )

    # ========================================================
    # Text Normalization
    #
    # NFKC：
    #   将全角字符等 Unicode 兼容形式标准化。
    #
    # lower：
    #   英文检测大小写无关。
    #
    # whitespace collapse：
    #   多个空格 / 换行统一为一个空格。
    # ========================================================

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        normalized = (
            unicodedata.normalize(
                "NFKC",
                text,
            )
        )

        normalized = (
            normalized.lower()
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        return normalized.strip()

    # ========================================================
    # 一条 Rule 只需要任意一个 Pattern 命中。
    #
    # 即使同一 Rule 中多个 Pattern 命中，
    # matched_rule_ids 也只记录一次。
    # ========================================================

    @staticmethod
    def _rule_matches(
        *,
        rule: InjectionRule,
        text: str,
    ) -> bool:
        return any(
            re.search(
                pattern,
                text,
            )
            is not None
            for pattern
            in rule.patterns
        )

    @staticmethod
    def _clean_result(
    ) -> InjectionDetectionResult:
        return InjectionDetectionResult(
            detected=False,
            severity="none",
            matched_rule_ids=(),
            reason=(
                "未检测到 Prompt Injection"
            ),
        )