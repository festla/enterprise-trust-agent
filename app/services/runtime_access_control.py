from __future__ import annotations

from dataclasses import (
    dataclass,
)

from app.schemas.tool_registry import (
    ToolPermission,
)
from app.schemas.trust import (
    UserRole,
)


# ============================================================
# Week7 - Step 4.1
#
# RBAC Core
#
# 这一层只负责：
#
# UserRole
#     ↓
# ToolPermission
#     ↓
# Allow / Deny
#
# 暂时不负责：
#
# - AgentState
# - Runtime Node
# - LangGraph
# - HITL
# - Risk Policy
# - Prompt Injection
#
# 这些会在后续 Step 中接入。
# ============================================================


class RuntimeAccessControlError(
    ValueError
):
    """Runtime RBAC 基础异常。"""


@dataclass(
    frozen=True,
    slots=True,
)
class AccessDecision:
    """一次确定性的 RBAC 权限判断结果。"""

    role: UserRole

    permission: ToolPermission

    allowed: bool

    reason: str


class RuntimePermissionDeniedError(
    RuntimeAccessControlError
):
    """当前角色没有请求的工具权限。"""

    def __init__(
        self,
        decision: AccessDecision,
    ) -> None:
        super().__init__(
            decision.reason
        )

        self.decision = decision


# ============================================================
# 系统当前支持的全部 Tool Permission。
#
# 与 app.schemas.tool_registry.ToolPermission 保持一致。
# ============================================================

_ALL_TOOL_PERMISSIONS: frozenset[
    ToolPermission
] = frozenset(
    {
        "read_financial_data",
        "read_documents",
        "execute_calculation",
    }
)


# ============================================================
# Role → Permission Matrix
#
# viewer:
#   只允许读取已有信息。
#
# reviewer:
#   除读取之外，可以执行确定性财务计算。
#
# admin:
#   当前版本拥有全部已有 Tool Permission。
#
# 后续如果增加：
#
#   export_report
#   update_registry
#   approve_answer
#   delete_resource
#
# admin / reviewer 的差异会进一步扩大。
# ============================================================

_ROLE_PERMISSIONS: dict[
    UserRole,
    frozenset[
        ToolPermission
    ],
] = {
    "viewer": frozenset(
        {
            "read_financial_data",
            "read_documents",
        }
    ),

    "reviewer": frozenset(
        {
            "read_financial_data",
            "read_documents",
            "execute_calculation",
        }
    ),

    "admin": frozenset(
        {
            "read_financial_data",
            "read_documents",
            "execute_calculation",
        }
    ),
}


@dataclass(
    frozen=True,
    slots=True,
)
class RuntimeAccessController:
    """Runtime 的确定性 RBAC 权限控制器。"""

    # ========================================================
    # 查询某个 Role 拥有哪些权限。
    # ========================================================

    def permissions_for_role(
        self,
        role: UserRole,
    ) -> frozenset[
        ToolPermission
    ]:
        permissions = (
            _ROLE_PERMISSIONS.get(
                role
            )
        )

        if permissions is None:
            raise RuntimeAccessControlError(
                "未知 UserRole："
                f"{role}"
            )

        return permissions

    # ========================================================
    # 非抛异常式检查。
    #
    # 适合：
    #
    # AccessController
    #      ↓
    # AccessDecision
    #      ↓
    # Audit / Policy / Runtime
    #
    # 无论允许还是拒绝，都返回结构化 Decision。
    # ========================================================

    def check(
        self,
        *,
        role: UserRole,
        permission: ToolPermission,
    ) -> AccessDecision:
        self._validate_permission(
            permission
        )

        permissions = (
            self.permissions_for_role(
                role
            )
        )

        allowed = (
            permission
            in permissions
        )

        if allowed:
            reason = (
                f"角色 {role} "
                f"允许权限 {permission}"
            )

        else:
            reason = (
                f"角色 {role} "
                f"不允许权限 {permission}"
            )

        return AccessDecision(
            role=role,
            permission=permission,
            allowed=allowed,
            reason=reason,
        )

    # ========================================================
    # 强制权限检查。
    #
    # ALLOW:
    #
    #   返回 AccessDecision
    #
    # DENY:
    #
    #   RuntimePermissionDeniedError
    #
    # 后续 Step4.3 Runtime 接入时，
    # Runtime 可以把这个异常转换成：
    #
    # controlled refusal
    # permission_denied
    #
    # 而不是 internal_error。
    # ========================================================

    def require(
        self,
        *,
        role: UserRole,
        permission: ToolPermission,
    ) -> AccessDecision:
        decision = self.check(
            role=role,
            permission=permission,
        )

        if not decision.allowed:
            raise (
                RuntimePermissionDeniedError(
                    decision
                )
            )

        return decision

    # ========================================================
    # Permission Runtime Validation
    #
    # ToolPermission 是 typing.Literal，
    # Python 运行时本身不会自动阻止：
    #
    # permission="delete_everything"
    #
    # 所以 Service 层仍然需要显式防御。
    # ========================================================

    @staticmethod
    def _validate_permission(
        permission: ToolPermission,
    ) -> None:
        if (
            permission
            not in _ALL_TOOL_PERMISSIONS
        ):
            raise RuntimeAccessControlError(
                "未知 ToolPermission："
                f"{permission}"
            )