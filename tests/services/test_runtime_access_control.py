from __future__ import annotations

import pytest

from app.services.runtime_access_control import (
    RuntimeAccessControlError,
    RuntimeAccessController,
    RuntimePermissionDeniedError,
)


# ============================================================
# Week7 - Step4.1
#
# RBAC Core Tests
# ============================================================


def _build_controller(
) -> RuntimeAccessController:
    return RuntimeAccessController()


# ============================================================
# Role → Permission Mapping
# ============================================================


def test_viewer_permissions(
) -> None:
    controller = (
        _build_controller()
    )

    permissions = (
        controller
        .permissions_for_role(
            "viewer"
        )
    )

    assert permissions == frozenset(
        {
            "read_financial_data",
            "read_documents",
        }
    )


def test_reviewer_permissions(
) -> None:
    controller = (
        _build_controller()
    )

    permissions = (
        controller
        .permissions_for_role(
            "reviewer"
        )
    )

    assert permissions == frozenset(
        {
            "read_financial_data",
            "read_documents",
            "execute_calculation",
        }
    )


def test_admin_permissions(
) -> None:
    controller = (
        _build_controller()
    )

    permissions = (
        controller
        .permissions_for_role(
            "admin"
        )
    )

    assert permissions == frozenset(
        {
            "read_financial_data",
            "read_documents",
            "execute_calculation",
        }
    )


# ============================================================
# Viewer
# ============================================================


def test_viewer_can_read_financial_data(
) -> None:
    controller = (
        _build_controller()
    )

    decision = controller.check(
        role="viewer",
        permission=(
            "read_financial_data"
        ),
    )

    assert decision.allowed is True

    assert (
        decision.role
        == "viewer"
    )

    assert (
        decision.permission
        == "read_financial_data"
    )

    assert (
        decision.reason
        == (
            "角色 viewer "
            "允许权限 "
            "read_financial_data"
        )
    )


def test_viewer_can_read_documents(
) -> None:
    controller = (
        _build_controller()
    )

    decision = controller.check(
        role="viewer",
        permission=(
            "read_documents"
        ),
    )

    assert decision.allowed is True


def test_viewer_cannot_execute_calculation(
) -> None:
    controller = (
        _build_controller()
    )

    decision = controller.check(
        role="viewer",
        permission=(
            "execute_calculation"
        ),
    )

    assert decision.allowed is False

    assert (
        decision.reason
        == (
            "角色 viewer "
            "不允许权限 "
            "execute_calculation"
        )
    )


# ============================================================
# Reviewer / Admin
# ============================================================


def test_reviewer_can_execute_calculation(
) -> None:
    controller = (
        _build_controller()
    )

    decision = controller.require(
        role="reviewer",
        permission=(
            "execute_calculation"
        ),
    )

    assert decision.allowed is True


def test_admin_can_execute_calculation(
) -> None:
    controller = (
        _build_controller()
    )

    decision = controller.require(
        role="admin",
        permission=(
            "execute_calculation"
        ),
    )

    assert decision.allowed is True


# ============================================================
# Hard Permission Gate
#
# check():
#     返回 allowed=False
#
# require():
#     直接阻止执行
# ============================================================


def test_require_blocks_missing_permission(
) -> None:
    controller = (
        _build_controller()
    )

    with pytest.raises(
        RuntimePermissionDeniedError,
        match=(
            "viewer .*"
            "execute_calculation"
        ),
    ) as exc_info:
        controller.require(
            role="viewer",
            permission=(
                "execute_calculation"
            ),
        )

    decision = (
        exc_info.value.decision
    )

    assert decision.allowed is False

    assert (
        decision.role
        == "viewer"
    )

    assert (
        decision.permission
        == "execute_calculation"
    )


# ============================================================
# Defensive Runtime Validation
#
# Literal 类型主要是静态类型约束；
# Service 仍应拒绝非法运行时字符串。
# ============================================================


def test_unknown_role_is_rejected(
) -> None:
    controller = (
        _build_controller()
    )

    with pytest.raises(
        RuntimeAccessControlError,
        match="未知 UserRole",
    ):
        controller.permissions_for_role(
            "superuser"  # type: ignore[arg-type]
        )


def test_unknown_permission_is_rejected(
) -> None:
    controller = (
        _build_controller()
    )

    with pytest.raises(
        RuntimeAccessControlError,
        match=(
            "未知 ToolPermission"
        ),
    ):
        controller.check(
            role="admin",
            permission=(
                "delete_everything"
            ),  # type: ignore[arg-type]
        )