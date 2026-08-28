"""由受信反向代理注入的身份声明校验。"""

from __future__ import annotations

from dataclasses import dataclass
from hmac import compare_digest

_MAX_ROLE_COUNT = 20
_MAX_ROLE_LENGTH = 64
_MAX_SUBJECT_LENGTH = 128
_ADMIN_ROLE = "logic_qa_admin"


class IdentityProviderNotConfiguredError(ValueError):
    """部署环境未配置可验证的身份提供方时抛出。"""


class IdentityAuthenticationError(ValueError):
    """请求未携带有效的受信身份声明时抛出。"""


@dataclass(frozen=True, slots=True)
class AuthenticatedIdentity:
    """经身份提供方验证后的最小主体与角色集合。"""

    subject: str
    roles: frozenset[str]

    @property
    def is_admin(self) -> bool:
        """是否具有逻辑题库管理角色。"""
        return _ADMIN_ROLE in self.roles


class TrustedProxyIdentityProvider:
    """校验由前置可信代理注入的用户和角色声明。"""

    def __init__(self, expected_proxy_token: str | None) -> None:
        self._expected_proxy_token = expected_proxy_token

    def authenticate(
        self,
        *,
        proxy_token: str | None,
        subject: str | None,
        roles: str | None,
    ) -> AuthenticatedIdentity:
        """验证代理凭据后规范化其传递的身份声明。"""
        expected_token = self._expected_proxy_token
        if not expected_token:
            raise IdentityProviderNotConfiguredError("身份提供方未配置")
        if proxy_token is None or not compare_digest(proxy_token, expected_token):
            raise IdentityAuthenticationError("身份认证失败")
        return AuthenticatedIdentity(
            subject=_normalize_subject(subject),
            roles=_normalize_roles(roles),
        )


def _normalize_subject(value: str | None) -> str:
    if value is None:
        raise IdentityAuthenticationError("身份提供方未提供用户标识")
    normalized = value.strip()
    if not normalized:
        raise IdentityAuthenticationError("身份提供方未提供用户标识")
    if len(normalized) > _MAX_SUBJECT_LENGTH:
        raise IdentityAuthenticationError("身份提供方用户标识超出长度限制")
    return normalized


def _normalize_roles(value: str | None) -> frozenset[str]:
    if value is None or not value.strip():
        return frozenset()
    normalized_roles = (role.strip() for role in value.split(","))
    roles = tuple(dict.fromkeys(role for role in normalized_roles if role))
    if len(roles) > _MAX_ROLE_COUNT:
        raise IdentityAuthenticationError("身份提供方角色数量超出限制")
    if any(len(role) > _MAX_ROLE_LENGTH for role in roles):
        raise IdentityAuthenticationError("身份提供方角色标识超出长度限制")
    return frozenset(roles)
