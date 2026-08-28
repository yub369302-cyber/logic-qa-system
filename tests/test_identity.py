"""受信代理身份声明的单元测试。"""

import pytest

from logic_qa.identity import (
    IdentityAuthenticationError,
    IdentityProviderNotConfiguredError,
    TrustedProxyIdentityProvider,
)


def test_trusted_proxy_identity_normalizes_subject_and_roles() -> None:
    """身份提供方应规范化主体和重复角色，并识别管理员角色。"""
    provider = TrustedProxyIdentityProvider("proxy-secret")

    identity = provider.authenticate(
        proxy_token="proxy-secret",
        subject=" learner-a ",
        roles=" learner, logic_qa_admin, learner ",
    )

    assert identity.subject == "learner-a"
    assert identity.roles == frozenset({"learner", "logic_qa_admin"})
    assert identity.is_admin is True


@pytest.mark.parametrize(
    ("proxy_token", "subject", "expected_message"),
    [
        (None, "learner-a", "身份认证失败"),
        ("wrong-secret", "learner-a", "身份认证失败"),
        ("proxy-secret", None, "用户标识"),
        ("proxy-secret", " ", "用户标识"),
    ],
)
def test_trusted_proxy_identity_rejects_invalid_claims(
    proxy_token: str | None,
    subject: str | None,
    expected_message: str,
) -> None:
    """无效代理凭据或主体声明不得被视作已认证用户。"""
    provider = TrustedProxyIdentityProvider("proxy-secret")

    with pytest.raises(IdentityAuthenticationError, match=expected_message):
        provider.authenticate(
            proxy_token=proxy_token,
            subject=subject,
            roles="learner",
        )


def test_trusted_proxy_identity_fails_closed_without_deployment_secret() -> None:
    """未配置可信代理密钥时，身份提供方必须保持关闭。"""
    provider = TrustedProxyIdentityProvider(None)

    with pytest.raises(IdentityProviderNotConfiguredError, match="未配置"):
        provider.authenticate(
            proxy_token="anything",
            subject="learner-a",
            roles="learner",
        )
