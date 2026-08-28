"""FastAPI 路由共享的可信代理身份依赖。"""

from __future__ import annotations

import os
from typing import Annotated

from fastapi import Depends, Header, HTTPException

from logic_qa.identity import (
    AuthenticatedIdentity,
    IdentityAuthenticationError,
    IdentityProviderNotConfiguredError,
    TrustedProxyIdentityProvider,
)


def require_authenticated_identity(
    x_logic_qa_proxy_token: str | None = Header(default=None),
    x_logic_qa_subject: str | None = Header(default=None),
    x_logic_qa_roles: str | None = Header(default=None),
) -> AuthenticatedIdentity:
    """仅接受受信代理验证后注入的用户与角色声明。"""
    provider = TrustedProxyIdentityProvider(
        os.environ.get("LOGIC_QA_TRUSTED_PROXY_TOKEN")
    )
    try:
        return provider.authenticate(
            proxy_token=x_logic_qa_proxy_token,
            subject=x_logic_qa_subject,
            roles=x_logic_qa_roles,
        )
    except IdentityProviderNotConfiguredError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except IdentityAuthenticationError as error:
        raise HTTPException(status_code=401, detail=str(error)) from error


def require_admin_identity(
    identity: Annotated[
        AuthenticatedIdentity,
        Depends(require_authenticated_identity),
    ],
) -> AuthenticatedIdentity:
    """要求当前经认证主体具备逻辑题库管理角色。"""
    if not identity.is_admin:
        raise HTTPException(status_code=403, detail="管理员访问未授权")
    return identity


CurrentIdentity = Annotated[
    AuthenticatedIdentity,
    Depends(require_authenticated_identity),
]
AdminIdentity = Annotated[AuthenticatedIdentity, Depends(require_admin_identity)]
