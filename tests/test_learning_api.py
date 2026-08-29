"""基于受信代理身份的学习档案 API 集成测试。"""

from pathlib import Path

from fastapi.testclient import TestClient

from logic_qa import api
from logic_qa.learning_profile import LearningProfileStore

_PROXY_TOKEN = "test-proxy-token"


def _identity_headers(
    subject: str,
    roles: str = "learner",
) -> dict[str, str]:
    """构造仅测试环境中的受信代理身份声明。"""
    return {
        "X-Logic-QA-Proxy-Token": _PROXY_TOKEN,
        "X-Logic-QA-Subject": subject,
        "X-Logic-QA-Roles": roles,
    }


def _client_with_store(tmp_path: Path, monkeypatch) -> TestClient:
    """使用独立学习数据库和测试代理密钥。"""
    store = LearningProfileStore(tmp_path / "learning-api.sqlite3")
    monkeypatch.setenv("LOGIC_QA_TRUSTED_PROXY_TOKEN", _PROXY_TOKEN)
    monkeypatch.setattr(api, "learning_store", store)
    return TestClient(api.app)


def test_learning_api_binds_records_profile_and_deletion_to_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """学习记录、画像和删除操作均使用认证主体，而非客户端用户字段。"""
    client = _client_with_store(tmp_path, monkeypatch)
    headers = _identity_headers("user-a")
    create_response = client.post(
        "/v1/learning/records",
        headers=headers,
        json={
            "question_id": "q-1",
            "question_type": "propositional",
            "is_correct": False,
            "error_tags": ["invalid_converse"],
            "knowledge_tags": ["逆命题与逆否命题"],
            "duration_seconds": 40,
        },
    )

    assert create_response.status_code == 200
    record = create_response.json()
    assert set(record) == {
        "record_id",
        "question_id",
        "question_type",
        "is_correct",
        "duration_seconds",
        "created_at",
    }

    profile_response = client.get("/v1/learning/profile", headers=headers)

    assert profile_response.status_code == 200
    profile = profile_response.json()
    assert set(profile) == {
        "total_attempts",
        "correct_attempts",
        "accuracy",
        "focus_areas",
    }
    assert profile["total_attempts"] == 1
    assert profile["accuracy"] == 0.0
    assert profile["focus_areas"][0]["title"] == "条件方向核验"
    for forbidden_field in (
        "user_id",
        "error_tags",
        "knowledge_tags",
        "error_counts",
        "knowledge_mastery",
        "recommendations",
        "label",
    ):
        assert forbidden_field not in profile
        assert forbidden_field not in profile["focus_areas"][0]

    delete_response = client.delete(
        f"/v1/learning/records/{record['record_id']}",
        headers=headers,
    )

    assert delete_response.status_code == 200
    assert delete_response.json() == {"deleted": True}
    empty_profile_response = client.get("/v1/learning/profile", headers=headers)
    assert empty_profile_response.json()["total_attempts"] == 0


def test_learning_api_rejects_spoofed_user_id_and_cross_user_delete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """客户端不可指定用户，另一认证主体也不能删除其记录。"""
    client = _client_with_store(tmp_path, monkeypatch)
    owner_headers = _identity_headers("user-a")
    create_response = client.post(
        "/v1/learning/records",
        headers=owner_headers,
        json={
            "user_id": "user-b",
            "question_id": "q-1",
            "question_type": "propositional",
            "is_correct": True,
        },
    )

    assert create_response.status_code == 422
    assert create_response.json()["detail"][0]["loc"] == ["body", "user_id"]

    valid_create_response = client.post(
        "/v1/learning/records",
        headers=owner_headers,
        json={
            "question_id": "q-1",
            "question_type": "propositional",
            "is_correct": True,
        },
    )
    record_id = valid_create_response.json()["record_id"]
    other_headers = _identity_headers("user-b")

    delete_response = client.delete(
        f"/v1/learning/records/{record_id}",
        headers=other_headers,
    )

    assert delete_response.status_code == 404
    assert client.get("/v1/learning/profile", headers=owner_headers).json()[
        "total_attempts"
    ] == 1


def test_learning_api_requires_configured_and_valid_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """学习接口在缺少身份提供方或代理凭据时不得读取或写入数据。"""
    client = _client_with_store(tmp_path, monkeypatch)
    monkeypatch.delenv("LOGIC_QA_TRUSTED_PROXY_TOKEN")

    unavailable = client.get("/v1/learning/profile")
    assert unavailable.status_code == 503
    assert unavailable.json()["detail"] == "身份提供方未配置"

    monkeypatch.setenv("LOGIC_QA_TRUSTED_PROXY_TOKEN", _PROXY_TOKEN)
    unauthenticated = client.get("/v1/learning/profile")
    assert unauthenticated.status_code == 401
    assert unauthenticated.json()["detail"] == "身份认证失败"


def test_learning_api_rejects_invalid_record_input(tmp_path: Path, monkeypatch) -> None:
    """负作答时长等不合法最小记录应被明确拒绝。"""
    client = _client_with_store(tmp_path, monkeypatch)

    response = client.post(
        "/v1/learning/records",
        headers=_identity_headers("user-a"),
        json={
            "question_id": "q-1",
            "question_type": "propositional",
            "is_correct": True,
            "duration_seconds": -1,
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "作答时长不能为负数"


def test_learning_profile_masks_unknown_internal_tags(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """未在展示映射表中的内部标签也不能透传到学习者概览。"""
    client = _client_with_store(tmp_path, monkeypatch)
    headers = _identity_headers("user-a")
    create_response = client.post(
        "/v1/learning/records",
        headers=headers,
        json={
            "question_id": "q-unknown",
            "question_type": "propositional",
            "is_correct": False,
            "error_tags": ["internal-only-tag"],
            "knowledge_tags": ["内部知识点"],
        },
    )

    profile_response = client.get("/v1/learning/profile", headers=headers)

    assert create_response.status_code == 200
    assert profile_response.status_code == 200
    profile = profile_response.json()
    assert profile["focus_areas"][0]["title"] == "推理步骤复盘"
    assert "internal-only-tag" not in str(profile)
    assert "内部知识点" not in str(profile)
