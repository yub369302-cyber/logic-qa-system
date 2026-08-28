"""选择题验证 API 的集成测试。"""

from fastapi.testclient import TestClient

from logic_qa.api import app

client = TestClient(app)


def test_choice_api_returns_counterexample_for_non_necessary_option() -> None:
    """“一定为真”失败时接口应返回违反选项的反例模型。"""
    response = client.post(
        "/v1/questions/verify-choice",
        json={
            "facts": [],
            "rules": [{"premise": "A", "conclusion": "B"}],
            "question_type": "must_be_true",
            "option": "B",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "not_verified"
    assert payload["verification_level"] == "fully_verified_by_enumeration"
    assert payload["witness_type"] == "counterexample"
    assert "¬B" in payload["witness_model"]
    assert "不一定为真" in payload["conclusion"]


def test_choice_api_returns_example_for_possible_option() -> None:
    """“可能为真”成功时接口应返回支持选项的正例模型。"""
    response = client.post(
        "/v1/questions/verify-choice",
        json={
            "facts": [],
            "rules": [{"premise": "A", "conclusion": "B"}],
            "question_type": "may_be_true",
            "option": "B",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "verified"
    assert payload["witness_type"] == "example"
    assert "B" in payload["witness_model"]
    assert "可能为真" in payload["conclusion"]


def test_choice_api_blocks_unsatisfiable_conditions() -> None:
    """矛盾题干不得被误判为任何选项类型。"""
    response = client.post(
        "/v1/questions/verify-choice",
        json={
            "facts": ["A", "!A"],
            "rules": [],
            "question_type": "must_be_true",
            "option": "B",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "blocked_unsatisfiable"
    assert payload["verification_level"] == "inconsistent_conditions"
    assert payload["witness_model"] is None
    assert "不存在合法模型" in payload["conclusion"]


def test_choice_api_rejects_invalid_option() -> None:
    """选项命题不合法时接口需要返回可理解的校验错误。"""
    response = client.post(
        "/v1/questions/verify-choice",
        json={
            "facts": [],
            "rules": [],
            "question_type": "may_be_true",
            "option": "!",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "否定符号后必须包含命题名称"
