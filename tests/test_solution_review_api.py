"""用户结构化解法批改 API 的集成测试。"""

from fastapi.testclient import TestClient

from logic_qa.api import app

client = TestClient(app)


def test_solution_review_api_accepts_complete_proof() -> None:
    """正确步骤应返回完成状态与已验证等级。"""
    response = client.post(
        "/v1/questions/review-solution",
        json={
            "facts": ["A"],
            "rules": [
                {"premise": "A", "conclusion": "B"},
                {"premise": "B", "conclusion": "C"},
            ],
            "target": "C",
            "steps": [
                {
                    "rule_id": 1,
                    "direction": "forward",
                    "premise": "A",
                    "conclusion": "B",
                },
                {
                    "rule_id": 2,
                    "direction": "forward",
                    "premise": "B",
                    "conclusion": "C",
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "correct"
    assert payload["checked_step_count"] == 2
    assert payload["diagnostic"] is None
    assert payload["verification_level"] == "fully_verified_by_logic_engine"


def test_solution_review_api_returns_first_converse_error() -> None:
    """误用逆命题时接口应返回首错位置与知识点标签。"""
    response = client.post(
        "/v1/questions/review-solution",
        json={
            "facts": ["A", "B"],
            "rules": [
                {"premise": "A", "conclusion": "B"},
                {"premise": "B", "conclusion": "C"},
            ],
            "target": "C",
            "steps": [
                {
                    "rule_id": 1,
                    "direction": "forward",
                    "premise": "B",
                    "conclusion": "A",
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "first_error"
    assert payload["checked_step_count"] == 0
    assert payload["diagnostic"]["code"] == "invalid_converse"
    assert payload["diagnostic"]["step_index"] == 1
    assert "逆命题与逆否命题" in payload["diagnostic"]["knowledge_tags"]


def test_solution_review_api_blocks_inconsistent_conditions() -> None:
    """题干矛盾时 API 必须阻断过程评判。"""
    response = client.post(
        "/v1/questions/review-solution",
        json={
            "facts": ["A", "!A"],
            "rules": [],
            "target": "C",
            "steps": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "blocked_inconsistent"
    assert payload["baseline_status"] == "inconsistent"
    assert payload["verification_level"] == "inconsistent_conditions"
    assert payload["diagnostic"]["code"] == "inconsistent_conditions"
