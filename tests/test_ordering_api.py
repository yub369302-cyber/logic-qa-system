"""排序题求解 API 的集成测试。"""

from fastapi.testclient import TestClient

from logic_qa.api import app

client = TestClient(app)


def test_ordering_api_returns_complete_solution_space_summary() -> None:
    """接口应返回完整枚举状态、精确解数及展示样例。"""
    response = client.post(
        "/v1/questions/solve-ordering",
        json={
            "items": ["A", "B", "C"],
            "constraints": [
                {"constraint_type": "before", "item": "A", "other_item": "B"},
                {"constraint_type": "fixed_position", "item": "C", "position": 3},
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "complete"
    assert payload["verification_level"] == "fully_verified_by_enumeration"
    assert payload["solution_count"] == 1
    assert payload["sample_solutions"] == [["A", "B", "C"]]


def test_ordering_api_reports_unsatisfiable_constraints() -> None:
    """冲突约束应返回无解与条件矛盾标识。"""
    response = client.post(
        "/v1/questions/solve-ordering",
        json={
            "items": ["A", "B"],
            "constraints": [
                {"constraint_type": "before", "item": "A", "other_item": "B"},
                {"constraint_type": "before", "item": "B", "other_item": "A"},
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unsatisfiable"
    assert payload["verification_level"] == "inconsistent_conditions"
    assert payload["solution_count"] == 0


def test_ordering_api_rejects_unknown_constraint_object() -> None:
    """未知对象不应进入枚举，接口应返回输入校验错误。"""
    response = client.post(
        "/v1/questions/solve-ordering",
        json={
            "items": ["A", "B"],
            "constraints": [
                {"constraint_type": "adjacent", "item": "A", "other_item": "C"}
            ],
        },
    )

    assert response.status_code == 422
    assert "约束对象不存在" in response.json()["detail"]
