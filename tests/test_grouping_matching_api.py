"""分组与匹配题求解 API 的集成测试。"""

from fastapi.testclient import TestClient

from logic_qa.api import app

client = TestClient(app)


def test_grouping_api_returns_complete_assignment_samples() -> None:
    """分组接口应返回完整验证状态、精确解数和分配样例。"""
    response = client.post(
        "/v1/questions/solve-grouping",
        json={
            "items": ["A", "B", "C"],
            "groups": ["G1", "G2"],
            "max_group_size": 2,
            "constraints": [
                {
                    "constraint_type": "same_group",
                    "item": "A",
                    "other_item": "B",
                },
                {
                    "constraint_type": "different_group",
                    "item": "A",
                    "other_item": "C",
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "complete"
    assert payload["verification_level"] == "fully_verified_by_enumeration"
    assert payload["solution_count"] == 2
    assert payload["sample_solutions"][0]["A"] == payload["sample_solutions"][0]["B"]
    assert payload["sample_solutions"][0]["A"] != payload["sample_solutions"][0]["C"]


def test_grouping_api_reports_search_limit() -> None:
    """超过分组搜索边界时接口须明确标记未验证状态。"""
    response = client.post(
        "/v1/questions/solve-grouping",
        json={
            "items": [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
                "K",
            ],
            "groups": ["G1", "G2", "G3"],
            "max_group_size": 11,
            "constraints": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "search_limit_exceeded"
    assert payload["verification_level"] == "not_verified_search_limit"
    assert payload["solution_count"] == 0


def test_matching_api_returns_complete_matching_sample() -> None:
    """匹配接口应返回固定及禁止配对约束下的完整结果。"""
    response = client.post(
        "/v1/questions/solve-matching",
        json={
            "items": ["A", "B", "C"],
            "targets": ["X", "Y", "Z"],
            "constraints": [
                {"constraint_type": "fixed_match", "item": "A", "target": "X"},
                {
                    "constraint_type": "forbidden_match",
                    "item": "B",
                    "target": "Y",
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "complete"
    assert payload["verification_level"] == "fully_verified_by_enumeration"
    assert payload["solution_count"] == 1
    assert payload["sample_solutions"] == [{"A": "X", "B": "Z", "C": "Y"}]


def test_matching_api_rejects_mismatched_input_size() -> None:
    """对象与目标数量不同时接口应在枚举前拒绝输入。"""
    response = client.post(
        "/v1/questions/solve-matching",
        json={
            "items": ["A"],
            "targets": ["X", "Y"],
            "constraints": [],
        },
    )

    assert response.status_code == 422
    assert "匹配对象与目标数量必须相等" in response.json()["detail"]
