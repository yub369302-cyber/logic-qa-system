"""逻辑求解 API 的集成测试。"""

from fastapi.testclient import TestClient

from logic_qa.api import app

client = TestClient(app)


def test_learner_home_serves_same_origin_interface() -> None:
    """根路径应提供仅调用公开中文求解接口的学习者页面。"""
    response = client.get("/")

    assert response.status_code == 200
    assert "可信逻辑答疑" in response.text
    assert 'action="/v1/questions/solve-chinese"' not in response.text
    assert "/assets/app.css" in response.text
    assert "/assets/app.js" in response.text
    assert "/v1/admin/" not in response.text


def test_learner_static_assets_are_available() -> None:
    """同源页面使用的样式与交互资源应可被浏览器读取。"""
    stylesheet = client.get("/assets/app.css")
    script = client.get("/assets/app.js")

    assert stylesheet.status_code == 200
    assert "--brand" in stylesheet.text
    assert script.status_code == 200
    assert 'fetch("/v1/questions/solve-chinese"' in script.text
    assert "/v1/admin/" not in script.text


def test_practice_and_progress_pages_expose_only_learner_paths() -> None:
    """练习与学习概览页面应同源可用，且不包含管理端入口。"""
    practice_page = client.get("/practice")
    practice_script = client.get("/assets/practice.js")
    progress_page = client.get("/progress")
    progress_script = client.get("/assets/progress.js")

    assert practice_page.status_code == 200
    assert "练习已审核发布的逻辑题" in practice_page.text
    assert "/v1/admin/" not in practice_page.text
    assert practice_script.status_code == 200
    assert "/v1/learning/recommendations" in practice_script.text
    assert "/v1/learning/questions/" in practice_script.text
    assert "/v1/admin/" not in practice_script.text
    assert progress_page.status_code == 200
    assert "把已完成的练习变成下一步行动" in progress_page.text
    assert "/v1/admin/" not in progress_page.text
    assert progress_script.status_code == 200
    assert 'fetch("/v1/learning/profile")' in progress_script.text
    assert "/v1/admin/" not in progress_script.text


def test_health_check() -> None:
    """服务应暴露健康检查接口。"""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_solve_returns_a_verifiable_proof_trace() -> None:
    """接口应返回用户可阅读的结论和每步依据。"""
    response = client.post(
        "/v1/questions/solve",
        json={
            "facts": ["A"],
            "rules": [
                {
                    "premise": "A",
                    "conclusion": "B",
                    "source_text": "如果 A，那么 B",
                },
                {
                    "premise": "B",
                    "conclusion": "C",
                    "source_text": "如果 B，那么 C",
                },
            ],
            "query": "C",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "proved"
    assert payload["conclusion"] == "可以由题干推出 C。"
    assert payload["verification_level"] == "fully_verified"
    assert [step["derived"] for step in payload["proof_steps"]] == ["A", "B", "C"]


def test_solve_rejects_empty_literals() -> None:
    """接口需要对不合法的结构化逻辑输入返回明确错误。"""
    response = client.post(
        "/v1/questions/solve",
        json={"facts": ["!"], "rules": [], "query": "A"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "否定符号后必须包含命题名称"


def test_solve_chinese_parses_conditions_before_verifying() -> None:
    """中文接口应回显解析结果，并复用确定性内核完成验证。"""
    response = client.post(
        "/v1/questions/solve-chinese",
        json={
            "conditions": "甲参加。若甲参加，则乙通过。只有丙入选，乙才通过。",
            "query": "丙入选",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "proved"
    assert payload["query"] == "丙入选"
    assert payload["parsed_facts"] == ["甲参加"]
    assert payload["parsed_rules"] == [
        {
            "premise": "甲参加",
            "conclusion": "乙通过",
            "source_text": "若甲参加，则乙通过",
        },
        {
            "premise": "乙通过",
            "conclusion": "丙入选",
            "source_text": "只有丙入选，乙才通过",
        },
    ]


def test_solve_chinese_returns_confirmation_request_for_complex_expression() -> None:
    """复杂中文语义不得自动求解，而应返回需人工转写的原句。"""
    response = client.post(
        "/v1/questions/solve-chinese",
        json={
            "conditions": "甲不参加。除非甲参加，否则乙通过。",
            "query": "乙通过",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "confirmation_required"
    assert payload["verification_level"] == "not_verified_pending_human_confirmation"
    assert payload["confirmation_required"] is True
    assert payload["confirmation_requests"] == [
        {
            "source_sentence": "除非甲参加，否则乙通过",
            "codes": ["unless"],
            "message": "“除非/否则”可能存在条件方向或范围歧义。",
        }
    ]


def test_solve_chinese_verifies_only_after_structured_confirmation() -> None:
    """复杂条件经人工转写为规则后，才可进入确定性推理内核。"""
    response = client.post(
        "/v1/questions/solve-chinese",
        json={
            "conditions": "甲不参加。除非甲参加，否则乙通过。",
            "query": "乙通过",
            "confirmations": [
                {
                    "source_sentence": "除非甲参加，否则乙通过",
                    "rules": [{"premise": "!甲参加", "conclusion": "乙通过"}],
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "proved"
    assert payload["confirmation_required"] is False
    assert payload["confirmation_requests"][0]["codes"] == ["unless"]
    assert payload["parsed_rules"] == [
        {
            "premise": "¬甲参加",
            "conclusion": "乙通过",
            "source_text": "除非甲参加，否则乙通过（人工确认）",
        }
    ]


def test_solve_reports_global_conflict_with_honest_level() -> None:
    """与查询无关的矛盾必须上报，且验证等级不能伪装成完全验证。"""
    response = client.post(
        "/v1/questions/solve",
        json={
            "facts": ["A", "!A"],
            "rules": [],
            "query": "C",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "inconsistent"
    assert payload["verification_level"] == "inconsistent_conditions"
    assert payload["conflict"] == ["A", "¬A"]
    assert "A 与 ¬A 冲突" in payload["conclusion"]
