"""内容绑定审核题库发布与学习者最小响应的集成测试。"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from logic_qa import api
from logic_qa.learning_profile import LearningProfileStore
from logic_qa.quality_operations import QuestionReviewStore, RuntimeMetrics
from logic_qa.question_bank import QuestionBankStore
from logic_qa.question_bank_api import router as question_bank_router

_PROXY_TOKEN = "test-proxy-token"
_ADMIN_HEADERS = {
    "X-Logic-QA-Proxy-Token": _PROXY_TOKEN,
    "X-Logic-QA-Subject": "admin-a",
    "X-Logic-QA-Roles": "logic_qa_admin",
}
_LEARNER_HEADERS = {
    "X-Logic-QA-Proxy-Token": _PROXY_TOKEN,
    "X-Logic-QA-Subject": "user-a",
    "X-Logic-QA-Roles": "learner",
}


def _client_with_stores(tmp_path: Path, monkeypatch) -> TestClient:
    """为每个接口测试注入独立数据库，避免全局状态相互影响。"""
    monkeypatch.setenv("LOGIC_QA_TRUSTED_PROXY_TOKEN", _PROXY_TOKEN)
    review_store = QuestionReviewStore(tmp_path / "reviews.sqlite3")
    question_bank_store = QuestionBankStore(tmp_path / "questions.sqlite3")
    learning_store = LearningProfileStore(tmp_path / "learning.sqlite3")
    monkeypatch.setattr(api, "review_store", review_store)
    monkeypatch.setattr(api, "question_bank_store", question_bank_store)
    monkeypatch.setattr(api, "learning_store", learning_store)
    monkeypatch.setattr(api, "runtime_metrics", RuntimeMetrics())
    return TestClient(api.app)


def _publish_payload(question_id: str, content_version: str) -> dict[str, object]:
    """构造包含全部可摘要内容的管理员发布请求。"""
    return {
        "question_id": question_id,
        "content_version": content_version,
        "question_type": "propositional",
        "stem": f"{question_id} 的题干",
        "options": ["A", "B"],
        "error_tags": ["invalid_converse"],
        "knowledge_tags": ["逆命题与逆否命题"],
        "formalization_version": "logic-v1",
        "formalization": {
            "kind": "propositional",
            "facts": ["A"],
            "rules": [
                {
                    "premise": "A",
                    "conclusion": "B",
                    "source_text": "如果 A，那么 B",
                }
            ],
            "query": "B",
            "expected_status": "proved",
            "expected_answer": "B",
            "option_assertions": [
                {
                    "option": "A",
                    "claim_status": "disproved",
                    "claim_solution_count": None,
                },
                {
                    "option": "B",
                    "claim_status": "proved",
                    "claim_solution_count": None,
                },
            ],
        },
    }


def _prepare_candidate(
    client: TestClient,
    payload: dict[str, object],
) -> dict[str, object]:
    """按正式接口取得候选内容的规范化摘要。"""
    response = client.post(
        "/v1/admin/question-candidates",
        headers=_ADMIN_HEADERS,
        json=payload,
    )

    assert response.status_code == 200
    return response.json()


def _approve_candidate(client: TestClient, candidate: dict[str, object]) -> None:
    """将候选内容的唯一标识和摘要一并绑定到审核结论。"""
    response = client.post(
        "/v1/admin/question-reviews",
        headers=_ADMIN_HEADERS,
        json={
            "question_id": candidate["question_id"],
            "content_version": candidate["content_version"],
            "content_hash": candidate["content_hash"],
            "status": "approved",
            "verified_answer": "B",
            "formalization_version": candidate["formalization_version"],
        },
    )

    assert response.status_code == 200
    assert response.json()["content_hash"] == candidate["content_hash"]


def _prepare_and_approve(
    client: TestClient,
    payload: dict[str, object],
) -> dict[str, object]:
    """完成候选内容计算和精确审核绑定。"""
    candidate = _prepare_candidate(client, payload)
    _approve_candidate(client, candidate)
    return candidate


def test_question_bank_routes_are_registered_once() -> None:
    """题库模块拆分后，管理端公开路径必须保持完整且不重复。"""
    expected_paths = {route.path for route in question_bank_router.routes}
    registered_paths = set(api.app.openapi()["paths"])

    assert expected_paths == {
        "/v1/admin/question-candidates",
        "/v1/admin/question-candidates/{question_id}/{content_version}/{content_hash}",
        "/v1/admin/question-reviews",
        "/v1/admin/question-reviews/{question_id}/{content_version}/{content_hash}",
        "/v1/admin/questions",
        "/v1/admin/questions/{question_id}/{content_version}/correction-republication-links",
    }
    assert expected_paths <= registered_paths


def test_question_publication_requires_admin_and_exact_approval(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """发布接口必须受管理员角色和精确候选审核的双重保护。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    payload = _publish_payload("q-1", "content-v1")

    unauthorized = client.post(
        "/v1/admin/questions",
        headers=_LEARNER_HEADERS,
        json=payload,
    )
    assert unauthorized.status_code == 403

    _prepare_candidate(client, payload)
    unreviewed = client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=payload,
    )
    assert unreviewed.status_code == 422
    assert "当前内容尚无审核记录" in unreviewed.json()["detail"]

    candidate = _prepare_and_approve(client, payload)
    published = client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=payload,
    )

    assert published.status_code == 200
    published_payload = published.json()
    assert published_payload["question_id"] == "q-1"
    assert published_payload["content_version"] == "content-v1"
    assert published_payload["content_hash"] == candidate["content_hash"]
    assert published_payload["error_tags"] == ["invalid_converse"]
    assert published_payload["formalization"] == candidate["formalization"]


def test_question_candidate_requires_complete_formalization_asset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """候选题必须带完整的结构化形式化资产，不能仅声明版本号。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    payload = _publish_payload("q-1", "content-v1")
    incomplete_payload = {
        key: value
        for key, value in payload.items()
        if key != "formalization"
    }

    response = client.post(
        "/v1/admin/question-candidates",
        headers=_ADMIN_HEADERS,
        json=incomplete_payload,
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "formalization"]


def test_question_candidate_requires_complete_ordered_option_assertions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """选择题必须为全部展示选项按顺序提供可复算断言。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    payload = _publish_payload("q-1", "content-v1")
    missing_assertions = {
        **payload,
        "formalization": {
            key: value
            for key, value in payload["formalization"].items()
            if key != "option_assertions"
        },
    }
    out_of_order_assertions = {
        **payload,
        "formalization": {
            **payload["formalization"],
            "option_assertions": list(
                reversed(payload["formalization"]["option_assertions"])
            ),
        },
    }

    missing_response = client.post(
        "/v1/admin/question-candidates",
        headers=_ADMIN_HEADERS,
        json=missing_assertions,
    )
    out_of_order_response = client.post(
        "/v1/admin/question-candidates",
        headers=_ADMIN_HEADERS,
        json=out_of_order_assertions,
    )

    assert missing_response.status_code == 422
    assert out_of_order_response.status_code == 422
    assert "选项断言必须按题目选项顺序逐一绑定" in out_of_order_response.json()[
        "detail"
    ]


def test_publish_rejects_mismatched_reproducible_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """审核通过也不能绕过发布时对形式化预期结果的重新计算。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    payload = _publish_payload("q-1", "content-v1")
    invalid_payload = {
        **payload,
        "formalization": {
            **payload["formalization"],
            "expected_status": "disproved",
        },
    }
    _prepare_and_approve(client, invalid_payload)

    response = client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=invalid_payload,
    )

    assert response.status_code == 422
    assert "形式化验证结果与预期不一致" in response.json()["detail"]


def test_publish_rejects_non_unique_option_semantics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """审核通过不能绕过发布时的唯一选项语义命中门禁。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    payload = _publish_payload("q-non-unique", "content-v1")
    invalid_payload = {
        **payload,
        "formalization": {
            **payload["formalization"],
            "option_assertions": [
                {
                    "option": "A",
                    "claim_status": "proved",
                    "claim_solution_count": None,
                },
                {
                    "option": "B",
                    "claim_status": "proved",
                    "claim_solution_count": None,
                },
            ],
        },
    }
    _prepare_and_approve(client, invalid_payload)

    response = client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=invalid_payload,
    )

    assert response.status_code == 422
    assert "选项语义必须唯一命中" in response.json()["detail"]


@pytest.mark.parametrize(
    ("question_type", "formalization", "expected_count"),
    [
        (
            "ordering",
            {
                "kind": "ordering",
                "items": ["A", "B", "C"],
                "constraints": [
                    {"constraint_type": "before", "item": "A", "other_item": "B"}
                ],
                "expected_status": "complete",
                "expected_solution_count": 3,
                "expected_answer": "B",
                "option_assertions": [
                    {
                        "option": "A",
                        "claim_status": "complete",
                        "claim_solution_count": 2,
                    },
                    {
                        "option": "B",
                        "claim_status": "complete",
                        "claim_solution_count": 3,
                    },
                ],
            },
            3,
        ),
        (
            "grouping",
            {
                "kind": "grouping",
                "items": ["A", "B"],
                "groups": ["G1", "G2"],
                "max_group_size": 2,
                "constraints": [
                    {
                        "constraint_type": "same_group",
                        "item": "A",
                        "other_item": "B",
                    }
                ],
                "expected_status": "complete",
                "expected_solution_count": 2,
                "expected_answer": "B",
                "option_assertions": [
                    {
                        "option": "A",
                        "claim_status": "complete",
                        "claim_solution_count": 1,
                    },
                    {
                        "option": "B",
                        "claim_status": "complete",
                        "claim_solution_count": 2,
                    },
                ],
            },
            2,
        ),
        (
            "matching",
            {
                "kind": "matching",
                "items": ["A", "B"],
                "targets": ["X", "Y"],
                "constraints": [
                    {"constraint_type": "fixed_match", "item": "A", "target": "X"}
                ],
                "expected_status": "complete",
                "expected_solution_count": 1,
                "expected_answer": "B",
                "option_assertions": [
                    {
                        "option": "A",
                        "claim_status": "complete",
                        "claim_solution_count": 2,
                    },
                    {
                        "option": "B",
                        "claim_status": "complete",
                        "claim_solution_count": 1,
                    },
                ],
            },
            1,
        ),
    ],
)
def test_constraint_question_publication_uses_correct_solver(
    tmp_path: Path,
    monkeypatch,
    question_type: str,
    formalization: dict[str, object],
    expected_count: int,
) -> None:
    """组合题候选必须携带求解器资产并经对应内核复算后才能发布。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    payload = {
        **_publish_payload(f"q-{question_type}", "content-v1"),
        "question_type": question_type,
        "formalization": formalization,
    }

    candidate = _prepare_and_approve(client, payload)
    published = client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=payload,
    )

    assert published.status_code == 200
    published_formalization = published.json()["formalization"]
    assert published_formalization["kind"] == question_type
    assert published_formalization["expected_solution_count"] == expected_count
    verification = api.question_bank_store.get_formalization_verification(
        candidate["question_id"],
        candidate["content_version"],
        candidate["content_hash"],
    )
    assert verification is not None
    assert verification.actual_solution_count == expected_count


def test_constraint_question_publication_rejects_wrong_expected_count(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """审核通过也不能掩盖组合约束题中错误的解空间数量声明。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    payload = {
        **_publish_payload("q-ordering", "content-v1"),
        "question_type": "ordering",
        "formalization": {
            "kind": "ordering",
            "items": ["A", "B"],
            "constraints": [],
            "expected_status": "complete",
            "expected_solution_count": 1,
            "expected_answer": "B",
            "option_assertions": [
                {
                    "option": "A",
                    "claim_status": "complete",
                    "claim_solution_count": 2,
                },
                {
                    "option": "B",
                    "claim_status": "complete",
                    "claim_solution_count": 1,
                },
            ],
        },
    }
    _prepare_and_approve(client, payload)

    response = client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=payload,
    )

    assert response.status_code == 422
    assert "解空间数量" in response.json()["detail"]


def test_changed_content_cannot_reuse_same_version_approval(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """题干变化后即使题号和内容版本未变，也不能沿用旧审核发布。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    original_payload = _publish_payload("q-1", "content-v1")
    original_candidate = _prepare_and_approve(client, original_payload)
    changed_payload = {**original_payload, "stem": "q-1 已修改的题干"}

    changed_candidate = _prepare_candidate(client, changed_payload)
    response = client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=changed_payload,
    )

    assert changed_candidate["content_hash"] != original_candidate["content_hash"]
    assert response.status_code == 422
    assert "当前内容尚无审核记录" in response.json()["detail"]


def test_question_review_requires_persisted_candidate_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """任意合法摘要若未对应服务端候选快照，均不可写入审核记录。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    response = client.post(
        "/v1/admin/question-reviews",
        headers=_ADMIN_HEADERS,
        json={
            "question_id": "q-missing",
            "content_version": "content-v1",
            "content_hash": "0" * 64,
            "status": "approved",
            "verified_answer": "B",
            "formalization_version": "logic-v1",
        },
    )

    assert response.status_code == 422
    assert "候选内容不存在" in response.json()["detail"]


def test_question_review_rejects_formalization_version_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """审核形式化版本必须与已持久化候选快照完全一致。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    payload = _publish_payload("q-1", "content-v1")
    candidate = _prepare_candidate(client, payload)
    review_response = client.post(
        "/v1/admin/question-reviews",
        headers=_ADMIN_HEADERS,
        json={
            "question_id": candidate["question_id"],
            "content_version": candidate["content_version"],
            "content_hash": candidate["content_hash"],
            "status": "approved",
            "verified_answer": "B",
            "formalization_version": "logic-v2",
        },
    )

    assert review_response.status_code == 422
    assert "审核形式化版本" in review_response.json()["detail"]


def test_candidate_snapshot_is_idempotent_and_retrievable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """候选接口重复提交同内容保持稳定，并支持以三元键精确回查。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    payload = _publish_payload("q-1", "content-v1")

    first = _prepare_candidate(client, payload)
    second = _prepare_candidate(client, payload)
    fetched = client.get(
        "/v1/admin/question-candidates/"
        f"{first['question_id']}/{first['content_version']}/{first['content_hash']}",
        headers=_ADMIN_HEADERS,
    )

    assert first == second
    assert fetched.status_code == 200
    assert fetched.json() == first


def test_active_practice_question_requires_identity_and_hides_internal_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """练习接口只向认证主体提供活动发布题目，且不泄露发布与审核资产。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    first_payload = _publish_payload("q-1", "content-v1")
    second_payload = {
        **_publish_payload("q-1", "content-v2"),
        "stem": "q-1 的新版题干",
    }
    for payload in (first_payload, second_payload):
        _prepare_and_approve(client, payload)
        published = client.post(
            "/v1/admin/questions",
            headers=_ADMIN_HEADERS,
            json=payload,
        )
        assert published.status_code == 200

    unauthenticated = client.get("/v1/learning/questions/q-1/content-v2")
    historical = client.get(
        "/v1/learning/questions/q-1/content-v1",
        headers=_LEARNER_HEADERS,
    )
    active = client.get(
        "/v1/learning/questions/q-1/content-v2",
        headers=_LEARNER_HEADERS,
    )

    assert unauthenticated.status_code == 401
    assert historical.status_code == 404
    assert active.status_code == 200
    learner_question = active.json()
    assert learner_question == {
        "question_id": "q-1",
        "content_version": "content-v2",
        "question_type": "propositional",
        "stem": "q-1 的新版题干",
        "options": ["A", "B"],
    }
    for forbidden_field in (
        "content_hash",
        "error_tags",
        "knowledge_tags",
        "formalization_version",
        "formalization",
        "reviewer_id",
        "verified_answer",
        "publisher_id",
    ):
        assert forbidden_field not in learner_question


def test_practice_attempt_is_scored_server_side_and_records_current_user(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """作答接口必须从审计答案判分，并把题目标签仅写入当前用户学习记录。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    payload = _publish_payload("q-1", "content-v1")
    _prepare_and_approve(client, payload)
    published = client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=payload,
    )
    assert published.status_code == 200

    unauthenticated = client.post(
        "/v1/learning/questions/q-1/content-v1/attempts",
        json={"selected_option": "B"},
    )
    manipulated = client.post(
        "/v1/learning/questions/q-1/content-v1/attempts",
        headers=_LEARNER_HEADERS,
        json={
            "selected_option": "B",
            "is_correct": False,
            "user_id": "another-user",
            "error_tags": ["client-controlled"],
        },
    )
    incorrect = client.post(
        "/v1/learning/questions/q-1/content-v1/attempts",
        headers=_LEARNER_HEADERS,
        json={"selected_option": "A", "duration_seconds": 12},
    )
    duplicate = client.post(
        "/v1/learning/questions/q-1/content-v1/attempts",
        headers=_LEARNER_HEADERS,
        json={"selected_option": "B"},
    )
    invalid_option = client.post(
        "/v1/learning/questions/q-1/content-v1/attempts",
        headers=_LEARNER_HEADERS,
        json={"selected_option": "not-an-option"},
    )

    assert unauthenticated.status_code == 401
    assert manipulated.status_code == 422
    assert incorrect.status_code == 200
    assert incorrect.json() == {
        "question_id": "q-1",
        "content_version": "content-v1",
        "selected_option": "A",
        "is_correct": False,
        "record_id": incorrect.json()["record_id"],
    }
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "该题目版本已完成练习，请选择下一题"
    assert invalid_option.status_code == 422
    assert invalid_option.json()["detail"] == "所选选项不属于该题目"

    profile = client.get("/v1/learning/profile", headers=_LEARNER_HEADERS)
    assert profile.status_code == 200
    assert profile.json()["total_attempts"] == 1
    assert profile.json()["correct_attempts"] == 0
    assert profile.json()["focus_areas"][0]["title"] == "条件方向核验"
    assert "error_counts" not in profile.json()
    assert "knowledge_mastery" not in profile.json()
    assert "invalid_converse" not in str(profile.json())
    assert "逆命题与逆否命题" not in str(profile.json())


def test_practice_attempt_records_are_isolated_per_authenticated_user(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """同一题只能在同一用户范围内去重，其他认证用户可独立完成练习。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    payload = _publish_payload("q-1", "content-v1")
    _prepare_and_approve(client, payload)
    published = client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=payload,
    )
    assert published.status_code == 200

    first_user = client.post(
        "/v1/learning/questions/q-1/content-v1/attempts",
        headers=_LEARNER_HEADERS,
        json={"selected_option": "A"},
    )
    second_user = client.post(
        "/v1/learning/questions/q-1/content-v1/attempts",
        headers={
            **_LEARNER_HEADERS,
            "X-Logic-QA-Subject": "user-b",
        },
        json={"selected_option": "B"},
    )

    assert first_user.status_code == 200
    assert second_user.status_code == 200
    assert second_user.json()["is_correct"] is True


def test_practice_correction_request_is_isolated_and_resolved_without_rewriting_attempt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """申请和管理员处置都受身份保护，且不会重开首次作答或泄露内部信息。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    payload = _publish_payload("q-1", "content-v1")
    _prepare_and_approve(client, payload)
    published = client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=payload,
    )
    assert published.status_code == 200
    attempt = client.post(
        "/v1/learning/questions/q-1/content-v1/attempts",
        headers=_LEARNER_HEADERS,
        json={"selected_option": "A"},
    )
    assert attempt.status_code == 200
    record_id = attempt.json()["record_id"]

    invalid_payload = client.post(
        "/v1/learning/practice-correction-requests",
        headers=_LEARNER_HEADERS,
        json={
            "record_id": record_id,
            "reason": "请复核服务端判分",
            "is_correct": True,
        },
    )
    unauthenticated = client.post(
        "/v1/learning/practice-correction-requests",
        json={"record_id": record_id, "reason": "请复核服务端判分"},
    )
    cross_user = client.post(
        "/v1/learning/practice-correction-requests",
        headers={**_LEARNER_HEADERS, "X-Logic-QA-Subject": "user-b"},
        json={"record_id": record_id, "reason": "请复核服务端判分"},
    )
    created = client.post(
        "/v1/learning/practice-correction-requests",
        headers=_LEARNER_HEADERS,
        json={"record_id": record_id, "reason": "请复核服务端判分"},
    )

    assert invalid_payload.status_code == 422
    assert unauthenticated.status_code == 401
    assert cross_user.status_code == 404
    assert created.status_code == 200
    learner_request = created.json()
    assert set(learner_request) == {
        "request_id",
        "record_id",
        "question_id",
        "content_version",
        "status",
        "created_at",
        "resolved_at",
    }
    assert learner_request["status"] == "pending"
    assert "reason" not in learner_request
    assert "user_id" not in learner_request
    assert "resolved_by" not in learner_request

    learner_list = client.get(
        "/v1/learning/practice-correction-requests",
        headers=_LEARNER_HEADERS,
    )
    other_learner_list = client.get(
        "/v1/learning/practice-correction-requests",
        headers={**_LEARNER_HEADERS, "X-Logic-QA-Subject": "user-b"},
    )
    learner_admin_list = client.get(
        "/v1/admin/practice-correction-requests",
        headers=_LEARNER_HEADERS,
    )
    pending_admin_list = client.get(
        "/v1/admin/practice-correction-requests?status=pending",
        headers=_ADMIN_HEADERS,
    )

    assert learner_list.status_code == 200
    assert learner_list.json() == [learner_request]
    assert other_learner_list.status_code == 200
    assert other_learner_list.json() == []
    assert learner_admin_list.status_code == 403
    assert pending_admin_list.status_code == 200
    assert pending_admin_list.json()[0]["reason"] == "请复核服务端判分"

    resolved = client.post(
        f"/v1/admin/practice-correction-requests/{learner_request['request_id']}/resolution",
        headers=_ADMIN_HEADERS,
        json={
            "resolution": "republication_required",
            "notes": "原始练习记录保持不变，后续按题库流程复核。",
        },
    )
    duplicate_resolution = client.post(
        f"/v1/admin/practice-correction-requests/{learner_request['request_id']}/resolution",
        headers=_ADMIN_HEADERS,
        json={"resolution": "record_confirmed"},
    )
    duplicate_request = client.post(
        "/v1/learning/practice-correction-requests",
        headers=_LEARNER_HEADERS,
        json={"record_id": record_id, "reason": "再次申请"},
    )
    repeat_attempt = client.post(
        "/v1/learning/questions/q-1/content-v1/attempts",
        headers=_LEARNER_HEADERS,
        json={"selected_option": "B"},
    )
    profile = client.get("/v1/learning/profile", headers=_LEARNER_HEADERS)
    resolved_learner_list = client.get(
        "/v1/learning/practice-correction-requests",
        headers=_LEARNER_HEADERS,
    )

    assert resolved.status_code == 200
    assert resolved.json()["status"] == "republication_required"
    assert resolved.json()["resolved_by"] == "admin-a"
    assert duplicate_resolution.status_code == 409
    assert duplicate_resolution.json()["detail"] == "复核申请已完成处置"
    assert duplicate_request.status_code == 409
    assert duplicate_request.json()["detail"] == "该练习记录已提交复核申请"
    assert repeat_attempt.status_code == 409
    assert repeat_attempt.json()["detail"] == "该题目版本已完成练习，请选择下一题"
    assert profile.json()["total_attempts"] == 1
    assert profile.json()["correct_attempts"] == 0
    assert resolved_learner_list.status_code == 200
    assert resolved_learner_list.json()[0]["status"] == "republication_required"
    assert "reason" not in resolved_learner_list.json()[0]
    assert "resolution_notes" not in resolved_learner_list.json()[0]


def test_republication_outcome_projects_safely_and_new_version_is_recommended(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """派生结果说明复核状态，新发布版本仍通过版本化推荐独立进入练习。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    first_payload = _publish_payload("q-1", "content-v1")
    _prepare_and_approve(client, first_payload)
    first_published = client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=first_payload,
    )
    assert first_published.status_code == 200
    attempt = client.post(
        "/v1/learning/questions/q-1/content-v1/attempts",
        headers=_LEARNER_HEADERS,
        json={"selected_option": "A"},
    )
    assert attempt.status_code == 200
    request = client.post(
        "/v1/learning/practice-correction-requests",
        headers=_LEARNER_HEADERS,
        json={"record_id": attempt.json()["record_id"], "reason": "请复核题目版本"},
    )
    assert request.status_code == 200
    resolved = client.post(
        f"/v1/admin/practice-correction-requests/{request.json()['request_id']}/resolution",
        headers=_ADMIN_HEADERS,
        json={
            "resolution": "republication_required",
            "notes": "题库复核中，内部说明不对学习者公开。",
        },
    )
    assert resolved.status_code == 200

    unauthenticated = client.get("/v1/learning/practice-correction-outcomes")
    other_user = client.get(
        "/v1/learning/practice-correction-outcomes",
        headers={**_LEARNER_HEADERS, "X-Logic-QA-Subject": "user-b"},
    )
    outcomes = client.get(
        "/v1/learning/practice-correction-outcomes",
        headers=_LEARNER_HEADERS,
    )

    assert unauthenticated.status_code == 401
    assert other_user.status_code == 200
    assert other_user.json() == []
    assert outcomes.status_code == 200
    assert outcomes.json() == [
        {
            "request_id": request.json()["request_id"],
            "record_id": attempt.json()["record_id"],
            "question_id": "q-1",
            "content_version": "content-v1",
            "kind": "republication_required",
            "message": (
                "复核已完成，该题目将按发布流程复核；若发布新版本，"
                "新版本会作为独立练习重新推荐。"
            ),
            "created_at": outcomes.json()[0]["created_at"],
            "resolved_at": resolved.json()["resolved_at"],
        }
    ]
    for forbidden_field in (
        "reason",
        "resolved_by",
        "resolution_notes",
        "is_correct",
        "content_hash",
        "formalization",
    ):
        assert forbidden_field not in outcomes.json()[0]

    second_payload = {
        **_publish_payload("q-1", "content-v2"),
        "stem": "q-1 的复核后新版题干",
    }
    _prepare_and_approve(client, second_payload)
    second_published = client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=second_payload,
    )
    recommendations = client.get(
        "/v1/learning/recommendations",
        headers=_LEARNER_HEADERS,
    )
    profile = client.get("/v1/learning/profile", headers=_LEARNER_HEADERS)

    assert second_published.status_code == 200
    assert recommendations.status_code == 200
    assert recommendations.json()[0]["question"]["question_id"] == "q-1"
    assert recommendations.json()[0]["question"]["content_version"] == "content-v2"
    assert profile.json()["total_attempts"] == 1
    assert profile.json()["correct_attempts"] == 0


def test_correction_republication_link_requires_published_active_version(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """关联只接受同题的活动新版本，并由题库摘要而非客户端参数作为审计事实。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    first_payload = _publish_payload("q-1", "content-v1")
    _prepare_and_approve(client, first_payload)
    assert client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=first_payload,
    ).status_code == 200
    question_backup = api.question_bank_store.create_backup(
        tmp_path / "question-backups"
    )
    attempt = client.post(
        "/v1/learning/questions/q-1/content-v1/attempts",
        headers=_LEARNER_HEADERS,
        json={"selected_option": "A"},
    )
    assert attempt.status_code == 200
    correction_request = client.post(
        "/v1/learning/practice-correction-requests",
        headers=_LEARNER_HEADERS,
        json={"record_id": attempt.json()["record_id"], "reason": "请复核题目版本"},
    )
    assert correction_request.status_code == 200
    request_id = correction_request.json()["request_id"]
    assert client.post(
        f"/v1/admin/practice-correction-requests/{request_id}/resolution",
        headers=_ADMIN_HEADERS,
        json={"resolution": "republication_required"},
    ).status_code == 200

    new_version_path = (
        "/v1/admin/questions/q-1/content-v2/correction-republication-links"
    )
    before_publication = client.post(
        new_version_path,
        headers=_ADMIN_HEADERS,
        json={"request_id": request_id, "previous_content_version": "content-v1"},
    )
    unauthenticated = client.post(
        new_version_path,
        json={"request_id": request_id, "previous_content_version": "content-v1"},
    )
    learner = client.post(
        new_version_path,
        headers=_LEARNER_HEADERS,
        json={"request_id": request_id, "previous_content_version": "content-v1"},
    )
    invalid_payload = client.post(
        new_version_path,
        headers=_ADMIN_HEADERS,
        json={
            "request_id": request_id,
            "previous_content_version": "content-v1",
            "new_content_hash": "f" * 64,
        },
    )

    assert before_publication.status_code == 422
    assert before_publication.json()["detail"] == "未找到可关联的已发布新版本"
    assert unauthenticated.status_code == 401
    assert learner.status_code == 403
    assert invalid_payload.status_code == 422

    second_payload = {
        **_publish_payload("q-1", "content-v2"),
        "stem": "q-1 的复核后新版题干",
    }
    candidate = _prepare_and_approve(client, second_payload)
    second_published = client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=second_payload,
    )
    assert second_published.status_code == 200

    linked = client.post(
        new_version_path,
        headers=_ADMIN_HEADERS,
        json={"request_id": request_id, "previous_content_version": "content-v1"},
    )
    duplicate = client.post(
        new_version_path,
        headers=_ADMIN_HEADERS,
        json={"request_id": request_id, "previous_content_version": "content-v1"},
    )
    outcomes = client.get(
        "/v1/learning/practice-correction-outcomes",
        headers=_LEARNER_HEADERS,
    )
    recommendations = client.get(
        "/v1/learning/recommendations",
        headers=_LEARNER_HEADERS,
    )
    profile = client.get("/v1/learning/profile", headers=_LEARNER_HEADERS)

    assert linked.status_code == 200
    assert linked.json()["question_id"] == "q-1"
    assert linked.json()["previous_content_version"] == "content-v1"
    assert linked.json()["new_content_version"] == "content-v2"
    assert linked.json()["new_content_hash"] == candidate["content_hash"]
    assert linked.json()["linked_by"] == "admin-a"
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "该复核申请已关联新的发布版本"
    assert outcomes.status_code == 200
    assert outcomes.json()[0]["message"] == "复核后的新版本已发布，可作为独立练习完成。"
    assert outcomes.json()[0]["republished_content_version"] == "content-v2"
    for forbidden_field in (
        "reason",
        "resolved_by",
        "resolution_notes",
        "content_hash",
        "linked_by",
        "new_content_hash",
    ):
        assert forbidden_field not in outcomes.json()[0]
    assert recommendations.status_code == 200
    assert recommendations.json()[0]["question"]["content_version"] == "content-v2"
    assert profile.json()["total_attempts"] == 1
    assert profile.json()["correct_attempts"] == 0

    api.question_bank_store.restore_backup(
        api.question_bank_store.load_backup(question_backup.manifest_path)
    )
    restored_outcomes = client.get(
        "/v1/learning/practice-correction-outcomes",
        headers=_LEARNER_HEADERS,
    )

    assert restored_outcomes.status_code == 200
    assert restored_outcomes.json()[0]["message"] == (
        "复核已完成，该题目将按发布流程复核；若发布新版本，"
        "新版本会作为独立练习重新推荐。"
    )
    assert "republished_content_version" not in restored_outcomes.json()[0]


def test_correction_republication_link_rejects_non_republication_or_mismatched_request(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """关联仅可用于同题、同旧版本且已标记重发布的申请。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    first_payload = _publish_payload("q-1", "content-v1")
    _prepare_and_approve(client, first_payload)
    assert client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=first_payload,
    ).status_code == 200
    attempt = client.post(
        "/v1/learning/questions/q-1/content-v1/attempts",
        headers=_LEARNER_HEADERS,
        json={"selected_option": "A"},
    )
    assert attempt.status_code == 200
    correction_request = client.post(
        "/v1/learning/practice-correction-requests",
        headers=_LEARNER_HEADERS,
        json={"record_id": attempt.json()["record_id"], "reason": "请确认判分"},
    )
    assert correction_request.status_code == 200
    request_id = correction_request.json()["request_id"]
    assert client.post(
        f"/v1/admin/practice-correction-requests/{request_id}/resolution",
        headers=_ADMIN_HEADERS,
        json={"resolution": "record_confirmed"},
    ).status_code == 200

    second_payload = _publish_payload("q-1", "content-v2")
    _prepare_and_approve(client, second_payload)
    assert client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=second_payload,
    ).status_code == 200
    new_version_path = (
        "/v1/admin/questions/q-1/content-v2/correction-republication-links"
    )
    confirmed = client.post(
        new_version_path,
        headers=_ADMIN_HEADERS,
        json={"request_id": request_id, "previous_content_version": "content-v1"},
    )
    old_version_path = (
        "/v1/admin/questions/q-1/content-v1/correction-republication-links"
    )
    inactive_old_version = client.post(
        old_version_path,
        headers=_ADMIN_HEADERS,
        json={"request_id": request_id, "previous_content_version": "content-v1"},
    )

    second_question_first_payload = _publish_payload("q-2", "content-v1")
    _prepare_and_approve(client, second_question_first_payload)
    assert client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=second_question_first_payload,
    ).status_code == 200
    second_attempt = client.post(
        "/v1/learning/questions/q-2/content-v1/attempts",
        headers=_LEARNER_HEADERS,
        json={"selected_option": "A"},
    )
    assert second_attempt.status_code == 200
    second_request = client.post(
        "/v1/learning/practice-correction-requests",
        headers=_LEARNER_HEADERS,
        json={
            "record_id": second_attempt.json()["record_id"],
            "reason": "请复核另一道题目",
        },
    )
    assert second_request.status_code == 200
    assert client.post(
        "/v1/admin/practice-correction-requests/"
        f"{second_request.json()['request_id']}/resolution",
        headers=_ADMIN_HEADERS,
        json={"resolution": "republication_required"},
    ).status_code == 200
    second_question_new_payload = _publish_payload("q-2", "content-v2")
    _prepare_and_approve(client, second_question_new_payload)
    assert client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=second_question_new_payload,
    ).status_code == 200
    cross_question = client.post(
        new_version_path,
        headers=_ADMIN_HEADERS,
        json={
            "request_id": second_request.json()["request_id"],
            "previous_content_version": "content-v1",
        },
    )

    assert confirmed.status_code == 422
    assert confirmed.json()["detail"] == "该复核申请不需要重新发布"
    assert inactive_old_version.status_code == 422
    assert inactive_old_version.json()["detail"] == "未找到可关联的已发布新版本"
    assert cross_question.status_code == 422
    assert cross_question.json()["detail"] == "复核申请与待关联题目版本不一致"


def test_practice_attempt_record_cannot_be_deleted_by_its_owner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """练习账本记录不可删除，避免用户通过删除重新获取同一版本。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    payload = _publish_payload("q-1", "content-v1")
    _prepare_and_approve(client, payload)
    published = client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=payload,
    )
    assert published.status_code == 200

    attempt = client.post(
        "/v1/learning/questions/q-1/content-v1/attempts",
        headers=_LEARNER_HEADERS,
        json={"selected_option": "B"},
    )
    delete_response = client.delete(
        f"/v1/learning/records/{attempt.json()['record_id']}",
        headers=_LEARNER_HEADERS,
    )
    duplicate = client.post(
        "/v1/learning/questions/q-1/content-v1/attempts",
        headers=_LEARNER_HEADERS,
        json={"selected_option": "B"},
    )

    assert attempt.status_code == 200
    assert delete_response.status_code == 409
    assert delete_response.json()["detail"] == "练习记录不可删除"
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "该题目版本已完成练习，请选择下一题"


def test_active_practice_question_rejects_invalid_identifiers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """练习读取的标识不合法时必须明确拒绝，而不是回退到其他题目。"""
    client = _client_with_stores(tmp_path, monkeypatch)

    response = client.get(
        "/v1/learning/questions/%20/content-v1",
        headers=_LEARNER_HEADERS,
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "题目标识不能为空"


def test_recommendation_returns_unattempted_question_without_internal_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """学习者推荐仅返回展示题目和通用原因，不泄露审核或内部标签。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    for question_id in ("q-1", "q-2"):
        payload = _publish_payload(question_id, "content-v1")
        _prepare_and_approve(client, payload)
        response = client.post(
            "/v1/admin/questions",
            headers=_ADMIN_HEADERS,
            json=payload,
        )
        assert response.status_code == 200

    attempt = client.post(
        "/v1/learning/questions/q-1/content-v1/attempts",
        headers=_LEARNER_HEADERS,
        json={"selected_option": "A"},
    )
    assert attempt.status_code == 200

    recommendations = client.get(
        "/v1/learning/recommendations?limit=3",
        headers=_LEARNER_HEADERS,
    )

    assert recommendations.status_code == 200
    payload = recommendations.json()
    assert len(payload) == 1
    recommendation = payload[0]
    assert set(recommendation) == {"question", "reason"}
    assert "高频错因" in recommendation["reason"]
    assert recommendation["question"]["question_id"] == "q-2"
    assert set(recommendation["question"]) == {
        "question_id",
        "content_version",
        "question_type",
        "stem",
        "options",
    }
    for forbidden_field in (
        "content_hash",
        "error_tags",
        "knowledge_tags",
        "formalization_version",
        "reviewer_id",
        "verified_answer",
        "notes",
        "matched_tags",
    ):
        assert forbidden_field not in recommendation
        assert forbidden_field not in recommendation["question"]


def test_generic_learning_record_does_not_exclude_published_practice_version(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """未绑定版本的通用记录不得影响发布题练习推荐。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    payload = _publish_payload("q-1", "content-v1")
    _prepare_and_approve(client, payload)
    published = client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=payload,
    )
    assert published.status_code == 200

    generic_record = client.post(
        "/v1/learning/records",
        headers=_LEARNER_HEADERS,
        json={
            "question_id": "q-1",
            "question_type": "propositional",
            "is_correct": False,
        },
    )
    recommendations = client.get(
        "/v1/learning/recommendations?limit=3",
        headers=_LEARNER_HEADERS,
    )

    assert generic_record.status_code == 200
    assert recommendations.status_code == 200
    assert recommendations.json()[0]["question"]["question_id"] == "q-1"
    assert recommendations.json()[0]["question"]["content_version"] == "content-v1"


def test_recommendation_includes_a_new_active_version_after_old_version_attempt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """同一题发布新版本后，应向已完成旧版本的用户重新开放新版本。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    first_payload = _publish_payload("q-1", "content-v1")
    _prepare_and_approve(client, first_payload)
    first_published = client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=first_payload,
    )
    assert first_published.status_code == 200

    first_attempt = client.post(
        "/v1/learning/questions/q-1/content-v1/attempts",
        headers=_LEARNER_HEADERS,
        json={"selected_option": "B"},
    )
    assert first_attempt.status_code == 200

    second_payload = {
        **_publish_payload("q-1", "content-v2"),
        "stem": "q-1 的新版题干",
    }
    _prepare_and_approve(client, second_payload)
    second_published = client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=second_payload,
    )
    assert second_published.status_code == 200

    recommendations = client.get(
        "/v1/learning/recommendations?limit=3",
        headers=_LEARNER_HEADERS,
    )

    assert recommendations.status_code == 200
    assert recommendations.json()[0]["question"] == {
        "question_id": "q-1",
        "content_version": "content-v2",
        "question_type": "propositional",
        "stem": "q-1 的新版题干",
        "options": ["A", "B"],
    }
