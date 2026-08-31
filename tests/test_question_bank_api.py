"""内容绑定审核题库发布与学习者最小响应的集成测试。"""

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from logic_qa import api
from logic_qa.learning_profile import (
    LearningProfileStore,
    PracticeCorrectionAudit,
    PracticeCorrectionRepublication,
    PracticeCorrectionRequest,
    PracticeCorrectionRequestStatus,
)
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


def _correction_audit_with_republication(
    *,
    republication_question_id: str = "q-1",
    previous_content_version: str = "content-v1",
) -> PracticeCorrectionAudit:
    """构造只用于审计异常核验的不可变关联视图。"""
    request = PracticeCorrectionRequest(
        request_id="request-1",
        record_id="record-1",
        user_id="user-a",
        question_id="q-1",
        content_version="content-v1",
        reason="需要重发布",
        status=PracticeCorrectionRequestStatus.REPUBLICATION_REQUIRED,
        created_at="2026-08-29T00:00:00+00:00",
        resolved_by="admin-a",
        resolution_notes=None,
        resolved_at="2026-08-29T00:00:00+00:00",
    )
    republication = PracticeCorrectionRepublication(
        request_id=request.request_id,
        question_id=republication_question_id,
        previous_content_version=previous_content_version,
        new_content_version="content-v2",
        new_content_hash="a" * 64,
        linked_by="admin-a",
        linked_at="2026-08-29T00:00:00+00:00",
    )
    return PracticeCorrectionAudit(
        request=request,
        republication=republication,
        events=(),
    )


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
        "/v1/admin/questions/{question_id}/{content_version}/deactivation",
        "/v1/admin/questions/{question_id}/{content_version}/reactivation",
        "/v1/admin/questions/{question_id}/version-lifecycle-events",
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


def test_admin_version_lifecycle_routes_preserve_immutable_practice_ledger(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """管理员可受控下线与回滚，学习者不能调用且首次练习记录保持不变。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    first_payload = _publish_payload("q-1", "content-v1")
    second_payload = {
        **_publish_payload("q-1", "content-v2"),
        "stem": "q-1 的受控下线新版题干",
    }
    first_candidate = _prepare_and_approve(client, first_payload)
    assert client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=first_payload,
    ).status_code == 200
    initial_attempt = client.post(
        "/v1/learning/questions/q-1/content-v1/attempts",
        headers=_LEARNER_HEADERS,
        json={"selected_option": "A"},
    )
    assert initial_attempt.status_code == 200
    second_candidate = _prepare_and_approve(client, second_payload)
    assert client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=second_payload,
    ).status_code == 200

    events_path = "/v1/admin/questions/q-1/version-lifecycle-events"
    events_after_publication = client.get(events_path, headers=_ADMIN_HEADERS)
    assert events_after_publication.status_code == 200
    assert events_after_publication.json() == [
        {
            "event_id": events_after_publication.json()[0]["event_id"],
            "question_id": "q-1",
            "content_version": "content-v1",
            "content_hash": first_candidate["content_hash"],
            "action": "superseded",
            "actor_id": "admin-a",
            "replaced_content_version": "content-v2",
            "reason": "发布新的已审核内容版本",
            "created_at": events_after_publication.json()[0]["created_at"],
        }
    ]

    deactivation_path = "/v1/admin/questions/q-1/content-v2/deactivation"
    reactivation_path = "/v1/admin/questions/q-1/content-v1/reactivation"
    unauthenticated = client.post(
        deactivation_path,
        json={"reason": "等待复核"},
    )
    learner = client.post(
        deactivation_path,
        headers=_LEARNER_HEADERS,
        json={"reason": "等待复核"},
    )
    malformed = client.post(
        deactivation_path,
        headers=_ADMIN_HEADERS,
        json={"reason": "等待复核", "actor_id": "forged-admin"},
    )
    deactivated = client.post(
        deactivation_path,
        headers=_ADMIN_HEADERS,
        json={"reason": "新版题干待核验，受控下线"},
    )
    duplicate_deactivation = client.post(
        deactivation_path,
        headers=_ADMIN_HEADERS,
        json={"reason": "重复下线"},
    )
    unavailable_question = client.get(
        "/v1/learning/questions/q-1/content-v2",
        headers=_LEARNER_HEADERS,
    )
    events_after_deactivation = client.get(events_path, headers=_ADMIN_HEADERS)
    learner_events = client.get(events_path, headers=_LEARNER_HEADERS)

    assert unauthenticated.status_code == 401
    assert learner.status_code == 403
    assert malformed.status_code == 422
    assert deactivated.status_code == 200
    assert deactivated.json() == {
        "event_id": deactivated.json()["event_id"],
        "question_id": "q-1",
        "content_version": "content-v2",
        "content_hash": second_candidate["content_hash"],
        "action": "deactivated",
        "actor_id": "admin-a",
        "replaced_content_version": None,
        "reason": "新版题干待核验，受控下线",
        "created_at": deactivated.json()["created_at"],
    }
    assert duplicate_deactivation.status_code == 422
    assert duplicate_deactivation.json()["detail"] == "该题目版本当前未活动，不能下线"
    assert unavailable_question.status_code == 404
    assert learner_events.status_code == 403
    assert events_after_deactivation.status_code == 200
    assert events_after_deactivation.json() == [
        events_after_publication.json()[0],
        deactivated.json(),
    ]

    reactivated = client.post(
        reactivation_path,
        headers=_ADMIN_HEADERS,
        json={"reason": "审核与确定性复验均通过，回滚历史版本"},
    )
    restored_question = client.get(
        "/v1/learning/questions/q-1/content-v1",
        headers=_LEARNER_HEADERS,
    )
    repeat_attempt = client.post(
        "/v1/learning/questions/q-1/content-v1/attempts",
        headers=_LEARNER_HEADERS,
        json={"selected_option": "B"},
    )
    profile = client.get("/v1/learning/profile", headers=_LEARNER_HEADERS)
    events = client.get(events_path, headers=_ADMIN_HEADERS)

    assert reactivated.status_code == 200
    assert reactivated.json() == {
        "event_id": reactivated.json()["event_id"],
        "question_id": "q-1",
        "content_version": "content-v1",
        "content_hash": first_candidate["content_hash"],
        "action": "reactivated",
        "actor_id": "admin-a",
        "replaced_content_version": None,
        "reason": "审核与确定性复验均通过，回滚历史版本",
        "created_at": reactivated.json()["created_at"],
    }
    assert restored_question.status_code == 200
    assert restored_question.json()["content_version"] == "content-v1"
    assert repeat_attempt.status_code == 409
    assert profile.status_code == 200
    assert profile.json()["total_attempts"] == 1
    assert profile.json()["correct_attempts"] == 0
    assert events.status_code == 200
    assert events.json() == [
        events_after_publication.json()[0],
        deactivated.json(),
        reactivated.json(),
    ]

    stale_review = client.post(
        "/v1/admin/question-reviews",
        headers=_ADMIN_HEADERS,
        json={
            "question_id": "q-1",
            "content_version": "content-v2",
            "content_hash": second_candidate["content_hash"],
            "status": "needs_revision",
            "verified_answer": None,
            "formalization_version": "logic-v1",
            "notes": "历史版本审核已撤销",
        },
    )
    rejected_reactivation = client.post(
        "/v1/admin/questions/q-1/content-v2/reactivation",
        headers=_ADMIN_HEADERS,
        json={"reason": "尝试绕过历史审核"},
    )

    assert stale_review.status_code == 200
    assert rejected_reactivation.status_code == 422
    assert rejected_reactivation.json()["detail"] == "题目当前内容未通过审核，不能发布"
    assert client.get(
        "/v1/learning/questions/q-1/content-v1",
        headers=_LEARNER_HEADERS,
    ).status_code == 200


def test_republication_outcome_downgrades_when_question_bank_is_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """题库独立存储暂不可读时，学习者结果必须安全降级而非返回 500。"""
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
        json={"record_id": attempt.json()["record_id"], "reason": "需要重发布"},
    )
    assert correction_request.status_code == 200
    request_id = correction_request.json()["request_id"]
    assert client.post(
        f"/v1/admin/practice-correction-requests/{request_id}/resolution",
        headers=_ADMIN_HEADERS,
        json={"resolution": "republication_required"},
    ).status_code == 200
    second_payload = {
        **_publish_payload("q-1", "content-v2"),
        "stem": "q-1 的暂不可读关联新版题干",
    }
    _prepare_and_approve(client, second_payload)
    assert client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=second_payload,
    ).status_code == 200
    assert client.post(
        "/v1/admin/questions/q-1/content-v2/correction-republication-links",
        headers=_ADMIN_HEADERS,
        json={"request_id": request_id, "previous_content_version": "content-v1"},
    ).status_code == 200

    class UnavailableQuestionBankStore:
        """模拟独立题库读取发生 SQLite 运行时错误。"""

        def get_active_published_question(
            self,
            question_id: str,
            content_version: str,
        ) -> None:
            raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(api, "question_bank_store", UnavailableQuestionBankStore())
    outcome = client.get(
        "/v1/learning/practice-correction-outcomes",
        headers=_LEARNER_HEADERS,
    )

    assert outcome.status_code == 200
    assert outcome.json()[0]["message"] == (
        "复核已完成，该题目将按发布流程复核；若发布新版本，"
        "新版本会作为独立练习重新推荐。"
    )
    assert "republished_content_version" not in outcome.json()[0]


def test_republication_outcome_downgrades_when_linked_version_is_deactivated(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """已关联版本下线后，学习者结果必须安全降级而不能继续声称已发布。"""
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
        json={"record_id": attempt.json()["record_id"], "reason": "需要重发布"},
    )
    assert correction_request.status_code == 200
    request_id = correction_request.json()["request_id"]
    assert client.post(
        f"/v1/admin/practice-correction-requests/{request_id}/resolution",
        headers=_ADMIN_HEADERS,
        json={"resolution": "republication_required"},
    ).status_code == 200
    second_payload = {
        **_publish_payload("q-1", "content-v2"),
        "stem": "q-1 的关联新版题干",
    }
    _prepare_and_approve(client, second_payload)
    assert client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=second_payload,
    ).status_code == 200
    assert client.post(
        "/v1/admin/questions/q-1/content-v2/correction-republication-links",
        headers=_ADMIN_HEADERS,
        json={"request_id": request_id, "previous_content_version": "content-v1"},
    ).status_code == 200

    published_outcome = client.get(
        "/v1/learning/practice-correction-outcomes",
        headers=_LEARNER_HEADERS,
    )
    deactivated = client.post(
        "/v1/admin/questions/q-1/content-v2/deactivation",
        headers=_ADMIN_HEADERS,
        json={"reason": "关联新版本需要暂时下线"},
    )
    downgraded_outcome = client.get(
        "/v1/learning/practice-correction-outcomes",
        headers=_LEARNER_HEADERS,
    )

    assert published_outcome.status_code == 200
    assert published_outcome.json()[0]["republished_content_version"] == "content-v2"
    assert deactivated.status_code == 200
    assert downgraded_outcome.status_code == 200
    assert downgraded_outcome.json()[0]["message"] == (
        "复核已完成，该题目将按发布流程复核；若发布新版本，"
        "新版本会作为独立练习重新推荐。"
    )
    assert "republished_content_version" not in downgraded_outcome.json()[0]

    audit = client.get(
        f"/v1/admin/practice-correction-audits/{request_id}",
        headers=_ADMIN_HEADERS,
    )

    assert audit.status_code == 200
    assert audit.json()["republication_verification"] == {
        "status": "historical_inactive",
        "observed_content_hash": audit.json()["republication"]["new_content_hash"],
    }


def test_admin_correction_audit_reports_missing_republication_version(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """题库独立恢复而缺少关联版本时，管理员审计必须标记为缺失。"""
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
        json={"record_id": attempt.json()["record_id"], "reason": "需要重发布"},
    )
    assert correction_request.status_code == 200
    request_id = correction_request.json()["request_id"]
    assert client.post(
        f"/v1/admin/practice-correction-requests/{request_id}/resolution",
        headers=_ADMIN_HEADERS,
        json={"resolution": "republication_required"},
    ).status_code == 200
    second_payload = {
        **_publish_payload("q-1", "content-v2"),
        "stem": "q-1 的缺失关联新版题干",
    }
    _prepare_and_approve(client, second_payload)
    assert client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=second_payload,
    ).status_code == 200
    assert client.post(
        "/v1/admin/questions/q-1/content-v2/correction-republication-links",
        headers=_ADMIN_HEADERS,
        json={"request_id": request_id, "previous_content_version": "content-v1"},
    ).status_code == 200
    api.question_bank_store.restore_backup(
        api.question_bank_store.load_backup(question_backup.manifest_path)
    )

    audit = client.get(
        f"/v1/admin/practice-correction-audits/{request_id}",
        headers=_ADMIN_HEADERS,
    )

    assert audit.status_code == 200
    assert audit.json()["republication_verification"] == {
        "status": "missing",
        "observed_content_hash": None,
    }
    reconciliation = client.get(
        "/v1/admin/practice-correction-reconciliations",
        headers=_ADMIN_HEADERS,
    )

    assert reconciliation.status_code == 200
    reconciliation_payload = reconciliation.json()
    assert reconciliation_payload["scan_boundary"] == 1
    assert reconciliation_payload["total_linked_audits"] == 1
    assert reconciliation_payload["offset"] == 0
    assert reconciliation_payload["scanned_linked_audits"] == 1
    assert reconciliation_payload["active_verified_audits"] == 0
    assert reconciliation_payload["next_offset"] is None
    assert reconciliation_payload["non_verified_audits"] == [audit.json()]


def test_admin_republication_reconciliation_paginates_linked_audits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """巡检按稳定偏移量扫描关联，并只在异常页返回完整审计视图。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    for question_id in ("q-1", "q-2"):
        first_payload = _publish_payload(question_id, "content-v1")
        _prepare_and_approve(client, first_payload)
        assert client.post(
            "/v1/admin/questions",
            headers=_ADMIN_HEADERS,
            json=first_payload,
        ).status_code == 200
        attempt = client.post(
            f"/v1/learning/questions/{question_id}/content-v1/attempts",
            headers=_LEARNER_HEADERS,
            json={"selected_option": "A"},
        )
        assert attempt.status_code == 200
        correction_request = client.post(
            "/v1/learning/practice-correction-requests",
            headers=_LEARNER_HEADERS,
            json={"record_id": attempt.json()["record_id"], "reason": "需要重发布"},
        )
        assert correction_request.status_code == 200
        request_id = correction_request.json()["request_id"]
        assert client.post(
            f"/v1/admin/practice-correction-requests/{request_id}/resolution",
            headers=_ADMIN_HEADERS,
            json={"resolution": "republication_required"},
        ).status_code == 200
        second_payload = {
            **_publish_payload(question_id, "content-v2"),
            "stem": f"{question_id} 的巡检新版题干",
        }
        _prepare_and_approve(client, second_payload)
        assert client.post(
            "/v1/admin/questions",
            headers=_ADMIN_HEADERS,
            json=second_payload,
        ).status_code == 200
        assert client.post(
            f"/v1/admin/questions/{question_id}/content-v2/"
            "correction-republication-links",
            headers=_ADMIN_HEADERS,
            json={"request_id": request_id, "previous_content_version": "content-v1"},
        ).status_code == 200

    assert client.post(
        "/v1/admin/questions/q-2/content-v2/deactivation",
        headers=_ADMIN_HEADERS,
        json={"reason": "巡检异常投影测试"},
    ).status_code == 200
    first_page = client.get(
        "/v1/admin/practice-correction-reconciliations?limit=1",
        headers=_ADMIN_HEADERS,
    )
    assert first_page.status_code == 200
    first_page_payload = first_page.json()
    scan_boundary = first_page_payload["scan_boundary"]
    assert scan_boundary == 2
    second_page = client.get(
        "/v1/admin/practice-correction-reconciliations"
        f"?limit=1&offset=1&scan_boundary={scan_boundary}",
        headers=_ADMIN_HEADERS,
    )
    empty_page = client.get(
        "/v1/admin/practice-correction-reconciliations"
        f"?limit=1&offset=2&scan_boundary={scan_boundary}",
        headers=_ADMIN_HEADERS,
    )

    assert first_page_payload["total_linked_audits"] == 2
    assert first_page_payload["offset"] == 0
    assert first_page_payload["scanned_linked_audits"] == 1
    assert first_page_payload["next_offset"] == 1
    assert second_page.status_code == 200
    second_page_payload = second_page.json()
    assert second_page_payload["scan_boundary"] == scan_boundary
    assert second_page_payload["total_linked_audits"] == 2
    assert second_page_payload["offset"] == 1
    assert second_page_payload["scanned_linked_audits"] == 1
    assert second_page_payload["next_offset"] is None
    assert (
        first_page_payload["active_verified_audits"]
        + second_page_payload["active_verified_audits"]
        == 1
    )
    non_verified_audits = (
        first_page_payload["non_verified_audits"]
        + second_page_payload["non_verified_audits"]
    )
    assert len(non_verified_audits) == 1
    assert (
        non_verified_audits[0]["republication_verification"]["status"]
        == "historical_inactive"
    )
    assert empty_page.status_code == 200
    assert empty_page.json() == {
        "scan_boundary": scan_boundary,
        "total_linked_audits": 2,
        "offset": 2,
        "scanned_linked_audits": 0,
        "active_verified_audits": 0,
        "next_offset": None,
        "non_verified_audits": [],
    }


def test_reconciliation_scan_boundary_excludes_new_links_until_next_scan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """管理端必须用首响应边界完成同一轮分页，不混入后续新关联。"""
    client = _client_with_stores(tmp_path, monkeypatch)
    for question_id in ("q-1", "q-2"):
        first_payload = _publish_payload(question_id, "content-v1")
        _prepare_and_approve(client, first_payload)
        assert client.post(
            "/v1/admin/questions",
            headers=_ADMIN_HEADERS,
            json=first_payload,
        ).status_code == 200
        attempt = client.post(
            f"/v1/learning/questions/{question_id}/content-v1/attempts",
            headers=_LEARNER_HEADERS,
            json={"selected_option": "A"},
        )
        assert attempt.status_code == 200
        correction_request = client.post(
            "/v1/learning/practice-correction-requests",
            headers=_LEARNER_HEADERS,
            json={"record_id": attempt.json()["record_id"], "reason": "需要重发布"},
        )
        assert correction_request.status_code == 200
        request_id = correction_request.json()["request_id"]
        assert client.post(
            f"/v1/admin/practice-correction-requests/{request_id}/resolution",
            headers=_ADMIN_HEADERS,
            json={"resolution": "republication_required"},
        ).status_code == 200
        second_payload = {
            **_publish_payload(question_id, "content-v2"),
            "stem": f"{question_id} 的边界巡检新版题干",
        }
        _prepare_and_approve(client, second_payload)
        assert client.post(
            "/v1/admin/questions",
            headers=_ADMIN_HEADERS,
            json=second_payload,
        ).status_code == 200
        assert client.post(
            f"/v1/admin/questions/{question_id}/content-v2/"
            "correction-republication-links",
            headers=_ADMIN_HEADERS,
            json={"request_id": request_id, "previous_content_version": "content-v1"},
        ).status_code == 200

    first_page = client.get(
        "/v1/admin/practice-correction-reconciliations?limit=1",
        headers=_ADMIN_HEADERS,
    )

    assert first_page.status_code == 200
    first_page_payload = first_page.json()
    scan_boundary = first_page_payload["scan_boundary"]
    assert scan_boundary == 2
    assert first_page_payload["total_linked_audits"] == 2
    third_payload = _publish_payload("q-3", "content-v1")
    _prepare_and_approve(client, third_payload)
    assert client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=third_payload,
    ).status_code == 200
    third_attempt = client.post(
        "/v1/learning/questions/q-3/content-v1/attempts",
        headers=_LEARNER_HEADERS,
        json={"selected_option": "A"},
    )
    assert third_attempt.status_code == 200
    third_correction_request = client.post(
        "/v1/learning/practice-correction-requests",
        headers=_LEARNER_HEADERS,
        json={"record_id": third_attempt.json()["record_id"], "reason": "需要重发布"},
    )
    assert third_correction_request.status_code == 200
    third_request_id = third_correction_request.json()["request_id"]
    assert client.post(
        f"/v1/admin/practice-correction-requests/{third_request_id}/resolution",
        headers=_ADMIN_HEADERS,
        json={"resolution": "republication_required"},
    ).status_code == 200
    third_second_payload = {
        **_publish_payload("q-3", "content-v2"),
        "stem": "q-3 的边界巡检新版题干",
    }
    _prepare_and_approve(client, third_second_payload)
    assert client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=third_second_payload,
    ).status_code == 200
    assert client.post(
        "/v1/admin/questions/q-3/content-v2/correction-republication-links",
        headers=_ADMIN_HEADERS,
        json={"request_id": third_request_id, "previous_content_version": "content-v1"},
    ).status_code == 200
    second_page = client.get(
        "/v1/admin/practice-correction-reconciliations"
        f"?limit=1&offset=1&scan_boundary={scan_boundary}",
        headers=_ADMIN_HEADERS,
    )
    exhausted_page = client.get(
        "/v1/admin/practice-correction-reconciliations"
        f"?limit=1&offset=2&scan_boundary={scan_boundary}",
        headers=_ADMIN_HEADERS,
    )
    fresh_scan = client.get(
        "/v1/admin/practice-correction-reconciliations?limit=10",
        headers=_ADMIN_HEADERS,
    )

    assert second_page.status_code == 200
    assert second_page.json()["scan_boundary"] == scan_boundary
    assert second_page.json()["total_linked_audits"] == 2
    assert exhausted_page.status_code == 200
    assert exhausted_page.json()["scan_boundary"] == scan_boundary
    assert exhausted_page.json()["total_linked_audits"] == 2
    assert exhausted_page.json()["scanned_linked_audits"] == 0
    assert fresh_scan.status_code == 200
    assert fresh_scan.json()["scan_boundary"] > scan_boundary
    assert fresh_scan.json()["total_linked_audits"] == 3
    invalid_boundary = client.get(
        "/v1/admin/practice-correction-reconciliations?scan_boundary=999",
        headers=_ADMIN_HEADERS,
    )
    assert invalid_boundary.status_code == 422
    assert invalid_boundary.json()["detail"] == "巡检扫描边界不存在"


def test_reconciliation_fails_closed_when_learning_store_is_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """学习库巡检读取失败时不得返回可能不一致的计数或异常清单。"""
    client = _client_with_stores(tmp_path, monkeypatch)

    class UnavailableLearningStore:
        """模拟学习库读取期间发生 SQLite 运行时错误。"""

        def list_linked_practice_correction_audit_page(
            self,
            *,
            limit: int,
            offset: int,
            scan_boundary: int | None,
        ) -> None:
            raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(api, "learning_store", UnavailableLearningStore())

    response = client.get(
        "/v1/admin/practice-correction-reconciliations",
        headers=_ADMIN_HEADERS,
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "学习库暂时不可用，无法完成巡检"


def test_republication_audit_verification_detects_abnormal_cross_store_facts(
    monkeypatch,
) -> None:
    """异常关联或不可读题库只能被标记，不会覆盖任何已有审计事实。"""
    binding_mismatch = _correction_audit_with_republication(
        republication_question_id="q-2"
    )

    binding_result = api._verify_republication_for_audit(binding_mismatch)

    assert binding_result is not None
    assert (
        binding_result.status
        is api.RepublicationVerificationStatus.REQUEST_BINDING_MISMATCH
    )
    assert binding_result.observed_content_hash is None

    class HashMismatchStore:
        """返回同一读取快照中的不同历史摘要。"""

        def get_published_question_verification_snapshot(
            self,
            question_id: str,
            content_version: str,
        ) -> SimpleNamespace:
            assert (question_id, content_version) == ("q-1", "content-v2")
            return SimpleNamespace(
                question=SimpleNamespace(content_hash="b" * 64),
                is_active=True,
            )

    monkeypatch.setattr(api, "question_bank_store", HashMismatchStore())
    hash_result = api._verify_republication_for_audit(
        _correction_audit_with_republication()
    )

    assert hash_result is not None
    assert (
        hash_result.status
        is api.RepublicationVerificationStatus.CONTENT_HASH_MISMATCH
    )
    assert hash_result.observed_content_hash == "b" * 64

    class UnavailableStore:
        """模拟题库独立存储暂时不可读取。"""

        def get_published_question_verification_snapshot(
            self,
            question_id: str,
            content_version: str,
        ) -> None:
            raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(api, "question_bank_store", UnavailableStore())
    unavailable_result = api._verify_republication_for_audit(
        _correction_audit_with_republication()
    )

    assert unavailable_result is not None
    assert unavailable_result.status is api.RepublicationVerificationStatus.UNVERIFIABLE
    assert unavailable_result.observed_content_hash is None


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


def test_admin_correction_audit_routes_return_link_and_complete_event_chain(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """管理员可按申请或新版本回查关联和事件链，学习者不能读取治理审计。"""
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
        json={"record_id": attempt.json()["record_id"], "reason": "请复核题目版本"},
    )
    assert correction_request.status_code == 200
    request_id = correction_request.json()["request_id"]
    assert client.post(
        f"/v1/admin/practice-correction-requests/{request_id}/resolution",
        headers=_ADMIN_HEADERS,
        json={"resolution": "republication_required", "notes": "进入重发布流程"},
    ).status_code == 200
    second_payload = {
        **_publish_payload("q-1", "content-v2"),
        "stem": "q-1 的审计查询新版题干",
    }
    candidate = _prepare_and_approve(client, second_payload)
    assert client.post(
        "/v1/admin/questions",
        headers=_ADMIN_HEADERS,
        json=second_payload,
    ).status_code == 200
    assert client.post(
        "/v1/admin/questions/q-1/content-v2/correction-republication-links",
        headers=_ADMIN_HEADERS,
        json={"request_id": request_id, "previous_content_version": "content-v1"},
    ).status_code == 200

    unauthenticated = client.get("/v1/admin/practice-correction-audits")
    learner = client.get(
        "/v1/admin/practice-correction-audits",
        headers=_LEARNER_HEADERS,
    )
    unauthenticated_reconciliation = client.get(
        "/v1/admin/practice-correction-reconciliations"
    )
    learner_reconciliation = client.get(
        "/v1/admin/practice-correction-reconciliations",
        headers=_LEARNER_HEADERS,
    )
    invalid_reconciliation_offset = client.get(
        "/v1/admin/practice-correction-reconciliations?offset=10001",
        headers=_ADMIN_HEADERS,
    )
    exact = client.get(
        f"/v1/admin/practice-correction-audits/{request_id}",
        headers=_ADMIN_HEADERS,
    )
    by_new_version = client.get(
        "/v1/admin/practice-correction-audits?new_content_version=content-v2",
        headers=_ADMIN_HEADERS,
    )
    linked_only = client.get(
        "/v1/admin/practice-correction-audits?linked=true",
        headers=_ADMIN_HEADERS,
    )
    unlinked_only = client.get(
        "/v1/admin/practice-correction-audits?linked=false",
        headers=_ADMIN_HEADERS,
    )
    malformed_filter = client.get(
        "/v1/admin/practice-correction-audits?question_id=%20",
        headers=_ADMIN_HEADERS,
    )
    missing = client.get(
        "/v1/admin/practice-correction-audits/missing-request",
        headers=_ADMIN_HEADERS,
    )
    profile = client.get("/v1/learning/profile", headers=_LEARNER_HEADERS)

    assert unauthenticated.status_code == 401
    assert learner.status_code == 403
    assert unauthenticated_reconciliation.status_code == 401
    assert learner_reconciliation.status_code == 403
    assert invalid_reconciliation_offset.status_code == 422
    assert exact.status_code == 200
    payload = exact.json()
    assert payload["request"]["request_id"] == request_id
    assert payload["request"]["reason"] == "请复核题目版本"
    assert payload["request"]["resolution_notes"] == "进入重发布流程"
    assert payload["republication"] == {
        "request_id": request_id,
        "question_id": "q-1",
        "previous_content_version": "content-v1",
        "new_content_version": "content-v2",
        "new_content_hash": candidate["content_hash"],
        "linked_by": "admin-a",
        "linked_at": payload["republication"]["linked_at"],
    }
    assert payload["republication_verification"] == {
        "status": "active_verified",
        "observed_content_hash": candidate["content_hash"],
    }
    reconciliation = client.get(
        "/v1/admin/practice-correction-reconciliations",
        headers=_ADMIN_HEADERS,
    )

    assert reconciliation.status_code == 200
    assert reconciliation.json() == {
        "scan_boundary": 1,
        "total_linked_audits": 1,
        "offset": 0,
        "scanned_linked_audits": 1,
        "active_verified_audits": 1,
        "next_offset": None,
        "non_verified_audits": [],
    }
    assert [event["event_type"] for event in payload["events"]] == [
        "requested",
        "resolved",
        "republication_linked",
    ]
    assert [event["actor_id"] for event in payload["events"]] == [
        "user-a",
        "admin-a",
        "admin-a",
    ]
    assert [event["notes"] for event in payload["events"]] == [
        "请复核题目版本",
        "进入重发布流程",
        "content-v2",
    ]
    assert by_new_version.status_code == 200
    assert by_new_version.json() == [payload]
    assert linked_only.status_code == 200
    assert linked_only.json() == [payload]
    assert unlinked_only.status_code == 200
    assert unlinked_only.json() == []
    assert malformed_filter.status_code == 422
    assert malformed_filter.json()["detail"] == "题目标识不能为空"
    assert missing.status_code == 404
    assert missing.json()["detail"] == "未找到复核申请审计"
    assert profile.json()["total_attempts"] == 1
    assert profile.json()["correct_attempts"] == 0


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
