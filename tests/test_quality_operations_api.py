"""基于受信代理身份的审核、质量看板与指标接口测试。"""

from hashlib import sha256
from pathlib import Path

from fastapi.testclient import TestClient

from logic_qa import api
from logic_qa.models import VerificationStatus
from logic_qa.quality_operations import QuestionReviewStore, RuntimeMetrics
from logic_qa.question_bank import (
    OptionAssertion,
    PropositionalFormalization,
    QuestionBankStore,
    QuestionPublicationInput,
)

_PROXY_TOKEN = "test-proxy-token"
_ADMIN_HEADERS = {
    "X-Logic-QA-Proxy-Token": _PROXY_TOKEN,
    "X-Logic-QA-Subject": "reviewer-a",
    "X-Logic-QA-Roles": "logic_qa_admin",
}
_LEARNER_HEADERS = {
    "X-Logic-QA-Proxy-Token": _PROXY_TOKEN,
    "X-Logic-QA-Subject": "learner-a",
    "X-Logic-QA-Roles": "learner",
}


def _content_hash(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _client_with_operations(tmp_path: Path, monkeypatch) -> TestClient:
    """为每个接口测试配置独立审核库、候选库和受信代理密钥。"""
    monkeypatch.setenv("LOGIC_QA_TRUSTED_PROXY_TOKEN", _PROXY_TOKEN)
    review_store = QuestionReviewStore(tmp_path / "reviews.sqlite3")
    question_bank_store = QuestionBankStore(tmp_path / "questions.sqlite3")
    monkeypatch.setattr(api, "review_store", review_store)
    monkeypatch.setattr(api, "question_bank_store", question_bank_store)
    monkeypatch.setattr(api, "runtime_metrics", RuntimeMetrics())
    return TestClient(api.app)


def _submit_candidate_for_review() -> str:
    """向当前注入的题库保存审核所需的候选快照，并返回其摘要。"""
    candidate = api.question_bank_store.submit_candidate(
        QuestionPublicationInput(
            question_id="q-1",
            content_version="content-v1",
            question_type="propositional",
            stem="q-1 的题干",
            options=("A", "B"),
            formalization_version="logic-v1",
            formalization=PropositionalFormalization(
                facts=("A",),
                rules=(),
                query="A",
                expected_status=VerificationStatus.PROVED,
                expected_answer="B",
                option_assertions=(
                    OptionAssertion("A", "disproved"),
                    OptionAssertion("B", "proved"),
                ),
            ),
        )
    )
    return candidate.content_hash


def test_admin_routes_fail_closed_without_identity_provider(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """身份提供方未配置时，管理接口不能回退至客户端令牌。"""
    client = _client_with_operations(tmp_path, monkeypatch)
    monkeypatch.delenv("LOGIC_QA_TRUSTED_PROXY_TOKEN")

    response = client.get("/v1/admin/review-dashboard", headers=_ADMIN_HEADERS)

    assert response.status_code == 503
    assert response.json()["detail"] == "身份提供方未配置"


def test_admin_routes_reject_non_admin_identity(tmp_path: Path, monkeypatch) -> None:
    """认证但无管理角色的主体不得访问审核和指标接口。"""
    client = _client_with_operations(tmp_path, monkeypatch)

    response = client.get("/v1/admin/review-dashboard", headers=_LEARNER_HEADERS)

    assert response.status_code == 403
    assert response.json()["detail"] == "管理员访问未授权"


def test_review_api_derives_reviewer_and_protects_dashboard_and_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """审核人取自认证主体，管理接口仅返回必要聚合信息。"""
    client = _client_with_operations(tmp_path, monkeypatch)
    content_hash = _submit_candidate_for_review()
    review_response = client.post(
        "/v1/admin/question-reviews",
        headers=_ADMIN_HEADERS,
        json={
            "question_id": "q-1",
            "content_version": "content-v1",
            "content_hash": content_hash,
            "reviewer_id": "spoofed-reviewer",
            "status": "approved",
            "verified_answer": "B",
            "formalization_version": "logic-v1",
        },
    )

    assert review_response.status_code == 422
    assert review_response.json()["detail"][0]["loc"] == ["body", "reviewer_id"]

    valid_review_response = client.post(
        "/v1/admin/question-reviews",
        headers=_ADMIN_HEADERS,
        json={
            "question_id": "q-1",
            "content_version": "content-v1",
            "content_hash": content_hash,
            "status": "approved",
            "verified_answer": "B",
            "formalization_version": "logic-v1",
        },
    )
    assert valid_review_response.status_code == 200
    assert valid_review_response.json()["status"] == "approved"
    assert valid_review_response.json()["reviewer_id"] == "reviewer-a"

    dashboard_response = client.get(
        "/v1/admin/review-dashboard",
        headers=_ADMIN_HEADERS,
    )
    metrics_response = client.get(
        "/v1/admin/runtime-metrics",
        headers=_ADMIN_HEADERS,
    )

    assert dashboard_response.status_code == 200
    dashboard = dashboard_response.json()
    assert dashboard["total_questions"] == 1
    assert dashboard["status_counts"] == [["approved", 1]]
    assert dashboard["total_review_events"] == 1

    assert metrics_response.status_code == 200
    metrics = metrics_response.json()
    assert metrics["total_requests"] >= 3
    assert metrics["route_counts"]
    assert "request_body" not in metrics
