"""内容绑定审核存储与非敏感运行指标的回归测试。"""

from hashlib import sha256
from pathlib import Path

import pytest

from logic_qa.quality_operations import (
    QuestionReviewInput,
    QuestionReviewStatus,
    QuestionReviewStore,
    RuntimeMetrics,
)


def _content_hash(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _store(tmp_path: Path) -> QuestionReviewStore:
    return QuestionReviewStore(tmp_path / "reviews.sqlite3")


def test_review_store_requires_answer_for_approved_content(tmp_path: Path) -> None:
    """通过审核必须包含该候选内容对应的已核验答案。"""
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="通过审核必须填写"):
        store.upsert_review(
            QuestionReviewInput(
                question_id="q-1",
                content_version="content-v1",
                content_hash=_content_hash("candidate"),
                reviewer_id="reviewer-a",
                status=QuestionReviewStatus.APPROVED,
                verified_answer=None,
                formalization_version="logic-v1",
            )
        )


def test_review_store_binds_review_to_version_and_content_hash(tmp_path: Path) -> None:
    """同题不同内容摘要必须拥有独立审核状态，不能相互覆盖。"""
    store = _store(tmp_path)
    first_hash = _content_hash("candidate-one")
    second_hash = _content_hash("candidate-two")
    first = store.upsert_review(
        QuestionReviewInput(
            question_id="q-1",
            content_version="content-v1",
            content_hash=first_hash,
            reviewer_id="reviewer-a",
            status=QuestionReviewStatus.APPROVED,
            verified_answer="B",
            formalization_version="logic-v1",
        )
    )
    second = store.upsert_review(
        QuestionReviewInput(
            question_id="q-1",
            content_version="content-v1",
            content_hash=second_hash,
            reviewer_id="reviewer-b",
            status=QuestionReviewStatus.NEEDS_REVISION,
            verified_answer=None,
            formalization_version="logic-v1",
        )
    )

    assert first.content_hash == first_hash
    assert second.content_hash == second_hash
    assert store.get_review("q-1", "content-v1", first_hash) == first
    assert store.get_review("q-1", "content-v1", second_hash) == second
    assert store.dashboard().total_questions == 1
    assert store.dashboard().total_review_events == 2


def test_review_store_archives_legacy_unbound_schema(tmp_path: Path) -> None:
    """旧审核表缺少摘要绑定时应归档，而不能被新发布流程继续使用。"""
    database_path = tmp_path / "legacy-reviews.sqlite3"
    import sqlite3

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE question_reviews (
                question_id TEXT PRIMARY KEY,
                reviewer_id TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE question_review_events (
                event_id TEXT PRIMARY KEY,
                question_id TEXT NOT NULL
            )
            """
        )

    store = QuestionReviewStore(database_path)
    review = store.upsert_review(
        QuestionReviewInput(
            question_id="q-1",
            content_version="content-v1",
            content_hash=_content_hash("candidate"),
            reviewer_id="reviewer-a",
            status=QuestionReviewStatus.PENDING,
            verified_answer=None,
            formalization_version="logic-v1",
        )
    )

    assert review.status is QuestionReviewStatus.PENDING


def test_runtime_metrics_aggregate_without_request_payloads() -> None:
    """运行指标只按路由与状态码聚合，并计算时延均值。"""
    metrics = RuntimeMetrics()
    metrics.record("/health", status_code=200, latency_ms=10)
    metrics.record("/health", status_code=200, latency_ms=20)
    metrics.record("/v1/questions/solve", status_code=422, latency_ms=30)

    snapshot = metrics.snapshot()

    assert snapshot.total_requests == 3
    assert snapshot.error_requests == 1
    assert snapshot.average_latency_ms == 20.0
    assert snapshot.route_counts == (("/health", 2), ("/v1/questions/solve", 1))
    assert snapshot.status_counts == (("200", 2), ("422", 1))


def test_review_store_restores_review_and_audit_snapshot(tmp_path: Path) -> None:
    """恢复审核库时应同时回退当前状态与追加式审核审计事件。"""
    store = _store(tmp_path)
    first_hash = _content_hash("first-candidate")
    store.upsert_review(
        QuestionReviewInput(
            question_id="q-1",
            content_version="content-v1",
            content_hash=first_hash,
            reviewer_id="reviewer-a",
            status=QuestionReviewStatus.PENDING,
            verified_answer=None,
            formalization_version="logic-v1",
        )
    )
    backup = store.create_backup(tmp_path / "backups")
    store.upsert_review(
        QuestionReviewInput(
            question_id="q-2",
            content_version="content-v1",
            content_hash=_content_hash("second-candidate"),
            reviewer_id="reviewer-b",
            status=QuestionReviewStatus.NEEDS_REVISION,
            verified_answer=None,
            formalization_version="logic-v1",
        )
    )

    store.restore_backup(store.load_backup(backup.manifest_path))

    assert store.schema_version() == 1
    assert store.get_review("q-1", "content-v1", first_hash) is not None
    assert store.dashboard().total_questions == 1
    assert store.dashboard().total_review_events == 1
