"""最小化学习档案、用户隔离和练习方向的回归测试。"""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from logic_qa.learning_profile import (
    DuplicatePracticeAttemptError,
    DuplicatePracticeCorrectionRequestError,
    ImmutablePracticeAttemptError,
    LearningProfileStore,
    LearningRecordInput,
    PracticeCorrectionOutcomeKind,
    PracticeCorrectionRepublicationAlreadyLinkedError,
    PracticeCorrectionRepublicationNotEligibleError,
    PracticeCorrectionRepublicationVersionAlreadyLinkedError,
    PracticeCorrectionRequestAlreadyResolvedError,
    PracticeCorrectionRequestInput,
    PracticeCorrectionResolution,
    PracticeCorrectionResolutionInput,
)


def _store(tmp_path: Path) -> LearningProfileStore:
    return LearningProfileStore(tmp_path / "learning.sqlite3")


def test_records_are_isolated_by_user_and_profile_uses_own_statistics(
    tmp_path: Path,
) -> None:
    """不同用户的作答和错因统计不得相互影响。"""
    store = _store(tmp_path)
    store.add_record(
        LearningRecordInput(
            user_id="user-a",
            question_id="q-1",
            question_type="propositional",
            is_correct=False,
            error_tags=("invalid_converse",),
            knowledge_tags=("逆命题与逆否命题",),
            duration_seconds=30,
        )
    )
    store.add_record(
        LearningRecordInput(
            user_id="user-b",
            question_id="q-2",
            question_type="ordering",
            is_correct=True,
            knowledge_tags=("排序约束",),
            duration_seconds=20,
        )
    )

    profile = store.get_profile("user-a")

    assert profile.total_attempts == 1
    assert profile.correct_attempts == 0
    assert profile.accuracy == 0.0
    assert profile.error_counts == (("invalid_converse", 1),)
    assert profile.knowledge_mastery == (("逆命题与逆否命题", 0.0),)
    assert profile.recommendations[0].label == "invalid_converse"


def test_deleting_record_requires_matching_user_and_updates_profile(
    tmp_path: Path,
) -> None:
    """删除必须同时匹配用户和记录编号，并即时更新该用户的画像。"""
    store = _store(tmp_path)
    record = store.add_record(
        LearningRecordInput(
            user_id="user-a",
            question_id="q-1",
            question_type="propositional",
            is_correct=False,
            error_tags=("premise_not_established",),
            knowledge_tags=("前提核验",),
        )
    )

    assert store.delete_record("user-b", record.record_id) is False
    assert store.get_profile("user-a").total_attempts == 1
    assert store.delete_record("user-a", record.record_id) is True
    assert store.get_profile("user-a").total_attempts == 0
    assert store.get_profile("user-a").recommendations == ()


def test_deleting_versioned_practice_record_is_rejected(tmp_path: Path) -> None:
    """发布题练习记录不可删除，避免重新打开已完成版本的推荐入口。"""
    store = _store(tmp_path)
    record = store.record_practice_attempt(
        LearningRecordInput(
            user_id="user-a",
            question_id="q-1",
            content_version="content-v1",
            question_type="propositional",
            is_correct=True,
        )
    )

    with pytest.raises(ImmutablePracticeAttemptError, match="练习记录不可删除"):
        store.delete_record("user-a", record.record_id)

    assert store.get_profile("user-a").total_attempts == 1
    assert store.attempted_practice_versions("user-a") == (
        ("q-1", "content-v1"),
    )


def test_profile_recommends_lowest_mastery_knowledge_tag(tmp_path: Path) -> None:
    """学习建议应以用户自身的低掌握度知识点为依据。"""
    store = _store(tmp_path)
    store.add_record(
        LearningRecordInput(
            user_id="user-a",
            question_id="q-1",
            question_type="propositional",
            is_correct=True,
            knowledge_tags=("条件推理",),
        )
    )
    store.add_record(
        LearningRecordInput(
            user_id="user-a",
            question_id="q-2",
            question_type="propositional",
            is_correct=False,
            knowledge_tags=("逆命题与逆否命题",),
        )
    )

    profile = store.get_profile("user-a")

    assert profile.knowledge_mastery == (("条件推理", 1.0), ("逆命题与逆否命题", 0.0))
    assert profile.recommendations[0].focus_type == "knowledge_tag"
    assert profile.recommendations[0].label == "逆命题与逆否命题"


def test_record_normalizes_duplicate_tags_and_rejects_invalid_duration(
    tmp_path: Path,
) -> None:
    """持久化前应去重标签并拒绝不合法的最小记录字段。"""
    store = _store(tmp_path)
    record = store.add_record(
        LearningRecordInput(
            user_id=" user-a ",
            question_id=" q-1 ",
            question_type=" propositional ",
            is_correct=True,
            error_tags=("tag-a", "tag-a", " "),
            knowledge_tags=("tag-b", "tag-b"),
        )
    )

    assert record.user_id == "user-a"
    assert record.error_tags == ("tag-a",)
    assert record.knowledge_tags == ("tag-b",)


def test_learning_store_migrates_once_and_restores_manifest_backup(
    tmp_path: Path,
) -> None:
    """重复初始化不重放迁移，且清单可在新存储实例中恢复一致快照。"""
    database_path = tmp_path / "learning.sqlite3"
    store = LearningProfileStore(database_path)
    first = store.add_record(
        LearningRecordInput(
            user_id="user-a",
            question_id="q-1",
            question_type="propositional",
            is_correct=True,
        )
    )
    backup = store.create_backup(tmp_path / "backups")
    reloaded_store = LearningProfileStore(database_path)
    reloaded_backup = reloaded_store.load_backup(backup.manifest_path)
    reloaded_store.add_record(
        LearningRecordInput(
            user_id="user-a",
            question_id="q-2",
            question_type="ordering",
            is_correct=False,
        )
    )

    reloaded_store.restore_backup(reloaded_backup)

    profile = reloaded_store.get_profile("user-a")
    assert reloaded_store.schema_version() == 4
    assert backup.manifest_path.is_file()
    assert profile.total_attempts == 1
    assert reloaded_store.attempted_question_ids("user-a") == (first.question_id,)


def test_learning_store_upgrades_v1_records_to_versioned_practice_schema(
    tmp_path: Path,
) -> None:
    """既有 v1 通用记录升级后保持可读且不阻断版本化练习。"""
    database_path = tmp_path / "learning.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO schema_migrations (version, name, applied_at)
            VALUES (1, 'create_learning_records', '2026-08-29T00:00:00+00:00')
            """
        )
        connection.execute(
            """
            CREATE TABLE learning_records (
                record_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                question_type TEXT NOT NULL,
                is_correct INTEGER NOT NULL CHECK (is_correct IN (0, 1)),
                error_tags TEXT NOT NULL,
                knowledge_tags TEXT NOT NULL,
                duration_seconds INTEGER,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO learning_records (
                record_id, user_id, question_id, question_type, is_correct,
                error_tags, knowledge_tags, duration_seconds, created_at
            ) VALUES ('legacy-record', 'user-a', 'q-1', 'propositional', 1,
                      '', '', NULL, '2026-08-29T00:00:00+00:00')
            """
        )

    store = LearningProfileStore(database_path)
    versioned_record = store.record_practice_attempt(
        LearningRecordInput(
            user_id="user-a",
            question_id="q-1",
            content_version="content-v1",
            question_type="propositional",
            is_correct=False,
        )
    )

    assert store.schema_version() == 4
    assert store.get_profile("user-a").total_attempts == 2
    assert versioned_record.content_version == "content-v1"
    assert store.attempted_practice_versions("user-a") == (
        ("q-1", "content-v1"),
    )


def test_versioned_practice_attempts_are_atomic_and_allow_new_versions(
    tmp_path: Path,
) -> None:
    """同一用户同一版本只可写入一次，新发布版本可独立完成练习。"""
    store = _store(tmp_path)
    first = store.record_practice_attempt(
        LearningRecordInput(
            user_id="user-a",
            question_id="q-1",
            content_version="content-v1",
            question_type="propositional",
            is_correct=False,
        )
    )

    with pytest.raises(DuplicatePracticeAttemptError, match="题目版本已完成"):
        store.record_practice_attempt(
            LearningRecordInput(
                user_id="user-a",
                question_id="q-1",
                content_version="content-v1",
                question_type="propositional",
                is_correct=True,
            )
        )

    second = store.record_practice_attempt(
        LearningRecordInput(
            user_id="user-a",
            question_id="q-1",
            content_version="content-v2",
            question_type="propositional",
            is_correct=True,
        )
    )

    assert first.content_version == "content-v1"
    assert second.content_version == "content-v2"
    assert store.attempted_practice_versions("user-a") == (
        ("q-1", "content-v1"),
        ("q-1", "content-v2"),
    )
    assert store.get_profile("user-a").total_attempts == 2


def test_versioned_practice_attempt_is_atomic_under_concurrent_submissions(
    tmp_path: Path,
) -> None:
    """并发提交相同发布版本时，唯一索引只允许一条学习记录落库。"""
    store = _store(tmp_path)

    def submit_attempt() -> str:
        try:
            record = store.record_practice_attempt(
                LearningRecordInput(
                    user_id="user-a",
                    question_id="q-1",
                    content_version="content-v1",
                    question_type="propositional",
                    is_correct=True,
                )
            )
        except DuplicatePracticeAttemptError:
            return "duplicate"
        return record.record_id

    with ThreadPoolExecutor(max_workers=4) as executor:
        outcomes = tuple(executor.map(lambda _: submit_attempt(), range(4)))

    assert sum(outcome != "duplicate" for outcome in outcomes) == 1
    assert outcomes.count("duplicate") == 3
    assert store.get_profile("user-a").total_attempts == 1


def test_practice_correction_request_is_atomic_under_concurrent_submissions(
    tmp_path: Path,
) -> None:
    """同一不可变练习记录的并发复核申请只能成功落一条。"""
    store = _store(tmp_path)
    record = store.record_practice_attempt(
        LearningRecordInput(
            user_id="user-a",
            question_id="q-1",
            content_version="content-v1",
            question_type="propositional",
            is_correct=False,
        )
    )

    def submit_request() -> str:
        try:
            request = store.create_practice_correction_request(
                PracticeCorrectionRequestInput(
                    user_id="user-a",
                    record_id=record.record_id,
                    reason="请复核该版本的判分依据",
                )
            )
        except DuplicatePracticeCorrectionRequestError:
            return "duplicate"
        assert request is not None
        return request.request_id

    with ThreadPoolExecutor(max_workers=4) as executor:
        outcomes = tuple(executor.map(lambda _: submit_request(), range(4)))

    assert sum(outcome != "duplicate" for outcome in outcomes) == 1
    assert outcomes.count("duplicate") == 3
    assert len(store.list_practice_correction_requests_for_user("user-a")) == 1
    with sqlite3.connect(tmp_path / "learning.sqlite3") as connection:
        event_count = connection.execute(
            "SELECT COUNT(*) FROM practice_correction_events"
        ).fetchone()[0]
    assert event_count == 1


def test_practice_correction_request_preserves_attempt_ledger_and_audit_history(
    tmp_path: Path,
) -> None:
    """更正申请只能追加治理记录，绝不修改原始练习账本或首次判分。"""
    store = _store(tmp_path)
    practice_record = store.record_practice_attempt(
        LearningRecordInput(
            user_id="user-a",
            question_id="q-1",
            content_version="content-v1",
            question_type="propositional",
            is_correct=False,
            error_tags=("invalid_converse",),
            knowledge_tags=("逆命题与逆否命题",),
        )
    )
    general_record = store.add_record(
        LearningRecordInput(
            user_id="user-a",
            question_id="note-1",
            question_type="reflection",
            is_correct=True,
        )
    )

    with pytest.raises(ValueError, match="仅已发布题目的练习记录"):
        store.create_practice_correction_request(
            PracticeCorrectionRequestInput(
                user_id="user-a",
                record_id=general_record.record_id,
                reason="希望复核",
            )
        )

    requested = store.create_practice_correction_request(
        PracticeCorrectionRequestInput(
            user_id="user-a",
            record_id=practice_record.record_id,
            reason="请复核该版本的判分依据",
        )
    )

    assert requested is not None
    assert requested.record_id == practice_record.record_id
    assert requested.question_id == "q-1"
    assert requested.content_version == "content-v1"
    assert requested.status.value == "pending"
    assert requested.resolved_by is None
    assert store.list_practice_correction_requests_for_user("user-a") == (requested,)

    with pytest.raises(DuplicatePracticeCorrectionRequestError, match="已提交复核申请"):
        store.create_practice_correction_request(
            PracticeCorrectionRequestInput(
                user_id="user-a",
                record_id=practice_record.record_id,
                reason="再次提交",
            )
        )

    resolved = store.resolve_practice_correction_request(
        PracticeCorrectionResolutionInput(
            request_id=requested.request_id,
            resolver_id="admin-a",
            resolution=PracticeCorrectionResolution.REPUBLICATION_REQUIRED,
            notes="当前作答账本保持不变，题目将按发布流程复核。",
        )
    )

    assert resolved is not None
    assert resolved.status.value == "republication_required"
    assert resolved.resolved_by == "admin-a"
    assert resolved.resolution_notes == "当前作答账本保持不变，题目将按发布流程复核。"
    assert store.get_profile("user-a").total_attempts == 2
    assert store.get_profile("user-a").correct_attempts == 1
    assert store.attempted_practice_versions("user-a") == (("q-1", "content-v1"),)

    with pytest.raises(
        PracticeCorrectionRequestAlreadyResolvedError,
        match="已完成处置",
    ):
        store.resolve_practice_correction_request(
            PracticeCorrectionResolutionInput(
                request_id=requested.request_id,
                resolver_id="admin-b",
                resolution=PracticeCorrectionResolution.RECORD_CONFIRMED,
            )
        )

    with sqlite3.connect(tmp_path / "learning.sqlite3") as connection:
        events = connection.execute(
            """
            SELECT event_type, status
            FROM practice_correction_events
            ORDER BY rowid ASC
            """
        ).fetchall()

    assert events == [
        ("requested", "pending"),
        ("resolved", "republication_required"),
    ]


def test_practice_correction_outcomes_are_safe_derived_views(tmp_path: Path) -> None:
    """派生处置视图只由申请状态生成，不能回写账本或暴露治理内部信息。"""
    store = _store(tmp_path)
    record = store.record_practice_attempt(
        LearningRecordInput(
            user_id="user-a",
            question_id="q-1",
            content_version="content-v1",
            question_type="propositional",
            is_correct=False,
        )
    )
    requested = store.create_practice_correction_request(
        PracticeCorrectionRequestInput(
            user_id="user-a",
            record_id=record.record_id,
            reason="请复核判分依据",
        )
    )
    assert requested is not None

    pending = store.list_practice_correction_outcomes_for_user("user-a")

    assert len(pending) == 1
    assert pending[0].kind is PracticeCorrectionOutcomeKind.PENDING
    assert pending[0].message == "复核申请已提交，正在处理中。"
    assert store.get_profile("user-a").correct_attempts == 0
    assert store.attempted_practice_versions("user-a") == (("q-1", "content-v1"),)

    resolved = store.resolve_practice_correction_request(
        PracticeCorrectionResolutionInput(
            request_id=requested.request_id,
            resolver_id="admin-a",
            resolution=PracticeCorrectionResolution.REPUBLICATION_REQUIRED,
            notes="需要更新形式化资产并重新发布。",
        )
    )
    assert resolved is not None

    outcomes = store.list_practice_correction_outcomes_for_user("user-a")

    assert len(outcomes) == 1
    assert outcomes[0].kind is PracticeCorrectionOutcomeKind.REPUBLICATION_REQUIRED
    assert outcomes[0].message == (
        "复核已完成，该题目将按发布流程复核；若发布新版本，"
        "新版本会作为独立练习重新推荐。"
    )
    assert outcomes[0].resolved_at == resolved.resolved_at
    assert not hasattr(outcomes[0], "reason")
    assert not hasattr(outcomes[0], "resolved_by")
    assert not hasattr(outcomes[0], "resolution_notes")
    assert store.get_profile("user-a").total_attempts == 1
    assert store.get_profile("user-a").correct_attempts == 0
    assert store.attempted_practice_versions("user-a") == (("q-1", "content-v1"),)
    assert store.list_practice_correction_outcomes_for_user("user-b") == ()


def test_record_confirmed_outcome_preserves_original_attempt(tmp_path: Path) -> None:
    """确认原记录的派生视图不能将管理员结论误表述为改分或重新作答。"""
    store = _store(tmp_path)
    record = store.record_practice_attempt(
        LearningRecordInput(
            user_id="user-a",
            question_id="q-1",
            content_version="content-v1",
            question_type="propositional",
            is_correct=False,
        )
    )
    requested = store.create_practice_correction_request(
        PracticeCorrectionRequestInput(
            user_id="user-a",
            record_id=record.record_id,
            reason="请确认判分记录",
        )
    )
    assert requested is not None
    resolved = store.resolve_practice_correction_request(
        PracticeCorrectionResolutionInput(
            request_id=requested.request_id,
            resolver_id="admin-a",
            resolution=PracticeCorrectionResolution.RECORD_CONFIRMED,
        )
    )
    assert resolved is not None

    outcomes = store.list_practice_correction_outcomes_for_user("user-a")

    assert outcomes[0].kind is PracticeCorrectionOutcomeKind.RECORD_CONFIRMED
    assert outcomes[0].message == "复核已完成，当前练习记录已确认。"
    assert outcomes[0].resolved_at == resolved.resolved_at
    assert store.get_profile("user-a").total_attempts == 1
    assert store.get_profile("user-a").correct_attempts == 0
    assert store.attempted_practice_versions("user-a") == (("q-1", "content-v1"),)


def test_correction_republication_link_is_immutable_and_updates_only_projection(
    tmp_path: Path,
) -> None:
    """关联已发布新版本只追加审计，不重写旧练习账本或原始处置。"""
    store = _store(tmp_path)
    record = store.record_practice_attempt(
        LearningRecordInput(
            user_id="user-a",
            question_id="q-1",
            content_version="content-v1",
            question_type="propositional",
            is_correct=False,
        )
    )
    requested = store.create_practice_correction_request(
        PracticeCorrectionRequestInput(
            user_id="user-a",
            record_id=record.record_id,
            reason="请复核题目版本",
        )
    )
    assert requested is not None
    resolved = store.resolve_practice_correction_request(
        PracticeCorrectionResolutionInput(
            request_id=requested.request_id,
            resolver_id="admin-a",
            resolution=PracticeCorrectionResolution.REPUBLICATION_REQUIRED,
        )
    )
    assert resolved is not None
    new_content_hash = "a" * 64

    linked = store.link_practice_correction_republication(
        request_id=requested.request_id,
        question_id="q-1",
        previous_content_version="content-v1",
        new_content_version="content-v2",
        new_content_hash=new_content_hash,
        linked_by="admin-b",
    )

    assert linked is not None
    assert linked.previous_content_version == "content-v1"
    assert linked.new_content_version == "content-v2"
    assert linked.new_content_hash == new_content_hash
    outcomes = store.list_practice_correction_outcomes_for_user("user-a")
    assert outcomes[0].kind is PracticeCorrectionOutcomeKind.REPUBLICATION_REQUIRED
    assert outcomes[0].message == "复核后的新版本已发布，可作为独立练习完成。"
    assert outcomes[0].republished_content_version == "content-v2"
    assert store.get_profile("user-a").total_attempts == 1
    assert store.get_profile("user-a").correct_attempts == 0
    assert store.attempted_practice_versions("user-a") == (("q-1", "content-v1"),)

    with pytest.raises(
        PracticeCorrectionRepublicationAlreadyLinkedError,
        match="已关联新的发布版本",
    ):
        store.link_practice_correction_republication(
            request_id=requested.request_id,
            question_id="q-1",
            previous_content_version="content-v1",
            new_content_version="content-v3",
            new_content_hash="b" * 64,
            linked_by="admin-c",
        )

    with sqlite3.connect(tmp_path / "learning.sqlite3") as connection:
        events = connection.execute(
            """
            SELECT event_type, status, notes
            FROM practice_correction_events
            ORDER BY rowid ASC
            """
        ).fetchall()
    assert events == [
        ("requested", "pending", "请复核题目版本"),
        ("resolved", "republication_required", None),
        ("republication_linked", "republication_required", "content-v2"),
    ]


def test_correction_republication_link_cannot_reuse_new_version(
    tmp_path: Path,
) -> None:
    """同题的新发布版本只能审计关联到一条更正申请。"""
    store = _store(tmp_path)
    request_ids = []
    for user_id in ("user-a", "user-b"):
        record = store.record_practice_attempt(
            LearningRecordInput(
                user_id=user_id,
                question_id="q-1",
                content_version="content-v1",
                question_type="propositional",
                is_correct=False,
            )
        )
        requested = store.create_practice_correction_request(
            PracticeCorrectionRequestInput(
                user_id=user_id,
                record_id=record.record_id,
                reason="请复核题目内容",
            )
        )
        assert requested is not None
        store.resolve_practice_correction_request(
            PracticeCorrectionResolutionInput(
                request_id=requested.request_id,
                resolver_id="admin-a",
                resolution=PracticeCorrectionResolution.REPUBLICATION_REQUIRED,
            )
        )
        request_ids.append(requested.request_id)

    store.link_practice_correction_republication(
        request_id=request_ids[0],
        question_id="q-1",
        previous_content_version="content-v1",
        new_content_version="content-v2",
        new_content_hash="a" * 64,
        linked_by="admin-b",
    )

    with pytest.raises(
        PracticeCorrectionRepublicationVersionAlreadyLinkedError,
        match="已关联另一复核申请",
    ):
        store.link_practice_correction_republication(
            request_id=request_ids[1],
            question_id="q-1",
            previous_content_version="content-v1",
            new_content_version="content-v2",
            new_content_hash="a" * 64,
            linked_by="admin-b",
        )


def test_correction_republication_link_requires_matching_resolved_request(
    tmp_path: Path,
) -> None:
    """非重发布结论、错误旧版本或同版本关联均不能写入关联审计。"""
    store = _store(tmp_path)
    record = store.record_practice_attempt(
        LearningRecordInput(
            user_id="user-a",
            question_id="q-1",
            content_version="content-v1",
            question_type="propositional",
            is_correct=False,
        )
    )
    requested = store.create_practice_correction_request(
        PracticeCorrectionRequestInput(
            user_id="user-a",
            record_id=record.record_id,
            reason="请确认判分",
        )
    )
    assert requested is not None
    confirmed = store.resolve_practice_correction_request(
        PracticeCorrectionResolutionInput(
            request_id=requested.request_id,
            resolver_id="admin-a",
            resolution=PracticeCorrectionResolution.RECORD_CONFIRMED,
        )
    )
    assert confirmed is not None

    with pytest.raises(
        PracticeCorrectionRepublicationNotEligibleError,
        match="不需要重新发布",
    ):
        store.link_practice_correction_republication(
            request_id=requested.request_id,
            question_id="q-1",
            previous_content_version="content-v1",
            new_content_version="content-v2",
            new_content_hash="a" * 64,
            linked_by="admin-b",
        )

    republish_record = store.record_practice_attempt(
        LearningRecordInput(
            user_id="user-a",
            question_id="q-2",
            content_version="content-v1",
            question_type="propositional",
            is_correct=False,
        )
    )
    republish_request = store.create_practice_correction_request(
        PracticeCorrectionRequestInput(
            user_id="user-a",
            record_id=republish_record.record_id,
            reason="请复核题目内容",
        )
    )
    assert republish_request is not None
    store.resolve_practice_correction_request(
        PracticeCorrectionResolutionInput(
            request_id=republish_request.request_id,
            resolver_id="admin-a",
            resolution=PracticeCorrectionResolution.REPUBLICATION_REQUIRED,
        )
    )

    with pytest.raises(ValueError, match="复核申请与待关联题目版本不一致"):
        store.link_practice_correction_republication(
            request_id=republish_request.request_id,
            question_id="q-2",
            previous_content_version="content-v0",
            new_content_version="content-v2",
            new_content_hash="a" * 64,
            linked_by="admin-b",
        )
    with pytest.raises(ValueError, match="必须关联新的内容版本"):
        store.link_practice_correction_republication(
            request_id=republish_request.request_id,
            question_id="q-2",
            previous_content_version="content-v1",
            new_content_version="content-v1",
            new_content_hash="a" * 64,
            linked_by="admin-b",
        )


def test_learning_store_upgrades_v3_correction_schema_to_republication_schema(
    tmp_path: Path,
) -> None:
    """已有更正申请账本升级只添加关联表，不回写历史申请或练习记录。"""
    database_path = tmp_path / "learning.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO schema_migrations (version, name, applied_at)
            VALUES (?, ?, '2026-08-29T00:00:00+00:00')
            """,
            [
                (1, "create_learning_records"),
                (2, "add_versioned_practice_attempts"),
                (3, "add_practice_correction_requests"),
            ],
        )
        connection.execute(
            """
            CREATE TABLE learning_records (
                record_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                question_type TEXT NOT NULL,
                is_correct INTEGER NOT NULL CHECK (is_correct IN (0, 1)),
                error_tags TEXT NOT NULL,
                knowledge_tags TEXT NOT NULL,
                duration_seconds INTEGER,
                created_at TEXT NOT NULL,
                content_version TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE practice_correction_requests (
                request_id TEXT PRIMARY KEY,
                record_id TEXT NOT NULL UNIQUE,
                user_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                content_version TEXT NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                resolved_by TEXT,
                resolution_notes TEXT,
                resolved_at TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE practice_correction_events (
                event_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                status TEXT NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO learning_records (
                record_id, user_id, question_id, question_type, is_correct,
                error_tags, knowledge_tags, duration_seconds, created_at,
                content_version
            ) VALUES ('attempt-v1', 'user-a', 'q-1', 'propositional', 0,
                      'invalid_converse', 'logic', NULL,
                      '2026-08-29T00:00:00+00:00', 'content-v1')
            """
        )
        connection.execute(
            """
            INSERT INTO practice_correction_requests (
                request_id, record_id, user_id, question_id, content_version,
                reason, status, created_at, resolved_by, resolution_notes, resolved_at
            ) VALUES ('request-v1', 'attempt-v1', 'user-a', 'q-1', 'content-v1',
                      '请复核历史练习记录', 'republication_required',
                      '2026-08-29T00:00:00+00:00', 'admin-a', NULL,
                      '2026-08-29T01:00:00+00:00')
            """
        )

    store = LearningProfileStore(database_path)

    assert store.schema_version() == 4
    assert store.get_profile("user-a").total_attempts == 1
    assert store.list_practice_correction_requests_for_user("user-a")[0].request_id == (
        "request-v1"
    )
    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(practice_correction_republications)"
            ).fetchall()
        }
    assert columns == {
        "request_id",
        "question_id",
        "previous_content_version",
        "new_content_version",
        "new_content_hash",
        "linked_by",
        "linked_at",
    }


def test_learning_store_upgrades_v2_ledger_to_correction_request_schema(
    tmp_path: Path,
) -> None:
    """已有版本化练习账本升级时，只添加申请表，不回写历史首次作答。"""
    database_path = tmp_path / "learning.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO schema_migrations (version, name, applied_at)
            VALUES (?, ?, '2026-08-29T00:00:00+00:00')
            """,
            [
                (1, "create_learning_records"),
                (2, "add_versioned_practice_attempts"),
            ],
        )
        connection.execute(
            """
            CREATE TABLE learning_records (
                record_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                question_type TEXT NOT NULL,
                is_correct INTEGER NOT NULL CHECK (is_correct IN (0, 1)),
                error_tags TEXT NOT NULL,
                knowledge_tags TEXT NOT NULL,
                duration_seconds INTEGER,
                created_at TEXT NOT NULL,
                content_version TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO learning_records (
                record_id, user_id, question_id, question_type, is_correct,
                error_tags, knowledge_tags, duration_seconds, created_at,
                content_version
            ) VALUES ('attempt-v1', 'user-a', 'q-1', 'propositional', 0,
                      'invalid_converse', 'logic', NULL,
                      '2026-08-29T00:00:00+00:00', 'content-v1')
            """
        )

    store = LearningProfileStore(database_path)
    request = store.create_practice_correction_request(
        PracticeCorrectionRequestInput(
            user_id="user-a",
            record_id="attempt-v1",
            reason="请复核历史练习记录",
        )
    )

    assert store.schema_version() == 4
    assert request is not None
    assert request.status.value == "pending"
    assert store.get_profile("user-a").total_attempts == 1
    assert store.get_profile("user-a").correct_attempts == 0
    assert store.attempted_practice_versions("user-a") == (("q-1", "content-v1"),)


def test_practice_correction_audit_query_returns_immutable_link_and_event_chain(
    tmp_path: Path,
) -> None:
    """管理员审计查询可按申请和版本回查完整事件链，不改写任何账本。"""
    store = _store(tmp_path)
    request_ids = []
    for question_id, user_id in (("q-1", "user-a"), ("q-2", "user-b")):
        record = store.record_practice_attempt(
            LearningRecordInput(
                user_id=user_id,
                question_id=question_id,
                content_version="content-v1",
                question_type="propositional",
                is_correct=False,
            )
        )
        correction_request = store.create_practice_correction_request(
            PracticeCorrectionRequestInput(
                user_id=user_id,
                record_id=record.record_id,
                reason=f"请复核 {question_id}",
            )
        )
        assert correction_request is not None
        resolved = store.resolve_practice_correction_request(
            PracticeCorrectionResolutionInput(
                request_id=correction_request.request_id,
                resolver_id="admin-a",
                resolution=PracticeCorrectionResolution.REPUBLICATION_REQUIRED,
            )
        )
        assert resolved is not None
        request_ids.append(correction_request.request_id)

    linked = store.link_practice_correction_republication(
        request_id=request_ids[0],
        question_id="q-1",
        previous_content_version="content-v1",
        new_content_version="content-v2",
        new_content_hash="a" * 64,
        linked_by="admin-b",
    )
    assert linked is not None

    exact = store.get_practice_correction_audit(request_ids[0])
    linked_only = store.list_practice_correction_audits(linked=True)
    unlinked_only = store.list_practice_correction_audits(linked=False)
    linked_page = store.list_practice_correction_audits(linked=True, limit=1)
    linked_audit_page = store.list_linked_practice_correction_audit_page(limit=1)
    empty_linked_page = store.list_practice_correction_audits(
        linked=True,
        limit=1,
        offset=1,
    )
    empty_linked_audit_page = store.list_linked_practice_correction_audit_page(
        limit=1,
        offset=1,
    )
    by_new_version = store.list_practice_correction_audits(
        new_content_version="content-v2"
    )
    by_old_version = store.list_practice_correction_audits(
        question_id="q-2",
        content_version="content-v1",
    )

    assert exact is not None
    assert exact.request.request_id == request_ids[0]
    assert exact.republication == linked
    assert [event.event_type for event in exact.events] == [
        "requested",
        "resolved",
        "republication_linked",
    ]
    assert [event.actor_id for event in exact.events] == [
        "user-a",
        "admin-a",
        "admin-b",
    ]
    assert [event.status.value for event in exact.events] == [
        "pending",
        "republication_required",
        "republication_required",
    ]
    assert linked_only == (exact,)
    assert linked_page == (exact,)
    assert linked_audit_page.total_linked_audits == 1
    assert linked_audit_page.audits == (exact,)
    assert empty_linked_page == ()
    assert empty_linked_audit_page.total_linked_audits == 1
    assert empty_linked_audit_page.audits == ()
    assert store.count_linked_practice_correction_audits() == 1
    assert len(unlinked_only) == 1
    assert unlinked_only[0].request.request_id == request_ids[1]
    assert unlinked_only[0].republication is None
    assert [event.event_type for event in unlinked_only[0].events] == [
        "requested",
        "resolved",
    ]
    assert by_new_version == (exact,)
    assert by_old_version == unlinked_only
    assert store.get_profile("user-a").total_attempts == 1
    assert store.get_profile("user-b").total_attempts == 1


def test_practice_correction_audit_query_validates_filters_and_limit(
    tmp_path: Path,
) -> None:
    """审计查询拒绝空标识与越界分页，不将错误条件静默放宽。"""
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="题目标识不能为空"):
        store.list_practice_correction_audits(question_id=" ")
    with pytest.raises(ValueError, match="内容版本不能为空"):
        store.list_practice_correction_audits(new_content_version=" ")
    with pytest.raises(ValueError, match="数量必须介于 1 到 100"):
        store.list_practice_correction_audits(limit=0)
    with pytest.raises(ValueError, match="数量必须介于 1 到 100"):
        store.list_practice_correction_audits(limit=101)
    with pytest.raises(ValueError, match="偏移量必须介于 0 到 10000"):
        store.list_practice_correction_audits(offset=-1)
    with pytest.raises(ValueError, match="偏移量必须介于 0 到 10000"):
        store.list_practice_correction_audits(offset=10_001)
    with pytest.raises(ValueError, match="数量必须介于 1 到 100"):
        store.list_linked_practice_correction_audit_page(limit=0)
    with pytest.raises(ValueError, match="偏移量必须介于 0 到 10000"):
        store.list_linked_practice_correction_audit_page(offset=10_001)
    with pytest.raises(ValueError, match="复核申请标识不能为空"):
        store.get_practice_correction_audit(" ")
    assert store.get_practice_correction_audit("missing-request") is None


def test_learning_store_restores_correction_request_and_audit_snapshot(
    tmp_path: Path,
) -> None:
    """恢复学习库快照时应一并回退关联、当前状态与追加式处置事件。"""
    store = _store(tmp_path)
    record = store.record_practice_attempt(
        LearningRecordInput(
            user_id="user-a",
            question_id="q-1",
            content_version="content-v1",
            question_type="propositional",
            is_correct=False,
        )
    )
    requested = store.create_practice_correction_request(
        PracticeCorrectionRequestInput(
            user_id="user-a",
            record_id=record.record_id,
            reason="请复核当前判分",
        )
    )
    assert requested is not None
    resolved = store.resolve_practice_correction_request(
        PracticeCorrectionResolutionInput(
            request_id=requested.request_id,
            resolver_id="admin-a",
            resolution=PracticeCorrectionResolution.REPUBLICATION_REQUIRED,
        )
    )
    assert resolved is not None
    backup = store.create_backup(tmp_path / "backups")
    store.link_practice_correction_republication(
        request_id=requested.request_id,
        question_id="q-1",
        previous_content_version="content-v1",
        new_content_version="content-v2",
        new_content_hash="a" * 64,
        linked_by="admin-a",
    )
    assert (
        store.list_practice_correction_outcomes_for_user("user-a")[0]
        .republished_content_version
        == "content-v2"
    )

    store.restore_backup(store.load_backup(backup.manifest_path))

    restored = store.list_practice_correction_requests_for_user("user-a")
    assert restored == (resolved,)
    assert store.get_practice_correction_republication(requested.request_id) is None
    assert (
        store.list_practice_correction_outcomes_for_user("user-a")[0]
        .republished_content_version
        is None
    )
    with sqlite3.connect(tmp_path / "learning.sqlite3") as connection:
        events = connection.execute(
            """
            SELECT event_type, status
            FROM practice_correction_events
            ORDER BY rowid ASC
            """
        ).fetchall()
    assert events == [
        ("requested", "pending"),
        ("resolved", "republication_required"),
    ]


def test_learning_store_rejects_tampered_backup(tmp_path: Path) -> None:
    """备份文件内容变化后，摘要校验必须在恢复前阻止覆盖当前档案。"""
    store = _store(tmp_path)
    store.add_record(
        LearningRecordInput(
            user_id="user-a",
            question_id="q-1",
            question_type="propositional",
            is_correct=True,
        )
    )
    backup = store.create_backup(tmp_path / "backups")
    backup.backup_path.write_bytes(b"tampered backup")

    with pytest.raises(ValueError, match="校验摘要"):
        store.restore_backup(backup)

    assert store.get_profile("user-a").total_attempts == 1
