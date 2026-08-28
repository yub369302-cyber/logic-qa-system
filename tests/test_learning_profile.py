"""最小化学习档案、用户隔离和练习方向的回归测试。"""

from pathlib import Path

import pytest

from logic_qa.learning_profile import LearningProfileStore, LearningRecordInput


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
    assert reloaded_store.schema_version() == 1
    assert backup.manifest_path.is_file()
    assert profile.total_attempts == 1
    assert reloaded_store.attempted_question_ids("user-a") == (first.question_id,)


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
