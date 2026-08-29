"""最小化、按用户隔离的学习档案与练习方向服务。"""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from logic_qa.database_governance import (
    DatabaseBackup,
    DatabaseMigration,
    SQLiteDatabaseManager,
)


@dataclass(frozen=True, slots=True)
class LearningRecordInput:
    """用户主动提交、允许持久化的最小学习记录。"""

    user_id: str
    question_id: str
    question_type: str
    is_correct: bool
    error_tags: tuple[str, ...] = ()
    knowledge_tags: tuple[str, ...] = ()
    duration_seconds: int | None = None
    content_version: str | None = None


@dataclass(frozen=True, slots=True)
class LearningRecord:
    """带服务端编号与创建时间的学习记录。"""

    record_id: str
    user_id: str
    question_id: str
    content_version: str | None
    question_type: str
    is_correct: bool
    error_tags: tuple[str, ...]
    knowledge_tags: tuple[str, ...]
    duration_seconds: int | None
    created_at: str


class DuplicatePracticeAttemptError(ValueError):
    """当前用户已完成同一不可变题目版本的练习。"""


@dataclass(frozen=True, slots=True)
class LearningRecommendation:
    """基于用户自身统计生成的练习方向，而不是虚构题目。"""

    focus_type: str
    label: str
    reason: str
    suggested_practice: str


@dataclass(frozen=True, slots=True)
class LearningProfile:
    """按单一用户聚合后的学习表现、弱项与练习方向。"""

    user_id: str
    total_attempts: int
    correct_attempts: int
    accuracy: float | None
    error_counts: tuple[tuple[str, int], ...]
    knowledge_mastery: tuple[tuple[str, float], ...]
    recommendations: tuple[LearningRecommendation, ...]


class LearningProfileStore:
    """使用参数化 SQLite 查询实现最小化学习记录存取。"""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._database = SQLiteDatabaseManager(
            database_path,
            (
                DatabaseMigration(
                    1,
                    "create_learning_records",
                    self._migrate_v1,
                ),
                DatabaseMigration(
                    2,
                    "add_versioned_practice_attempts",
                    self._migrate_v2,
                ),
            ),
        )
        self._database.migrate()

    def schema_version(self) -> int:
        """返回当前学习档案数据库已完成的最高迁移版本。"""
        return self._database.schema_version()

    def create_backup(self, destination_directory: Path) -> DatabaseBackup:
        """创建经过 SQLite 完整性校验的学习档案一致性备份。"""
        return self._database.backup(destination_directory)

    def load_backup(self, manifest_path: Path) -> DatabaseBackup:
        """从已持久化的备份清单读取学习档案恢复元数据。"""
        return self._database.load_backup(manifest_path)

    def restore_backup(self, backup: DatabaseBackup) -> None:
        """仅从经校验且属于当前存储的备份恢复学习档案。"""
        self._database.restore(backup)

    def add_record(self, record: LearningRecordInput) -> LearningRecord:
        """写入一条不绑定发布版本的最小学习记录。"""
        if record.content_version is not None:
            raise ValueError("通用学习记录不能绑定题目内容版本")
        saved = _new_learning_record(record)
        with self._connect() as connection:
            _insert_learning_record(connection, saved)
        return saved

    def record_practice_attempt(self, record: LearningRecordInput) -> LearningRecord:
        """原子写入当前用户对同一发布版本的唯一练习作答。"""
        if record.content_version is None:
            raise ValueError("练习作答必须指定题目内容版本")
        saved = _new_learning_record(record)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO learning_records (
                    record_id, user_id, question_id, content_version, question_type,
                    is_correct, error_tags, knowledge_tags, duration_seconds, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _learning_record_values(saved),
            )
        if cursor.rowcount != 1:
            raise DuplicatePracticeAttemptError("该题目版本已完成练习")
        return saved

    def get_profile(self, user_id: str) -> LearningProfile:
        """仅基于指定用户自己的记录生成学习画像与练习方向。"""
        normalized_user_id = _validate_identifier(user_id, "用户标识")
        records = self._records_for_user(normalized_user_id)
        total_attempts = len(records)
        correct_attempts = sum(record.is_correct for record in records)
        accuracy = correct_attempts / total_attempts if total_attempts else None
        error_counts = _count_tags(record.error_tags for record in records)
        knowledge_mastery = _knowledge_mastery(records)
        recommendations = _recommend(error_counts, knowledge_mastery)
        return LearningProfile(
            user_id=normalized_user_id,
            total_attempts=total_attempts,
            correct_attempts=correct_attempts,
            accuracy=accuracy,
            error_counts=error_counts,
            knowledge_mastery=knowledge_mastery,
            recommendations=recommendations,
        )

    def delete_record(self, user_id: str, record_id: str) -> bool:
        """仅当记录属于该用户时删除，避免跨用户误删。"""
        normalized_user_id = _validate_identifier(user_id, "用户标识")
        normalized_record_id = _validate_identifier(record_id, "记录标识")
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM learning_records WHERE record_id = ? AND user_id = ?",
                (normalized_record_id, normalized_user_id),
            )
        return cursor.rowcount == 1

    def attempted_question_ids(self, user_id: str) -> tuple[str, ...]:
        """兼容旧调用：返回当前用户全部学习记录涉及的题目标识。"""
        normalized_user_id = _validate_identifier(user_id, "用户标识")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT question_id
                FROM learning_records
                WHERE user_id = ?
                ORDER BY question_id ASC
                """,
                (normalized_user_id,),
            ).fetchall()
        return tuple(row["question_id"] for row in rows)

    def attempted_practice_versions(self, user_id: str) -> tuple[tuple[str, str], ...]:
        """返回当前用户已完成的精确发布版本，用于版本感知的推荐去重。"""
        normalized_user_id = _validate_identifier(user_id, "用户标识")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT question_id, content_version
                FROM learning_records
                WHERE user_id = ? AND content_version IS NOT NULL
                ORDER BY question_id ASC, content_version ASC
                """,
                (normalized_user_id,),
            ).fetchall()
        return tuple(
            (row["question_id"], row["content_version"])
            for row in rows
        )

    def _migrate_v1(self, connection: sqlite3.Connection) -> None:
        """创建学习记录表及其用户范围查询索引。"""
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_records (
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
            CREATE INDEX IF NOT EXISTS idx_learning_records_user
            ON learning_records (user_id, created_at)
            """
        )

    def _migrate_v2(self, connection: sqlite3.Connection) -> None:
        """为发布题练习添加内容版本与用户范围唯一性约束。"""
        connection.execute(
            "ALTER TABLE learning_records ADD COLUMN content_version TEXT"
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX idx_learning_records_practice_version
            ON learning_records (user_id, question_id, content_version)
            WHERE content_version IS NOT NULL
            """
        )

    def _records_for_user(self, user_id: str) -> tuple[LearningRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT record_id, user_id, question_id, content_version,
                       question_type, is_correct, error_tags, knowledge_tags,
                       duration_seconds, created_at
                FROM learning_records
                WHERE user_id = ?
                ORDER BY created_at ASC
                """,
                (user_id,),
            ).fetchall()
        return tuple(_record_from_row(row) for row in rows)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """在每次操作后提交或回滚并关闭连接，避免恢复时保留文件锁。"""
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()


def _new_learning_record(record: LearningRecordInput) -> LearningRecord:
    """校验并规范化输入后，生成尚未持久化的学习记录。"""
    _validate_record_input(record)
    content_version = (
        _validate_identifier(record.content_version, "内容版本")
        if record.content_version is not None
        else None
    )
    return LearningRecord(
        record_id=str(uuid4()),
        user_id=record.user_id.strip(),
        question_id=record.question_id.strip(),
        content_version=content_version,
        question_type=record.question_type.strip(),
        is_correct=record.is_correct,
        error_tags=_normalize_tags(record.error_tags),
        knowledge_tags=_normalize_tags(record.knowledge_tags),
        duration_seconds=record.duration_seconds,
        created_at=datetime.now(UTC).isoformat(),
    )


def _insert_learning_record(
    connection: sqlite3.Connection,
    record: LearningRecord,
) -> None:
    connection.execute(
        """
        INSERT INTO learning_records (
            record_id, user_id, question_id, content_version, question_type,
            is_correct, error_tags, knowledge_tags, duration_seconds, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _learning_record_values(record),
    )


def _learning_record_values(record: LearningRecord) -> tuple[object, ...]:
    return (
        record.record_id,
        record.user_id,
        record.question_id,
        record.content_version,
        record.question_type,
        int(record.is_correct),
        _serialize_tags(record.error_tags),
        _serialize_tags(record.knowledge_tags),
        record.duration_seconds,
        record.created_at,
    )


def _validate_record_input(record: LearningRecordInput) -> None:
    _validate_identifier(record.user_id, "用户标识")
    _validate_identifier(record.question_id, "题目标识")
    _validate_identifier(record.question_type, "题型")
    if record.content_version is not None:
        _validate_identifier(record.content_version, "内容版本")
    if record.duration_seconds is not None and record.duration_seconds < 0:
        raise ValueError("作答时长不能为负数")
    _normalize_tags(record.error_tags)
    _normalize_tags(record.knowledge_tags)


def _validate_identifier(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label}不能为空")
    if len(normalized) > 128:
        raise ValueError(f"{label}不能超过 128 个字符")
    return normalized


def _normalize_tags(tags: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(tag.strip() for tag in tags if tag.strip()))
    if len(normalized) > 20:
        raise ValueError("标签数量不能超过 20 个")
    if any(len(tag) > 64 for tag in normalized):
        raise ValueError("单个标签不能超过 64 个字符")
    return normalized


def _serialize_tags(tags: tuple[str, ...]) -> str:
    return "\x1f".join(tags)


def _deserialize_tags(value: str) -> tuple[str, ...]:
    return tuple(tag for tag in value.split("\x1f") if tag)


def _record_from_row(row: sqlite3.Row) -> LearningRecord:
    return LearningRecord(
        record_id=row["record_id"],
        user_id=row["user_id"],
        question_id=row["question_id"],
        content_version=row["content_version"],
        question_type=row["question_type"],
        is_correct=bool(row["is_correct"]),
        error_tags=_deserialize_tags(row["error_tags"]),
        knowledge_tags=_deserialize_tags(row["knowledge_tags"]),
        duration_seconds=row["duration_seconds"],
        created_at=row["created_at"],
    )


def _count_tags(
    tag_groups: Iterable[tuple[str, ...]],
) -> tuple[tuple[str, int], ...]:
    counts: Counter[str] = Counter()
    for tags in tag_groups:
        counts.update(tags)
    return tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _knowledge_mastery(
    records: tuple[LearningRecord, ...],
) -> tuple[tuple[str, float], ...]:
    outcomes: dict[str, list[bool]] = {}
    for record in records:
        for tag in record.knowledge_tags:
            outcomes.setdefault(tag, []).append(record.is_correct)
    return tuple(
        sorted(
            (
                tag,
                sum(result) / len(result),
            )
            for tag, result in outcomes.items()
        )
    )


def _recommend(
    error_counts: tuple[tuple[str, int], ...],
    knowledge_mastery: tuple[tuple[str, float], ...],
) -> tuple[LearningRecommendation, ...]:
    recommendations: list[LearningRecommendation] = []
    if error_counts:
        error_tag, count = error_counts[0]
        recommendations.append(
            LearningRecommendation(
                focus_type="error_tag",
                label=error_tag,
                reason=f"该错因已出现 {count} 次，是当前最高频问题。",
                suggested_practice=(
                    "先复习对应规则，再完成 3 道同类微练习并核对推理步骤。"
                ),
            )
        )
    if knowledge_mastery:
        knowledge_tag, mastery = min(knowledge_mastery, key=lambda item: item[1])
        recommendations.append(
            LearningRecommendation(
                focus_type="knowledge_tag",
                label=knowledge_tag,
                reason=f"该知识点当前正确率为 {mastery:.0%}，低于其他已记录知识点。",
                suggested_practice="使用从基础条件翻译到完整证明的渐进式练习巩固该知识点。",
            )
        )
    return tuple(recommendations)
