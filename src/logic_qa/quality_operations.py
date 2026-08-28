"""内容绑定审核、运行指标与发布质量门禁的基础能力。"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import Lock
from uuid import uuid4

from logic_qa.database_governance import (
    DatabaseBackup,
    DatabaseMigration,
    SQLiteDatabaseManager,
)

_CONTENT_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class QuestionReviewStatus(StrEnum):
    """候选题目内容进入正式题库前的审核状态。"""

    PENDING = "pending"
    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class QuestionReviewInput:
    """审核人员提交的、精确绑定候选内容的审核结论。"""

    question_id: str
    content_version: str
    content_hash: str
    reviewer_id: str
    status: QuestionReviewStatus
    verified_answer: str | None
    formalization_version: str
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class QuestionReviewRecord:
    """某一不可变候选内容的最新审核状态和可追溯元数据。"""

    question_id: str
    content_version: str
    content_hash: str
    reviewer_id: str
    status: QuestionReviewStatus
    verified_answer: str | None
    formalization_version: str
    notes: str | None
    updated_at: str


@dataclass(frozen=True, slots=True)
class ReviewDashboard:
    """审核状态统计，不包含题干、答案或任何用户学习数据。"""

    total_questions: int
    status_counts: tuple[tuple[str, int], ...]
    total_review_events: int


@dataclass(frozen=True, slots=True)
class RuntimeMetricsSnapshot:
    """进程内 HTTP 指标，不记录请求体、用户标识或文本内容。"""

    total_requests: int
    error_requests: int
    average_latency_ms: float
    route_counts: tuple[tuple[str, int], ...]
    status_counts: tuple[tuple[str, int], ...]


class QuestionReviewStore:
    """保存精确内容审核状态和追加式审计历史。"""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._database = SQLiteDatabaseManager(
            database_path,
            (
                DatabaseMigration(
                    1,
                    "create_content_bound_review_tables",
                    self._migrate_v1,
                ),
            ),
        )
        self._database.migrate()

    def schema_version(self) -> int:
        """返回当前审核数据库已完成的最高迁移版本。"""
        return self._database.schema_version()

    def create_backup(self, destination_directory: Path) -> DatabaseBackup:
        """创建经过完整性校验的审核数据一致性备份。"""
        return self._database.backup(destination_directory)

    def load_backup(self, manifest_path: Path) -> DatabaseBackup:
        """从已持久化的备份清单读取审核库恢复元数据。"""
        return self._database.load_backup(manifest_path)

    def restore_backup(self, backup: DatabaseBackup) -> None:
        """仅从经校验且属于当前审核库的备份恢复数据。"""
        self._database.restore(backup)

    def upsert_review(self, review: QuestionReviewInput) -> QuestionReviewRecord:
        """更新同一候选内容的审核状态，并追加不可变审核事件。"""
        normalized = _normalize_review_input(review)
        record = QuestionReviewRecord(
            question_id=normalized.question_id,
            content_version=normalized.content_version,
            content_hash=normalized.content_hash,
            reviewer_id=normalized.reviewer_id,
            status=normalized.status,
            verified_answer=normalized.verified_answer,
            formalization_version=normalized.formalization_version,
            notes=normalized.notes,
            updated_at=datetime.now(UTC).isoformat(),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO question_reviews (
                    question_id, content_version, content_hash, reviewer_id, status,
                    verified_answer, formalization_version, notes, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(question_id, content_version, content_hash) DO UPDATE SET
                    reviewer_id = excluded.reviewer_id,
                    status = excluded.status,
                    verified_answer = excluded.verified_answer,
                    formalization_version = excluded.formalization_version,
                    notes = excluded.notes,
                    updated_at = excluded.updated_at
                """,
                (
                    record.question_id,
                    record.content_version,
                    record.content_hash,
                    record.reviewer_id,
                    record.status.value,
                    record.verified_answer,
                    record.formalization_version,
                    record.notes,
                    record.updated_at,
                ),
            )
            connection.execute(
                """
                INSERT INTO question_review_events (
                    event_id, question_id, content_version, content_hash, reviewer_id,
                    status, verified_answer, formalization_version, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    record.question_id,
                    record.content_version,
                    record.content_hash,
                    record.reviewer_id,
                    record.status.value,
                    record.verified_answer,
                    record.formalization_version,
                    record.notes,
                    record.updated_at,
                ),
            )
        return record

    def get_review(
        self,
        question_id: str,
        content_version: str,
        content_hash: str,
    ) -> QuestionReviewRecord | None:
        """读取与题目版本和内容摘要完全匹配的审核记录。"""
        normalized_question_id = _validate_text(question_id, "题目标识", max_length=128)
        normalized_content_version = _validate_text(
            content_version,
            "内容版本",
            max_length=128,
        )
        normalized_content_hash = _validate_content_hash(content_hash)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT question_id, content_version, content_hash, reviewer_id, status,
                       verified_answer, formalization_version, notes, updated_at
                FROM question_reviews
                WHERE question_id = ? AND content_version = ? AND content_hash = ?
                """,
                (
                    normalized_question_id,
                    normalized_content_version,
                    normalized_content_hash,
                ),
            ).fetchone()
        return _review_from_row(row) if row else None

    def dashboard(self) -> ReviewDashboard:
        """返回审核候选状态计数和审计事件总数。"""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM question_reviews GROUP BY status"
            ).fetchall()
            question_count = connection.execute(
                "SELECT COUNT(DISTINCT question_id) AS count FROM question_reviews"
            ).fetchone()["count"]
            event_count = connection.execute(
                "SELECT COUNT(*) AS count FROM question_review_events"
            ).fetchone()["count"]
        status_counts = tuple(sorted((row["status"], row["count"]) for row in rows))
        return ReviewDashboard(
            total_questions=question_count,
            status_counts=status_counts,
            total_review_events=event_count,
        )

    def _migrate_v1(self, connection: sqlite3.Connection) -> None:
        """建立精确绑定候选内容的审核表和审计索引。"""
        self._archive_unbound_table(
            connection,
            "question_reviews",
            {"question_id", "content_version", "content_hash"},
        )
        self._archive_unbound_table(
            connection,
            "question_review_events",
            {"question_id", "content_version", "content_hash"},
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS question_reviews (
                question_id TEXT NOT NULL,
                content_version TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                reviewer_id TEXT NOT NULL,
                status TEXT NOT NULL,
                verified_answer TEXT,
                formalization_version TEXT NOT NULL,
                notes TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (question_id, content_version, content_hash)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS question_review_events (
                event_id TEXT PRIMARY KEY,
                question_id TEXT NOT NULL,
                content_version TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                reviewer_id TEXT NOT NULL,
                status TEXT NOT NULL,
                verified_answer TEXT,
                formalization_version TEXT NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_question_review_events_candidate
            ON question_review_events (
                question_id, content_version, content_hash, created_at
            )
            """
        )

    @staticmethod
    def _archive_unbound_table(
        connection: sqlite3.Connection,
        table_name: str,
        required_columns: set[str],
    ) -> None:
        """保留无法精确绑定内容的旧表，避免将旧审核误用于新发布。"""
        rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        if not rows:
            return
        existing_columns = {row["name"] for row in rows}
        if required_columns.issubset(existing_columns):
            return

        base_name = f"{table_name}_legacy_unbound"
        archive_name = base_name
        suffix = 2
        existing_tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        while archive_name in existing_tables:
            archive_name = f"{base_name}_{suffix}"
            suffix += 1
        connection.execute(f"ALTER TABLE {table_name} RENAME TO {archive_name}")

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


class RuntimeMetrics:
    """线程安全地聚合非敏感 HTTP 请求状态和时延信息。"""

    def __init__(self) -> None:
        self._lock = Lock()
        self._total_requests = 0
        self._error_requests = 0
        self._total_latency_ms = 0.0
        self._route_counts: Counter[str] = Counter()
        self._status_counts: Counter[str] = Counter()

    def record(self, route_template: str, status_code: int, latency_ms: float) -> None:
        """记录一条不含请求内容和用户数据的请求指标。"""
        with self._lock:
            self._total_requests += 1
            self._total_latency_ms += max(latency_ms, 0.0)
            self._route_counts[route_template] += 1
            self._status_counts[str(status_code)] += 1
            if status_code >= 400:
                self._error_requests += 1

    def snapshot(self) -> RuntimeMetricsSnapshot:
        """返回当前进程内指标的稳定快照。"""
        with self._lock:
            average_latency = (
                self._total_latency_ms / self._total_requests
                if self._total_requests
                else 0.0
            )
            return RuntimeMetricsSnapshot(
                total_requests=self._total_requests,
                error_requests=self._error_requests,
                average_latency_ms=round(average_latency, 3),
                route_counts=tuple(sorted(self._route_counts.items())),
                status_counts=tuple(sorted(self._status_counts.items())),
            )


def _normalize_review_input(review: QuestionReviewInput) -> QuestionReviewInput:
    question_id = _validate_text(review.question_id, "题目标识", max_length=128)
    content_version = _validate_text(review.content_version, "内容版本", max_length=128)
    content_hash = _validate_content_hash(review.content_hash)
    reviewer_id = _validate_text(review.reviewer_id, "审核人标识", max_length=128)
    verified_answer = _normalize_optional_text(
        review.verified_answer,
        "已核验答案",
        max_length=2_000,
    )
    formalization_version = _validate_text(
        review.formalization_version,
        "形式化版本",
        max_length=128,
    )
    notes = _normalize_optional_text(review.notes, "审核备注", max_length=4_000)
    if review.status is QuestionReviewStatus.APPROVED and not verified_answer:
        raise ValueError("通过审核必须填写已核验答案")
    return QuestionReviewInput(
        question_id=question_id,
        content_version=content_version,
        content_hash=content_hash,
        reviewer_id=reviewer_id,
        status=review.status,
        verified_answer=verified_answer,
        formalization_version=formalization_version,
        notes=notes,
    )


def _validate_content_hash(value: str) -> str:
    normalized = value.strip().lower()
    if not _CONTENT_HASH_PATTERN.fullmatch(normalized):
        raise ValueError("内容摘要必须是 64 位 SHA-256 十六进制字符串")
    return normalized


def _validate_text(value: str, label: str, max_length: int) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label}不能为空")
    if len(normalized) > max_length:
        raise ValueError(f"{label}不能超过 {max_length} 个字符")
    return normalized


def _normalize_optional_text(
    value: str | None,
    label: str,
    max_length: int,
) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized) > max_length:
        raise ValueError(f"{label}不能超过 {max_length} 个字符")
    return normalized


def _review_from_row(row: sqlite3.Row) -> QuestionReviewRecord:
    return QuestionReviewRecord(
        question_id=row["question_id"],
        content_version=row["content_version"],
        content_hash=row["content_hash"],
        reviewer_id=row["reviewer_id"],
        status=QuestionReviewStatus(row["status"]),
        verified_answer=row["verified_answer"],
        formalization_version=row["formalization_version"],
        notes=row["notes"],
        updated_at=row["updated_at"],
    )
