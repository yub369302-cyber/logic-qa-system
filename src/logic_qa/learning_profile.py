"""最小化、按用户隔离的学习档案与练习方向服务。"""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from logic_qa.database_governance import (
    DatabaseBackup,
    DatabaseMigration,
    SQLiteDatabaseManager,
)

_PRACTICE_CORRECTION_REQUEST_COLUMNS = """
    request_id, record_id, user_id, question_id, content_version, reason, status,
    created_at, resolved_by, resolution_notes, resolved_at
"""
_PRACTICE_CORRECTION_REPUBLICATION_COLUMNS = """
    request_id, question_id, previous_content_version, new_content_version,
    new_content_hash, linked_by, linked_at
"""


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


class ImmutablePracticeAttemptError(ValueError):
    """已审核发布题目的练习记录不得由学习者删除。"""


class PracticeCorrectionRequestStatus(StrEnum):
    """不可变练习记录更正申请的当前处置状态。"""

    PENDING = "pending"
    RECORD_CONFIRMED = "record_confirmed"
    REPUBLICATION_REQUIRED = "republication_required"


class PracticeCorrectionResolution(StrEnum):
    """管理员可写入的终态处置结论。"""

    RECORD_CONFIRMED = "record_confirmed"
    REPUBLICATION_REQUIRED = "republication_required"


class PracticeCorrectionOutcomeKind(StrEnum):
    """学习者可见的派生复核结论类型，不改变原始练习账本。"""

    PENDING = "pending"
    RECORD_CONFIRMED = "record_confirmed"
    REPUBLICATION_REQUIRED = "republication_required"


class DuplicatePracticeCorrectionRequestError(ValueError):
    """同一练习记录已经提交过受控更正申请。"""


class PracticeCorrectionRequestAlreadyResolvedError(ValueError):
    """更正申请已处置，不能覆盖既有治理结论。"""


class PracticeCorrectionRepublicationNotEligibleError(ValueError):
    """仅需要重新发布的更正申请可关联到新的已发布题目版本。"""


class PracticeCorrectionRepublicationAlreadyLinkedError(ValueError):
    """同一更正申请已经不可变地关联到一个新发布版本。"""


class PracticeCorrectionRepublicationVersionAlreadyLinkedError(ValueError):
    """同一题目的一个新发布版本已经不可变地关联到另一申请。"""


@dataclass(frozen=True, slots=True)
class PracticeCorrectionRequestInput:
    """学习者对自身不可变练习记录提交的最小复核申请。"""

    user_id: str
    record_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class PracticeCorrectionResolutionInput:
    """管理员对更正申请记录的终态处置。"""

    request_id: str
    resolver_id: str
    resolution: PracticeCorrectionResolution
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class PracticeCorrectionRequest:
    """不可变练习记录的受控更正申请及其当前治理状态。"""

    request_id: str
    record_id: str
    user_id: str
    question_id: str
    content_version: str
    reason: str
    status: PracticeCorrectionRequestStatus
    created_at: str
    resolved_by: str | None
    resolution_notes: str | None
    resolved_at: str | None


@dataclass(frozen=True, slots=True)
class PracticeCorrectionOutcome:
    """由处置状态推导的学习者安全视图，不包含题库或管理员内部信息。"""

    request_id: str
    record_id: str
    question_id: str
    content_version: str
    kind: PracticeCorrectionOutcomeKind
    message: str
    created_at: str
    resolved_at: str | None
    republished_content_version: str | None = None


@dataclass(frozen=True, slots=True)
class PracticeCorrectionRepublication:
    """管理员将复核申请关联到独立新发布版本的不可变审计记录。"""

    request_id: str
    question_id: str
    previous_content_version: str
    new_content_version: str
    new_content_hash: str
    linked_by: str
    linked_at: str


@dataclass(frozen=True, slots=True)
class PracticeCorrectionEvent:
    """更正申请生命周期中的一条追加式审计事件。"""

    event_id: str
    request_id: str
    actor_id: str
    event_type: str
    status: PracticeCorrectionRequestStatus
    notes: str | None
    created_at: str


@dataclass(frozen=True, slots=True)
class PracticeCorrectionAudit:
    """管理员回查一条更正申请、其重发布关联和完整事件链的只读视图。"""

    request: PracticeCorrectionRequest
    republication: PracticeCorrectionRepublication | None
    events: tuple[PracticeCorrectionEvent, ...]


@dataclass(frozen=True, slots=True)
class LinkedPracticeCorrectionAuditPage:
    """同一学习库读取快照内的已关联申请总数和稳定分页审计视图。"""

    total_linked_audits: int
    audits: tuple[PracticeCorrectionAudit, ...]


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
                DatabaseMigration(
                    3,
                    "add_practice_correction_requests",
                    self._migrate_v3,
                ),
                DatabaseMigration(
                    4,
                    "add_practice_correction_republications",
                    self._migrate_v4,
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
        """只删除当前用户的通用记录，保留已审核发布题目的练习账本。"""
        normalized_user_id = _validate_identifier(user_id, "用户标识")
        normalized_record_id = _validate_identifier(record_id, "记录标识")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT content_version
                FROM learning_records
                WHERE record_id = ? AND user_id = ?
                """,
                (normalized_record_id, normalized_user_id),
            ).fetchone()
            if row is None:
                return False
            if row["content_version"] is not None:
                raise ImmutablePracticeAttemptError("练习记录不可删除")
            cursor = connection.execute(
                "DELETE FROM learning_records WHERE record_id = ? AND user_id = ?",
                (normalized_record_id, normalized_user_id),
            )
        return cursor.rowcount == 1

    def create_practice_correction_request(
        self,
        request: PracticeCorrectionRequestInput,
    ) -> PracticeCorrectionRequest | None:
        """为当前用户的不可变练习记录原子创建一次受控复核申请。"""
        normalized_user_id = _validate_identifier(request.user_id, "用户标识")
        normalized_record_id = _validate_identifier(request.record_id, "记录标识")
        normalized_reason = _validate_reason(request.reason)
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            record = connection.execute(
                """
                SELECT record_id, question_id, content_version
                FROM learning_records
                WHERE record_id = ? AND user_id = ?
                """,
                (normalized_record_id, normalized_user_id),
            ).fetchone()
            if record is None:
                return None
            if record["content_version"] is None:
                raise ValueError("仅已发布题目的练习记录可申请复核")
            correction_request = PracticeCorrectionRequest(
                request_id=str(uuid4()),
                record_id=record["record_id"],
                user_id=normalized_user_id,
                question_id=record["question_id"],
                content_version=record["content_version"],
                reason=normalized_reason,
                status=PracticeCorrectionRequestStatus.PENDING,
                created_at=created_at,
                resolved_by=None,
                resolution_notes=None,
                resolved_at=None,
            )
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO practice_correction_requests (
                    request_id, record_id, user_id, question_id, content_version,
                    reason, status, created_at, resolved_by, resolution_notes,
                    resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _practice_correction_request_values(correction_request),
            )
            if cursor.rowcount != 1:
                raise DuplicatePracticeCorrectionRequestError(
                    "该练习记录已提交复核申请"
                )
            _insert_practice_correction_event(
                connection,
                request_id=correction_request.request_id,
                actor_id=normalized_user_id,
                event_type="requested",
                status=correction_request.status,
                notes=correction_request.reason,
                created_at=created_at,
            )
        return correction_request

    def list_practice_correction_requests_for_user(
        self,
        user_id: str,
    ) -> tuple[PracticeCorrectionRequest, ...]:
        """仅返回当前用户自身的更正申请，避免跨用户浏览。"""
        normalized_user_id = _validate_identifier(user_id, "用户标识")
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT {_PRACTICE_CORRECTION_REQUEST_COLUMNS}
                FROM practice_correction_requests
                WHERE user_id = ?
                ORDER BY created_at DESC, request_id DESC
                """,
                (normalized_user_id,),
            ).fetchall()
        return tuple(_practice_correction_request_from_row(row) for row in rows)

    def list_practice_correction_requests(
        self,
        status: PracticeCorrectionRequestStatus | None = None,
    ) -> tuple[PracticeCorrectionRequest, ...]:
        """供管理员按可选当前状态读取更正申请，不改写练习账本。"""
        where_clause = ""
        parameters: tuple[str, ...] = ()
        if status is not None:
            where_clause = "WHERE status = ?"
            parameters = (status.value,)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT {_PRACTICE_CORRECTION_REQUEST_COLUMNS}
                FROM practice_correction_requests
                {where_clause}
                ORDER BY created_at ASC, request_id ASC
                """,
                parameters,
            ).fetchall()
        return tuple(_practice_correction_request_from_row(row) for row in rows)

    def list_practice_correction_outcomes_for_user(
        self,
        user_id: str,
    ) -> tuple[PracticeCorrectionOutcome, ...]:
        """返回当前用户的最小派生处置视图，不改写历史作答或题库状态。"""
        requests = self.list_practice_correction_requests_for_user(user_id)
        republications = self._republications_by_request_id(
            request.request_id for request in requests
        )
        return tuple(
            _practice_correction_outcome_from(
                request,
                republications.get(request.request_id),
            )
            for request in requests
        )

    def link_practice_correction_republication(
        self,
        *,
        request_id: str,
        question_id: str,
        previous_content_version: str,
        new_content_version: str,
        new_content_hash: str,
        linked_by: str,
    ) -> PracticeCorrectionRepublication | None:
        """把需要重发布的申请原子关联到同题的独立新版本，不修改练习账本。"""
        normalized_request_id = _validate_identifier(request_id, "复核申请标识")
        normalized_question_id = _validate_identifier(question_id, "题目标识")
        normalized_previous_version = _validate_identifier(
            previous_content_version,
            "原内容版本",
        )
        normalized_new_version = _validate_identifier(new_content_version, "新内容版本")
        normalized_new_hash = _validate_content_hash(new_content_hash)
        normalized_linked_by = _validate_identifier(linked_by, "关联人标识")
        if normalized_new_version == normalized_previous_version:
            raise ValueError("复核重发布必须关联新的内容版本")
        linked_at = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            request = connection.execute(
                f"""
                SELECT {_PRACTICE_CORRECTION_REQUEST_COLUMNS}
                FROM practice_correction_requests
                WHERE request_id = ?
                """,
                (normalized_request_id,),
            ).fetchone()
            if request is None:
                return None
            correction_request = _practice_correction_request_from_row(request)
            if (
                correction_request.status
                is not PracticeCorrectionRequestStatus.REPUBLICATION_REQUIRED
            ):
                raise PracticeCorrectionRepublicationNotEligibleError(
                    "该复核申请不需要重新发布"
                )
            if (
                correction_request.question_id != normalized_question_id
                or correction_request.content_version != normalized_previous_version
            ):
                raise ValueError("复核申请与待关联题目版本不一致")
            existing_for_request = connection.execute(
                """
                SELECT request_id
                FROM practice_correction_republications
                WHERE request_id = ?
                """,
                (normalized_request_id,),
            ).fetchone()
            if existing_for_request is not None:
                raise PracticeCorrectionRepublicationAlreadyLinkedError(
                    "该复核申请已关联新的发布版本"
                )
            existing_for_version = connection.execute(
                """
                SELECT request_id
                FROM practice_correction_republications
                WHERE question_id = ? AND new_content_version = ?
                """,
                (normalized_question_id, normalized_new_version),
            ).fetchone()
            if existing_for_version is not None:
                raise PracticeCorrectionRepublicationVersionAlreadyLinkedError(
                    "该题目新发布版本已关联另一复核申请"
                )
            republication = PracticeCorrectionRepublication(
                request_id=normalized_request_id,
                question_id=normalized_question_id,
                previous_content_version=normalized_previous_version,
                new_content_version=normalized_new_version,
                new_content_hash=normalized_new_hash,
                linked_by=normalized_linked_by,
                linked_at=linked_at,
            )
            try:
                connection.execute(
                    """
                    INSERT INTO practice_correction_republications (
                        request_id, question_id, previous_content_version,
                        new_content_version, new_content_hash, linked_by, linked_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    _practice_correction_republication_values(republication),
                )
            except sqlite3.IntegrityError as error:
                existing_for_request = connection.execute(
                    """
                    SELECT request_id
                    FROM practice_correction_republications
                    WHERE request_id = ?
                    """,
                    (normalized_request_id,),
                ).fetchone()
                if existing_for_request is not None:
                    raise PracticeCorrectionRepublicationAlreadyLinkedError(
                        "该复核申请已关联新的发布版本"
                    ) from error
                raise PracticeCorrectionRepublicationVersionAlreadyLinkedError(
                    "该题目新发布版本已关联另一复核申请"
                ) from error
            _insert_practice_correction_event(
                connection,
                request_id=normalized_request_id,
                actor_id=normalized_linked_by,
                event_type="republication_linked",
                status=correction_request.status,
                notes=normalized_new_version,
                created_at=linked_at,
            )
        return republication

    def get_practice_correction_republication(
        self,
        request_id: str,
    ) -> PracticeCorrectionRepublication | None:
        """读取申请的不可变重发布关联，供跨存储投影精确核验。"""
        normalized_request_id = _validate_identifier(request_id, "复核申请标识")
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT {_PRACTICE_CORRECTION_REPUBLICATION_COLUMNS}
                FROM practice_correction_republications
                WHERE request_id = ?
                """,
                (normalized_request_id,),
            ).fetchone()
        return _practice_correction_republication_from_row(row) if row else None

    def get_practice_correction_audit(
        self,
        request_id: str,
    ) -> PracticeCorrectionAudit | None:
        """按申请标识回查关联与完整追加式事件链，不修改任何治理记录。"""
        audits = self.list_practice_correction_audits(request_id=request_id, limit=1)
        return audits[0] if audits else None

    def list_practice_correction_audits(
        self,
        *,
        request_id: str | None = None,
        question_id: str | None = None,
        content_version: str | None = None,
        new_content_version: str | None = None,
        linked: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[PracticeCorrectionAudit, ...]:
        """按精确申请、题目或版本筛选管理员审计视图，单次读取数量受限。"""
        normalized_request_id = (
            _validate_identifier(request_id, "复核申请标识")
            if request_id is not None
            else None
        )
        normalized_question_id = (
            _validate_identifier(question_id, "题目标识")
            if question_id is not None
            else None
        )
        normalized_content_version = (
            _validate_identifier(content_version, "内容版本")
            if content_version is not None
            else None
        )
        normalized_new_content_version = (
            _validate_identifier(new_content_version, "新内容版本")
            if new_content_version is not None
            else None
        )
        _validate_audit_page_arguments(limit=limit, offset=offset)

        filters: list[str] = []
        parameters: list[object] = []
        if normalized_request_id is not None:
            filters.append("request.request_id = ?")
            parameters.append(normalized_request_id)
        if normalized_question_id is not None:
            filters.append("request.question_id = ?")
            parameters.append(normalized_question_id)
        if normalized_content_version is not None:
            filters.append("request.content_version = ?")
            parameters.append(normalized_content_version)
        if normalized_new_content_version is not None:
            filters.append(
                """
                EXISTS (
                    SELECT 1
                    FROM practice_correction_republications AS republication
                    WHERE republication.request_id = request.request_id
                      AND republication.new_content_version = ?
                )
                """
            )
            parameters.append(normalized_new_content_version)
        if linked is not None:
            link_predicate = "EXISTS" if linked else "NOT EXISTS"
            filters.append(
                f"""
                {link_predicate} (
                    SELECT 1
                    FROM practice_correction_republications AS republication
                    WHERE republication.request_id = request.request_id
                )
                """
            )
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        parameters.extend((limit, offset))
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT {_PRACTICE_CORRECTION_REQUEST_COLUMNS}
                FROM practice_correction_requests AS request
                {where_clause}
                ORDER BY request.created_at ASC, request.request_id ASC
                LIMIT ? OFFSET ?
                """,
                tuple(parameters),
            ).fetchall()
        requests = tuple(_practice_correction_request_from_row(row) for row in rows)
        republications = self._republications_by_request_id(
            request.request_id for request in requests
        )
        events = self._events_by_request_id(request.request_id for request in requests)
        return tuple(
            PracticeCorrectionAudit(
                request=request,
                republication=republications.get(request.request_id),
                events=events.get(request.request_id, ()),
            )
            for request in requests
        )

    def list_linked_practice_correction_audit_page(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> LinkedPracticeCorrectionAuditPage:
        """在同一学习库读取快照中统计并分页读取不可变重发布关联。"""
        _validate_audit_page_arguments(limit=limit, offset=offset)
        with self._connect() as connection:
            connection.execute("BEGIN")
            total_linked_audits = _count_linked_practice_correction_audits(connection)
            rows = connection.execute(
                f"""
                SELECT {_PRACTICE_CORRECTION_REQUEST_COLUMNS}
                FROM practice_correction_requests AS request
                WHERE EXISTS (
                    SELECT 1
                    FROM practice_correction_republications AS republication
                    WHERE republication.request_id = request.request_id
                )
                ORDER BY request.created_at ASC, request.request_id ASC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
            requests = tuple(
                _practice_correction_request_from_row(row) for row in rows
            )
            republications = _republications_by_request_id_for_connection(
                connection,
                (request.request_id for request in requests),
            )
            events = _events_by_request_id_for_connection(
                connection,
                (request.request_id for request in requests),
            )
        return LinkedPracticeCorrectionAuditPage(
            total_linked_audits=total_linked_audits,
            audits=tuple(
                PracticeCorrectionAudit(
                    request=request,
                    republication=republications.get(request.request_id),
                    events=events.get(request.request_id, ()),
                )
                for request in requests
            ),
        )

    def count_linked_practice_correction_audits(self) -> int:
        """返回当前学习库中不可变重发布关联的精确数量。"""
        with self._connect() as connection:
            return _count_linked_practice_correction_audits(connection)

    def resolve_practice_correction_request(
        self,
        resolution: PracticeCorrectionResolutionInput,
    ) -> PracticeCorrectionRequest | None:
        """追加管理员处置事件，不修改原始练习记录或服务端判分。"""
        request_id = _validate_identifier(resolution.request_id, "复核申请标识")
        resolver_id = _validate_identifier(resolution.resolver_id, "处置人标识")
        notes = _normalize_optional_text(resolution.notes, "处置备注", max_length=2_000)
        status = PracticeCorrectionRequestStatus(resolution.resolution.value)
        resolved_at = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE practice_correction_requests
                SET status = ?, resolved_by = ?, resolution_notes = ?, resolved_at = ?
                WHERE request_id = ? AND status = ?
                """,
                (
                    status.value,
                    resolver_id,
                    notes,
                    resolved_at,
                    request_id,
                    PracticeCorrectionRequestStatus.PENDING.value,
                ),
            )
            if cursor.rowcount != 1:
                existing = connection.execute(
                    """
                    SELECT request_id
                    FROM practice_correction_requests
                    WHERE request_id = ?
                    """,
                    (request_id,),
                ).fetchone()
                if existing is None:
                    return None
                raise PracticeCorrectionRequestAlreadyResolvedError(
                    "复核申请已完成处置"
                )
            _insert_practice_correction_event(
                connection,
                request_id=request_id,
                actor_id=resolver_id,
                event_type="resolved",
                status=status,
                notes=notes,
                created_at=resolved_at,
            )
            row = connection.execute(
                f"""
                SELECT {_PRACTICE_CORRECTION_REQUEST_COLUMNS}
                FROM practice_correction_requests
                WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()
        return _practice_correction_request_from_row(row)

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

    def _migrate_v3(self, connection: sqlite3.Connection) -> None:
        """创建不可变练习账本的更正申请与追加式处置事件表。"""
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
            CREATE INDEX idx_practice_correction_requests_user_created
            ON practice_correction_requests (user_id, created_at)
            """
        )
        connection.execute(
            """
            CREATE INDEX idx_practice_correction_requests_status_created
            ON practice_correction_requests (status, created_at)
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
            CREATE INDEX idx_practice_correction_events_request_created
            ON practice_correction_events (request_id, created_at)
            """
        )

    def _migrate_v4(self, connection: sqlite3.Connection) -> None:
        """创建复核申请到新发布版本的一对一不可变关联与审计索引。"""
        connection.execute(
            """
            CREATE TABLE practice_correction_republications (
                request_id TEXT PRIMARY KEY,
                question_id TEXT NOT NULL,
                previous_content_version TEXT NOT NULL,
                new_content_version TEXT NOT NULL,
                new_content_hash TEXT NOT NULL,
                linked_by TEXT NOT NULL,
                linked_at TEXT NOT NULL,
                UNIQUE (question_id, new_content_version)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX idx_practice_correction_republications_question_version
            ON practice_correction_republications (
                question_id, previous_content_version, new_content_version
            )
            """
        )

    def _republications_by_request_id(
        self,
        request_ids: Iterable[str],
    ) -> dict[str, PracticeCorrectionRepublication]:
        with self._connect() as connection:
            return _republications_by_request_id_for_connection(connection, request_ids)

    def _events_by_request_id(
        self,
        request_ids: Iterable[str],
    ) -> dict[str, tuple[PracticeCorrectionEvent, ...]]:
        with self._connect() as connection:
            return _events_by_request_id_for_connection(connection, request_ids)

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


def _validate_audit_page_arguments(*, limit: int, offset: int) -> None:
    if not 1 <= limit <= 100:
        raise ValueError("审计查询数量必须介于 1 到 100")
    if not 0 <= offset <= 10_000:
        raise ValueError("审计查询偏移量必须介于 0 到 10000")


def _count_linked_practice_correction_audits(
    connection: sqlite3.Connection,
) -> int:
    row = connection.execute(
        """
        SELECT COUNT(*) AS audit_count
        FROM practice_correction_requests AS request
        WHERE EXISTS (
            SELECT 1
            FROM practice_correction_republications AS republication
            WHERE republication.request_id = request.request_id
        )
        """
    ).fetchone()
    return int(row["audit_count"])


def _republications_by_request_id_for_connection(
    connection: sqlite3.Connection,
    request_ids: Iterable[str],
) -> dict[str, PracticeCorrectionRepublication]:
    normalized_request_ids = tuple(request_ids)
    if not normalized_request_ids:
        return {}
    placeholders = ", ".join("?" for _ in normalized_request_ids)
    rows = connection.execute(
        f"""
        SELECT {_PRACTICE_CORRECTION_REPUBLICATION_COLUMNS}
        FROM practice_correction_republications
        WHERE request_id IN ({placeholders})
        """,
        normalized_request_ids,
    ).fetchall()
    return {
        row["request_id"]: _practice_correction_republication_from_row(row)
        for row in rows
    }


def _events_by_request_id_for_connection(
    connection: sqlite3.Connection,
    request_ids: Iterable[str],
) -> dict[str, tuple[PracticeCorrectionEvent, ...]]:
    normalized_request_ids = tuple(request_ids)
    if not normalized_request_ids:
        return {}
    placeholders = ", ".join("?" for _ in normalized_request_ids)
    rows = connection.execute(
        f"""
        SELECT event_id, request_id, actor_id, event_type, status, notes,
               created_at
        FROM practice_correction_events
        WHERE request_id IN ({placeholders})
        ORDER BY request_id ASC, created_at ASC, rowid ASC
        """,
        normalized_request_ids,
    ).fetchall()
    events_by_request_id: dict[str, list[PracticeCorrectionEvent]] = {}
    for row in rows:
        event = _practice_correction_event_from_row(row)
        events_by_request_id.setdefault(event.request_id, []).append(event)
    return {
        request_id: tuple(events)
        for request_id, events in events_by_request_id.items()
    }


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


def _practice_correction_request_values(
    request: PracticeCorrectionRequest,
) -> tuple[object, ...]:
    return (
        request.request_id,
        request.record_id,
        request.user_id,
        request.question_id,
        request.content_version,
        request.reason,
        request.status.value,
        request.created_at,
        request.resolved_by,
        request.resolution_notes,
        request.resolved_at,
    )


def _practice_correction_republication_values(
    republication: PracticeCorrectionRepublication,
) -> tuple[str, ...]:
    return (
        republication.request_id,
        republication.question_id,
        republication.previous_content_version,
        republication.new_content_version,
        republication.new_content_hash,
        republication.linked_by,
        republication.linked_at,
    )


def _insert_practice_correction_event(
    connection: sqlite3.Connection,
    *,
    request_id: str,
    actor_id: str,
    event_type: str,
    status: PracticeCorrectionRequestStatus,
    notes: str | None,
    created_at: str,
) -> None:
    connection.execute(
        """
        INSERT INTO practice_correction_events (
            event_id, request_id, actor_id, event_type, status, notes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid4()),
            request_id,
            actor_id,
            event_type,
            status.value,
            notes,
            created_at,
        ),
    )


def _practice_correction_request_from_row(
    row: sqlite3.Row,
) -> PracticeCorrectionRequest:
    return PracticeCorrectionRequest(
        request_id=row["request_id"],
        record_id=row["record_id"],
        user_id=row["user_id"],
        question_id=row["question_id"],
        content_version=row["content_version"],
        reason=row["reason"],
        status=PracticeCorrectionRequestStatus(row["status"]),
        created_at=row["created_at"],
        resolved_by=row["resolved_by"],
        resolution_notes=row["resolution_notes"],
        resolved_at=row["resolved_at"],
    )


def _practice_correction_outcome_from(
    request: PracticeCorrectionRequest,
    republication: PracticeCorrectionRepublication | None,
) -> PracticeCorrectionOutcome:
    kind = PracticeCorrectionOutcomeKind(request.status.value)
    messages = {
        PracticeCorrectionOutcomeKind.PENDING: "复核申请已提交，正在处理中。",
        PracticeCorrectionOutcomeKind.RECORD_CONFIRMED: (
            "复核已完成，当前练习记录已确认。"
        ),
        PracticeCorrectionOutcomeKind.REPUBLICATION_REQUIRED: (
            "复核已完成，该题目将按发布流程复核；若发布新版本，"
            "新版本会作为独立练习重新推荐。"
        ),
    }
    republished_content_version = (
        republication.new_content_version if republication is not None else None
    )
    if republished_content_version is not None:
        messages[PracticeCorrectionOutcomeKind.REPUBLICATION_REQUIRED] = (
            "复核后的新版本已发布，可作为独立练习完成。"
        )
    return PracticeCorrectionOutcome(
        request_id=request.request_id,
        record_id=request.record_id,
        question_id=request.question_id,
        content_version=request.content_version,
        kind=kind,
        message=messages[kind],
        created_at=request.created_at,
        resolved_at=request.resolved_at,
        republished_content_version=republished_content_version,
    )


def _practice_correction_republication_from_row(
    row: sqlite3.Row,
) -> PracticeCorrectionRepublication:
    return PracticeCorrectionRepublication(
        request_id=row["request_id"],
        question_id=row["question_id"],
        previous_content_version=row["previous_content_version"],
        new_content_version=row["new_content_version"],
        new_content_hash=row["new_content_hash"],
        linked_by=row["linked_by"],
        linked_at=row["linked_at"],
    )


def _practice_correction_event_from_row(row: sqlite3.Row) -> PracticeCorrectionEvent:
    return PracticeCorrectionEvent(
        event_id=row["event_id"],
        request_id=row["request_id"],
        actor_id=row["actor_id"],
        event_type=row["event_type"],
        status=PracticeCorrectionRequestStatus(row["status"]),
        notes=row["notes"],
        created_at=row["created_at"],
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


def _validate_content_hash(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("内容摘要必须是 64 位 SHA-256 十六进制字符串")
    return normalized


def _validate_reason(value: str) -> str:
    return _validate_text(value, "复核理由", max_length=2_000)


def _normalize_optional_text(
    value: str | None,
    label: str,
    *,
    max_length: int,
) -> str | None:
    if value is None:
        return None
    return _validate_text(value, label, max_length=max_length)


def _validate_text(value: str, label: str, *, max_length: int) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label}不能为空")
    if len(normalized) > max_length:
        raise ValueError(f"{label}不能超过 {max_length} 个字符")
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
