"""内容摘要绑定审核的版本化题库与个人练习推荐服务。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from logic_qa.database_governance import (
    DatabaseBackup,
    DatabaseMigration,
    SQLiteDatabaseManager,
)
from logic_qa.engine import InferenceEngine
from logic_qa.grouping_matching_solver import (
    GroupConstraint,
    GroupConstraintType,
    GroupingSolver,
    GroupingSolveStatus,
    MatchConstraint,
    MatchConstraintType,
    MatchingSolver,
    MatchingSolveStatus,
)
from logic_qa.learning_profile import LearningProfile
from logic_qa.models import ImplicationRule, Literal, ProofStep, VerificationStatus
from logic_qa.ordering_solver import (
    OrderingConstraint,
    OrderingConstraintType,
    OrderingSolver,
    OrderingSolveStatus,
)
from logic_qa.quality_operations import QuestionReviewRecord, QuestionReviewStatus

_MAX_STEM_LENGTH = 5_000
_MAX_OPTION_COUNT = 8
_MAX_OPTION_LENGTH = 1_000
_MAX_TAG_COUNT = 20
_MAX_TAG_LENGTH = 64
_MAX_FORMALIZATION_FACT_COUNT = 50
_MAX_FORMALIZATION_RULE_COUNT = 100
_MAX_FORMALIZATION_SOURCE_LENGTH = 5_000
_MAX_CONSTRAINT_ITEM_COUNT = 50
_MAX_CONSTRAINT_GROUP_COUNT = 50
_MAX_CONSTRAINT_COUNT = 100
_MAX_CONSTRAINT_NAME_LENGTH = 128


class FormalizationKind(StrEnum):
    """题库可复现形式化资产支持的确定性求解器类型。"""

    PROPOSITIONAL = "propositional"
    ORDERING = "ordering"
    GROUPING = "grouping"
    MATCHING = "matching"


class QuestionVersionLifecycleAction(StrEnum):
    """已发布版本可由受控治理链路追加的生命周期动作。"""

    DEACTIVATED = "deactivated"
    REACTIVATED = "reactivated"
    SUPERSEDED = "superseded"


@dataclass(frozen=True, slots=True)
class FormalizationRule:
    """候选题形式化资产中的单前提蕴含规则。"""

    premise: str
    conclusion: str
    source_text: str | None = None


@dataclass(frozen=True, slots=True)
class OptionAssertion:
    """一个选项所声明的可复现验证结果，用于确定唯一正确答案。"""

    option: str
    claim_status: str
    claim_solution_count: int | None = None


@dataclass(frozen=True, slots=True)
class PropositionalFormalization:
    """可由确定性命题推理内核复现的题目形式化资产。"""

    facts: tuple[str, ...]
    rules: tuple[FormalizationRule, ...]
    query: str
    expected_status: VerificationStatus
    expected_answer: str
    option_assertions: tuple[OptionAssertion, ...] = ()


@dataclass(frozen=True, slots=True)
class OrderingFormalization:
    """由完整排列枚举器复现的排序题形式化资产。"""

    items: tuple[str, ...]
    constraints: tuple[OrderingConstraint, ...]
    expected_status: OrderingSolveStatus
    expected_solution_count: int
    expected_answer: str
    option_assertions: tuple[OptionAssertion, ...] = ()


@dataclass(frozen=True, slots=True)
class GroupingFormalization:
    """由完整分配枚举器复现的分组题形式化资产。"""

    items: tuple[str, ...]
    groups: tuple[str, ...]
    max_group_size: int
    constraints: tuple[GroupConstraint, ...]
    expected_status: GroupingSolveStatus
    expected_solution_count: int
    expected_answer: str
    option_assertions: tuple[OptionAssertion, ...] = ()


@dataclass(frozen=True, slots=True)
class MatchingFormalization:
    """由一对一排列枚举器复现的匹配题形式化资产。"""

    items: tuple[str, ...]
    targets: tuple[str, ...]
    constraints: tuple[MatchConstraint, ...]
    expected_status: MatchingSolveStatus
    expected_solution_count: int
    expected_answer: str
    option_assertions: tuple[OptionAssertion, ...] = ()


type QuestionFormalization = (
    PropositionalFormalization
    | OrderingFormalization
    | GroupingFormalization
    | MatchingFormalization
)
type FormalizationStatus = (
    VerificationStatus
    | OrderingSolveStatus
    | GroupingSolveStatus
    | MatchingSolveStatus
)


@dataclass(frozen=True, slots=True)
class FormalizationVerification:
    """一次可复现形式化验证的结果与可审计证明信息。"""

    kind: FormalizationKind
    expected_status: FormalizationStatus
    actual_status: FormalizationStatus
    expected_solution_count: int | None
    actual_solution_count: int | None
    proof_steps: tuple[ProofStep, ...]
    known_literals: tuple[Literal, ...]
    conflict: tuple[Literal, Literal] | None
    evidence: str | None
    selected_option: str
    matching_options: tuple[str, ...]

    @property
    def matches_expected_status(self) -> bool:
        return self.actual_status == self.expected_status


@dataclass(frozen=True, slots=True)
class FormalizationVerificationEvent:
    """已发布候选的一次不可变形式化验证审计事件。"""

    question_id: str
    content_version: str
    content_hash: str
    formalization_version: str
    kind: FormalizationKind
    expected_status: FormalizationStatus
    actual_status: FormalizationStatus
    expected_solution_count: int | None
    actual_solution_count: int | None
    proof_steps: tuple[ProofStep, ...]
    known_literals: tuple[Literal, ...]
    conflict: tuple[Literal, Literal] | None
    evidence: str | None
    selected_option: str
    matching_options: tuple[str, ...]
    created_at: str


@dataclass(frozen=True, slots=True)
class QuestionPublicationInput:
    """管理员提交、待审核或发布的规范化题目内容。"""

    question_id: str
    content_version: str
    question_type: str
    stem: str
    options: tuple[str, ...] = ()
    error_tags: tuple[str, ...] = ()
    knowledge_tags: tuple[str, ...] = ()
    formalization_version: str = ""
    formalization: QuestionFormalization | None = None


@dataclass(frozen=True, slots=True)
class QuestionCandidate:
    """候选题目的完整管理员视图，含不可变内容摘要。"""

    publication: QuestionPublicationInput
    content_hash: str


@dataclass(frozen=True, slots=True)
class PublishedQuestion:
    """题目版本的内部发布记录，含内容摘要和内部标签。"""

    question_id: str
    content_version: str
    question_type: str
    stem: str
    options: tuple[str, ...]
    error_tags: tuple[str, ...]
    knowledge_tags: tuple[str, ...]
    formalization_version: str
    formalization: QuestionFormalization
    content_hash: str
    published_at: str


@dataclass(frozen=True, slots=True)
class PublishedQuestionVerificationSnapshot:
    """同一题库读取快照中的历史发布事实及其活动状态。"""

    question: PublishedQuestion | None
    is_active: bool | None


@dataclass(frozen=True, slots=True)
class QuestionVersionLifecycleEvent:
    """已发布版本状态变更时追加的不可变治理事件。"""

    event_id: str
    question_id: str
    content_version: str
    content_hash: str
    action: QuestionVersionLifecycleAction
    actor_id: str
    replaced_content_version: str | None
    reason: str
    created_at: str


@dataclass(frozen=True, slots=True)
class LearnerQuestion:
    """学习者可见题目，不包含审核、答案、摘要或内部标签。"""

    question_id: str
    content_version: str
    question_type: str
    stem: str
    options: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PracticeRecommendation:
    """基于单一用户画像匹配的学习者题目与展示级原因。"""

    question: LearnerQuestion
    matched_tags: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class GradedPracticeAttempt:
    """服务端判分后的学习记录输入，内部标签不应返回给学习者。"""

    question: LearnerQuestion
    is_correct: bool
    error_tags: tuple[str, ...]
    knowledge_tags: tuple[str, ...]


class QuestionBankStore:
    """保存不可变题目版本，只发布与审核内容精确匹配的候选。"""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._database = SQLiteDatabaseManager(
            database_path,
            (
                DatabaseMigration(
                    1,
                    "create_versioned_question_bank",
                    self._migrate_v1,
                ),
                DatabaseMigration(
                    2,
                    "add_question_version_lifecycle_events",
                    self._migrate_v2,
                ),
                DatabaseMigration(
                    3,
                    "allow_published_version_supersession_events",
                    self._migrate_v3,
                ),
                DatabaseMigration(
                    4,
                    "enforce_one_active_question_version",
                    self._migrate_v4,
                ),
                DatabaseMigration(
                    5,
                    "protect_question_version_lifecycle_audit",
                    self._migrate_v5,
                ),
                DatabaseMigration(
                    6,
                    "validate_question_version_lifecycle_references",
                    self._migrate_v6,
                ),
                DatabaseMigration(
                    7,
                    "protect_formalization_verification_audit",
                    self._migrate_v7,
                ),
                DatabaseMigration(
                    8,
                    "validate_formalization_verification_references",
                    self._migrate_v8,
                ),
                DatabaseMigration(
                    9,
                    "protect_candidate_snapshots_and_publication_events",
                    self._migrate_v9,
                ),
            ),
        )
        self._database.migrate()

    def schema_version(self) -> int:
        """返回当前题库数据库已完成的最高迁移版本。"""
        return self._database.schema_version()

    def create_backup(self, destination_directory: Path) -> DatabaseBackup:
        """创建经过完整性校验的题库一致性备份。"""
        return self._database.backup(destination_directory)

    def load_backup(self, manifest_path: Path) -> DatabaseBackup:
        """从已持久化的备份清单读取题库恢复元数据。"""
        return self._database.load_backup(manifest_path)

    def restore_backup(self, backup: DatabaseBackup) -> None:
        """仅从经校验且属于当前题库的备份恢复数据。"""
        self._database.restore(backup)

    def prepare_candidate(
        self,
        publication: QuestionPublicationInput,
    ) -> QuestionCandidate:
        """规范化候选内容并计算稳定 SHA-256 摘要，供审核和发布共用。"""
        normalized = _normalize_publication(publication)
        return QuestionCandidate(
            publication=normalized,
            content_hash=_content_hash_for(normalized),
        )

    def submit_candidate(
        self,
        publication: QuestionPublicationInput,
    ) -> QuestionCandidate:
        """持久化不可变候选快照；重复提交同一内容时返回原快照。"""
        candidate = self.prepare_candidate(publication)
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO question_candidates (
                    question_id, content_version, content_hash, question_type, stem,
                    options, error_tags, knowledge_tags, formalization_version,
                    formalization, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(question_id, content_version, content_hash) DO NOTHING
                """,
                (
                    candidate.publication.question_id,
                    candidate.publication.content_version,
                    candidate.content_hash,
                    candidate.publication.question_type,
                    candidate.publication.stem,
                    _serialize_values(candidate.publication.options),
                    _serialize_values(candidate.publication.error_tags),
                    _serialize_values(candidate.publication.knowledge_tags),
                    candidate.publication.formalization_version,
                    _serialize_formalization(candidate.publication.formalization),
                    created_at,
                ),
            )
            row = connection.execute(
                """
                SELECT question_id, content_version, content_hash, question_type, stem,
                       options, error_tags, knowledge_tags, formalization_version,
                       formalization
                FROM question_candidates
                WHERE question_id = ? AND content_version = ? AND content_hash = ?
                """,
                (
                    candidate.publication.question_id,
                    candidate.publication.content_version,
                    candidate.content_hash,
                ),
            ).fetchone()
        if row is None:
            raise RuntimeError("候选题目快照保存失败")
        stored_candidate = _candidate_from_row(row)
        if stored_candidate != candidate:
            raise ValueError("候选内容摘要与已保存快照冲突，拒绝覆盖不可变候选内容")
        return stored_candidate

    def get_candidate(
        self,
        question_id: str,
        content_version: str,
        content_hash: str,
    ) -> QuestionCandidate | None:
        """精确读取已持久化的候选题目快照。"""
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
                SELECT question_id, content_version, content_hash, question_type, stem,
                       options, error_tags, knowledge_tags, formalization_version,
                       formalization
                FROM question_candidates
                WHERE question_id = ? AND content_version = ? AND content_hash = ?
                """,
                (
                    normalized_question_id,
                    normalized_content_version,
                    normalized_content_hash,
                ),
            ).fetchone()
        return _candidate_from_row(row) if row else None

    def publish(
        self,
        candidate: QuestionCandidate,
        publisher_id: str,
        review: QuestionReviewRecord | None,
    ) -> PublishedQuestion:
        """仅发布与审核题号、版本、摘要及形式化版本完全一致的内容。"""
        normalized_publisher = _validate_text(
            publisher_id,
            "发布人标识",
            max_length=128,
        )
        stored_candidate = self.get_candidate(
            candidate.publication.question_id,
            candidate.publication.content_version,
            candidate.content_hash,
        )
        if stored_candidate is None:
            raise ValueError("候选内容尚未提交，不能发布")
        if stored_candidate != candidate:
            raise ValueError("候选快照与待发布内容不一致，不能发布")
        verification = self.verify_candidate_formalization(candidate)
        self._validate_formalization_verification(verification)
        verification = self._evaluate_option_semantics(candidate, verification)
        self._validate_review(candidate, review)

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT 1 FROM question_versions
                WHERE question_id = ? AND content_version = ?
                """,
                (
                    candidate.publication.question_id,
                    candidate.publication.content_version,
                ),
            ).fetchone()
            if existing is not None:
                raise ValueError("该题目内容版本已发布，已发布版本不可覆盖")
            active_rows = connection.execute(
                """
                SELECT question_id, content_version, content_hash, question_type, stem,
                       options, error_tags, knowledge_tags, formalization_version,
                       formalization, published_at
                FROM question_versions
                WHERE question_id = ? AND is_active = 1
                ORDER BY content_version ASC
                """,
                (candidate.publication.question_id,),
            ).fetchall()
            if len(active_rows) > 1:
                raise ValueError("该题目存在多个活动版本，拒绝发布新版本")
            superseded_question = (
                _question_from_row(active_rows[0]) if active_rows else None
            )
            published_at = datetime.now(UTC).isoformat()
            question = PublishedQuestion(
                question_id=candidate.publication.question_id,
                content_version=candidate.publication.content_version,
                question_type=candidate.publication.question_type,
                stem=candidate.publication.stem,
                options=candidate.publication.options,
                error_tags=candidate.publication.error_tags,
                knowledge_tags=candidate.publication.knowledge_tags,
                formalization_version=candidate.publication.formalization_version,
                formalization=candidate.publication.formalization,
                content_hash=candidate.content_hash,
                published_at=published_at,
            )
            connection.execute(
                "UPDATE question_versions SET is_active = 0 WHERE question_id = ?",
                (question.question_id,),
            )
            connection.execute(
                """
                INSERT INTO question_versions (
                    question_id, content_version, content_hash, question_type, stem,
                    options, error_tags, knowledge_tags, formalization_version,
                    formalization, is_active, publisher_id, published_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    question.question_id,
                    question.content_version,
                    question.content_hash,
                    question.question_type,
                    question.stem,
                    _serialize_values(question.options),
                    _serialize_values(question.error_tags),
                    _serialize_values(question.knowledge_tags),
                    question.formalization_version,
                    _serialize_formalization(question.formalization),
                    normalized_publisher,
                    question.published_at,
                ),
            )
            connection.execute(
                """
                INSERT INTO question_publication_events (
                    event_id, question_id, content_version, content_hash, publisher_id,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    question.question_id,
                    question.content_version,
                    question.content_hash,
                    normalized_publisher,
                    question.published_at,
                ),
            )
            _insert_formalization_verification_event(
                connection,
                question=question,
                verification=verification,
                created_at=question.published_at,
            )
            if superseded_question is not None:
                _insert_question_version_lifecycle_event(
                    connection,
                    question=superseded_question,
                    action=QuestionVersionLifecycleAction.SUPERSEDED,
                    actor_id=normalized_publisher,
                    replaced_content_version=question.content_version,
                    reason="发布新的已审核内容版本",
                    created_at=question.published_at,
                )
        return question

    def active_questions(self) -> tuple[PublishedQuestion, ...]:
        """返回当前活动题目版本的内部记录，按稳定顺序排列。"""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT question_id, content_version, content_hash, question_type, stem,
                       options, error_tags, knowledge_tags, formalization_version,
                       formalization, published_at
                FROM question_versions
                WHERE is_active = 1
                ORDER BY question_id ASC, content_version ASC
                """
            ).fetchall()
        return tuple(_question_from_row(row) for row in rows)

    def get_published_question(
        self,
        question_id: str,
        content_version: str,
    ) -> PublishedQuestion | None:
        """精确读取已发布版本，供跨存储投影核验历史发布事实。"""
        return self.get_published_question_verification_snapshot(
            question_id,
            content_version,
        ).question

    def get_active_published_question(
        self,
        question_id: str,
        content_version: str,
    ) -> PublishedQuestion | None:
        """精确读取当前活动发布版本，供受控治理链路核验新关联资格。"""
        snapshot = self.get_published_question_verification_snapshot(
            question_id,
            content_version,
        )
        return snapshot.question if snapshot.is_active else None

    def get_published_question_verification_snapshot(
        self,
        question_id: str,
        content_version: str,
    ) -> PublishedQuestionVerificationSnapshot:
        """在同一题库读取快照中取得精确发布版本及其活动状态。"""
        normalized_question_id = _validate_text(question_id, "题目标识", max_length=128)
        normalized_content_version = _validate_text(
            content_version,
            "内容版本",
            max_length=128,
        )
        with self._connect() as connection:
            connection.execute("BEGIN")
            row = connection.execute(
                """
                SELECT question_id, content_version, content_hash, question_type, stem,
                       options, error_tags, knowledge_tags, formalization_version,
                       formalization, published_at, is_active
                FROM question_versions
                WHERE question_id = ? AND content_version = ?
                """,
                (normalized_question_id, normalized_content_version),
            ).fetchone()
        if row is None:
            return PublishedQuestionVerificationSnapshot(
                question=None,
                is_active=None,
            )
        return PublishedQuestionVerificationSnapshot(
            question=_question_from_row(row),
            is_active=bool(row["is_active"]),
        )

    def deactivate_active_version(
        self,
        question_id: str,
        content_version: str,
        *,
        actor_id: str,
        reason: str,
    ) -> QuestionVersionLifecycleEvent | None:
        """下线当前活动版本并追加治理事件，不删除历史版本或审计事实。"""
        normalized_question_id = _validate_text(question_id, "题目标识", max_length=128)
        normalized_content_version = _validate_text(
            content_version,
            "内容版本",
            max_length=128,
        )
        normalized_actor_id = _validate_text(actor_id, "操作人标识", max_length=128)
        normalized_reason = _validate_text(reason, "下线原因", max_length=2_000)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT question_id, content_version, content_hash, question_type, stem,
                       options, error_tags, knowledge_tags, formalization_version,
                       formalization, published_at, is_active
                FROM question_versions
                WHERE question_id = ? AND content_version = ?
                """,
                (normalized_question_id, normalized_content_version),
            ).fetchone()
            if row is None:
                return None
            if not row["is_active"]:
                raise ValueError("该题目版本当前未活动，不能下线")
            question = _question_from_row(row)
            created_at = datetime.now(UTC).isoformat()
            cursor = connection.execute(
                """
                UPDATE question_versions
                SET is_active = 0
                WHERE question_id = ? AND content_version = ? AND is_active = 1
                """,
                (normalized_question_id, normalized_content_version),
            )
            if cursor.rowcount != 1:
                raise ValueError("该题目版本活动状态已变化，请重新查询")
            event = _insert_question_version_lifecycle_event(
                connection,
                question=question,
                action=QuestionVersionLifecycleAction.DEACTIVATED,
                actor_id=normalized_actor_id,
                replaced_content_version=None,
                reason=normalized_reason,
                created_at=created_at,
            )
        return event

    def reactivate_published_version(
        self,
        question_id: str,
        content_version: str,
        *,
        actor_id: str,
        reason: str,
        review: QuestionReviewRecord | None,
    ) -> QuestionVersionLifecycleEvent | None:
        """复核既有候选、审核和形式化资产后重新激活历史发布版本。"""
        normalized_question_id = _validate_text(question_id, "题目标识", max_length=128)
        normalized_content_version = _validate_text(
            content_version,
            "内容版本",
            max_length=128,
        )
        normalized_actor_id = _validate_text(actor_id, "操作人标识", max_length=128)
        normalized_reason = _validate_text(reason, "重新激活原因", max_length=2_000)
        question = self.get_published_question(
            normalized_question_id,
            normalized_content_version,
        )
        if question is None:
            return None
        candidate = self.get_candidate(
            question.question_id,
            question.content_version,
            question.content_hash,
        )
        if candidate is None:
            raise ValueError("已发布版本缺少精确候选快照，不能重新激活")
        _validate_candidate_matches_published_question(candidate, question)
        verification = self.verify_candidate_formalization(candidate)
        self._validate_formalization_verification(verification)
        verification = self._evaluate_option_semantics(candidate, verification)
        self._validate_review(candidate, review)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            target = connection.execute(
                """
                SELECT is_active
                FROM question_versions
                WHERE question_id = ? AND content_version = ?
                """,
                (normalized_question_id, normalized_content_version),
            ).fetchone()
            if target is None:
                return None
            if target["is_active"]:
                raise ValueError("该题目版本当前已活动，无需重新激活")
            active_rows = connection.execute(
                """
                SELECT question_id, content_version, content_hash, question_type, stem,
                       options, error_tags, knowledge_tags, formalization_version,
                       formalization, published_at
                FROM question_versions
                WHERE question_id = ? AND is_active = 1
                ORDER BY content_version ASC
                """,
                (normalized_question_id,),
            ).fetchall()
            if len(active_rows) > 1:
                raise ValueError("该题目存在多个活动版本，拒绝执行重新激活")
            superseded_question = (
                _question_from_row(active_rows[0]) if active_rows else None
            )
            replaced_content_version = (
                superseded_question.content_version if superseded_question else None
            )
            created_at = datetime.now(UTC).isoformat()
            connection.execute(
                "UPDATE question_versions SET is_active = 0 WHERE question_id = ?",
                (normalized_question_id,),
            )
            cursor = connection.execute(
                """
                UPDATE question_versions
                SET is_active = 1
                WHERE question_id = ? AND content_version = ? AND is_active = 0
                """,
                (normalized_question_id, normalized_content_version),
            )
            if cursor.rowcount != 1:
                raise ValueError("该题目版本活动状态已变化，请重新查询")
            _insert_formalization_verification_event(
                connection,
                question=question,
                verification=verification,
                created_at=created_at,
            )
            if superseded_question is not None:
                _insert_question_version_lifecycle_event(
                    connection,
                    question=superseded_question,
                    action=QuestionVersionLifecycleAction.SUPERSEDED,
                    actor_id=normalized_actor_id,
                    replaced_content_version=question.content_version,
                    reason="重新激活已审核历史版本",
                    created_at=created_at,
                )
            event = _insert_question_version_lifecycle_event(
                connection,
                question=question,
                action=QuestionVersionLifecycleAction.REACTIVATED,
                actor_id=normalized_actor_id,
                replaced_content_version=replaced_content_version,
                reason=normalized_reason,
                created_at=created_at,
            )
        return event

    def list_question_version_lifecycle_events(
        self,
        question_id: str,
    ) -> tuple[QuestionVersionLifecycleEvent, ...]:
        """按题号回查下线、替换与重新激活的不可变治理事件。"""
        normalized_question_id = _validate_text(question_id, "题目标识", max_length=128)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, question_id, content_version, content_hash, action,
                       actor_id, replaced_content_version, reason, created_at
                FROM question_version_lifecycle_events
                WHERE question_id = ?
                ORDER BY created_at ASC, rowid ASC
                """,
                (normalized_question_id,),
            ).fetchall()
        return tuple(_question_version_lifecycle_event_from_row(row) for row in rows)

    def get_active_learner_question(
        self,
        question_id: str,
        content_version: str,
    ) -> LearnerQuestion | None:
        """按题号和版本读取活动题目的最小学习者视图。"""
        question = self.get_active_published_question(question_id, content_version)
        return _learner_question_from(question) if question else None

    def grade_active_learner_answer(
        self,
        question_id: str,
        content_version: str,
        selected_option: str,
    ) -> GradedPracticeAttempt | None:
        """使用活动发布版本的已审计正确选项进行服务端判分。"""
        normalized_question_id = _validate_text(question_id, "题目标识", max_length=128)
        normalized_content_version = _validate_text(
            content_version,
            "内容版本",
            max_length=128,
        )
        normalized_selected_option = _validate_text(
            selected_option,
            "所选选项",
            max_length=_MAX_OPTION_LENGTH,
        )
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT version.question_id, version.content_version,
                       version.content_hash, version.question_type, version.stem,
                       version.options, version.error_tags, version.knowledge_tags,
                       version.formalization_version, version.formalization,
                       version.published_at, verification.selected_option
                FROM question_versions AS version
                INNER JOIN question_formalization_verification_events AS verification
                    ON verification.question_id = version.question_id
                    AND verification.content_version = version.content_version
                    AND verification.content_hash = version.content_hash
                WHERE version.question_id = ?
                    AND version.content_version = ?
                    AND version.is_active = 1
                ORDER BY verification.created_at DESC, verification.event_id DESC
                LIMIT 1
                """,
                (normalized_question_id, normalized_content_version),
            ).fetchone()
        if row is None:
            return None
        question = _question_from_row(row)
        learner_question = _learner_question_from(question)
        if normalized_selected_option not in learner_question.options:
            raise ValueError("所选选项不属于该题目")
        return GradedPracticeAttempt(
            question=learner_question,
            is_correct=normalized_selected_option == row["selected_option"],
            error_tags=question.error_tags,
            knowledge_tags=question.knowledge_tags,
        )

    def get_formalization_verification(
        self,
        question_id: str,
        content_version: str,
        content_hash: str,
    ) -> FormalizationVerificationEvent | None:
        """精确回查已发布候选的可复现验证审计事件。"""
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
                SELECT question_id, content_version, content_hash,
                       formalization_version, formalization_kind, expected_status,
                       actual_status, expected_solution_count, actual_solution_count,
                       proof_steps, known_literals, conflict, evidence, selected_option,
                       matching_options, created_at
                FROM question_formalization_verification_events
                WHERE question_id = ? AND content_version = ? AND content_hash = ?
                ORDER BY created_at DESC, event_id DESC
                LIMIT 1
                """,
                (
                    normalized_question_id,
                    normalized_content_version,
                    normalized_content_hash,
                ),
            ).fetchone()
        return _formalization_verification_event_from_row(row) if row else None

    def recommend(
        self,
        profile: LearningProfile,
        attempted_question_versions: tuple[tuple[str, str], ...],
        limit: int,
    ) -> tuple[PracticeRecommendation, ...]:
        """从未尝试的活动发布版本中生成不泄露内部标签的个人推荐。"""
        if not 1 <= limit <= 20:
            raise ValueError("推荐数量必须位于 1 到 20 之间")
        attempted = set(attempted_question_versions)
        error_weights = dict(profile.error_counts)
        knowledge_weights = {
            tag: 1.0 - mastery for tag, mastery in profile.knowledge_mastery
        }
        scored: list[tuple[float, PublishedQuestion, tuple[str, ...], str]] = []

        for question in self.active_questions():
            if (question.question_id, question.content_version) in attempted:
                continue
            error_matches = tuple(
                tag for tag in question.error_tags if tag in error_weights
            )
            knowledge_matches = tuple(
                tag for tag in question.knowledge_tags if tag in knowledge_weights
            )
            score = sum(error_weights[tag] * 100 for tag in error_matches)
            score += sum(knowledge_weights[tag] * 10 for tag in knowledge_matches)
            matched_tags = tuple(dict.fromkeys((*error_matches, *knowledge_matches)))
            reason = _recommendation_reason(error_matches, knowledge_matches)
            scored.append((score, question, matched_tags, reason))

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1].question_id,
                item[1].content_version,
            )
        )
        return tuple(
            PracticeRecommendation(
                question=_learner_question_from(question),
                matched_tags=matched_tags,
                reason=reason,
            )
            for _, question, matched_tags, reason in scored[:limit]
        )

    @staticmethod
    def _validate_review(
        candidate: QuestionCandidate,
        review: QuestionReviewRecord | None,
    ) -> None:
        if review is None:
            raise ValueError("题目当前内容尚无审核记录，不能发布")
        if review.status is not QuestionReviewStatus.APPROVED:
            raise ValueError("题目当前内容未通过审核，不能发布")
        if review.question_id != candidate.publication.question_id:
            raise ValueError("审核题目标识与待发布内容不一致，不能发布")
        if review.content_version != candidate.publication.content_version:
            raise ValueError("审核内容版本与待发布内容不一致，不能发布")
        if review.content_hash != candidate.content_hash:
            raise ValueError("审核内容摘要与待发布内容不一致，不能发布")
        if review.formalization_version != candidate.publication.formalization_version:
            raise ValueError("题目形式化版本与审核记录不一致，不能发布")
        formalization = candidate.publication.formalization
        if formalization is None:
            raise ValueError("候选题目缺少可复现的形式化资产")
        if review.verified_answer != _formalization_expected_answer(formalization):
            raise ValueError("审核核验答案与形式化预期答案不一致，不能发布")

    @staticmethod
    def verify_candidate_formalization(
        candidate: QuestionCandidate,
    ) -> FormalizationVerification:
        """使用题型对应的确定性内核复跑候选结构化资产。"""
        formalization = candidate.publication.formalization
        if formalization is None:
            raise ValueError("候选题目缺少可复现的形式化资产")
        match formalization:
            case PropositionalFormalization():
                return _verify_propositional_formalization(formalization)
            case OrderingFormalization():
                return _verify_ordering_formalization(formalization)
            case GroupingFormalization():
                return _verify_grouping_formalization(formalization)
            case MatchingFormalization():
                return _verify_matching_formalization(formalization)
        raise ValueError("候选题目形式化资产类型不受支持")

    @staticmethod
    def _evaluate_option_semantics(
        candidate: QuestionCandidate,
        verification: FormalizationVerification,
    ) -> FormalizationVerification:
        """验证全部选项断言，并产生可审计的唯一语义正确答案。"""
        formalization = candidate.publication.formalization
        if formalization is None:
            raise ValueError("候选题目缺少可复现的形式化资产")
        assertions = _formalization_option_assertions(formalization)
        if not candidate.publication.options:
            if assertions:
                raise ValueError("无选项题目不能提交选项断言")
            return replace(
                verification,
                selected_option=_formalization_expected_answer(formalization),
                matching_options=(),
            )
        if not assertions:
            raise ValueError("选择题必须提供全部选项的语义断言")
        assertion_options = tuple(assertion.option for assertion in assertions)
        if assertion_options != candidate.publication.options:
            raise ValueError("选项断言必须按题目选项顺序逐一绑定")
        matching_options = tuple(
            assertion.option
            for assertion in assertions
            if _option_assertion_matches(assertion, verification)
        )
        if len(matching_options) != 1:
            raise ValueError("选项语义必须唯一命中，不能发布")
        if matching_options[0] != _formalization_expected_answer(formalization):
            raise ValueError("形式化预期答案与选项语义断言不一致，不能发布")
        return replace(
            verification,
            selected_option=matching_options[0],
            matching_options=matching_options,
        )

    @staticmethod
    def _validate_formalization_verification(
        verification: FormalizationVerification,
    ) -> None:
        if not verification.matches_expected_status:
            raise ValueError(
                "形式化验证结果与预期不一致，不能发布："
                f"预期 {verification.expected_status.value}，"
                f"实际 {verification.actual_status.value}"
            )
        if verification.expected_solution_count is not None and (
            verification.actual_solution_count
            != verification.expected_solution_count
        ):
            raise ValueError(
                "形式化解空间数量与预期不一致，不能发布："
                f"预期 {verification.expected_solution_count}，"
                f"实际 {verification.actual_solution_count}"
            )
        if verification.kind is FormalizationKind.PROPOSITIONAL:
            if verification.actual_status is VerificationStatus.INCONSISTENT:
                raise ValueError("形式化条件存在矛盾，不能发布")
            if verification.actual_status is VerificationStatus.UNKNOWN:
                raise ValueError("形式化查询无法确定，不能发布")
            return
        if verification.actual_status in {
            OrderingSolveStatus.ITEM_LIMIT_EXCEEDED,
            GroupingSolveStatus.SEARCH_LIMIT_EXCEEDED,
            MatchingSolveStatus.ITEM_LIMIT_EXCEEDED,
        }:
            raise ValueError("形式化搜索超出安全上限，不能发布")
        if verification.actual_status in {
            OrderingSolveStatus.UNSATISFIABLE,
            GroupingSolveStatus.UNSATISFIABLE,
            MatchingSolveStatus.UNSATISFIABLE,
        }:
            raise ValueError("形式化约束无解，不能发布")

    def _migrate_v2(self, connection: sqlite3.Connection) -> None:
        """为已发布版本添加下线与重新激活的追加式治理审计表。"""
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS question_version_lifecycle_events (
                event_id TEXT PRIMARY KEY,
                question_id TEXT NOT NULL,
                content_version TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                action TEXT NOT NULL CHECK (action IN ('deactivated', 'reactivated')),
                actor_id TEXT NOT NULL,
                replaced_content_version TEXT,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_question_version_lifecycle_events_q_created
            ON question_version_lifecycle_events (question_id, created_at)
            """
        )

    def _migrate_v3(self, connection: sqlite3.Connection) -> None:
        """扩展生命周期动作约束，并原样保留既有不可变审计事件。"""
        connection.execute(
            """
            ALTER TABLE question_version_lifecycle_events
            RENAME TO question_version_lifecycle_events_v2
            """
        )
        connection.execute(
            """
            CREATE TABLE question_version_lifecycle_events (
                event_id TEXT PRIMARY KEY,
                question_id TEXT NOT NULL,
                content_version TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                action TEXT NOT NULL CHECK (
                    action IN ('deactivated', 'reactivated', 'superseded')
                ),
                actor_id TEXT NOT NULL,
                replaced_content_version TEXT,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO question_version_lifecycle_events (
                event_id, question_id, content_version, content_hash, action, actor_id,
                replaced_content_version, reason, created_at
            )
            SELECT event_id, question_id, content_version, content_hash, action,
                   actor_id, replaced_content_version, reason, created_at
            FROM question_version_lifecycle_events_v2
            ORDER BY rowid ASC
            """
        )
        connection.execute("DROP TABLE question_version_lifecycle_events_v2")
        connection.execute(
            """
            CREATE INDEX idx_question_version_lifecycle_events_q_created
            ON question_version_lifecycle_events (question_id, created_at)
            """
        )

    def _migrate_v4(self, connection: sqlite3.Connection) -> None:
        """在存储边界强制同一题目至多保留一个活动版本。"""
        duplicate = connection.execute(
            """
            SELECT question_id
            FROM question_versions
            WHERE is_active = 1
            GROUP BY question_id
            HAVING COUNT(*) > 1
            ORDER BY question_id ASC
            LIMIT 1
            """
        ).fetchone()
        if duplicate is not None:
            raise RuntimeError(
                "题库存在同题多个活动版本，拒绝启用单活动版本约束："
                f"{duplicate['question_id']}"
            )
        connection.execute(
            """
            CREATE UNIQUE INDEX idx_question_versions_one_active_per_question
            ON question_versions (question_id)
            WHERE is_active = 1
            """
        )

    def _migrate_v5(self, connection: sqlite3.Connection) -> None:
        """禁止直接改写或删除版本生命周期审计事件。"""
        connection.execute(
            """
            CREATE TRIGGER trg_question_version_lifecycle_events_no_update
            BEFORE UPDATE ON question_version_lifecycle_events
            BEGIN
                SELECT RAISE(ABORT, '版本生命周期审计事件不可修改');
            END
            """
        )
        connection.execute(
            """
            CREATE TRIGGER trg_question_version_lifecycle_events_no_delete
            BEFORE DELETE ON question_version_lifecycle_events
            BEGIN
                SELECT RAISE(ABORT, '版本生命周期审计事件不可删除');
            END
            """
        )

    def _migrate_v6(self, connection: sqlite3.Connection) -> None:
        """确保生命周期审计只引用真实且摘要一致的已发布版本。"""
        invalid = connection.execute(
            """
            SELECT event_id
            FROM question_version_lifecycle_events AS event
            LEFT JOIN question_versions AS version
                ON version.question_id = event.question_id
                AND version.content_version = event.content_version
                AND version.content_hash = event.content_hash
            LEFT JOIN question_versions AS replacement
                ON replacement.question_id = event.question_id
                AND replacement.content_version = event.replaced_content_version
            WHERE version.question_id IS NULL
                OR (
                    event.replaced_content_version IS NOT NULL
                    AND replacement.question_id IS NULL
                )
            ORDER BY event.rowid ASC
            LIMIT 1
            """
        ).fetchone()
        if invalid is not None:
            raise RuntimeError(
                "生命周期审计事件引用不存在或摘要不一致的题目版本，拒绝启用引用完整性约束："
                f"{invalid['event_id']}"
            )
        connection.execute(
            """
            CREATE TRIGGER trg_question_version_lifecycle_events_reference_versions
            BEFORE INSERT ON question_version_lifecycle_events
            BEGIN
                SELECT CASE
                    WHEN NOT EXISTS (
                        SELECT 1
                        FROM question_versions
                        WHERE question_id = NEW.question_id
                            AND content_version = NEW.content_version
                            AND content_hash = NEW.content_hash
                    )
                    THEN RAISE(ABORT, '生命周期审计事件必须引用摘要一致的已发布版本')
                END;
                SELECT CASE
                    WHEN NEW.replaced_content_version IS NOT NULL
                        AND NOT EXISTS (
                            SELECT 1
                            FROM question_versions
                            WHERE question_id = NEW.question_id
                                AND content_version = NEW.replaced_content_version
                        )
                    THEN RAISE(ABORT, '生命周期审计替代版本必须已发布且属于同一题目')
                END;
            END
            """
        )

    def _migrate_v7(self, connection: sqlite3.Connection) -> None:
        """禁止直接改写或删除形式化验证审计事件。"""
        connection.execute(
            """
            CREATE TRIGGER trg_question_formalization_verification_events_no_update
            BEFORE UPDATE ON question_formalization_verification_events
            BEGIN
                SELECT RAISE(ABORT, '形式化验证审计事件不可修改');
            END
            """
        )
        connection.execute(
            """
            CREATE TRIGGER trg_question_formalization_verification_events_no_delete
            BEFORE DELETE ON question_formalization_verification_events
            BEGIN
                SELECT RAISE(ABORT, '形式化验证审计事件不可删除');
            END
            """
        )

    def _migrate_v8(self, connection: sqlite3.Connection) -> None:
        """确保验证审计只引用真实且摘要一致的已发布版本。"""
        invalid = connection.execute(
            """
            SELECT event_id
            FROM question_formalization_verification_events AS event
            LEFT JOIN question_versions AS version
                ON version.question_id = event.question_id
                AND version.content_version = event.content_version
                AND version.content_hash = event.content_hash
            WHERE version.question_id IS NULL
            ORDER BY event.rowid ASC
            LIMIT 1
            """
        ).fetchone()
        if invalid is not None:
            raise RuntimeError(
                "形式化验证事件引用不存在或摘要不一致的题目版本，"
                "拒绝启用引用完整性约束："
                f"{invalid['event_id']}"
            )
        connection.execute(
            """
            CREATE TRIGGER
                trg_question_formalization_verification_events_reference_versions
            BEFORE INSERT ON question_formalization_verification_events
            BEGIN
                SELECT CASE
                    WHEN NOT EXISTS (
                        SELECT 1
                        FROM question_versions
                        WHERE question_id = NEW.question_id
                            AND content_version = NEW.content_version
                            AND content_hash = NEW.content_hash
                        )
                    THEN RAISE(ABORT, '形式化验证事件必须引用摘要一致的已发布版本')
                END;
            END
            """
        )

    def _migrate_v9(self, connection: sqlite3.Connection) -> None:
        """禁止改写或删除候选快照与发布事件。"""
        connection.execute(
            """
            CREATE TRIGGER trg_question_candidates_no_update
            BEFORE UPDATE ON question_candidates
            BEGIN
                SELECT RAISE(ABORT, '题目候选快照不可修改');
            END
            """
        )
        connection.execute(
            """
            CREATE TRIGGER trg_question_candidates_no_delete
            BEFORE DELETE ON question_candidates
            BEGIN
                SELECT RAISE(ABORT, '题目候选快照不可删除');
            END
            """
        )
        connection.execute(
            """
            CREATE TRIGGER trg_question_publication_events_no_update
            BEFORE UPDATE ON question_publication_events
            BEGIN
                SELECT RAISE(ABORT, '题目发布事件不可修改');
            END
            """
        )
        connection.execute(
            """
            CREATE TRIGGER trg_question_publication_events_no_delete
            BEFORE DELETE ON question_publication_events
            BEGIN
                SELECT RAISE(ABORT, '题目发布事件不可删除');
            END
            """
        )

    def _migrate_v1(self, connection: sqlite3.Connection) -> None:
        """建立候选、版本、发布与形式化验证审计表。"""
        self._archive_unbound_table(
            connection,
            "question_versions",
            {"content_hash", "formalization"},
        )
        self._archive_unbound_table(
            connection,
            "question_formalization_verification_events",
            {
                "formalization_kind",
                "expected_solution_count",
                "actual_solution_count",
                "evidence",
                "selected_option",
                "matching_options",
            },
        )
        self._archive_unbound_table(
            connection,
            "question_publication_events",
            {"content_hash"},
        )
        self._archive_unbound_table(
            connection,
            "question_candidates",
            {
                "question_id",
                "content_version",
                "content_hash",
                "question_type",
                "stem",
                "options",
                "error_tags",
                "knowledge_tags",
                "formalization_version",
                "formalization",
                "created_at",
            },
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS question_candidates (
                question_id TEXT NOT NULL,
                content_version TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                question_type TEXT NOT NULL,
                stem TEXT NOT NULL,
                options TEXT NOT NULL,
                error_tags TEXT NOT NULL,
                knowledge_tags TEXT NOT NULL,
                formalization_version TEXT NOT NULL,
                formalization TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (question_id, content_version, content_hash)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS question_versions (
                question_id TEXT NOT NULL,
                content_version TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                question_type TEXT NOT NULL,
                stem TEXT NOT NULL,
                options TEXT NOT NULL,
                error_tags TEXT NOT NULL,
                knowledge_tags TEXT NOT NULL,
                formalization_version TEXT NOT NULL,
                formalization TEXT NOT NULL,
                is_active INTEGER NOT NULL CHECK (is_active IN (0, 1)),
                publisher_id TEXT NOT NULL,
                published_at TEXT NOT NULL,
                PRIMARY KEY (question_id, content_version)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS question_publication_events (
                event_id TEXT PRIMARY KEY,
                question_id TEXT NOT NULL,
                content_version TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                publisher_id TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS question_formalization_verification_events (
                event_id TEXT PRIMARY KEY,
                question_id TEXT NOT NULL,
                content_version TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                formalization_version TEXT NOT NULL,
                formalization_kind TEXT NOT NULL,
                expected_status TEXT NOT NULL,
                actual_status TEXT NOT NULL,
                expected_solution_count INTEGER,
                actual_solution_count INTEGER,
                proof_steps TEXT NOT NULL,
                known_literals TEXT NOT NULL,
                conflict TEXT,
                evidence TEXT,
                selected_option TEXT NOT NULL,
                matching_options TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_question_candidates_lookup
            ON question_candidates (question_id, content_version, content_hash)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_question_versions_active
            ON question_versions (is_active, question_id, content_version)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_question_publication_events_candidate
            ON question_publication_events (
                question_id, content_version, content_hash, created_at
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_question_formalization_events_candidate
            ON question_formalization_verification_events (
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
        """归档缺少内容摘要列的旧表，杜绝旧记录绕过新审核绑定。"""
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


def _normalize_publication(
    publication: QuestionPublicationInput,
) -> QuestionPublicationInput:
    question_id = _validate_text(publication.question_id, "题目标识", max_length=128)
    content_version = _validate_text(
        publication.content_version,
        "内容版本",
        max_length=128,
    )
    question_type = _validate_text(publication.question_type, "题型", max_length=128)
    stem = _validate_text(publication.stem, "题干", max_length=_MAX_STEM_LENGTH)
    formalization_version = _validate_text(
        publication.formalization_version,
        "形式化版本",
        max_length=128,
    )
    options = _normalize_options(publication.options)
    formalization = _normalize_formalization(
        question_type,
        publication.formalization,
    )
    formalization = _bind_option_assertions(formalization, options)
    if options and _formalization_expected_answer(formalization) not in options:
        raise ValueError("形式化预期答案必须对应一个题目选项")
    error_tags = _normalize_tags(publication.error_tags)
    knowledge_tags = _normalize_tags(publication.knowledge_tags)

    return QuestionPublicationInput(
        question_id=question_id,
        content_version=content_version,
        question_type=question_type,
        stem=stem,
        options=options,
        error_tags=error_tags,
        knowledge_tags=knowledge_tags,
        formalization_version=formalization_version,
        formalization=formalization,
    )


def _content_hash_for(publication: QuestionPublicationInput) -> str:
    """为规范化题目内容生成跨进程稳定的 SHA-256 摘要。"""
    canonical_content = {
        "content_version": publication.content_version,
        "error_tags": publication.error_tags,
        "formalization": _formalization_to_payload(publication.formalization),
        "formalization_version": publication.formalization_version,
        "knowledge_tags": publication.knowledge_tags,
        "options": publication.options,
        "question_id": publication.question_id,
        "question_type": publication.question_type,
        "stem": publication.stem,
    }
    serialized = json.dumps(
        canonical_content,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalize_formalization(
    question_type: str,
    formalization: QuestionFormalization | None,
) -> QuestionFormalization:
    """按题型规范化结构化资产，拒绝跨题型或超限的求解输入。"""
    if formalization is None:
        raise ValueError("候选题目必须提供可复现的形式化资产")
    match formalization:
        case PropositionalFormalization():
            if question_type != FormalizationKind.PROPOSITIONAL.value:
                raise ValueError("题型与形式化资产类型不一致")
            return _normalize_propositional_formalization(formalization)
        case OrderingFormalization():
            if question_type != FormalizationKind.ORDERING.value:
                raise ValueError("题型与形式化资产类型不一致")
            return _normalize_ordering_formalization(formalization)
        case GroupingFormalization():
            if question_type != FormalizationKind.GROUPING.value:
                raise ValueError("题型与形式化资产类型不一致")
            return _normalize_grouping_formalization(formalization)
        case MatchingFormalization():
            if question_type != FormalizationKind.MATCHING.value:
                raise ValueError("题型与形式化资产类型不一致")
            return _normalize_matching_formalization(formalization)
    raise ValueError("候选题目形式化资产类型不受支持")


def _normalize_propositional_formalization(
    formalization: PropositionalFormalization,
) -> PropositionalFormalization:
    if len(formalization.facts) > _MAX_FORMALIZATION_FACT_COUNT:
        raise ValueError(f"形式化事实数量不能超过 {_MAX_FORMALIZATION_FACT_COUNT} 条")
    if len(formalization.rules) > _MAX_FORMALIZATION_RULE_COUNT:
        raise ValueError(f"形式化规则数量不能超过 {_MAX_FORMALIZATION_RULE_COUNT} 条")
    facts = tuple(
        dict.fromkeys(
            _normalize_literal(fact, "形式化事实") for fact in formalization.facts
        )
    )
    rules = tuple(
        dict.fromkeys(
            _normalize_formalization_rule(rule) for rule in formalization.rules
        )
    )
    return PropositionalFormalization(
        facts=facts,
        rules=rules,
        query=_normalize_literal(formalization.query, "形式化查询"),
        expected_status=_normalize_propositional_status(formalization.expected_status),
        expected_answer=_normalize_expected_answer(formalization.expected_answer),
        option_assertions=formalization.option_assertions,
    )


def _normalize_ordering_formalization(
    formalization: OrderingFormalization,
) -> OrderingFormalization:
    items = _normalize_constraint_names(formalization.items, "排序对象")
    constraints = _normalize_ordering_constraints(formalization.constraints)
    status = _normalize_ordering_status(formalization.expected_status)
    solution_count = _normalize_expected_solution_count(
        formalization.expected_solution_count
    )
    return OrderingFormalization(
        items=items,
        constraints=constraints,
        expected_status=status,
        expected_solution_count=solution_count,
        expected_answer=_normalize_expected_answer(formalization.expected_answer),
        option_assertions=formalization.option_assertions,
    )


def _normalize_grouping_formalization(
    formalization: GroupingFormalization,
) -> GroupingFormalization:
    items = _normalize_constraint_names(formalization.items, "分组对象")
    groups = _normalize_constraint_names(
        formalization.groups,
        "分组名称",
        max_count=_MAX_CONSTRAINT_GROUP_COUNT,
    )
    if not isinstance(formalization.max_group_size, int):
        raise ValueError("每组最大容量必须是整数")
    if formalization.max_group_size < 1:
        raise ValueError("每组最大容量必须至少为 1")
    constraints = _normalize_grouping_constraints(formalization.constraints)
    status = _normalize_grouping_status(formalization.expected_status)
    solution_count = _normalize_expected_solution_count(
        formalization.expected_solution_count
    )
    return GroupingFormalization(
        items=items,
        groups=groups,
        max_group_size=formalization.max_group_size,
        constraints=constraints,
        expected_status=status,
        expected_solution_count=solution_count,
        expected_answer=_normalize_expected_answer(formalization.expected_answer),
        option_assertions=formalization.option_assertions,
    )


def _normalize_matching_formalization(
    formalization: MatchingFormalization,
) -> MatchingFormalization:
    items = _normalize_constraint_names(formalization.items, "匹配对象")
    targets = _normalize_constraint_names(formalization.targets, "匹配目标")
    constraints = _normalize_matching_constraints(formalization.constraints)
    status = _normalize_matching_status(formalization.expected_status)
    solution_count = _normalize_expected_solution_count(
        formalization.expected_solution_count
    )
    return MatchingFormalization(
        items=items,
        targets=targets,
        constraints=constraints,
        expected_status=status,
        expected_solution_count=solution_count,
        expected_answer=_normalize_expected_answer(formalization.expected_answer),
        option_assertions=formalization.option_assertions,
    )


def _bind_option_assertions(
    formalization: QuestionFormalization,
    options: tuple[str, ...],
) -> QuestionFormalization:
    """规范化每个选项的语义声明，并确保其类型匹配求解器输出。"""
    assertions = formalization.option_assertions
    if not options:
        if assertions:
            raise ValueError("无选项题目不能提交选项断言")
        return formalization
    if len(assertions) != len(options):
        raise ValueError("选择题必须为每个选项提交一条语义断言")
    normalized: list[OptionAssertion] = []
    for option, assertion in zip(options, assertions, strict=True):
        if not isinstance(assertion, OptionAssertion):
            raise ValueError("选项断言格式不合法")
        assertion_option = _validate_text(
            assertion.option,
            "选项断言选项",
            max_length=_MAX_OPTION_LENGTH,
        )
        if assertion_option != option:
            raise ValueError("选项断言必须按题目选项顺序逐一绑定")
        normalized.append(
            OptionAssertion(
                option=assertion_option,
                claim_status=_normalize_option_claim_status(
                    formalization,
                    assertion.claim_status,
                ),
                claim_solution_count=_normalize_option_claim_solution_count(
                    formalization,
                    assertion.claim_solution_count,
                ),
            )
        )
    return replace(formalization, option_assertions=tuple(normalized))


def _normalize_option_claim_status(
    formalization: QuestionFormalization,
    value: str,
) -> str:
    if not isinstance(value, str):
        raise ValueError("选项断言状态必须是文本")
    try:
        match formalization:
            case PropositionalFormalization():
                return VerificationStatus(value).value
            case OrderingFormalization():
                return OrderingSolveStatus(value).value
            case GroupingFormalization():
                return GroupingSolveStatus(value).value
            case MatchingFormalization():
                return MatchingSolveStatus(value).value
    except ValueError as error:
        raise ValueError("选项断言状态与题型不匹配") from error
    raise ValueError("候选题目形式化资产类型不受支持")


def _normalize_option_claim_solution_count(
    formalization: QuestionFormalization,
    value: int | None,
) -> int | None:
    if isinstance(formalization, PropositionalFormalization):
        if value is not None:
            raise ValueError("命题题选项断言不能声明解数量")
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("组合题选项断言必须声明非负解数量")
    return value


def _formalization_option_assertions(
    formalization: QuestionFormalization,
) -> tuple[OptionAssertion, ...]:
    return formalization.option_assertions


def _option_assertion_matches(
    assertion: OptionAssertion,
    verification: FormalizationVerification,
) -> bool:
    if assertion.claim_status != verification.actual_status.value:
        return False
    if verification.actual_solution_count is None:
        return assertion.claim_solution_count is None
    return assertion.claim_solution_count == verification.actual_solution_count


def _normalize_constraint_names(
    values: tuple[str, ...],
    label: str,
    max_count: int = _MAX_CONSTRAINT_ITEM_COUNT,
) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"{label}不能为空")
    if len(values) > max_count:
        raise ValueError(f"{label}数量不能超过 {max_count} 个")
    normalized = tuple(
        _validate_text(value, label, max_length=_MAX_CONSTRAINT_NAME_LENGTH)
        for value in values
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label}不能重复")
    return normalized


def _normalize_ordering_constraints(
    constraints: tuple[OrderingConstraint, ...],
) -> tuple[OrderingConstraint, ...]:
    if len(constraints) > _MAX_CONSTRAINT_COUNT:
        raise ValueError(f"排序约束数量不能超过 {_MAX_CONSTRAINT_COUNT} 条")
    normalized: list[OrderingConstraint] = []
    for constraint in constraints:
        if not isinstance(constraint, OrderingConstraint):
            raise ValueError("排序约束格式不合法")
        try:
            constraint_type = OrderingConstraintType(constraint.constraint_type)
        except ValueError as error:
            raise ValueError("排序约束类型不合法") from error
        normalized.append(
            OrderingConstraint(
                constraint_type=constraint_type,
                item=_validate_text(
                    constraint.item,
                    "排序约束对象",
                    max_length=_MAX_CONSTRAINT_NAME_LENGTH,
                ),
                other_item=(
                    _validate_text(
                        constraint.other_item,
                        "排序约束对象",
                        max_length=_MAX_CONSTRAINT_NAME_LENGTH,
                    )
                    if constraint.other_item is not None
                    else None
                ),
                position=constraint.position,
            )
        )
    return tuple(dict.fromkeys(normalized))


def _normalize_grouping_constraints(
    constraints: tuple[GroupConstraint, ...],
) -> tuple[GroupConstraint, ...]:
    if len(constraints) > _MAX_CONSTRAINT_COUNT:
        raise ValueError(f"分组约束数量不能超过 {_MAX_CONSTRAINT_COUNT} 条")
    normalized: list[GroupConstraint] = []
    for constraint in constraints:
        if not isinstance(constraint, GroupConstraint):
            raise ValueError("分组约束格式不合法")
        try:
            constraint_type = GroupConstraintType(constraint.constraint_type)
        except ValueError as error:
            raise ValueError("分组约束类型不合法") from error
        normalized.append(
            GroupConstraint(
                constraint_type=constraint_type,
                item=_validate_text(
                    constraint.item,
                    "分组约束对象",
                    max_length=_MAX_CONSTRAINT_NAME_LENGTH,
                ),
                other_item=_validate_text(
                    constraint.other_item,
                    "分组约束对象",
                    max_length=_MAX_CONSTRAINT_NAME_LENGTH,
                ),
            )
        )
    return tuple(dict.fromkeys(normalized))


def _normalize_matching_constraints(
    constraints: tuple[MatchConstraint, ...],
) -> tuple[MatchConstraint, ...]:
    if len(constraints) > _MAX_CONSTRAINT_COUNT:
        raise ValueError(f"匹配约束数量不能超过 {_MAX_CONSTRAINT_COUNT} 条")
    normalized: list[MatchConstraint] = []
    for constraint in constraints:
        if not isinstance(constraint, MatchConstraint):
            raise ValueError("匹配约束格式不合法")
        try:
            constraint_type = MatchConstraintType(constraint.constraint_type)
        except ValueError as error:
            raise ValueError("匹配约束类型不合法") from error
        normalized.append(
            MatchConstraint(
                constraint_type=constraint_type,
                item=_validate_text(
                    constraint.item,
                    "匹配约束对象",
                    max_length=_MAX_CONSTRAINT_NAME_LENGTH,
                ),
                target=_validate_text(
                    constraint.target,
                    "匹配约束目标",
                    max_length=_MAX_CONSTRAINT_NAME_LENGTH,
                ),
            )
        )
    return tuple(dict.fromkeys(normalized))


def _normalize_expected_answer(value: str) -> str:
    return _validate_text(value, "形式化预期答案", max_length=_MAX_OPTION_LENGTH)


def _normalize_expected_solution_count(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("形式化预期解数量必须是非负整数")
    return value


def _normalize_formalization_rule(rule: FormalizationRule) -> FormalizationRule:
    if not isinstance(rule, FormalizationRule):
        raise ValueError("形式化规则格式不合法")
    source_text = rule.source_text
    if source_text is not None:
        if not isinstance(source_text, str):
            raise ValueError("形式化规则原文必须是文本")
        source_text = source_text.strip() or None
        if (
            source_text is not None
            and len(source_text) > _MAX_FORMALIZATION_SOURCE_LENGTH
        ):
            raise ValueError(
                f"形式化规则原文不能超过 {_MAX_FORMALIZATION_SOURCE_LENGTH} 个字符"
            )
    return FormalizationRule(
        premise=_normalize_literal(rule.premise, "形式化规则前提"),
        conclusion=_normalize_literal(rule.conclusion, "形式化规则结论"),
        source_text=source_text,
    )


def _normalize_literal(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label}必须是文本")
    try:
        return Literal.parse(value).display()
    except ValueError as error:
        raise ValueError(f"{label}无效：{error}") from error


def _normalize_propositional_status(value: VerificationStatus) -> VerificationStatus:
    try:
        return VerificationStatus(value)
    except ValueError as error:
        raise ValueError("命题形式化预期状态不合法") from error


def _normalize_ordering_status(value: OrderingSolveStatus) -> OrderingSolveStatus:
    try:
        return OrderingSolveStatus(value)
    except ValueError as error:
        raise ValueError("排序形式化预期状态不合法") from error


def _normalize_grouping_status(value: GroupingSolveStatus) -> GroupingSolveStatus:
    try:
        return GroupingSolveStatus(value)
    except ValueError as error:
        raise ValueError("分组形式化预期状态不合法") from error


def _normalize_matching_status(value: MatchingSolveStatus) -> MatchingSolveStatus:
    try:
        return MatchingSolveStatus(value)
    except ValueError as error:
        raise ValueError("匹配形式化预期状态不合法") from error


def _formalization_to_payload(
    formalization: QuestionFormalization | None,
) -> dict[str, object] | None:
    """将各题型形式化资产转换为参与摘要和持久化的稳定载荷。"""
    if formalization is None:
        return None
    match formalization:
        case PropositionalFormalization():
            return {
                "kind": FormalizationKind.PROPOSITIONAL.value,
                "facts": formalization.facts,
                "rules": [
                    {
                        "premise": rule.premise,
                        "conclusion": rule.conclusion,
                        "source_text": rule.source_text,
                    }
                    for rule in formalization.rules
                ],
                "query": formalization.query,
                "expected_status": formalization.expected_status.value,
                "expected_answer": formalization.expected_answer,
                "option_assertions": _option_assertions_to_payload(
                    formalization.option_assertions
                ),
            }
        case OrderingFormalization():
            return {
                "kind": FormalizationKind.ORDERING.value,
                "items": formalization.items,
                "constraints": [
                    {
                        "constraint_type": constraint.constraint_type.value,
                        "item": constraint.item,
                        "other_item": constraint.other_item,
                        "position": constraint.position,
                    }
                    for constraint in formalization.constraints
                ],
                "expected_status": formalization.expected_status.value,
                "expected_solution_count": formalization.expected_solution_count,
                "expected_answer": formalization.expected_answer,
                "option_assertions": _option_assertions_to_payload(
                    formalization.option_assertions
                ),
            }
        case GroupingFormalization():
            return {
                "kind": FormalizationKind.GROUPING.value,
                "items": formalization.items,
                "groups": formalization.groups,
                "max_group_size": formalization.max_group_size,
                "constraints": [
                    {
                        "constraint_type": constraint.constraint_type.value,
                        "item": constraint.item,
                        "other_item": constraint.other_item,
                    }
                    for constraint in formalization.constraints
                ],
                "expected_status": formalization.expected_status.value,
                "expected_solution_count": formalization.expected_solution_count,
                "expected_answer": formalization.expected_answer,
                "option_assertions": _option_assertions_to_payload(
                    formalization.option_assertions
                ),
            }
        case MatchingFormalization():
            return {
                "kind": FormalizationKind.MATCHING.value,
                "items": formalization.items,
                "targets": formalization.targets,
                "constraints": [
                    {
                        "constraint_type": constraint.constraint_type.value,
                        "item": constraint.item,
                        "target": constraint.target,
                    }
                    for constraint in formalization.constraints
                ],
                "expected_status": formalization.expected_status.value,
                "expected_solution_count": formalization.expected_solution_count,
                "expected_answer": formalization.expected_answer,
                "option_assertions": _option_assertions_to_payload(
                    formalization.option_assertions
                ),
            }
    raise ValueError("候选题目形式化资产类型不受支持")


def _option_assertions_to_payload(
    assertions: tuple[OptionAssertion, ...],
) -> list[dict[str, object]]:
    return [
        {
            "option": assertion.option,
            "claim_status": assertion.claim_status,
            "claim_solution_count": assertion.claim_solution_count,
        }
        for assertion in assertions
    ]


def _formalization_expected_answer(formalization: QuestionFormalization) -> str:
    return formalization.expected_answer


def _serialize_formalization(
    formalization: QuestionFormalization | None,
) -> str:
    payload = _formalization_to_payload(formalization)
    if payload is None:
        raise ValueError("候选题目必须提供可复现的形式化资产")
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _deserialize_formalization(value: str) -> QuestionFormalization:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("题库形式化资产不是合法 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("题库形式化资产格式不合法")
    kind = payload.get("kind")
    if kind == FormalizationKind.PROPOSITIONAL.value:
        return _deserialize_propositional_formalization(payload)
    if kind == FormalizationKind.ORDERING.value:
        return _deserialize_ordering_formalization(payload)
    if kind == FormalizationKind.GROUPING.value:
        return _deserialize_grouping_formalization(payload)
    if kind == FormalizationKind.MATCHING.value:
        return _deserialize_matching_formalization(payload)
    raise ValueError("题库形式化资产类型不合法")


def _deserialize_propositional_formalization(
    payload: dict[str, object],
) -> PropositionalFormalization:
    facts = _read_string_list(payload, "facts", "题库形式化事实")
    rules = _read_formalization_rules(payload)
    query = payload.get("query")
    expected_status = payload.get("expected_status")
    expected_answer = payload.get("expected_answer")
    if not isinstance(query, str) or not isinstance(expected_answer, str):
        raise ValueError("题库形式化查询或预期答案格式不合法")
    try:
        status = VerificationStatus(expected_status)
    except ValueError as error:
        raise ValueError("题库形式化预期状态格式不合法") from error
    return _normalize_formalization(
        FormalizationKind.PROPOSITIONAL.value,
        PropositionalFormalization(
            facts=facts,
            rules=rules,
            query=query,
            expected_status=status,
            expected_answer=expected_answer,
            option_assertions=_read_option_assertions(payload),
        ),
    )


def _deserialize_ordering_formalization(
    payload: dict[str, object],
) -> OrderingFormalization:
    items = _read_string_list(payload, "items", "题库排序对象")
    constraints = _read_ordering_constraints(payload)
    expected_answer = _read_expected_answer(payload)
    try:
        status = OrderingSolveStatus(payload.get("expected_status"))
    except ValueError as error:
        raise ValueError("题库排序形式化预期状态格式不合法") from error
    return _normalize_formalization(
        FormalizationKind.ORDERING.value,
        OrderingFormalization(
            items=items,
            constraints=constraints,
            expected_status=status,
            expected_solution_count=_read_expected_solution_count(payload),
            expected_answer=expected_answer,
            option_assertions=_read_option_assertions(payload),
        ),
    )


def _deserialize_grouping_formalization(
    payload: dict[str, object],
) -> GroupingFormalization:
    items = _read_string_list(payload, "items", "题库分组对象")
    groups = _read_string_list(payload, "groups", "题库分组名称")
    max_group_size = payload.get("max_group_size")
    if not isinstance(max_group_size, int) or isinstance(max_group_size, bool):
        raise ValueError("题库分组最大容量格式不合法")
    try:
        status = GroupingSolveStatus(payload.get("expected_status"))
    except ValueError as error:
        raise ValueError("题库分组形式化预期状态格式不合法") from error
    return _normalize_formalization(
        FormalizationKind.GROUPING.value,
        GroupingFormalization(
            items=items,
            groups=groups,
            max_group_size=max_group_size,
            constraints=_read_grouping_constraints(payload),
            expected_status=status,
            expected_solution_count=_read_expected_solution_count(payload),
            expected_answer=_read_expected_answer(payload),
            option_assertions=_read_option_assertions(payload),
        ),
    )


def _deserialize_matching_formalization(
    payload: dict[str, object],
) -> MatchingFormalization:
    items = _read_string_list(payload, "items", "题库匹配对象")
    targets = _read_string_list(payload, "targets", "题库匹配目标")
    try:
        status = MatchingSolveStatus(payload.get("expected_status"))
    except ValueError as error:
        raise ValueError("题库匹配形式化预期状态格式不合法") from error
    return _normalize_formalization(
        FormalizationKind.MATCHING.value,
        MatchingFormalization(
            items=items,
            targets=targets,
            constraints=_read_matching_constraints(payload),
            expected_status=status,
            expected_solution_count=_read_expected_solution_count(payload),
            expected_answer=_read_expected_answer(payload),
            option_assertions=_read_option_assertions(payload),
        ),
    )


def _read_string_list(
    payload: dict[str, object],
    key: str,
    label: str,
) -> tuple[str, ...]:
    values = payload.get(key)
    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise ValueError(f"{label}格式不合法")
    return tuple(values)


def _read_expected_answer(payload: dict[str, object]) -> str:
    answer = payload.get("expected_answer")
    if not isinstance(answer, str):
        raise ValueError("题库形式化预期答案格式不合法")
    return answer


def _read_expected_solution_count(payload: dict[str, object]) -> int:
    count = payload.get("expected_solution_count")
    if not isinstance(count, int) or isinstance(count, bool):
        raise ValueError("题库形式化预期解数量格式不合法")
    return count


def _read_option_assertions(
    payload: dict[str, object],
) -> tuple[OptionAssertion, ...]:
    raw_assertions = payload.get("option_assertions")
    if not isinstance(raw_assertions, list):
        raise ValueError("题库选项断言格式不合法")
    assertions: list[OptionAssertion] = []
    for raw_assertion in raw_assertions:
        if not isinstance(raw_assertion, dict):
            raise ValueError("题库选项断言格式不合法")
        option = raw_assertion.get("option")
        claim_status = raw_assertion.get("claim_status")
        claim_solution_count = raw_assertion.get("claim_solution_count")
        if not isinstance(option, str) or not isinstance(claim_status, str):
            raise ValueError("题库选项断言格式不合法")
        if claim_solution_count is not None and (
            not isinstance(claim_solution_count, int)
            or isinstance(claim_solution_count, bool)
        ):
            raise ValueError("题库选项断言解数量格式不合法")
        assertions.append(
            OptionAssertion(option, claim_status, claim_solution_count)
        )
    return tuple(assertions)


def _read_formalization_rules(
    payload: dict[str, object],
) -> tuple[FormalizationRule, ...]:
    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, list):
        raise ValueError("题库形式化规则格式不合法")
    rules: list[FormalizationRule] = []
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            raise ValueError("题库形式化规则格式不合法")
        premise = raw_rule.get("premise")
        conclusion = raw_rule.get("conclusion")
        source_text = raw_rule.get("source_text")
        if not isinstance(premise, str) or not isinstance(conclusion, str):
            raise ValueError("题库形式化规则格式不合法")
        if source_text is not None and not isinstance(source_text, str):
            raise ValueError("题库形式化规则原文格式不合法")
        rules.append(FormalizationRule(premise, conclusion, source_text))
    return tuple(rules)


def _read_ordering_constraints(
    payload: dict[str, object],
) -> tuple[OrderingConstraint, ...]:
    raw_constraints = payload.get("constraints")
    if not isinstance(raw_constraints, list):
        raise ValueError("题库排序约束格式不合法")
    constraints: list[OrderingConstraint] = []
    for raw_constraint in raw_constraints:
        if not isinstance(raw_constraint, dict):
            raise ValueError("题库排序约束格式不合法")
        try:
            constraint_type = OrderingConstraintType(
                raw_constraint.get("constraint_type")
            )
        except ValueError as error:
            raise ValueError("题库排序约束类型不合法") from error
        item = raw_constraint.get("item")
        other_item = raw_constraint.get("other_item")
        position = raw_constraint.get("position")
        if not isinstance(item, str):
            raise ValueError("题库排序约束对象格式不合法")
        if other_item is not None and not isinstance(other_item, str):
            raise ValueError("题库排序约束对象格式不合法")
        if position is not None and (
            not isinstance(position, int) or isinstance(position, bool)
        ):
            raise ValueError("题库排序约束位置格式不合法")
        constraints.append(
            OrderingConstraint(constraint_type, item, other_item, position)
        )
    return tuple(constraints)


def _read_grouping_constraints(
    payload: dict[str, object],
) -> tuple[GroupConstraint, ...]:
    raw_constraints = payload.get("constraints")
    if not isinstance(raw_constraints, list):
        raise ValueError("题库分组约束格式不合法")
    constraints: list[GroupConstraint] = []
    for raw_constraint in raw_constraints:
        if not isinstance(raw_constraint, dict):
            raise ValueError("题库分组约束格式不合法")
        try:
            constraint_type = GroupConstraintType(raw_constraint.get("constraint_type"))
        except ValueError as error:
            raise ValueError("题库分组约束类型不合法") from error
        item = raw_constraint.get("item")
        other_item = raw_constraint.get("other_item")
        if not isinstance(item, str) or not isinstance(other_item, str):
            raise ValueError("题库分组约束对象格式不合法")
        constraints.append(GroupConstraint(constraint_type, item, other_item))
    return tuple(constraints)


def _read_matching_constraints(
    payload: dict[str, object],
) -> tuple[MatchConstraint, ...]:
    raw_constraints = payload.get("constraints")
    if not isinstance(raw_constraints, list):
        raise ValueError("题库匹配约束格式不合法")
    constraints: list[MatchConstraint] = []
    for raw_constraint in raw_constraints:
        if not isinstance(raw_constraint, dict):
            raise ValueError("题库匹配约束格式不合法")
        try:
            constraint_type = MatchConstraintType(raw_constraint.get("constraint_type"))
        except ValueError as error:
            raise ValueError("题库匹配约束类型不合法") from error
        item = raw_constraint.get("item")
        target = raw_constraint.get("target")
        if not isinstance(item, str) or not isinstance(target, str):
            raise ValueError("题库匹配约束对象格式不合法")
        constraints.append(MatchConstraint(constraint_type, item, target))
    return tuple(constraints)


def _serialize_proof_steps(proof_steps: tuple[ProofStep, ...]) -> str:
    payload = [
        {
            "derived": step.derived.display(),
            "reason": step.reason,
            "source_rule": (
                {
                    "premise": step.source_rule.premise.display(),
                    "conclusion": step.source_rule.conclusion.display(),
                    "source_text": step.source_rule.source_text,
                }
                if step.source_rule is not None
                else None
            ),
            "dependencies": [
                dependency.display() for dependency in step.dependencies
            ],
        }
        for step in proof_steps
    ]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _serialize_literals(literals: tuple[Literal, ...]) -> str:
    return json.dumps(
        [literal.display() for literal in literals],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _serialize_conflict(conflict: tuple[Literal, Literal] | None) -> str | None:
    if conflict is None:
        return None
    return json.dumps(
        [literal.display() for literal in conflict],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _verify_propositional_formalization(
    formalization: PropositionalFormalization,
) -> FormalizationVerification:
    try:
        facts = tuple(Literal.parse(fact) for fact in formalization.facts)
        rules = tuple(
            ImplicationRule(
                premise=Literal.parse(rule.premise),
                conclusion=Literal.parse(rule.conclusion),
                source_text=rule.source_text,
            )
            for rule in formalization.rules
        )
        query = Literal.parse(formalization.query)
    except ValueError as error:
        raise ValueError(f"题目形式化资产无效：{error}") from error
    result = InferenceEngine().verify(facts, rules, query)
    return FormalizationVerification(
        kind=FormalizationKind.PROPOSITIONAL,
        expected_status=formalization.expected_status,
        actual_status=result.status,
        expected_solution_count=None,
        actual_solution_count=None,
        proof_steps=result.proof_steps,
        known_literals=result.known_literals,
        conflict=result.conflict,
        evidence=None,
        selected_option="",
        matching_options=(),
    )


def _verify_ordering_formalization(
    formalization: OrderingFormalization,
) -> FormalizationVerification:
    try:
        result = OrderingSolver().solve(
            items=formalization.items,
            constraints=formalization.constraints,
        )
    except ValueError as error:
        raise ValueError(f"题目形式化资产无效：{error}") from error
    return FormalizationVerification(
        kind=FormalizationKind.ORDERING,
        expected_status=formalization.expected_status,
        actual_status=result.status,
        expected_solution_count=formalization.expected_solution_count,
        actual_solution_count=result.solution_count,
        proof_steps=(),
        known_literals=(),
        conflict=None,
        evidence=_serialize_ordering_solutions(result.sample_solutions),
        selected_option="",
        matching_options=(),
    )


def _verify_grouping_formalization(
    formalization: GroupingFormalization,
) -> FormalizationVerification:
    try:
        result = GroupingSolver().solve(
            items=formalization.items,
            groups=formalization.groups,
            max_group_size=formalization.max_group_size,
            constraints=formalization.constraints,
        )
    except ValueError as error:
        raise ValueError(f"题目形式化资产无效：{error}") from error
    return FormalizationVerification(
        kind=FormalizationKind.GROUPING,
        expected_status=formalization.expected_status,
        actual_status=result.status,
        expected_solution_count=formalization.expected_solution_count,
        actual_solution_count=result.solution_count,
        proof_steps=(),
        known_literals=(),
        conflict=None,
        evidence=_serialize_grouping_solutions(result.sample_solutions),
        selected_option="",
        matching_options=(),
    )


def _verify_matching_formalization(
    formalization: MatchingFormalization,
) -> FormalizationVerification:
    try:
        result = MatchingSolver().solve(
            items=formalization.items,
            targets=formalization.targets,
            constraints=formalization.constraints,
        )
    except ValueError as error:
        raise ValueError(f"题目形式化资产无效：{error}") from error
    return FormalizationVerification(
        kind=FormalizationKind.MATCHING,
        expected_status=formalization.expected_status,
        actual_status=result.status,
        expected_solution_count=formalization.expected_solution_count,
        actual_solution_count=result.solution_count,
        proof_steps=(),
        known_literals=(),
        conflict=None,
        evidence=_serialize_matching_solutions(result.sample_solutions),
        selected_option="",
        matching_options=(),
    )


def _serialize_ordering_solutions(
    solutions: tuple[tuple[str, ...], ...],
) -> str:
    return json.dumps(solutions, ensure_ascii=False, separators=(",", ":"))


def _serialize_grouping_solutions(solutions: tuple[object, ...]) -> str:
    payload = [dict(solution.assignments) for solution in solutions]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _serialize_matching_solutions(solutions: tuple[object, ...]) -> str:
    payload = [dict(solution.assignments) for solution in solutions]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _deserialize_proof_steps(value: str) -> tuple[ProofStep, ...]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("形式化验证证明步骤不是合法 JSON") from error
    if not isinstance(payload, list):
        raise ValueError("形式化验证证明步骤格式不合法")
    proof_steps: list[ProofStep] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("形式化验证证明步骤格式不合法")
        derived = item.get("derived")
        reason = item.get("reason")
        dependencies = item.get("dependencies")
        if not isinstance(derived, str) or not isinstance(reason, str):
            raise ValueError("形式化验证证明步骤格式不合法")
        if not isinstance(dependencies, list) or not all(
            isinstance(dependency, str) for dependency in dependencies
        ):
            raise ValueError("形式化验证证明依赖格式不合法")
        proof_steps.append(
            ProofStep(
                derived=Literal.parse(derived),
                reason=reason,
                source_rule=_deserialize_proof_source_rule(item.get("source_rule")),
                dependencies=tuple(Literal.parse(value) for value in dependencies),
            )
        )
    return tuple(proof_steps)


def _deserialize_proof_source_rule(value: object) -> ImplicationRule | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("形式化验证证明规则格式不合法")
    premise = value.get("premise")
    conclusion = value.get("conclusion")
    source_text = value.get("source_text")
    if not isinstance(premise, str) or not isinstance(conclusion, str):
        raise ValueError("形式化验证证明规则格式不合法")
    if source_text is not None and not isinstance(source_text, str):
        raise ValueError("形式化验证证明规则原文格式不合法")
    return ImplicationRule(
        premise=Literal.parse(premise),
        conclusion=Literal.parse(conclusion),
        source_text=source_text,
    )


def _deserialize_literals(value: str, label: str) -> tuple[Literal, ...]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label}不是合法 JSON") from error
    if not isinstance(payload, list) or not all(
        isinstance(item, str) for item in payload
    ):
        raise ValueError(f"{label}格式不合法")
    return tuple(Literal.parse(item) for item in payload)


def _deserialize_conflict(value: str | None) -> tuple[Literal, Literal] | None:
    if value is None:
        return None
    conflict = _deserialize_literals(value, "形式化验证冲突")
    if len(conflict) != 2 or conflict[0].opposite() != conflict[1]:
        raise ValueError("形式化验证冲突格式不合法")
    return conflict[0], conflict[1]


def _insert_formalization_verification_event(
    connection: sqlite3.Connection,
    *,
    question: PublishedQuestion,
    verification: FormalizationVerification,
    created_at: str,
) -> None:
    """追加一次形式化复验事实，供发布和历史版本重激活共同审计。"""
    connection.execute(
        """
        INSERT INTO question_formalization_verification_events (
            event_id, question_id, content_version, content_hash,
            formalization_version, formalization_kind, expected_status,
            actual_status, expected_solution_count, actual_solution_count,
            proof_steps, known_literals, conflict, evidence, selected_option,
            matching_options, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid4()),
            question.question_id,
            question.content_version,
            question.content_hash,
            question.formalization_version,
            verification.kind.value,
            verification.expected_status.value,
            verification.actual_status.value,
            verification.expected_solution_count,
            verification.actual_solution_count,
            _serialize_proof_steps(verification.proof_steps),
            _serialize_literals(verification.known_literals),
            _serialize_conflict(verification.conflict),
            verification.evidence,
            verification.selected_option,
            _serialize_values(verification.matching_options),
            created_at,
        ),
    )


def _insert_question_version_lifecycle_event(
    connection: sqlite3.Connection,
    *,
    question: PublishedQuestion,
    action: QuestionVersionLifecycleAction,
    actor_id: str,
    replaced_content_version: str | None,
    reason: str,
    created_at: str,
) -> QuestionVersionLifecycleEvent:
    """写入版本状态变更的不可变审计事实，并返回刚创建的事件。"""
    event = QuestionVersionLifecycleEvent(
        event_id=str(uuid4()),
        question_id=question.question_id,
        content_version=question.content_version,
        content_hash=question.content_hash,
        action=action,
        actor_id=actor_id,
        replaced_content_version=replaced_content_version,
        reason=reason,
        created_at=created_at,
    )
    connection.execute(
        """
        INSERT INTO question_version_lifecycle_events (
            event_id, question_id, content_version, content_hash, action, actor_id,
            replaced_content_version, reason, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.event_id,
            event.question_id,
            event.content_version,
            event.content_hash,
            event.action.value,
            event.actor_id,
            event.replaced_content_version,
            event.reason,
            event.created_at,
        ),
    )
    return event


def _formalization_verification_event_from_row(
    row: sqlite3.Row,
) -> FormalizationVerificationEvent:
    try:
        kind = FormalizationKind(row["formalization_kind"])
        expected_status = _status_from_value(kind, row["expected_status"])
        actual_status = _status_from_value(kind, row["actual_status"])
    except ValueError as error:
        raise ValueError("形式化验证事件状态格式不合法") from error
    return FormalizationVerificationEvent(
        question_id=row["question_id"],
        content_version=row["content_version"],
        content_hash=row["content_hash"],
        formalization_version=row["formalization_version"],
        kind=kind,
        expected_status=expected_status,
        actual_status=actual_status,
        expected_solution_count=row["expected_solution_count"],
        actual_solution_count=row["actual_solution_count"],
        proof_steps=_deserialize_proof_steps(row["proof_steps"]),
        known_literals=_deserialize_literals(
            row["known_literals"],
            "形式化验证已知命题",
        ),
        conflict=_deserialize_conflict(row["conflict"]),
        evidence=row["evidence"],
        selected_option=row["selected_option"],
        matching_options=_deserialize_values(row["matching_options"]),
        created_at=row["created_at"],
    )


def _status_from_value(kind: FormalizationKind, value: str) -> FormalizationStatus:
    match kind:
        case FormalizationKind.PROPOSITIONAL:
            return VerificationStatus(value)
        case FormalizationKind.ORDERING:
            return OrderingSolveStatus(value)
        case FormalizationKind.GROUPING:
            return GroupingSolveStatus(value)
        case FormalizationKind.MATCHING:
            return MatchingSolveStatus(value)
    raise ValueError("形式化资产类型不受支持")


def _normalize_options(options: tuple[str, ...]) -> tuple[str, ...]:
    if len(options) > _MAX_OPTION_COUNT:
        raise ValueError(f"选项数量不能超过 {_MAX_OPTION_COUNT} 个")
    normalized = tuple(option.strip() for option in options)
    if any(not option for option in normalized):
        raise ValueError("选项不能包含空文本")
    if any(len(option) > _MAX_OPTION_LENGTH for option in normalized):
        raise ValueError(f"单个选项不能超过 {_MAX_OPTION_LENGTH} 个字符")
    if len(set(normalized)) != len(normalized):
        raise ValueError("选项不能重复")
    return normalized


def _normalize_tags(tags: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(tag.strip() for tag in tags if tag.strip()))
    if len(normalized) > _MAX_TAG_COUNT:
        raise ValueError(f"标签数量不能超过 {_MAX_TAG_COUNT} 个")
    if any(len(tag) > _MAX_TAG_LENGTH for tag in normalized):
        raise ValueError(f"单个标签不能超过 {_MAX_TAG_LENGTH} 个字符")
    return normalized


def _validate_text(value: str, label: str, max_length: int) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label}不能为空")
    if len(normalized) > max_length:
        raise ValueError(f"{label}不能超过 {max_length} 个字符")
    return normalized


def _validate_content_hash(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("内容摘要必须是 64 位 SHA-256 十六进制字符串")
    return normalized


def _serialize_values(values: tuple[str, ...]) -> str:
    return json.dumps(values, ensure_ascii=False)


def _deserialize_values(value: str) -> tuple[str, ...]:
    loaded = json.loads(value)
    is_string_list = isinstance(loaded, list) and all(
        isinstance(item, str) for item in loaded
    )
    if not is_string_list:
        raise ValueError("题库数据格式不合法")
    return tuple(loaded)


def _validate_candidate_matches_published_question(
    candidate: QuestionCandidate,
    question: PublishedQuestion,
) -> None:
    """拒绝缺失、漂移或伪造的候选快照重新激活历史发布版本。"""
    expected_publication = QuestionPublicationInput(
        question_id=question.question_id,
        content_version=question.content_version,
        question_type=question.question_type,
        stem=question.stem,
        options=question.options,
        error_tags=question.error_tags,
        knowledge_tags=question.knowledge_tags,
        formalization_version=question.formalization_version,
        formalization=question.formalization,
    )
    if candidate.publication != expected_publication:
        raise ValueError("候选快照与已发布版本不一致，不能重新激活")
    if candidate.content_hash != question.content_hash:
        raise ValueError("候选快照摘要与已发布版本不一致，不能重新激活")
    if _content_hash_for(candidate.publication) != candidate.content_hash:
        raise ValueError("候选快照内容摘要校验失败，不能重新激活")


def _question_version_lifecycle_event_from_row(
    row: sqlite3.Row,
) -> QuestionVersionLifecycleEvent:
    try:
        action = QuestionVersionLifecycleAction(row["action"])
    except ValueError as error:
        raise ValueError("题目版本生命周期动作格式不合法") from error
    return QuestionVersionLifecycleEvent(
        event_id=row["event_id"],
        question_id=row["question_id"],
        content_version=row["content_version"],
        content_hash=row["content_hash"],
        action=action,
        actor_id=row["actor_id"],
        replaced_content_version=row["replaced_content_version"],
        reason=row["reason"],
        created_at=row["created_at"],
    )


def _candidate_from_row(row: sqlite3.Row) -> QuestionCandidate:
    return QuestionCandidate(
        publication=QuestionPublicationInput(
            question_id=row["question_id"],
            content_version=row["content_version"],
            question_type=row["question_type"],
            stem=row["stem"],
            options=_deserialize_values(row["options"]),
            error_tags=_deserialize_values(row["error_tags"]),
            knowledge_tags=_deserialize_values(row["knowledge_tags"]),
            formalization_version=row["formalization_version"],
            formalization=_deserialize_formalization(row["formalization"]),
        ),
        content_hash=row["content_hash"],
    )


def _question_from_row(row: sqlite3.Row) -> PublishedQuestion:
    return PublishedQuestion(
        question_id=row["question_id"],
        content_version=row["content_version"],
        content_hash=row["content_hash"],
        question_type=row["question_type"],
        stem=row["stem"],
        options=_deserialize_values(row["options"]),
        error_tags=_deserialize_values(row["error_tags"]),
        knowledge_tags=_deserialize_values(row["knowledge_tags"]),
        formalization_version=row["formalization_version"],
        formalization=_deserialize_formalization(row["formalization"]),
        published_at=row["published_at"],
    )


def _learner_question_from(question: PublishedQuestion) -> LearnerQuestion:
    """从内部题目记录构造最小学习者展示视图。"""
    return LearnerQuestion(
        question_id=question.question_id,
        content_version=question.content_version,
        question_type=question.question_type,
        stem=question.stem,
        options=question.options,
    )


def _recommendation_reason(
    error_matches: tuple[str, ...],
    knowledge_matches: tuple[str, ...],
) -> str:
    if error_matches:
        return "匹配你的高频错因，建议重点练习条件方向与反向推理。"
    if knowledge_matches:
        return "匹配你待巩固的知识点，建议从基础条件翻译开始练习。"
    return "用于基础练习和当前能力校准。"
