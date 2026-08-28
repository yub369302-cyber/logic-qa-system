"""SQLite 数据库版本迁移、校验备份与受控恢复。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

_SCHEMA_TABLE = "schema_migrations"
_BACKUP_PREFIX = "logic_qa_backup"
_BACKUP_MANIFEST_SUFFIX = ".manifest.json"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class DatabaseMigration:
    """一个按版本严格递增执行的 SQLite 架构迁移。"""

    version: int
    name: str
    apply: Callable[[sqlite3.Connection], None]


@dataclass(frozen=True, slots=True)
class DatabaseBackup:
    """已校验的 SQLite 一致性备份及其持久化清单。"""

    source_path: Path
    backup_path: Path
    manifest_path: Path
    source_schema_version: int
    content_sha256: str
    created_at: str


class SQLiteDatabaseManager:
    """为单个 SQLite 文件提供可审计迁移、备份和恢复能力。"""

    def __init__(
        self,
        database_path: Path,
        migrations: Iterable[DatabaseMigration],
    ) -> None:
        self._database_path = database_path.resolve()
        self._migrations = _validate_migrations(migrations)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def database_path(self) -> Path:
        """返回受管理的 SQLite 数据库路径。"""
        return self._database_path

    def migrate(self) -> int:
        """在单个立即事务中应用所有尚未执行的迁移。"""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {_SCHEMA_TABLE} (
                        version INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        applied_at TEXT NOT NULL
                    )
                    """
                )
                applied_rows = connection.execute(
                    f"SELECT version, name FROM {_SCHEMA_TABLE}"
                ).fetchall()
                self._validate_applied_migrations(applied_rows)
                applied_versions = {row["version"] for row in applied_rows}
                for migration in self._migrations:
                    if migration.version in applied_versions:
                        continue
                    migration.apply(connection)
                    connection.execute(
                        f"""
                        INSERT INTO {_SCHEMA_TABLE} (version, name, applied_at)
                        VALUES (?, ?, ?)
                        """,
                        (
                            migration.version,
                            migration.name,
                            datetime.now(UTC).isoformat(),
                        ),
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        finally:
            connection.close()
        return self.schema_version()

    def schema_version(self) -> int:
        """返回最高已迁移版本；未初始化数据库返回零。"""
        if not self._database_path.exists():
            return 0
        connection = self._connect()
        try:
            return _schema_version_for_connection(connection)
        finally:
            connection.close()

    def backup(self, destination_directory: Path) -> DatabaseBackup:
        """创建一致性快照、校验其完整性并写入可恢复清单。"""
        if not self._database_path.exists():
            raise ValueError("数据库文件不存在，无法创建备份")
        normalized_destination = destination_directory.resolve()
        normalized_destination.mkdir(parents=True, exist_ok=True)
        created_at = datetime.now(UTC).isoformat()
        backup_path = normalized_destination / _backup_filename(self._database_path)
        temporary_path = backup_path.with_suffix(".sqlite3.tmp")
        manifest_path = Path(f"{backup_path}{_BACKUP_MANIFEST_SUFFIX}")
        try:
            source = self._connect()
            target = sqlite3.connect(temporary_path)
            try:
                source.backup(target)
            finally:
                target.close()
                source.close()
            _validate_sqlite_database(temporary_path)
            source_schema_version = _schema_version_for_path(temporary_path)
            temporary_path.replace(backup_path)
            content_sha256 = _sha256_for_file(backup_path)
            backup = DatabaseBackup(
                source_path=self._database_path,
                backup_path=backup_path,
                manifest_path=manifest_path,
                source_schema_version=source_schema_version,
                content_sha256=content_sha256,
                created_at=created_at,
            )
            _write_backup_manifest(backup)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            backup_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
            raise
        return backup

    def load_backup(self, manifest_path: Path) -> DatabaseBackup:
        """从持久化清单读取备份元数据，以支持跨进程恢复。"""
        normalized_manifest_path = manifest_path.resolve()
        if not normalized_manifest_path.is_file():
            raise ValueError("备份清单不存在，无法恢复")
        try:
            payload = json.loads(normalized_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("备份清单格式不合法，无法恢复") from error
        backup = _backup_from_manifest(normalized_manifest_path, payload)
        if backup.source_path != self._database_path:
            raise ValueError("备份不属于当前数据库，拒绝恢复")
        return backup

    def restore(self, backup: DatabaseBackup) -> None:
        """验证备份完整性后原子替换当前数据库，并校验恢复结果。"""
        if backup.source_path != self._database_path:
            raise ValueError("备份不属于当前数据库，拒绝恢复")
        if backup.source_schema_version != self.schema_version():
            raise ValueError("备份架构版本与当前数据库不一致，拒绝恢复")
        if not backup.backup_path.is_file():
            raise ValueError("备份文件不存在，无法恢复")
        if _sha256_for_file(backup.backup_path) != backup.content_sha256:
            raise ValueError("备份校验摘要不匹配，拒绝恢复")
        _validate_sqlite_database(backup.backup_path)
        temporary_path = self._database_path.with_suffix(".restore.tmp")
        source = sqlite3.connect(backup.backup_path)
        replacement = sqlite3.connect(temporary_path)
        try:
            source.backup(replacement)
        finally:
            replacement.close()
            source.close()
        try:
            _validate_sqlite_database(temporary_path)
            os.replace(temporary_path, self._database_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        if self.schema_version() != backup.source_schema_version:
            raise RuntimeError("恢复后的数据库版本与备份元数据不一致")

    def _validate_applied_migrations(self, rows: list[sqlite3.Row]) -> None:
        expected_names = {
            migration.version: migration.name for migration in self._migrations
        }
        for row in rows:
            version = row["version"]
            name = row["name"]
            if version not in expected_names:
                raise RuntimeError("数据库包含当前程序不支持的迁移版本")
            if name != expected_names[version]:
                raise RuntimeError("数据库迁移记录与当前迁移定义不一致")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        return connection


def _validate_migrations(
    migrations: Iterable[DatabaseMigration],
) -> tuple[DatabaseMigration, ...]:
    normalized = tuple(migrations)
    if not normalized:
        raise ValueError("至少需要一条数据库迁移")
    if any(
        not isinstance(migration, DatabaseMigration) for migration in normalized
    ):
        raise ValueError("数据库迁移格式不合法")
    versions = tuple(migration.version for migration in normalized)
    if any(
        not isinstance(version, int) or isinstance(version, bool) or version < 1
        for version in versions
    ):
        raise ValueError("数据库迁移版本必须是正整数")
    if versions != tuple(sorted(versions)) or len(set(versions)) != len(versions):
        raise ValueError("数据库迁移版本必须严格递增且不可重复")
    if any(
        not isinstance(migration.name, str) or not migration.name.strip()
        for migration in normalized
    ):
        raise ValueError("数据库迁移名称不能为空")
    if any(not callable(migration.apply) for migration in normalized):
        raise ValueError("数据库迁移执行器必须可调用")
    return normalized


def _backup_filename(database_path: Path) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{_BACKUP_PREFIX}_{database_path.stem}_{timestamp}_{uuid4().hex}.sqlite3"


def _write_backup_manifest(backup: DatabaseBackup) -> None:
    payload = {
        "backup_filename": backup.backup_path.name,
        "content_sha256": backup.content_sha256,
        "created_at": backup.created_at,
        "source_path": str(backup.source_path),
        "source_schema_version": backup.source_schema_version,
    }
    temporary_path = backup.manifest_path.with_name(
        f"{backup.manifest_path.name}.{uuid4().hex}.tmp"
    )
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, backup.manifest_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _backup_from_manifest(manifest_path: Path, payload: object) -> DatabaseBackup:
    if not isinstance(payload, dict):
        raise ValueError("备份清单格式不合法，无法恢复")
    required_keys = {
        "backup_filename",
        "content_sha256",
        "created_at",
        "source_path",
        "source_schema_version",
    }
    if set(payload) != required_keys:
        raise ValueError("备份清单字段不合法，无法恢复")
    backup_filename = payload["backup_filename"]
    content_sha256 = payload["content_sha256"]
    created_at = payload["created_at"]
    source_path = payload["source_path"]
    schema_version = payload["source_schema_version"]
    if (
        not isinstance(backup_filename, str)
        or Path(backup_filename).name != backup_filename
        or not backup_filename.endswith(".sqlite3")
    ):
        raise ValueError("备份清单中的备份文件名不合法")
    if (
        not isinstance(content_sha256, str)
        or not _SHA256_PATTERN.fullmatch(content_sha256)
    ):
        raise ValueError("备份清单中的校验摘要不合法")
    if not isinstance(created_at, str) or not created_at.strip():
        raise ValueError("备份清单中的创建时间不合法")
    if not isinstance(source_path, str) or not source_path.strip():
        raise ValueError("备份清单中的来源路径不合法")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version < 0
    ):
        raise ValueError("备份清单中的架构版本不合法")
    return DatabaseBackup(
        source_path=Path(source_path).resolve(),
        backup_path=manifest_path.parent / backup_filename,
        manifest_path=manifest_path,
        source_schema_version=schema_version,
        content_sha256=content_sha256,
        created_at=created_at,
    )


def _sha256_for_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sqlite_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        row = connection.execute("PRAGMA integrity_check").fetchone()
    finally:
        connection.close()
    if row is None or row[0] != "ok":
        raise ValueError("SQLite 完整性校验失败")


def _schema_version_for_path(path: Path) -> int:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        return _schema_version_for_connection(connection)
    finally:
        connection.close()


def _schema_version_for_connection(connection: sqlite3.Connection) -> int:
    if not _table_exists(connection, _SCHEMA_TABLE):
        return 0
    row = connection.execute(
        f"SELECT COALESCE(MAX(version), 0) AS version FROM {_SCHEMA_TABLE}"
    ).fetchone()
    return int(row["version"])


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None
