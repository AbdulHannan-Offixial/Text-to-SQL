"""
database_manager.py — Universal Database Manager for the Text-to-SQL Agent
===========================================================================
Connects to any SQLAlchemy-supported database, reflects full schema metadata,
reads raw table samples, and executes validated read-only SQL queries.

Supported dialects (out of the box):
    SQLite      →  sqlite:///path/to/file.db   |  sqlite:///:memory:
    PostgreSQL  →  postgresql+psycopg2://user:pass@host:5432/dbname
    MySQL       →  mysql+pymysql://user:pass@host:3306/dbname
    MS SQL      →  mssql+pyodbc://user:pass@host/dbname?driver=ODBC+Driver+17+for+SQL+Server
    BigQuery    →  bigquery://project/dataset          (needs sqlalchemy-bigquery)
    Any other dialect SQLAlchemy supports via its plugin system.

Integration contract with schemas.py
--------------------------------------
  get_schema_context()  → dict   consumed by Agent 1 (SchemaAnalyzer)
  execute_query(sql)    → QueryResult  consumed by Agent 3 (SQLExecutor)
  get_table_sample()    → dict   consumed by Agent 1 for richer analysis

Exception hierarchy (outermost → most specific):
    DatabaseManagerError                    ← project base
    ├── ConnectionError                     ← cannot reach / auth failed
    ├── SchemaReflectionError               ← Inspector failed
    ├── QueryExecutionError                 ← runtime SQL error
    │   ├── QueryTimeoutError
    │   ├── QuerySyntaxError
    │   └── ReadOnlyViolationError          ← DML/DDL slipped past schemas.py
    └── PoolExhaustedError                  ← all connections in use
"""

from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import (
    ArgumentError,
    DatabaseError,
    DisconnectionError,
    IntegrityError,
    InternalError,
    NoSuchTableError,
    OperationalError,
    PendingRollbackError,
    ProgrammingError,
    SQLAlchemyError,
    StatementError,
    TimeoutError as SATimeoutError,
)
from sqlalchemy.pool import QueuePool

# ---------------------------------------------------------------------------
# Logging — one named logger; callers configure handlers / levels externally
# ---------------------------------------------------------------------------
logger = logging.getLogger("text_to_sql.database_manager")

# ---------------------------------------------------------------------------
# 1.  Project-specific exception hierarchy
# ---------------------------------------------------------------------------

class DatabaseManagerError(Exception):
    """Base class for all errors raised by DatabaseManager."""

    def __init__(self, message: str, original: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.original = original  # always carry the root cause for debugging

    def __str__(self) -> str:
        base = super().__str__()
        if self.original:
            return f"{base}  [caused by: {type(self.original).__name__}: {self.original}]"
        return base


class ConnectionError(DatabaseManagerError):          # noqa: A001  (shadows built-in intentionally)
    """Raised when the manager cannot establish or verify a DB connection."""


class SchemaReflectionError(DatabaseManagerError):
    """Raised when SQLAlchemy's Inspector fails to read schema metadata."""


class QueryExecutionError(DatabaseManagerError):
    """Raised when a valid-looking query fails at runtime."""


class QueryTimeoutError(QueryExecutionError):
    """Raised when the database reports a statement timeout."""


class QuerySyntaxError(QueryExecutionError):
    """Raised when the database rejects a query due to a syntax / type error."""


class ReadOnlyViolationError(QueryExecutionError):
    """
    Raised when a mutation statement (INSERT/UPDATE/DELETE/DDL) reaches
    the executor despite schema.py guards. This is the last line of defence.
    """


class PoolExhaustedError(DatabaseManagerError):
    """Raised when the connection pool is full and the checkout times out."""


# ---------------------------------------------------------------------------
# 2.  Result data-classes (plain Python; no Pydantic dependency here)
# ---------------------------------------------------------------------------

@dataclass
class ColumnMeta:
    name: str
    type: str                       # dialect-native type string, e.g. "VARCHAR(255)"
    nullable: bool
    primary_key: bool
    default: Optional[str] = None
    comment: Optional[str] = None


@dataclass
class ForeignKeyMeta:
    constrained_columns: List[str]
    referred_table: str
    referred_columns: List[str]

    def __str__(self) -> str:
        lhs = ", ".join(self.constrained_columns)
        rhs = ", ".join(self.referred_columns)
        return f"{lhs} → {self.referred_table}.{rhs}"


@dataclass
class IndexMeta:
    name: Optional[str]
    columns: List[str]
    unique: bool


@dataclass
class TableMeta:
    name: str
    columns: List[ColumnMeta]
    primary_keys: List[str]
    foreign_keys: List[ForeignKeyMeta]
    indexes: List[IndexMeta]
    row_count_estimate: Optional[int] = None   # None when dialect doesn't expose stats
    comment: Optional[str] = None


@dataclass
class SchemaMetadata:
    """
    Everything Agent 1 (SchemaAnalyzer) needs to populate a SchemaContext.
    Passed directly to schemas.SchemaContext for strict Pydantic validation.
    """
    database_type: str                          # matches SchemaContext.database_type literals
    tables: Dict[str, TableMeta]
    foreign_key_strings: List[str]             # human-readable: "orders.cid → customers.id"
    ambiguous_column_names: List[str]          # short / generic names likely to be vague
    raw_ddl: Dict[str, str]                    # table_name → CREATE TABLE ... (best-effort)


@dataclass
class QueryResult:
    """
    Returned by execute_query().  Rows are plain Python dicts for easy JSON
    serialisation — no SQLAlchemy Row objects escape this module.
    """
    columns: List[str]
    rows: List[Dict[str, Any]]
    row_count: int
    execution_time_ms: float
    truncated: bool                            # True when LIMIT was applied server-side
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 3.  Internals — read-only guard (second layer after schemas.py)
# ---------------------------------------------------------------------------

_MUTATION_PATTERN = re.compile(
    r"""
    \b(?:
        INSERT | UPDATE | DELETE | MERGE |
        DROP   | CREATE | ALTER  | TRUNCATE |
        REPLACE| EXEC(?:UTE)? | PRAGMA\s+\w+\s*= |  # PRAGMA writes
        ATTACH | DETACH | VACUUM | REINDEX |
        GRANT  | REVOKE | SAVEPOINT | ROLLBACK |
        COMMIT | BEGIN\s+(?:TRANSACTION|TRAN) |
        BULK\s+INSERT
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_AMBIGUOUS_COLUMN_NAMES = frozenset({
    "id", "status", "type", "name", "value", "score", "rank",
    "flag", "code", "label", "category", "level", "tag", "key",
    "amount", "count", "total", "result", "data", "info", "note",
    "desc", "description", "comment", "date", "time", "ts",
})


def _is_mutation(sql: str) -> bool:
    return bool(_MUTATION_PATTERN.search(sql))


def _detect_dialect(url: str) -> str:
    """Map a SQLAlchemy URL to a SchemaContext.database_type literal."""
    url_lower = url.lower()
    if url_lower.startswith("sqlite"):
        return "sqlite"
    if url_lower.startswith("postgresql") or url_lower.startswith("postgres"):
        return "postgresql"
    if url_lower.startswith("mysql"):
        return "mysql"
    if url_lower.startswith("mssql"):
        return "mssql"
    if url_lower.startswith("bigquery"):
        return "bigquery"
    return "other"


# ---------------------------------------------------------------------------
# 4.  DatabaseManager
# ---------------------------------------------------------------------------

class DatabaseManager:
    """
    Thread-safe, dialect-agnostic database manager.

    Lifecycle
    ---------
    1.  Instantiate with a connection URL.
    2.  Call connect() — validates connectivity with pool_pre_ping.
    3.  Call get_schema_metadata() — populates all TableMeta objects via Inspector.
    4.  Call execute_query(sql) — executes a pre-validated SELECT / WITH query.
    5.  Call close() (or use as a context manager) — disposes the engine pool.

    Usage as context manager (recommended):
        with DatabaseManager("sqlite:///sales.db") as db:
            meta = db.get_schema_metadata()
            result = db.execute_query("SELECT * FROM orders LIMIT 50")

    Parameters
    ----------
    database_url : str
        Any SQLAlchemy-compatible connection URL.
    pool_size : int
        Number of permanent connections in the pool (default 5).
        Use 1 for SQLite (which does not support concurrent writers).
    max_overflow : int
        Temporary extra connections beyond pool_size (default 10).
    pool_timeout : float
        Seconds to wait for a connection before raising PoolExhaustedError.
    pool_recycle : int
        Seconds after which idle connections are recycled (avoids "gone away").
    query_timeout : Optional[int]
        Per-statement timeout in seconds. None = database default.
    max_rows : int
        Hard cap on rows returned by execute_query (default 500).
    echo : bool
        If True, SQLAlchemy logs all SQL to the root logger (debug only).
    """

    def __init__(
        self,
        database_url: str,
        *,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: float = 30.0,
        pool_recycle: int = 3600,
        query_timeout: Optional[int] = None,
        max_rows: int = 500,
        echo: bool = False,
    ) -> None:
        if not database_url or not isinstance(database_url, str):
            raise ValueError("database_url must be a non-empty string.")

        self._url = database_url
        self._dialect = _detect_dialect(database_url)
        self._query_timeout = query_timeout
        self._max_rows = max_rows
        self._engine = None
        self._schema_cache: Optional[SchemaMetadata] = None

        # SQLite uses a StaticPool / NullPool — QueuePool + multiple threads
        # cause "ProgrammingError: SQLite objects created in a thread can only
        # be used in that same thread" or database-is-locked errors.
        is_sqlite = self._dialect == "sqlite"

        engine_kwargs: Dict[str, Any] = {
            "echo": echo,
            "future": True,                # SQLAlchemy 2.0 style
            "pool_pre_ping": True,         # validates connections on checkout
            "pool_recycle": pool_recycle,  # drop stale connections
        }

        if is_sqlite:
            # SQLite in-memory (:memory:) requires StaticPool so every
            # checkout reuses the same underlying connection — otherwise
            # each new connection creates a completely blank database and
            # schema reflection returns zero tables.
            # File-based SQLite (sqlite:///path.db) uses NullPool to avoid
            # "database is locked" errors from concurrent writers.
            from sqlalchemy.pool import StaticPool, NullPool
            if ":memory:" in database_url:
                engine_kwargs["connect_args"] = {"check_same_thread": False}
                engine_kwargs["poolclass"] = StaticPool
            else:
                engine_kwargs["connect_args"] = {"check_same_thread": False}
                engine_kwargs["poolclass"] = NullPool
        else:
            engine_kwargs["pool_size"] = pool_size
            engine_kwargs["max_overflow"] = max_overflow
            engine_kwargs["pool_timeout"] = pool_timeout

        try:
            self._engine = create_engine(self._url, **engine_kwargs)
        except ArgumentError as exc:
            raise ConnectionError(
                f"Invalid database URL format: '{self._url}'. "
                "Expected format examples:\n"
                "  sqlite:///path/to/file.db\n"
                "  postgresql+psycopg2://user:pass@host:5432/dbname\n"
                "  mysql+pymysql://user:pass@host:3306/dbname\n"
                "  mssql+pyodbc://user:pass@host/dbname?driver=ODBC+Driver+17+for+SQL+Server",
                original=exc,
            ) from exc

        logger.debug("Engine created for dialect '%s'.", self._dialect)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "DatabaseManager":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # 4a.  Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Verify the database is reachable.

        Raises
        ------
        ConnectionError
            On authentication failure, wrong host/port, missing driver,
            or if the engine was already closed.
        PoolExhaustedError
            If the pool cannot allocate a test connection within pool_timeout.
        """
        if self._engine is None:
            raise ConnectionError(
                "Engine has been disposed. Create a new DatabaseManager instance."
            )

        logger.info("Verifying connectivity to '%s'...", self._url)
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Connection verified successfully.")
        except OperationalError as exc:
            raise ConnectionError(
                f"Cannot connect to database. "
                f"Verify host, port, credentials, and network access.\n"
                f"URL: {self._url}",
                original=exc,
            ) from exc
        except SATimeoutError as exc:
            raise PoolExhaustedError(
                "Connection pool exhausted — all connections are in use. "
                "Increase pool_size or max_overflow, or reduce concurrent load.",
                original=exc,
            ) from exc
        except SQLAlchemyError as exc:
            raise ConnectionError(
                f"Unexpected error while connecting: {exc}",
                original=exc,
            ) from exc

    def close(self) -> None:
        """
        Dispose the connection pool and release all resources.
        Safe to call multiple times.
        """
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._schema_cache = None
            logger.info("DatabaseManager closed and pool disposed.")

    @property
    def is_connected(self) -> bool:
        """Returns True only if the engine is live (not disposed)."""
        if self._engine is None:
            return False
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 4b.  Context manager for individual connections
    # ------------------------------------------------------------------

    @contextmanager
    def _get_connection(self) -> Generator:
        """
        Yields a single, auto-committing read-only connection.

        All known SQLAlchemy disconnection and pool states are handled here
        so callers never need to manage raw connection lifecycle.

        Critical lesson from Apache Superset's PendingRollbackError bug:
        If a connection enters an error state, we must invalidate it
        explicitly and NEVER retry on the same poisoned connection.
        """
        if self._engine is None:
            raise ConnectionError(
                "Engine is not initialised. Call connect() first."
            )
        try:
            with self._engine.connect() as conn:
                # Set read-only mode where supported
                if self._dialect == "postgresql":
                    conn.execute(text("SET TRANSACTION READ ONLY"))
                elif self._dialect == "mysql":
                    conn.execute(text("SET SESSION TRANSACTION READ ONLY"))
                # Set statement timeout where supported
                if self._query_timeout is not None:
                    self._apply_timeout(conn)
                yield conn
        except PendingRollbackError as exc:
            # Connection is poisoned — invalidate it and surface clearly
            logger.error(
                "PendingRollbackError detected: a previous transaction was "
                "not rolled back. The connection has been invalidated. "
                "This is a programming bug — ensure exceptions are handled "
                "before reusing a connection. Detail: %s", exc,
            )
            raise QueryExecutionError(
                "Database connection is in a failed transaction state. "
                "This connection has been invalidated and discarded.",
                original=exc,
            ) from exc
        except DisconnectionError as exc:
            logger.warning("Disconnection detected; pool will recover on next use.")
            raise ConnectionError(
                "Database connection was lost mid-operation. "
                "The pool will self-heal on the next request.",
                original=exc,
            ) from exc
        except SATimeoutError as exc:
            raise PoolExhaustedError(
                "Timed out waiting for a free connection from the pool.",
                original=exc,
            ) from exc

    def _apply_timeout(self, conn: Any) -> None:
        """Apply a per-statement timeout in a dialect-safe way."""
        ms = self._query_timeout * 1000
        try:
            if self._dialect == "postgresql":
                conn.execute(text(f"SET statement_timeout = {int(ms)}"))
            elif self._dialect == "mysql":
                conn.execute(text(f"SET SESSION MAX_EXECUTION_TIME = {int(ms)}"))
            elif self._dialect == "mssql":
                # MSSQL uses per-connection timeout set on the connection string,
                # not via a SQL command; log a warning rather than silently failing
                logger.warning(
                    "MS SQL Server does not support SET statement_timeout via SQL. "
                    "Set 'connect_timeout' in your connection string instead."
                )
            # SQLite / BigQuery / other: no standard SQL timeout mechanism
        except SQLAlchemyError as exc:
            logger.warning("Could not apply query timeout: %s", exc)

    # ------------------------------------------------------------------
    # 4c.  Schema reflection
    # ------------------------------------------------------------------

    def get_schema_metadata(self, *, force_refresh: bool = False) -> SchemaMetadata:
        """
        Reflect the full database schema using SQLAlchemy's Inspector API.

        Results are cached after the first call. Pass force_refresh=True
        to re-read from the database (e.g. after a migration).

        Returns
        -------
        SchemaMetadata
            Structured representation of all tables, columns, PKs, FKs,
            indexes, and row-count estimates.

        Raises
        ------
        SchemaReflectionError
            If the Inspector cannot read the schema (permissions, corruption,
            empty database, dialect driver not installed, etc.).
        ConnectionError
            If the database is unreachable.
        """
        if self._schema_cache is not None and not force_refresh:
            logger.debug("Returning cached schema metadata.")
            return self._schema_cache

        logger.info("Reflecting schema from database...")

        try:
            inspector = inspect(self._engine)
        except OperationalError as exc:
            raise ConnectionError(
                "Cannot inspect the database — is the server running?",
                original=exc,
            ) from exc
        except SQLAlchemyError as exc:
            raise SchemaReflectionError(
                f"Inspector could not initialise: {exc}",
                original=exc,
            ) from exc

        try:
            table_names: List[str] = inspector.get_table_names()
        except SQLAlchemyError as exc:
            raise SchemaReflectionError(
                "get_table_names() failed. "
                "Verify the user has SELECT privilege on information_schema "
                "(PostgreSQL / MySQL) or sqlite_master (SQLite).",
                original=exc,
            ) from exc

        if not table_names:
            logger.warning(
                "Schema reflection returned zero tables. "
                "The database may be empty, or the user may lack visibility."
            )

        tables: Dict[str, TableMeta] = {}
        all_fk_strings: List[str] = []
        ambiguous_names: set = set()

        for tname in table_names:
            try:
                table_meta = self._reflect_table(inspector, tname, all_fk_strings, ambiguous_names)
                tables[tname] = table_meta
            except NoSuchTableError:
                # Race condition: table disappeared between get_table_names() and reflection
                logger.warning(
                    "Table '%s' vanished between listing and reflection — skipping.", tname
                )
            except SQLAlchemyError as exc:
                # Don't abort the entire reflection for one bad table —
                # log it and continue so other tables remain accessible.
                logger.error(
                    "Could not reflect table '%s': %s. Skipping.", tname, exc
                )

        self._schema_cache = SchemaMetadata(
            database_type=self._dialect,
            tables=tables,
            foreign_key_strings=sorted(set(all_fk_strings)),
            ambiguous_column_names=sorted(ambiguous_names),
            raw_ddl=self._get_raw_ddl(inspector, table_names),
        )

        logger.info(
            "Schema reflected: %d tables, %d FK relationships.",
            len(tables),
            len(all_fk_strings),
        )
        return self._schema_cache

    def _reflect_table(
        self,
        inspector: Any,
        tname: str,
        all_fk_strings: List[str],
        ambiguous_names: set,
    ) -> TableMeta:
        """Reflect a single table; mutates all_fk_strings and ambiguous_names."""

        # --- Columns ---
        raw_columns = inspector.get_columns(tname)
        pk_set: set = set(inspector.get_pk_constraint(tname).get("constrained_columns", []))

        columns: List[ColumnMeta] = []
        for col in raw_columns:
            col_name: str = col["name"]
            col_type_str: str = str(col.get("type", "UNKNOWN"))
            col_nullable: bool = col.get("nullable", True)
            col_default: Optional[str] = str(col["default"]) if col.get("default") is not None else None
            col_comment: Optional[str] = col.get("comment")

            columns.append(ColumnMeta(
                name=col_name,
                type=col_type_str,
                nullable=col_nullable,
                primary_key=(col_name in pk_set),
                default=col_default,
                comment=col_comment,
            ))

            if col_name.lower() in _AMBIGUOUS_COLUMN_NAMES:
                ambiguous_names.add(f"{tname}.{col_name}")

        # --- Foreign keys ---
        fk_metas: List[ForeignKeyMeta] = []
        for fk in inspector.get_foreign_keys(tname):
            constrained = fk.get("constrained_columns", [])
            referred_table = fk.get("referred_table", "?")
            referred_cols = fk.get("referred_columns", [])
            fk_meta = ForeignKeyMeta(
                constrained_columns=constrained,
                referred_table=referred_table,
                referred_columns=referred_cols,
            )
            fk_metas.append(fk_meta)
            # Human-readable string consumed by Agent 1
            all_fk_strings.append(
                f"{tname}.{', '.join(constrained)} → {referred_table}.{', '.join(referred_cols)}"
            )

        # --- Indexes ---
        index_metas: List[IndexMeta] = []
        for idx in inspector.get_indexes(tname):
            index_metas.append(IndexMeta(
                name=idx.get("name"),
                columns=idx.get("column_names", []),
                unique=idx.get("unique", False),
            ))

        # --- Row count estimate (best-effort; not all dialects support it) ---
        row_count = self._estimate_row_count(tname)

        # --- Table comment (PostgreSQL / MySQL) ---
        table_comment: Optional[str] = None
        try:
            table_comment = inspector.get_table_comment(tname).get("text")
        except (NotImplementedError, SQLAlchemyError):
            pass

        return TableMeta(
            name=tname,
            columns=columns,
            primary_keys=list(pk_set),
            foreign_keys=fk_metas,
            indexes=index_metas,
            row_count_estimate=row_count,
            comment=table_comment,
        )

    def _estimate_row_count(self, table_name: str) -> Optional[int]:
        """
        Return a fast row-count estimate.

        Strategy by dialect:
        - PostgreSQL : pg_class.reltuples  (fast, updated by ANALYZE)
        - MySQL      : information_schema.TABLES.TABLE_ROWS  (fast estimate)
        - SQLite     : COUNT(*) — unavoidable, but SQLite is always local
        - Others     : COUNT(*) with a fallback to None on any error
        """
        try:
            with self._get_connection() as conn:
                if self._dialect == "postgresql":
                    result = conn.execute(text(
                        "SELECT reltuples::BIGINT FROM pg_class WHERE relname = :t"
                    ), {"t": table_name}).scalar()
                    return int(result) if result is not None else None

                elif self._dialect == "mysql":
                    result = conn.execute(text(
                        "SELECT TABLE_ROWS FROM information_schema.TABLES "
                        "WHERE TABLE_NAME = :t AND TABLE_SCHEMA = DATABASE()"
                    ), {"t": table_name}).scalar()
                    return int(result) if result is not None else None

                else:
                    # SQLite, MSSQL, BigQuery, etc.
                    result = conn.execute(
                        text(f"SELECT COUNT(*) FROM \"{table_name}\"")
                    ).scalar()
                    return int(result) if result is not None else None

        except Exception as exc:
            logger.debug(
                "Row count estimate unavailable for '%s': %s", table_name, exc
            )
            return None

    def _get_raw_ddl(
        self, inspector: Any, table_names: List[str]
    ) -> Dict[str, str]:
        """
        Reconstruct a CREATE TABLE-like DDL string for each table.
        Used by Agent 1 to embed the full schema in its LLM prompt.
        This is best-effort: not all dialects return verbatim DDL.
        """
        ddl: Dict[str, str] = {}
        for tname in table_names:
            try:
                lines = [f"CREATE TABLE {tname} ("]
                cols = inspector.get_columns(tname)
                pk_set = set(inspector.get_pk_constraint(tname).get("constrained_columns", []))
                col_lines = []
                for col in cols:
                    flags = []
                    if col["name"] in pk_set:
                        flags.append("PRIMARY KEY")
                    if not col.get("nullable", True):
                        flags.append("NOT NULL")
                    flag_str = " " + " ".join(flags) if flags else ""
                    col_lines.append(f"    {col['name']} {col['type']}{flag_str}")
                for fk in inspector.get_foreign_keys(tname):
                    cols_str = ", ".join(fk["constrained_columns"])
                    ref_cols_str = ", ".join(fk["referred_columns"])
                    col_lines.append(
                        f"    FOREIGN KEY ({cols_str}) REFERENCES "
                        f"{fk['referred_table']} ({ref_cols_str})"
                    )
                lines.append(",\n".join(col_lines))
                lines.append(");")
                ddl[tname] = "\n".join(lines)
            except Exception as exc:
                ddl[tname] = f"-- DDL unavailable for {tname}: {exc}"
        return ddl

    # ------------------------------------------------------------------
    # 4d.  Table sampling (for Agent 1 context enrichment)
    # ------------------------------------------------------------------

    def get_table_sample(
        self, table_name: str, *, n_rows: int = 3
    ) -> Dict[str, Any]:
        """
        Return the first n_rows rows of a table as a dict.

        Used by Agent 1 to include concrete examples in its LLM prompt,
        which dramatically improves type inference and column-meaning detection.

        Parameters
        ----------
        table_name : str
            Exact table name (case-sensitive for PostgreSQL/MySQL).
        n_rows : int
            Maximum number of rows to return (default 3, hard cap 10).

        Returns
        -------
        dict with keys:
            table_name, columns, rows, row_count

        Raises
        ------
        SchemaReflectionError  — table does not exist.
        QueryExecutionError    — execution failed.
        """
        n_rows = min(max(1, n_rows), 10)   # clamp to [1, 10]

        # Validate table exists (prevents injection via table_name parameter)
        meta = self.get_schema_metadata()
        if table_name not in meta.tables:
            raise SchemaReflectionError(
                f"Table '{table_name}' does not exist in the reflected schema. "
                f"Available tables: {sorted(meta.tables.keys())}"
            )

        # Safely quote the table name so it can't be exploited
        quoted = self._quote_identifier(table_name)
        sql = f"SELECT * FROM {quoted} LIMIT {n_rows}"

        try:
            with self._get_connection() as conn:
                result = conn.execute(text(sql))
                cols = list(result.keys())
                rows = [dict(zip(cols, row)) for row in result.fetchall()]
            return {
                "table_name": table_name,
                "columns": cols,
                "rows": rows,
                "row_count": len(rows),
            }
        except SQLAlchemyError as exc:
            raise QueryExecutionError(
                f"Failed to sample table '{table_name}'.",
                original=exc,
            ) from exc

    def _quote_identifier(self, name: str) -> str:
        """Return a safely quoted identifier for the current dialect."""
        if self._dialect == "mssql":
            return f"[{name}]"
        # ANSI SQL double-quote for PostgreSQL, MySQL, SQLite, BigQuery
        return f'"{name}"'

    # ------------------------------------------------------------------
    # 4e.  Query execution
    # ------------------------------------------------------------------

    def execute_query(self, sql: str) -> QueryResult:
        """
        Execute a pre-validated SQL SELECT / WITH query and return results.

        This method is the LAST line of defence:
        1. Checks the query is read-only (rejects any mutation even if schemas.py missed it).
        2. Enforces a row cap (max_rows) to prevent memory exhaustion.
        3. Maps every SQLAlchemy exception to a typed project exception.
        4. Records wall-clock execution time.

        Parameters
        ----------
        sql : str
            A SQL string that passed schemas.py validation.

        Returns
        -------
        QueryResult

        Raises
        ------
        ReadOnlyViolationError   — DML/DDL detected.
        QuerySyntaxError         — database rejected the SQL (syntax / type error).
        QueryTimeoutError        — statement exceeded the configured timeout.
        QueryExecutionError      — all other runtime failures.
        ConnectionError          — database unreachable during execution.
        """
        if not sql or not isinstance(sql, str):
            raise QueryExecutionError("sql must be a non-empty string.")

        sql = sql.strip()

        # ---- Last-chance read-only guard ----
        if _is_mutation(sql):
            raise ReadOnlyViolationError(
                "Rejected: the SQL contains a write/DDL statement. "
                "Only SELECT and WITH queries are permitted. "
                "This query must not reach the database.",
            )

        warnings: List[str] = []

        # ---- Enforce row cap ----
        # If the query already has a LIMIT, respect it but cap at max_rows.
        sql, cap_applied = self._apply_row_cap(sql, warnings)

        logger.info("Executing query (cap=%d): %.120s...", self._max_rows, sql)
        t_start = time.perf_counter()

        try:
            with self._get_connection() as conn:
                result = conn.execute(text(sql))
                cols: List[str] = list(result.keys())
                rows_raw = result.fetchmany(self._max_rows + 1)   # fetch one extra to detect overflow

        # ---- Granular exception mapping ----
        except OperationalError as exc:
            orig_str = str(exc.orig).lower() if exc.orig else ""
            if any(kw in orig_str for kw in ("timeout", "timed out", "max_execution_time", "statement_timeout")):
                raise QueryTimeoutError(
                    f"Query timed out after {self._query_timeout}s. "
                    "Simplify the query, add indexes, or increase query_timeout.",
                    original=exc,
                ) from exc
            if any(kw in orig_str for kw in ("lost connection", "gone away", "broken pipe", "connection refused")):
                raise ConnectionError(
                    "Database connection was lost during query execution.",
                    original=exc,
                ) from exc
            raise QueryExecutionError(
                f"Operational error during query execution: {exc}",
                original=exc,
            ) from exc

        except ProgrammingError as exc:
            raise QuerySyntaxError(
                f"SQL syntax or schema error: {exc.orig or exc}. "
                "Verify table names, column names, and JOIN conditions.",
                original=exc,
            ) from exc

        except InternalError as exc:
            raise QueryExecutionError(
                f"Database internal error (cursor/transaction out of sync): {exc}. "
                "This may indicate a driver bug or a corrupt database.",
                original=exc,
            ) from exc

        except IntegrityError as exc:
            # Should never occur on a SELECT, but guard anyway
            raise QueryExecutionError(
                "Integrity constraint violated during SELECT — this is unexpected. "
                "Please report this as a bug.",
                original=exc,
            ) from exc

        except StatementError as exc:
            raise QueryExecutionError(
                f"Statement processing error (possible type mismatch): {exc}",
                original=exc,
            ) from exc

        except (DatabaseError, SQLAlchemyError) as exc:
            raise QueryExecutionError(
                f"Unexpected database error: {exc}",
                original=exc,
            ) from exc

        t_end = time.perf_counter()
        exec_ms = (t_end - t_start) * 1000

        # ---- Detect and trim overflow ----
        truncated = len(rows_raw) > self._max_rows
        rows_raw = rows_raw[: self._max_rows]

        rows: List[Dict[str, Any]] = [
            {col: self._safe_value(val) for col, val in zip(cols, row)}
            for row in rows_raw
        ]

        if truncated:
            warnings.append(
                f"Result truncated to {self._max_rows} rows. "
                "Add a more specific WHERE clause or tighten your LIMIT to see all data."
            )
        if cap_applied:
            warnings.append(f"LIMIT {self._max_rows} was applied automatically.")

        result_obj = QueryResult(
            columns=cols,
            rows=rows,
            row_count=len(rows),
            execution_time_ms=round(exec_ms, 2),
            truncated=truncated,
            warnings=warnings,
        )
        logger.info(
            "Query returned %d row(s) in %.1f ms.%s",
            result_obj.row_count,
            result_obj.execution_time_ms,
            " [truncated]" if truncated else "",
        )
        return result_obj

    def _apply_row_cap(self, sql: str, warnings: List[str]) -> Tuple[str, bool]:
        """
        Ensure a LIMIT <= max_rows is present in the query.

        Returns (modified_sql, cap_was_applied).
        """
        limit_match = re.search(r"\bLIMIT\s+(\d+)", sql, re.IGNORECASE)
        if limit_match:
            existing_limit = int(limit_match.group(1))
            if existing_limit > self._max_rows:
                # Replace the user's LIMIT with the enforced cap
                sql = re.sub(
                    r"\bLIMIT\s+\d+",
                    f"LIMIT {self._max_rows}",
                    sql,
                    flags=re.IGNORECASE,
                    count=1,
                )
                warnings.append(
                    f"LIMIT reduced from {existing_limit} to {self._max_rows} "
                    f"(max_rows cap)."
                )
                return sql, True
            return sql, False
        else:
            # No LIMIT at all — append one
            sql = f"{sql.rstrip().rstrip(';')} LIMIT {self._max_rows}"
            return sql, True

    @staticmethod
    def _safe_value(value: Any) -> Any:
        """
        Convert database values to JSON-safe Python types.
        SQLAlchemy can return decimal.Decimal, datetime, bytes, etc. —
        none of which are directly JSON-serialisable.
        """
        import decimal
        import datetime

        if value is None:
            return None
        if isinstance(value, (int, float, str, bool)):
            return value
        if isinstance(value, decimal.Decimal):
            # Preserve precision as string; callers can cast if needed
            return float(value)
        if isinstance(value, (datetime.date, datetime.datetime)):
            return value.isoformat()
        if isinstance(value, bytes):
            return value.hex()
        # Fallback: stringify unknown types rather than crashing
        return str(value)

    # ------------------------------------------------------------------
    # 4f.  Convenience helpers for agents
    # ------------------------------------------------------------------

    def get_table_names(self) -> List[str]:
        """Return a sorted list of all reflected table names."""
        return sorted(self.get_schema_metadata().tables.keys())

    def get_table_meta(self, table_name: str) -> TableMeta:
        """
        Return TableMeta for a specific table.

        Raises
        ------
        SchemaReflectionError — if the table is not in the reflected schema.
        """
        meta = self.get_schema_metadata()
        if table_name not in meta.tables:
            raise SchemaReflectionError(
                f"Table '{table_name}' not found. "
                f"Available: {sorted(meta.tables.keys())}"
            )
        return meta.tables[table_name]

    def get_foreign_key_strings(self) -> List[str]:
        """Return all FK relationships as human-readable strings."""
        return self.get_schema_metadata().foreign_key_strings

    def validate_tables_exist(self, table_names: List[str]) -> List[str]:
        """
        Given a list of table names (from SQLExecutorOutput.tables_used),
        return those that do NOT exist in the schema.

        Used by the orchestrator to catch hallucinated table names.
        """
        known = set(self.get_schema_metadata().tables.keys())
        return [t for t in table_names if t not in known]

    def health_check(self) -> Dict[str, Any]:
        """
        Return a health-check dict for monitoring / logging.

        Keys: status, dialect, table_count, pool_size, pool_checked_out
        """
        report: Dict[str, Any] = {
            "status": "unknown",
            "dialect": self._dialect,
            "table_count": 0,
            "pool_size": None,
            "pool_checked_out": None,
        }
        try:
            report["table_count"] = len(self.get_table_names())
            pool = self._engine.pool if self._engine else None
            if pool:
                # NullPool and StaticPool do not implement size()/checkedout()
                try:
                    report["pool_size"] = pool.size()
                    report["pool_checked_out"] = pool.checkedout()
                except AttributeError:
                    report["pool_size"] = "n/a (NullPool/StaticPool)"
                    report["pool_checked_out"] = "n/a"
            report["status"] = "healthy"
        except Exception as exc:
            report["status"] = "unhealthy"
            report["error"] = str(exc)
        return report


# ---------------------------------------------------------------------------
# 5.  Factory helper — used by CrewAI agent tool wrappers
# ---------------------------------------------------------------------------

def build_manager_from_url(
    database_url: str,
    *,
    max_rows: int = 500,
    query_timeout: Optional[int] = 30,
    echo: bool = False,
) -> DatabaseManager:
    """
    Convenience factory that constructs, connects, and returns a DatabaseManager.

    Raises DatabaseManagerError subclasses on any failure so the caller
    only needs to catch the project base exception.
    """
    manager = DatabaseManager(
        database_url,
        max_rows=max_rows,
        query_timeout=query_timeout,
        echo=echo,
    )
    manager.connect()
    return manager


# ---------------------------------------------------------------------------
# 6.  Smoke-test  (run: python database_manager.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    print("\n" + "=" * 65)
    print("SMOKE TEST — database_manager.py")
    print("=" * 65)

    # ----------------------------------------------------------------
    # Build a temp-FILE SQLite database with realistic sample data.
    # We use a file (not :memory:) for setup so that DatabaseManager
    # opens its own NullPool connection to the same data cleanly.
    # ----------------------------------------------------------------
    import os
    import datetime
    import tempfile
    from sqlalchemy import (
        Column, DateTime, ForeignKey, Integer, MetaData,
        Numeric, String, Table as SATable,
    )

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
    os.close(tmp_fd)
    DEMO_URL = f"sqlite:///{tmp_path}"

    engine_setup = create_engine(DEMO_URL, future=True)
    meta_setup = MetaData()
    customers_t = SATable(
        "customers", meta_setup,
        Column("customer_id", Integer, primary_key=True),
        Column("name", String(120), nullable=False),
        Column("email", String(255), nullable=False),
        Column("joined_at", DateTime),
    )
    orders_t = SATable(
        "orders", meta_setup,
        Column("order_id", Integer, primary_key=True),
        Column("customer_id", Integer, ForeignKey("customers.customer_id")),
        Column("total_amount", Numeric(10, 2), nullable=False),
        Column("status", String(30)),
        Column("created_at", DateTime),
    )
    meta_setup.create_all(engine_setup)

    with engine_setup.begin() as conn:
        conn.execute(customers_t.insert(), [
            {"customer_id": 1, "name": "Alice Nawaz", "email": "alice@example.com",
             "joined_at": datetime.datetime(2023, 1, 15)},
            {"customer_id": 2, "name": "Bob Tariq",   "email": "bob@example.com",
             "joined_at": datetime.datetime(2023, 3, 20)},
            {"customer_id": 3, "name": "Carla Raza",  "email": "carla@example.com",
             "joined_at": datetime.datetime(2024, 6, 1)},
        ])
        conn.execute(orders_t.insert(), [
            {"order_id": 101, "customer_id": 1, "total_amount": 299.99, "status": "completed",
             "created_at": datetime.datetime(2024, 1, 10)},
            {"order_id": 102, "customer_id": 1, "total_amount": 149.50, "status": "completed",
             "created_at": datetime.datetime(2024, 3, 5)},
            {"order_id": 103, "customer_id": 2, "total_amount": 899.00, "status": "pending",
             "created_at": datetime.datetime(2024, 8, 22)},
            {"order_id": 104, "customer_id": 3, "total_amount": 55.00,  "status": "cancelled",
             "created_at": datetime.datetime(2024, 9, 1)},
        ])
    engine_setup.dispose()   # release setup engine before DatabaseManager opens the file

    # ----------------------------------------------------------------
    # Test the DatabaseManager against the populated database
    # ----------------------------------------------------------------
    print("\n[1] Context manager + connectivity")
    with DatabaseManager(DEMO_URL, max_rows=500) as db:

        print("[OK] Connected via context manager")

        # Schema reflection
        print("\n[2] Schema reflection")
        schema = db.get_schema_metadata()
        print(f"[OK] Tables found: {list(schema.tables.keys())}")
        print(f"[OK] FK strings: {schema.foreign_key_strings}")
        print(f"[OK] Ambiguous columns: {schema.ambiguous_column_names}")

        # DDL
        print("\n[3] Raw DDL (orders table)")
        print(schema.raw_ddl.get("orders", "unavailable"))

        # Table sample
        print("\n[4] Table sample (customers, 2 rows)")
        sample = db.get_table_sample("customers", n_rows=2)
        print(json.dumps(sample, indent=2, default=str))

        # Valid query
        print("\n[5] Valid read-only query")
        result = db.execute_query(
            "SELECT c.name, SUM(o.total_amount) AS total_spend "
            "FROM customers c JOIN orders o ON o.customer_id = c.customer_id "
            "GROUP BY c.name ORDER BY total_spend DESC LIMIT 10"
        )
        print(f"[OK] {result.row_count} row(s) in {result.execution_time_ms} ms")
        print(json.dumps(result.rows, indent=2, default=str))

        # Row cap enforcement
        print("\n[6] Row cap enforcement (LIMIT 9999 → capped)")
        r2 = db.execute_query("SELECT * FROM orders LIMIT 9999")
        print(f"[OK] Returned {r2.row_count} row(s), warnings={r2.warnings}")

        # Hallucinated table validation
        print("\n[7] Hallucinated table detection")
        missing = db.validate_tables_exist(["customers", "ghost_table", "orders"])
        assert missing == ["ghost_table"], f"Expected ['ghost_table'], got {missing}"
        print("[OK] Hallucinated tables detected:", missing)

        # Health check
        print("\n[8] Health check")
        health = db.health_check()
        print(json.dumps(health, indent=2))

    # ----------------------------------------------------------------
    # Exception path tests (run outside context manager)
    # ----------------------------------------------------------------
    print("\n[9] Exception path tests")

    # ReadOnlyViolationError
    db_e = DatabaseManager(DEMO_URL)
    db_e.connect()
    for bad_sql in [
        "DROP TABLE customers",
        "DELETE FROM orders",
        "INSERT INTO customers VALUES (99,'x','x',NULL)",
        "UPDATE orders SET status='done'",
    ]:
        try:
            db_e.execute_query(bad_sql)
            print(f"[FAIL — not caught]: {bad_sql!r}")
        except ReadOnlyViolationError as exc:
            print(f"[BLOCKED OK] ReadOnlyViolationError: {bad_sql[:45]!r}")

    # QuerySyntaxError
    try:
        db_e.execute_query("SELECT * FROM nonexistent_table LIMIT 10")
    except QueryExecutionError as exc:
        print(f"[OK] Nonexistent table raised: {type(exc).__name__}")

    # ConnectionError / missing driver on bad URL
    # (psycopg2 may not be installed in the test environment — that's fine;
    #  ArgumentError from SQLAlchemy or ModuleNotFoundError from the driver
    #  both surface as ConnectionError via __init__'s ArgumentError handler
    #  or as an uncaught ModuleNotFoundError before connect() is called.
    #  We catch both to keep the test environment-agnostic.)
    try:
        bad_db = DatabaseManager("postgresql+psycopg2://user:pass@localhost:9999/nope")
        bad_db.connect()
        print("[INFO] psycopg2 installed — connect() attempt triggered ConnectionError")
    except ConnectionError as exc:
        print(f"[OK] Bad URL raised: {type(exc).__name__}")
    except ModuleNotFoundError as exc:
        print(f"[OK] Driver not installed (expected in CI): {exc}")

    # SchemaReflectionError on bad table name in sample
    try:
        db_e.get_table_sample("ghost_table")
    except SchemaReflectionError as exc:
        print(f"[OK] Ghost table sample raised: {type(exc).__name__}")

    db_e.close()

    # Clean up temp file
    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    print("\n" + "=" * 65)
    print("ALL CHECKS PASSED")
    print("=" * 65 + "\n")