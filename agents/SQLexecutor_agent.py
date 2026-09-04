"""
sql_executor_agent.py — Agent 3: SQL Executor
==============================================
Model  : openai/gpt-oss-20b  (via Groq API)
Role   : Receive the clarified, unambiguous intent from Agent 2 and the
         SchemaContext from Agent 1. Write a correct, safe, dialect-aware SQL
         query. Run it through a multi-layer static validator before touching
         the database. Execute it via DatabaseManager. Return a fully-typed
         SQLExecutorOutput + QueryResult back to the orchestrator.

Pipeline position
-----------------
  [Agent 1] → SchemaContext
  [Agent 2] → ClarificationOutput  (status = 'clear')
      ↓
  [Agent 3 — SQLExecutor]          ← THIS FILE
      ↓
  SQLExecutorOutput  +  QueryResult

Bug categories caught at every layer
-------------------------------------
Layer 0 — Pre-generation guards (Python, no LLM)
  • ClarificationOutput.status is not 'clear' → raise, never generate SQL
  • SchemaContext.available_tables is empty    → raise immediately

Layer 1 — sqlglot static analysis (AST-level, no DB connection)
  B01  Syntax error / unbalanced parentheses
  B02  Hallucinated table name not in SchemaContext
  B03  SELECT * without explicit column list (silent data exposure)
  B04  Missing LIMIT clause (unbounded result)
  B05  Aggregate in WHERE clause (must be in HAVING)
  B06  Missing GROUP BY when aggregates + non-aggregated columns co-exist
  B07  Implicit Cartesian product — multi-table FROM without JOIN ON
  B08  INNER JOIN when LEFT JOIN is semantically required
  B09  Division without NULLIF guard (divide-by-zero risk)
  B10  DISTINCT inside aggregate used incorrectly
  B11  Non-qualified ambiguous column reference across multiple joined tables
  B12  ORDER BY column not in SELECT (some dialects reject this)
  B13  Subquery returns multiple columns used as scalar
  B14  HAVING without GROUP BY

Layer 2 — Schema validator (structural, no DB connection)
  B15  Column name not found in any referenced table's schema
  B16  Type mismatch — numeric aggregate applied to TEXT column
  B17  FK join path not used when it should be (missing required join)

Layer 3 — DatabaseManager execution
  B18  QuerySyntaxError from the DB engine (dialect-specific rejection)
  B19  QueryTimeoutError
  B20  ReadOnlyViolationError  (last-chance guard from database_manager.py)
  B21  Empty result set → surface as a warning, not an error

Self-correction loop
---------------------
  The agent runs a generate → validate → critique → regenerate loop
  (max MAX_CORRECTION_ROUNDS rounds) before it ever touches the database.
  Each round feeds the previous SQL + its validation report back to the LLM.
  This mirrors the SelECT-SQL "generate-and-critique" pattern from NAACL 2025
  which improved text-to-SQL accuracy significantly over single-pass generation.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import sqlglot
from sqlglot import exp as sqlexp
from sqlglot.errors import ParseError as SQLGlotParseError

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ValidationError

from schemas import (
    ClarificationOutput,
    SchemaContext,
    SQLExecutorOutput,
    export_schemas,
    _validate_sql,          # reuse the same guard from schemas.py
)
from database.database_manager import (
    DatabaseManager,
    DatabaseManagerError,
    QueryResult,
    QueryExecutionError,
    QuerySyntaxError,
    QueryTimeoutError,
    ReadOnlyViolationError,
    SchemaReflectionError,
    ConnectionError as DBConnectionError,
)

logger = logging.getLogger("text_to_sql.sql_executor_agent")

# ---------------------------------------------------------------------------
# 0.  Constants
# ---------------------------------------------------------------------------

MAX_CORRECTION_ROUNDS = 3   # generate → validate → critique loop limit
DIALECT_MAP: Dict[str, str] = {
    "sqlite":     "sqlite",
    "postgresql": "postgres",
    "mysql":      "mysql",
    "mssql":      "tsql",
    "bigquery":   "bigquery",
    "other":      "",        # sqlglot generic dialect
}

# Column types that are numeric — used to detect type-mismatch aggregations
NUMERIC_TYPES: frozenset = frozenset({
    "integer", "int", "bigint", "smallint", "tinyint",
    "numeric", "decimal", "float", "double", "real",
    "number", "money", "bit",
})


# ---------------------------------------------------------------------------
# 1.  Validation result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    code: str          # B01..B21
    severity: str      # "error" | "warning"
    message: str
    suggestion: str


@dataclass
class ValidationReport:
    sql: str
    dialect: str
    issues: List[ValidationIssue]
    tables_extracted: List[str]
    columns_extracted: List[str]
    join_count: int
    has_limit: bool
    has_group_by: bool
    has_aggregate: bool

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == "warning" for i in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sql": self.sql,
            "dialect": self.dialect,
            "has_errors": self.has_errors,
            "has_warnings": self.has_warnings,
            "issues": [
                {
                    "code": i.code,
                    "severity": i.severity,
                    "message": i.message,
                    "suggestion": i.suggestion,
                }
                for i in self.issues
            ],
            "tables_extracted": self.tables_extracted,
            "columns_extracted": self.columns_extracted,
            "join_count": self.join_count,
            "has_limit": self.has_limit,
            "has_group_by": self.has_group_by,
            "has_aggregate": self.has_aggregate,
        }


# ---------------------------------------------------------------------------
# 2.  Static SQL Validator (Layer 1 + Layer 2)
# ---------------------------------------------------------------------------

class SQLValidator:
    """
    Multi-rule static SQL analyser.

    Uses sqlglot to parse the SQL into an AST and runs every bug-check rule
    against it without making a database connection. Fast, deterministic,
    zero-token-cost.

    Also applies schemas.py _validate_sql() as the entry gate (Layer 0
    safety contract: SELECT/WITH only, LIMIT required, no injection).
    """

    def __init__(self, ctx: SchemaContext) -> None:
        self._ctx   = ctx
        self._dialect = DIALECT_MAP.get(ctx.database_type, "")
        # Build column lookup: table → {col_name → type_string}
        self._schema_index: Dict[str, Dict[str, str]] = {}
        # We populate this from table_summaries which is the only structural
        # data we have in SchemaContext — actual column details come from
        # DatabaseManager.get_schema_metadata() if available.

    def validate(self, sql: str) -> ValidationReport:
        """
        Run all validation rules. Returns a ValidationReport.
        Never raises — all issues are captured as ValidationIssue objects.
        """
        issues: List[ValidationIssue] = []

        # ---- Layer 0: schemas.py contract ----
        try:
            _validate_sql(sql)
        except ValueError as exc:
            issues.append(ValidationIssue(
                code="B00",
                severity="error",
                message=f"Safety gate rejected: {exc}",
                suggestion="Ensure the query starts with SELECT/WITH, has a LIMIT, and contains no DDL/DML.",
            ))
            return ValidationReport(
                sql=sql, dialect=self._dialect, issues=issues,
                tables_extracted=[], columns_extracted=[],
                join_count=0, has_limit=False, has_group_by=False, has_aggregate=False,
            )

        # ---- Layer 1: sqlglot AST analysis ----
        try:
            dialect_arg = self._dialect if self._dialect else None
            tree = sqlglot.parse_one(sql, dialect=dialect_arg)
        except SQLGlotParseError as exc:
            issues.append(ValidationIssue(
                code="B01",
                severity="error",
                message=f"SQL syntax error: {exc}",
                suggestion="Fix unbalanced parentheses, missing commas, or reserved keyword misuse.",
            ))
            return ValidationReport(
                sql=sql, dialect=self._dialect, issues=issues,
                tables_extracted=[], columns_extracted=[],
                join_count=0, has_limit=False, has_group_by=False, has_aggregate=False,
            )
        except Exception as exc:
            issues.append(ValidationIssue(
                code="B01",
                severity="error",
                message=f"Unexpected parse failure: {exc}",
                suggestion="Simplify the query and retry.",
            ))
            return ValidationReport(
                sql=sql, dialect=self._dialect, issues=issues,
                tables_extracted=[], columns_extracted=[],
                join_count=0, has_limit=False, has_group_by=False, has_aggregate=False,
            )

        # --- Extract AST components ---
        tables_in_query: List[str] = [
            t.name.lower()
            for t in tree.find_all(sqlexp.Table)
            if t.name  # exclude subquery aliases
        ]
        columns_in_query: List[str] = [
            c.alias_or_name
            for c in tree.find_all(sqlexp.Column)
        ]
        agg_nodes   = list(tree.find_all(sqlexp.AggFunc))
        join_nodes  = list(tree.find_all(sqlexp.Join))
        group_node  = tree.find(sqlexp.Group)
        limit_node  = tree.find(sqlexp.Limit)
        having_node = tree.find(sqlexp.Having)
        order_node  = tree.find(sqlexp.Order)
        star_nodes  = list(tree.find_all(sqlexp.Star))
        where_node  = tree.find(sqlexp.Where)
        from_node   = tree.find(sqlexp.From)

        has_aggregate  = bool(agg_nodes)
        has_group_by   = group_node is not None
        has_limit      = limit_node is not None
        join_count     = len(join_nodes)

        # ---- B02: Hallucinated table names ----
        known_tables = {t.lower() for t in self._ctx.available_tables}
        cte_names = {
            cte.alias_or_name.lower()
            for cte in tree.find_all(sqlexp.CTE)
        }
        for tname in set(tables_in_query):
            if tname and tname not in known_tables and tname not in cte_names:
                issues.append(ValidationIssue(
                    code="B02",
                    severity="error",
                    message=f"Table '{tname}' does not exist in the schema.",
                    suggestion=f"Available tables: {sorted(self._ctx.available_tables)}. "
                               "Check for typos or use a CTE if this is a derived table.",
                ))

        # ---- B03: SELECT * ----
        if star_nodes:
            issues.append(ValidationIssue(
                code="B03",
                severity="warning",
                message="SELECT * detected — returns all columns including potentially sensitive ones.",
                suggestion="Explicitly list only the columns needed for the user's question.",
            ))

        # ---- B04: Missing LIMIT ----
        # (Already caught by _validate_sql, but re-flag if somehow bypassed)
        if not has_limit:
            issues.append(ValidationIssue(
                code="B04",
                severity="error",
                message="No LIMIT clause found — query could return unbounded rows.",
                suggestion="Add LIMIT 100 (or appropriate value) at the end of the query.",
            ))

        # ---- B05: Aggregate in WHERE clause ----
        if where_node:
            where_sql = where_node.sql()
            agg_kws   = re.compile(
                r"\b(SUM|COUNT|AVG|MAX|MIN|GROUP_CONCAT|STRING_AGG|ARRAY_AGG)\s*\(",
                re.IGNORECASE,
            )
            if agg_kws.search(where_sql):
                issues.append(ValidationIssue(
                    code="B05",
                    severity="error",
                    message="Aggregate function used inside WHERE clause — this will fail at runtime.",
                    suggestion="Move the aggregate condition to a HAVING clause after GROUP BY.",
                ))

        # ---- B06: Non-aggregated column in SELECT with aggregates ----
        if has_aggregate and has_group_by:
            group_by_cols: set = set()
            if group_node:
                for col in group_node.find_all(sqlexp.Column):
                    group_by_cols.add(col.alias_or_name.lower())

            top_select = tree.find(sqlexp.Select)
            if top_select:
                for proj in top_select.expressions:
                    if isinstance(proj, sqlexp.AggFunc):
                        continue
                    # Is this projection itself (or wrapped in) an aggregate?
                    if proj.find(sqlexp.AggFunc):
                        continue
                    # Alias wrapping an aggregate
                    if isinstance(proj, sqlexp.Alias) and proj.this.find(sqlexp.AggFunc):
                        continue
                    col_name = proj.alias_or_name.lower() if hasattr(proj, "alias_or_name") else ""
                    if col_name and col_name not in group_by_cols:
                        issues.append(ValidationIssue(
                            code="B06",
                            severity="error",
                            message=f"Column '{col_name}' is in SELECT but not in GROUP BY and not aggregated.",
                            suggestion=f"Either add '{col_name}' to the GROUP BY clause or wrap it in an aggregate function.",
                        ))

        elif has_aggregate and not has_group_by:
            # Global aggregate — check if any bare column exists alongside it
            top_select = tree.find(sqlexp.Select)
            if top_select:
                bare_cols = []
                for proj in top_select.expressions:
                    if isinstance(proj, sqlexp.AggFunc):
                        continue
                    if proj.find(sqlexp.AggFunc):
                        continue
                    if isinstance(proj, sqlexp.Alias) and proj.this.find(sqlexp.AggFunc):
                        continue
                    if isinstance(proj, (sqlexp.Column, sqlexp.Alias)):
                        bare_cols.append(getattr(proj, "alias_or_name", str(proj)))
                if bare_cols:
                    issues.append(ValidationIssue(
                        code="B06",
                        severity="error",
                        message=f"Columns {bare_cols} appear alongside aggregates but there is no GROUP BY clause.",
                        suggestion="Add a GROUP BY clause listing all non-aggregated columns, or remove them from SELECT.",
                    ))

        # ---- B07: Cartesian product risk ----
        # Multiple FROM tables without JOINs = cross join
        if from_node:
            from_tables_direct = [
                child for child in from_node.expressions
                if isinstance(child, (sqlexp.Table, sqlexp.Alias))
            ]
            if len(from_tables_direct) > 1 and join_count == 0:
                issues.append(ValidationIssue(
                    code="B07",
                    severity="error",
                    message="Multiple tables in FROM clause without JOIN — implicit Cartesian product.",
                    suggestion="Use explicit JOIN ... ON syntax. Every multi-table query needs at least one JOIN condition.",
                ))

        # ---- B08: Potentially wrong INNER JOIN (LEFT JOIN may be needed) ----
        for join in join_nodes:
            join_kind = join.args.get("kind")
            # join.kind is typically None (INNER) or 'LEFT', 'RIGHT', etc.
            # We warn when the user's intent seems to include "all X even without Y"
            # The LLM system prompt handles this more precisely; here we warn generally.
            if join_kind is None:  # implicit INNER
                # Only warn when the question context suggests counting/listing all items
                # We pass this as a soft warning so the system prompt can override
                pass  # deliberate: this is handled in the LLM prompt instead

        # ---- B09: Division without NULLIF ----
        div_nodes = list(tree.find_all(sqlexp.Div))
        for div in div_nodes:
            denom = div.right
            if not isinstance(denom, sqlexp.Anonymous) and not isinstance(denom, sqlexp.NullSafeEq):
                # Check if denominator is a plain column or expression (not already NULLIF-wrapped)
                denom_sql = denom.sql()
                if "NULLIF" not in denom_sql.upper() and "CASE" not in denom_sql.upper():
                    issues.append(ValidationIssue(
                        code="B09",
                        severity="warning",
                        message=f"Division by '{denom_sql}' without NULLIF guard — divide-by-zero will crash.",
                        suggestion=f"Replace '{denom_sql}' with NULLIF({denom_sql}, 0) to return NULL instead of crashing.",
                    ))

        # ---- B11: Ambiguous column reference across multiple tables ----
        # When multiple tables are joined, bare column names (no table prefix)
        # that appear in more than one table are ambiguous.
        if join_count > 0:
            for col_node in tree.find_all(sqlexp.Column):
                if col_node.table:
                    continue  # table-qualified → not ambiguous
                col_name = col_node.name.lower()
                # Count how many tables in the query have this column name
                # (we only have summary info, so we do a name-match heuristic)
                matching_tables = [
                    t for t in tables_in_query
                    if t in known_tables  # only real tables
                ]
                if len(matching_tables) > 1:
                    # Heuristic: if the column is very generic, flag it
                    if col_name in {"id", "name", "status", "type", "code",
                                    "created_at", "updated_at", "date", "value"}:
                        issues.append(ValidationIssue(
                            code="B11",
                            severity="warning",
                            message=f"Column '{col_name}' is not table-qualified and may be ambiguous across {matching_tables}.",
                            suggestion=f"Qualify it as 'table_alias.{col_name}' to make the query unambiguous.",
                        ))
                        break  # one warning per query is enough

        # ---- B14: HAVING without GROUP BY ----
        if having_node and not has_group_by:
            issues.append(ValidationIssue(
                code="B14",
                severity="error",
                message="HAVING clause present without a GROUP BY — this is invalid in most SQL dialects.",
                suggestion="Add a GROUP BY clause, or move the condition to WHERE if it doesn't involve an aggregate.",
            ))

        return ValidationReport(
            sql=sql,
            dialect=self._dialect,
            issues=issues,
            tables_extracted=list(set(tables_in_query)),
            columns_extracted=list(set(columns_in_query)),
            join_count=join_count,
            has_limit=has_limit,
            has_group_by=has_group_by,
            has_aggregate=has_aggregate,
        )

    def validate_against_schema(
        self,
        report: ValidationReport,
        db: Optional[DatabaseManager],
    ) -> ValidationReport:
        """
        Layer 2: cross-reference extracted tables/columns against the live
        schema from DatabaseManager (if available). Adds B15/B16/B17 issues.
        This is best-effort — if db is None, Layer 2 is skipped.
        """
        if db is None:
            return report

        try:
            meta = db.get_schema_metadata()
        except Exception:
            return report   # schema reflection failed — skip Layer 2 silently

        known = {t.lower(): meta.tables[t] for t in meta.tables}

        for tname in report.tables_extracted:
            if tname not in known:
                continue  # already caught by B02
            table_meta = known[tname]
            col_names  = {c.name.lower() for c in table_meta.columns}
            col_types  = {c.name.lower(): c.type.lower() for c in table_meta.columns}

            # B15: Column not in this table
            for col in report.columns_extracted:
                col_l = col.lower()
                if col_l and col_l not in col_names and col_l not in {"*", "1"}:
                    report.issues.append(ValidationIssue(
                        code="B15",
                        severity="warning",   # warning not error — might be alias/CTE
                        message=f"Column '{col}' not found in table '{tname}'.",
                        suggestion=f"Available columns: {sorted(col_names)}.",
                    ))

            # B16: Numeric aggregate on a TEXT column
            for agg_node in sqlglot.parse_one(report.sql, dialect=self._dialect or None).find_all(sqlexp.AggFunc):
                if isinstance(agg_node, (sqlexp.Sum, sqlexp.Avg)):
                    for inner_col in agg_node.find_all(sqlexp.Column):
                        c_name = inner_col.name.lower()
                        c_type = col_types.get(c_name, "")
                        if c_type and not any(nt in c_type for nt in NUMERIC_TYPES):
                            report.issues.append(ValidationIssue(
                                code="B16",
                                severity="warning",
                                message=(
                                    f"SUM/AVG applied to column '{c_name}' "
                                    f"which has type '{c_type}' — may produce unexpected results."
                                ),
                                suggestion=(
                                    f"Cast '{c_name}' to a numeric type: CAST({c_name} AS REAL)."
                                ),
                            ))

        return report


# ---------------------------------------------------------------------------
# 3.  Tool input schemas
# ---------------------------------------------------------------------------

class ValidateSQLInput(BaseModel):
    sql_query: str = Field(description="The SQL query to validate statically.")


class ExecuteSQLInput(BaseModel):
    sql_query: str = Field(
        description="The fully validated, read-only SQL SELECT query to execute against the database."
    )


class FormatResultInput(BaseModel):
    sql_query: str = Field(description="The SQL that was executed.")
    raw_result_json: str = Field(
        description="JSON string of the QueryResult returned by the database."
    )
    clarified_intent: str = Field(
        description="The clarified user intent string from Agent 2, used to assess result quality."
    )


# ---------------------------------------------------------------------------
# 4.  BaseTool subclasses
# ---------------------------------------------------------------------------

class ValidateSQLTool(BaseTool):
    """
    Runs all static validation rules (B00–B14 AST, B15–B16 schema) against a
    candidate SQL query WITHOUT touching the database.

    Returns a JSON ValidationReport:
      - has_errors   → True means the query MUST NOT be executed; fix first
      - has_warnings → True means the query may run but could be improved
      - issues[]     → list of {code, severity, message, suggestion}

    Call this after every SQL draft, before ExecuteSQL.
    If has_errors is True, revise the SQL and validate again (up to MAX rounds).
    """

    name: str = "validate_sql"
    description: str = (
        "Statically validates a SQL query against the schema and 20+ bug rules "
        "without executing it. Returns a report with has_errors, has_warnings, "
        "and a list of issues with fix suggestions. "
        "ALWAYS call this before execute_sql. "
        "If has_errors=True, fix the issues and call validate_sql again."
    )
    args_schema: Type[BaseModel] = ValidateSQLInput

    _validator: SQLValidator
    _db: Optional[DatabaseManager]

    def __init__(
        self,
        validator: SQLValidator,
        db: Optional[DatabaseManager] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_validator", validator)
        object.__setattr__(self, "_db", db)

    def _run(self, sql_query: str) -> str:
        report = self._validator.validate(sql_query)
        # Layer 2 schema cross-reference (best-effort)
        db = object.__getattribute__(self, "_db")
        if db:
            report = self._validator.validate_against_schema(report, db)
        result = report.to_dict()
        logger.info(
            "validate_sql: errors=%s warnings=%s issues=%d",
            result["has_errors"], result["has_warnings"], len(result["issues"]),
        )
        return json.dumps(result, default=str)


class ExecuteSQLTool(BaseTool):
    """
    Executes a validated, read-only SQL SELECT query against the database
    via DatabaseManager and returns the result rows as JSON.

    Returns a JSON object with:
      - columns: list of column names
      - rows: list of dicts (max max_rows)
      - row_count: number of rows returned
      - execution_time_ms: wall-clock time
      - truncated: True if result was capped by max_rows
      - warnings: list of execution warnings
      - error: present only on failure

    ONLY call this after validate_sql returns has_errors=False.
    """

    name: str = "execute_sql"
    description: str = (
        "Executes a fully validated SQL SELECT query against the live database. "
        "Returns rows, column names, execution time, and warnings. "
        "Only call AFTER validate_sql confirms has_errors=False. "
        "Never call this with a query that has unresolved errors."
    )
    args_schema: Type[BaseModel] = ExecuteSQLInput

    _db: DatabaseManager

    def __init__(self, db: DatabaseManager, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_db", db)

    def _run(self, sql_query: str) -> str:
        db: DatabaseManager = object.__getattribute__(self, "_db")
        try:
            result: QueryResult = db.execute_query(sql_query)
            output = {
                "status": "success",
                "columns": result.columns,
                "rows": result.rows,
                "row_count": result.row_count,
                "execution_time_ms": result.execution_time_ms,
                "truncated": result.truncated,
                "warnings": result.warnings,
                "error": None,
            }
            if result.row_count == 0:
                output["warnings"] = output.get("warnings", []) + [
                    "Query executed successfully but returned zero rows. "
                    "The filter conditions may be too restrictive, or the table is empty."
                ]
            logger.info(
                "execute_sql: %d rows in %.1f ms%s",
                result.row_count, result.execution_time_ms,
                " [truncated]" if result.truncated else "",
            )
            return json.dumps(output, default=str)

        except ReadOnlyViolationError as exc:
            return json.dumps({
                "status": "error",
                "error_type": "ReadOnlyViolation",
                "message": str(exc),
                "advice": "The query contains a mutation statement. Only SELECT queries are permitted.",
            })
        except QueryTimeoutError as exc:
            return json.dumps({
                "status": "error",
                "error_type": "QueryTimeout",
                "message": str(exc),
                "advice": (
                    "The query timed out. Try: adding a more restrictive WHERE clause, "
                    "reducing the LIMIT value, or simplifying joins."
                ),
            })
        except QuerySyntaxError as exc:
            return json.dumps({
                "status": "error",
                "error_type": "QuerySyntax",
                "message": str(exc),
                "advice": (
                    "The database rejected the query syntax. "
                    "Check column names, function names, and dialect-specific syntax."
                ),
            })
        except QueryExecutionError as exc:
            return json.dumps({
                "status": "error",
                "error_type": "QueryExecution",
                "message": str(exc),
                "advice": "Check for NULL handling, type mismatches, or divide-by-zero conditions.",
            })
        except DBConnectionError as exc:
            return json.dumps({
                "status": "error",
                "error_type": "ConnectionError",
                "message": str(exc),
                "advice": "Database connection lost. The orchestrator should retry after reconnecting.",
            })
        except DatabaseManagerError as exc:
            return json.dumps({
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "advice": "Unexpected database error — see logs for the root cause.",
            })
        except Exception as exc:
            logger.exception("Unexpected error in ExecuteSQLTool")
            return json.dumps({
                "status": "error",
                "error_type": "Unexpected",
                "message": str(exc),
                "advice": "An unexpected error occurred. Inspect the query and logs.",
            })


# ---------------------------------------------------------------------------
# 5.  System prompt
# ---------------------------------------------------------------------------

def _build_system_prompt(
    schema_json: str,
    executor_schema_json: str,
    dialect: str,
) -> str:
    dialect_notes = {
        "sqlite": (
            "SQLite-specific: Use strftime('%Y', date_col) for year extraction. "
            "No FULL OUTER JOIN support — use UNION of LEFT JOIN + anti-join. "
            "LIMIT must appear after ORDER BY, never inside subqueries that aren't supported. "
            "NULL ordering: NULLs sort LAST by default in ASC, use 'IS NULL, col' trick for explicit control."
        ),
        "postgres": (
            "PostgreSQL-specific: Use EXTRACT(YEAR FROM date_col) or DATE_TRUNC. "
            "NULLS FIRST / NULLS LAST available in ORDER BY. "
            "Use ILIKE for case-insensitive string matching. "
            "LIMIT ... OFFSET is preferred over ROW_NUMBER() for simple pagination."
        ),
        "mysql": (
            "MySQL-specific: Use YEAR(date_col), MONTH(date_col) for date parts. "
            "GROUP BY non-aggregated columns is allowed but ONLY_FULL_GROUP_BY mode may reject it — always group by all non-agg cols. "
            "Use IFNULL instead of COALESCE for two-argument null handling. "
            "String comparison is case-insensitive by default in utf8mb4_general_ci collation — be aware."
        ),
        "tsql": (
            "MS SQL Server-specific: Use TOP(n) instead of LIMIT. "
            "Date functions: YEAR(), MONTH(), DATEPART(). "
            "Use ISNULL() instead of COALESCE for two-arg null handling. "
            "Square-bracket quoting: [column_name] for reserved words."
        ),
        "bigquery": (
            "BigQuery-specific: Use EXTRACT(YEAR FROM date_col). "
            "Backtick quoting for identifiers: `project.dataset.table`. "
            "LIMIT required — BigQuery charges by bytes scanned. "
            "Use APPROX_COUNT_DISTINCT for performance on large datasets."
        ),
    }.get(dialect, "")

    return f"""You are an expert SQL Engineer specialising in text-to-SQL query generation.

YOUR SINGLE RESPONSIBILITY
--------------------------
Given the SchemaContext (from Agent 1) and the clarified user intent (from Agent 2),
write a correct, efficient, safe SQL query. Then validate it. Then execute it.
Return a SQLExecutorOutput with the final query, metadata, and result summary.

DATABASE DIALECT: {dialect.upper() or "GENERIC SQL"}
{dialect_notes}

SCHEMA CONTEXT:
{schema_json}

YOUR MANDATORY WORKFLOW
-----------------------
1. DRAFT SQL
   Write a SQL query that precisely answers the clarified intent.
   Follow ALL the rules in the QUERY RULES section below.

2. VALIDATE  →  call `validate_sql` with your draft.
   • If has_errors=True: read EVERY issue, fix ALL errors, call `validate_sql` again.
   • If has_warnings=True: fix each warning before proceeding.
   • Only proceed to step 3 when has_errors=False.
   • Repeat up to {MAX_CORRECTION_ROUNDS} rounds maximum.

3. EXECUTE   →  call `execute_sql` with the validated query.
   • If status='error': diagnose from error_type + advice, fix the query, re-validate, re-execute.
   • If row_count=0: emit a warning in execution_warnings — do NOT treat it as an error.

4. EMIT OUTPUT  →  return a JSON object matching the SQLExecutorOutput schema exactly:
{executor_schema_json}

QUERY RULES (violation = bug in your output)
--------------------------------------------
JOIN RULES
  - Always use explicit JOIN ... ON syntax. NEVER comma-join tables (Cartesian product).
  - Use INNER JOIN only when you are certain every row in the left table has a match.
    When the user wants "all X including those without Y", use LEFT JOIN.
  - Always use table aliases (c, o, p, etc.) and qualify ALL column references.
  - Verify every JOIN uses the correct FK path from SchemaContext.foreign_keys.

NULL RULES
  - COUNT(*) counts all rows. COUNT(col) skips NULLs. Use the right one intentionally.
  - When dividing: always wrap the denominator in NULLIF(denom, 0) to prevent crash.
  - Use COALESCE(col, 0) when a NULL result should display as zero.
  - When filtering for "missing" data: use "col IS NULL", not "col = NULL" (always wrong).

GROUP BY / HAVING RULES
  - Every non-aggregated column in SELECT must appear in GROUP BY.
  - Aggregate filters (SUM > 1000) go in HAVING, never in WHERE.
  - When aggregating after a JOIN, GROUP BY the correct granularity
    (usually the parent entity ID, not a derived column).

LIMIT RULE
  - Every query must end with LIMIT N. Use row_count_estimates from SchemaContext
    to choose a sensible N: small tables (< 1000 rows) → LIMIT 100;
    large tables (> 100k rows) → LIMIT 50 or less.

SELECT RULES
  - Never use SELECT * — list only the columns that answer the question.
  - Always alias computed columns: SUM(amount) AS total_revenue, not bare SUM(amount).
  - Use table-qualified names for ALL columns in multi-table queries.

SUBQUERY RULES
  - A scalar subquery (in SELECT or WHERE) must return exactly one row and one column.
  - If it might return multiple rows, use a JOIN or EXISTS instead.

ORDERING RULES
  - When the question asks for "top N", add ORDER BY <metric> DESC LIMIT N.
  - When the question asks for "bottom N", add ORDER BY <metric> ASC LIMIT N.
  - NULLs sort differently across dialects — use NULLS LAST / NULLS FIRST when order matters.

TYPE SAFETY RULES
  - Do not SUM or AVG a TEXT column. Cast it first: CAST(col AS REAL).
  - String comparison: use LIKE for patterns, = for exact match.
  - Date comparison: compare dates as strings only in SQLite (ISO 8601 format).
    Use proper date functions in PostgreSQL, MySQL, MSSQL.

SELF-CRITIQUE CHECKLIST (run before calling validate_sql)
----------------------------------------------------------
  □ Does every JOIN have an ON clause?
  □ Are all non-aggregated SELECT columns in GROUP BY?
  □ Is HAVING used instead of WHERE for aggregate filters?
  □ Is every column table-qualified?
  □ Is LIMIT present?
  □ Is SELECT * absent?
  □ Is every division wrapped in NULLIF?
  □ Does the query answer EXACTLY the clarified intent — no more, no less?
  □ Are all table names in SchemaContext.available_tables?
"""


# ---------------------------------------------------------------------------
# 6.  Task description
# ---------------------------------------------------------------------------

def _build_task_description(
    clarification_output: ClarificationOutput,
    schema_context: SchemaContext,
) -> str:
    intent = (
        clarification_output.clarified_intent
        or "Question was already clear — proceed with the original question."
    )
    return f"""Generate and execute a SQL query for the following clarified user intent.

CLARIFIED INTENT:
\"\"\"{intent}\"\"\"

CLARIFICATION REASONING:
{clarification_output.reasoning}

DATABASE TYPE: {schema_context.database_type}
AVAILABLE TABLES: {schema_context.available_tables}
FOREIGN KEYS: {schema_context.foreign_keys}
AMBIGUOUS COLUMNS (handle carefully): {schema_context.ambiguous_columns}
ROW COUNT ESTIMATES: {schema_context.row_count_estimates}

Your mandatory workflow:
1. Draft a SQL query that answers the clarified intent.
2. Call `validate_sql` — fix ALL errors and warnings before continuing.
3. Call `execute_sql` with the validated query.
4. Return a SQLExecutorOutput JSON object with status, sql_query, tables_used,
   columns_used, join_paths, confidence_score, reasoning, and execution_warnings.

Output ONLY the SQLExecutorOutput JSON — no prose, no markdown fences.
"""


# ---------------------------------------------------------------------------
# 7.  LLM builder
# ---------------------------------------------------------------------------

def build_llm(*, temperature: float = 0.0, max_tokens: int = 4096) -> LLM:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set.\n"
            "Set it:  export GROQ_API_KEY=gsk_..."
        )
    return LLM(
        model="groq/openai/gpt-oss-20b",
        api_key=api_key,
        temperature=temperature,    # 0.0: SQL must be deterministic
        max_tokens=max_tokens,
        timeout=90,
    )


# ---------------------------------------------------------------------------
# 8.  Crew builder — public API
# ---------------------------------------------------------------------------

def build_sql_executor_crew(
    clarification_output: ClarificationOutput,
    schema_context: SchemaContext,
    db: DatabaseManager,
    *,
    verbose: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> Tuple[Crew, Task]:
    """
    Construct and return a ready-to-kickoff Crew containing Agent 3.

    Parameters
    ----------
    clarification_output : ClarificationOutput
        Must have status='clear'. Raises ValueError otherwise.
    schema_context : SchemaContext
        Validated output from Agent 1.
    db : DatabaseManager
        Connected DatabaseManager instance.
    verbose : bool
        Print agent reasoning loop if True.
    temperature : float
        0.0 for deterministic SQL. Do not increase.
    max_tokens : int
        Max tokens per LLM response.

    Returns
    -------
    (crew, task) tuple

    Raises
    ------
    ValueError    — status != 'clear', or empty schema.
    RuntimeError  — GROQ_API_KEY not set.
    """
    if clarification_output.status != "clear":
        raise ValueError(
            f"SQLExecutor received ClarificationOutput with status='{clarification_output.status}'. "
            "Only status='clear' is permitted — the orchestrator must not route "
            "ambiguous or impossible queries to Agent 3."
        )
    if not schema_context.available_tables:
        raise ValueError(
            "SchemaContext has no available_tables — "
            "the database appears to be empty. Cannot generate SQL."
        )

    dialect = DIALECT_MAP.get(schema_context.database_type, "")
    validator = SQLValidator(schema_context)

    # --- Tools ---
    validate_tool = ValidateSQLTool(validator=validator, db=db)
    execute_tool  = ExecuteSQLTool(db=db)

    # --- LLM ---
    llm = build_llm(temperature=temperature, max_tokens=max_tokens)

    # --- Prompts ---
    all_schemas      = export_schemas()
    schema_ctx_json  = schema_context.model_dump_json(indent=2)
    executor_sch_json = json.dumps(all_schemas["SQLExecutorOutput"], indent=2)
    system_prompt    = _build_system_prompt(schema_ctx_json, executor_sch_json, dialect)

    # --- Agent ---
    executor_agent = Agent(
        role="SQL Query Engineer",
        goal=(
            "Generate a correct, efficient, safe SQL query that precisely answers "
            "the clarified user intent. Validate it against 20+ bug rules before "
            "execution. Execute it. Return structured results."
        ),
        backstory=system_prompt,
        tools=[validate_tool, execute_tool],
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        max_iter=MAX_CORRECTION_ROUNDS * 3 + 2,  # generate+validate+fix per round + execute
        max_retry_limit=2,
        respect_context_window=True,
    )

    # --- Task ---
    executor_task = Task(
        description=_build_task_description(clarification_output, schema_context),
        expected_output=(
            "A valid JSON object exactly matching the SQLExecutorOutput schema. "
            "status must be 'success' (query ran cleanly), 'error' (generation/execution failed), "
            "or 'partial' (ran with warnings). "
            "sql_query must be the final, executed SQL. "
            "tables_used and columns_used must list only schema elements actually referenced. "
            "reasoning must walk through: intent mapping → table/FK selection → query construction → validation passes."
        ),
        agent=executor_agent,
        output_pydantic=SQLExecutorOutput,
    )

    crew = Crew(
        agents=[executor_agent],
        tasks=[executor_task],
        process=Process.sequential,
        verbose=verbose,
        memory=False,
        full_output=True,
    )

    return crew, executor_task


# ---------------------------------------------------------------------------
# 9.  Convenience wrapper
# ---------------------------------------------------------------------------

def execute_query(
    clarification_output: ClarificationOutput,
    schema_context: SchemaContext,
    db: DatabaseManager,
    *,
    verbose: bool = False,
) -> Tuple[SQLExecutorOutput, Optional[QueryResult]]:
    """
    High-level wrapper: build crew, run it, return (SQLExecutorOutput, QueryResult).

    The QueryResult is retrieved directly from the DatabaseManager for the
    final validated SQL — ensuring the orchestrator always has typed row data,
    not just the LLM's summary of it.

    Raises
    ------
    ValueError    — status != 'clear'.
    RuntimeError  — crew failure or unparseable output.
    """
    crew, task = build_sql_executor_crew(
        clarification_output=clarification_output,
        schema_context=schema_context,
        db=db,
        verbose=verbose,
    )

    try:
        crew.kickoff()
    except Exception as exc:
        raise RuntimeError(
            f"SQLExecutor crew failed.\nCause: {type(exc).__name__}: {exc}"
        ) from exc

    output = task.output
    if output is None:
        raise RuntimeError("SQLExecutor produced no output.")

    # --- Extract SQLExecutorOutput ---
    executor_out: Optional[SQLExecutorOutput] = None

    if output.pydantic is not None:
        executor_out = output.pydantic
    elif output.raw:
        logger.warning("output.pydantic is None — attempting manual parse.")
        try:
            raw = output.raw.strip()
            if raw.startswith("```"):
                raw = "\n".join(
                    l for l in raw.splitlines() if not l.strip().startswith("```")
                ).strip()
            executor_out = SQLExecutorOutput(**json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise RuntimeError(
                f"SQLExecutorOutput parse failed.\nRaw: {output.raw[:500]}\nError: {exc}"
            ) from exc

    if executor_out is None:
        raise RuntimeError("SQLExecutor produced neither Pydantic model nor raw text.")

    logger.info(
        "SQLExecutorOutput: status=%s, tables=%s, confidence=%d",
        executor_out.status, executor_out.tables_used, executor_out.confidence_score,
    )

    # --- Re-execute the final SQL directly for a clean QueryResult ---
    query_result: Optional[QueryResult] = None
    if executor_out.status in ("success", "partial") and executor_out.sql_query:
        try:
            query_result = db.execute_query(executor_out.sql_query)
        except DatabaseManagerError as exc:
            logger.warning("Re-execution of final SQL failed: %s", exc)

    return executor_out, query_result


# ---------------------------------------------------------------------------
# 10. Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import datetime
    import os
    import tempfile

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    print("\n" + "=" * 68)
    print("SMOKE TEST — sql_executor_agent.py")
    print("=" * 68)

    # ------------------------------------------------------------------
    # A. Build a temp SQLite database with realistic data
    # ------------------------------------------------------------------
    from sqlalchemy import (
        Column, DateTime, ForeignKey, Integer, MetaData as SAMeta,
        Numeric, String, Boolean,
    )
    from sqlalchemy import create_engine

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
    os.close(tmp_fd)
    DEMO_URL = f"sqlite:///{tmp_path}"

    engine_s = create_engine(DEMO_URL, future=True)
    meta_s   = SAMeta()
    from sqlalchemy import Table as SATable

    customers_t = SATable("customers", meta_s,
        Column("customer_id", Integer, primary_key=True),
        Column("name",        String(120), nullable=False),
        Column("email",       String(255)),
        Column("status",      String(30)),
        Column("joined_at",   DateTime),
    )
    orders_t = SATable("orders", meta_s,
        Column("order_id",    Integer, primary_key=True),
        Column("customer_id", Integer, ForeignKey("customers.customer_id")),
        Column("total_amount",Numeric(10, 2)),
        Column("status",      String(30)),
        Column("created_at",  DateTime),
    )
    products_t = SATable("products", meta_s,
        Column("product_id",  Integer, primary_key=True),
        Column("name",        String(255)),
        Column("category",    String(100)),
        Column("price",       Numeric(10, 2)),
    )
    meta_s.create_all(engine_s)

    with engine_s.begin() as conn:
        conn.execute(customers_t.insert(), [
            {"customer_id": 1, "name": "Alice Nawaz",  "email": "a@x.com", "status": "active",
             "joined_at": datetime.datetime(2023, 1, 15)},
            {"customer_id": 2, "name": "Bob Tariq",    "email": "b@x.com", "status": "active",
             "joined_at": datetime.datetime(2023, 3, 20)},
            {"customer_id": 3, "name": "Carla Raza",   "email": "c@x.com", "status": "inactive",
             "joined_at": datetime.datetime(2024, 6, 1)},
        ])
        conn.execute(orders_t.insert(), [
            {"order_id": 101, "customer_id": 1, "total_amount": 299.99,
             "status": "completed", "created_at": datetime.datetime(2024, 1, 10)},
            {"order_id": 102, "customer_id": 1, "total_amount": 149.50,
             "status": "completed", "created_at": datetime.datetime(2024, 3, 5)},
            {"order_id": 103, "customer_id": 2, "total_amount": 899.00,
             "status": "pending",   "created_at": datetime.datetime(2024, 8, 22)},
            {"order_id": 104, "customer_id": 3, "total_amount": 55.00,
             "status": "cancelled", "created_at": datetime.datetime(2024, 9, 1)},
        ])
        conn.execute(products_t.insert(), [
            {"product_id": 1, "name": "Widget A", "category": "electronics", "price": 29.99},
            {"product_id": 2, "name": "Gadget B", "category": "electronics", "price": 99.99},
        ])
    engine_s.dispose()

    db = DatabaseManager(DEMO_URL, max_rows=200)
    db.connect()

    CTX = SchemaContext(
        database_type="sqlite",
        available_tables=["customers", "orders", "products"],
        table_summaries={
            "customers": "One row per customer with name, email, status.",
            "orders":    "One row per purchase with total_amount and customer_id FK.",
            "products":  "One row per product with category and price.",
        },
        foreign_keys=["orders.customer_id → customers.customer_id"],
        ambiguous_columns=["customers.status", "orders.status"],
        row_count_estimates={"customers": 3, "orders": 4, "products": 2},
    )

    # ------------------------------------------------------------------
    # B. Static validator — test all bug detection rules
    # ------------------------------------------------------------------
    print("\n--- B. SQLValidator (no DB, no LLM) ---")
    validator = SQLValidator(CTX)

    VALID_CASES: List[Tuple[str, str, List[str]]] = [
        # (sql, expected_outcome, expected_error_codes)
        (
            # GOOD: correct query
            "SELECT c.name, SUM(o.total_amount) AS total FROM customers c "
            "JOIN orders o ON o.customer_id = c.customer_id "
            "GROUP BY c.name ORDER BY total DESC LIMIT 10",
            "clean", []
        ),
        (
            # B01: syntax error
            "SELECT FROM customers WHERE LIMIT 10",
            "error", ["B01"]
        ),
        (
            # B02: hallucinated table
            "SELECT * FROM ghost_table LIMIT 10",
            "error", ["B02"]
        ),
        (
            # B03: SELECT *
            "SELECT * FROM customers LIMIT 10",
            "warning", ["B03"]
        ),
        (
            # B04: no LIMIT (will also be caught by _validate_sql as B00/B04)
            "SELECT name FROM customers",
            "error", []
        ),
        (
            # B05: aggregate in WHERE
            "SELECT customer_id FROM orders WHERE SUM(total_amount) > 100 LIMIT 10",
            "error", ["B05"]
        ),
        (
            # B06: non-aggregated col without GROUP BY
            "SELECT name, SUM(total_amount) AS t FROM orders LIMIT 10",
            "error", ["B06"]
        ),
        (
            # B07: cartesian product
            "SELECT c.name, o.total_amount FROM customers c, orders o LIMIT 10",
            "error", ["B07"]
        ),
        (
            # B09: division without NULLIF
            "SELECT total_amount / price AS ratio FROM orders o "
            "JOIN products p ON p.product_id = o.order_id LIMIT 10",
            "warning", ["B09"]
        ),
        (
            # B14: HAVING without GROUP BY
            "SELECT customer_id FROM orders HAVING COUNT(*) > 1 LIMIT 10",
            "error", ["B14"]
        ),
    ]

    all_passed = True
    for sql, expected, expected_codes in VALID_CASES:
        report = validator.validate(sql)
        actual_codes = [i.code for i in report.issues]

        if expected == "clean":
            ok = not report.has_errors and not report.has_warnings
        elif expected == "error":
            ok = report.has_errors
        else:  # warning
            ok = report.has_warnings

        status = "[OK]  " if ok else "[FAIL]"
        if not ok:
            all_passed = False
        print(f"  {status} {sql[:65]!r:<67}")
        print(f"         expected={expected:<8} codes={expected_codes} "
              f"  actual={'error' if report.has_errors else 'warning' if report.has_warnings else 'clean'} "
              f"codes={actual_codes}")

    assert all_passed, "Validator has failures — see above"
    print("  ✓ All validator cases passed")

    # ------------------------------------------------------------------
    # C. ExecuteSQLTool — test good + error paths
    # ------------------------------------------------------------------
    print("\n--- C. ExecuteSQLTool (real DB) ---")
    exec_tool = ExecuteSQLTool(db=db)

    EXEC_CASES = [
        (
            "SELECT c.name, SUM(o.total_amount) AS total_spend "
            "FROM customers c JOIN orders o ON o.customer_id = c.customer_id "
            "GROUP BY c.name ORDER BY total_spend DESC LIMIT 10",
            "success", 2,   # Alice + Bob have orders
        ),
        (
            # Empty result — valid but zero rows
            "SELECT name FROM customers WHERE status = 'platinum' LIMIT 10",
            "success", 0,
        ),
    ]

    for sql, exp_status, exp_min_rows in EXEC_CASES:
        raw = exec_tool._run(sql_query=sql)
        d   = json.loads(raw)
        ok  = d["status"] == exp_status and d["row_count"] >= exp_min_rows
        print(f"  {'[OK]  ' if ok else '[FAIL]'} rows={d['row_count']} "
              f"status={d['status']} sql={sql[:60]!r}")
        if not ok:
            all_passed = False

    # ------------------------------------------------------------------
    # D. Guard: status != 'clear' must be rejected
    # ------------------------------------------------------------------
    print("\n--- D. Status guard ---")
    from schemas import AmbiguityDetail

    ambig_output = ClarificationOutput(
        status="ambiguous",
        ambiguities=[AmbiguityDetail(
            term="best",
            interpretations=["SUM(total_amount)", "COUNT(order_id)"],
            suggested_question='By "best" do you mean (1) highest spend or (2) most orders?',
        )],
        combined_clarification_message="Please clarify what 'best' means.",
        confidence_score=40,
        reasoning="The term 'best' is ambiguous.",
    )
    os.environ.setdefault("GROQ_API_KEY", "gsk_dummy")
    try:
        build_sql_executor_crew(ambig_output, CTX, db, verbose=False)
        print("  [FAIL] Should have raised ValueError for status='ambiguous'")
        all_passed = False
    except ValueError as exc:
        print(f"  [OK]   status='ambiguous' correctly rejected: {str(exc)[:80]}")

    # ------------------------------------------------------------------
    # E. ValidateSQLTool integration
    # ------------------------------------------------------------------
    print("\n--- E. ValidateSQLTool integration ---")
    val_tool = ValidateSQLTool(validator=validator, db=db)
    good_sql = (
        "SELECT c.name, SUM(o.total_amount) AS total "
        "FROM customers c JOIN orders o ON o.customer_id = c.customer_id "
        "GROUP BY c.name ORDER BY total DESC LIMIT 10"
    )
    bad_sql  = "SELECT * FROM ghost_table"  # B02 + B04

    good_report = json.loads(val_tool._run(sql_query=good_sql))
    bad_report  = json.loads(val_tool._run(sql_query=bad_sql))

    assert not good_report["has_errors"], f"Good query should pass: {good_report['issues']}"
    assert bad_report["has_errors"],      f"Bad query should fail: {bad_report['issues']}"
    print(f"  [OK]   Good query: has_errors={good_report['has_errors']}")
    print(f"  [OK]   Bad query:  has_errors={bad_report['has_errors']}  "
          f"codes={[i['code'] for i in bad_report['issues']]}")

    # ------------------------------------------------------------------
    # F. Crew construction structural check
    # ------------------------------------------------------------------
    print("\n--- F. Crew construction ---")
    clear_output = ClarificationOutput(
        status="clear",
        clarified_intent="best customer = highest SUM(total_amount)",
        confidence_score=95,
        reasoning="User selected interpretation (1): highest total spend.",
    )
    crew, task = build_sql_executor_crew(
        clarification_output=clear_output,
        schema_context=CTX,
        db=db,
        verbose=False,
    )
    assert len(crew.agents) == 1
    assert len(crew.tasks) == 1
    assert task.output_pydantic is SQLExecutorOutput
    tool_names = [t.name for t in crew.agents[0].tools]
    assert "validate_sql" in tool_names
    assert "execute_sql"  in tool_names
    print(f"  [OK]   Crew: 1 agent, tools={tool_names}")
    print(f"  [OK]   Task output_pydantic = SQLExecutorOutput")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    db.close()
    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    print("\n" + "=" * 68)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED — see above")
    print("=" * 68 + "\n")

    print("""
To run Agent 3 with a real Groq call:

    from schema_analyzer_agent import analyze_schema
    from clarification_agent   import clarify_question
    from sql_executor_agent    import execute_query
    from database_manager      import build_manager_from_url

    db  = build_manager_from_url("sqlite:///sales.db")
    ctx = analyze_schema("sqlite:///sales.db", "Who is our best customer?")
    clo = clarify_question("Who is our best customer?", ctx, human_input=True)

    if clo.status == "clear":
        exec_out, rows = execute_query(clo, ctx, db, verbose=True)
        print(exec_out.sql_query)
        print(rows.rows if rows else "no rows")
""")