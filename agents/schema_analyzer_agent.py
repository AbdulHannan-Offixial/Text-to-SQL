"""
schema_analyzer_agent.py — Agent 1: Schema Analyzer
=====================================================
Model  : openai/gpt-oss-20b  (via Groq API)
Role   : Read the database via DatabaseManager, reason about every table,
         column, relationship and ambiguity, then emit a validated SchemaContext
         that downstream agents (ClarificationAgent, SQLExecutor) consume.

Pipeline position
-----------------
  User question
      ↓
  [Agent 1 — SchemaAnalyzer]   ← THIS FILE
      ↓ SchemaContext (Pydantic-validated)
  [Agent 2 — ClarificationAgent]
      ↓ ClarificationOutput
  [Agent 3 — SQLExecutor]
      ↓ QueryResult

Architecture
------------
  Three BaseTool subclasses are registered on the agent, each wrapping one
  public method of DatabaseManager:

    1. FetchSchemaTool        → db.get_schema_metadata()
    2. SampleTableTool        → db.get_table_sample()
    3. ValidateTablesTool     → db.validate_tables_exist()

  The agent calls these tools autonomously during its reasoning loop.
  The Task is configured with output_pydantic=SchemaContext so CrewAI forces
  the LLM to emit a JSON payload that Pydantic validates on arrival — no
  string parsing, no hallucinated field names, no silent type coercion.

Contracts honoured
------------------
  • schemas.py  →  SchemaContext (the agent's sole output type)
  • database_manager.py  →  DatabaseManager, SchemaMetadata, TableMeta,
                             ColumnMeta, ForeignKeyMeta, all exception types

Usage (standalone / orchestrator)
----------------------------------
  from schema_analyzer_agent import build_schema_analyzer_crew

  crew, task = build_schema_analyzer_crew(
      database_url="sqlite:///sales.db",
      user_question="Who is our best customer?",
  )
  result = crew.kickoff()
  schema_ctx: SchemaContext = task.output.pydantic   # fully validated
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Type

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------------------------
# Local project imports — schemas.py and database_manager.py must be on
# PYTHONPATH or in the same directory as this file.
# ---------------------------------------------------------------------------
from schemas import SchemaContext, export_schemas
from database.database_manager import (
    DatabaseManager,
    DatabaseManagerError,
    ConnectionError as DBConnectionError,
    SchemaReflectionError,
    SchemaMetadata,
    TableMeta,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("text_to_sql.schema_analyzer_agent")

# ---------------------------------------------------------------------------
# 1.  LLM Configuration
# ---------------------------------------------------------------------------
# CrewAI's LLM wrapper talks to Groq via LiteLLM under the hood.
# The prefix "groq/" tells LiteLLM which provider adapter to use.
# The model string after the prefix is exactly what Groq expects.
#
# Key settings for a schema-analysis agent:
#   temperature=0      → deterministic; schema facts are not creative
#   max_tokens=4096    → schema DDL + sample rows can be large
#   timeout=90         → Groq is fast but large schemas need headroom
# ---------------------------------------------------------------------------

def build_llm(*, temperature: float = 0.0, max_tokens: int = 4096) -> LLM:
    """
    Construct the LLM object.

    Reads GROQ_API_KEY from the environment.  Raises a clear RuntimeError
    rather than letting CrewAI surface a cryptic auth failure later.
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set.\n"
            "Set it before running:  export GROQ_API_KEY=gsk_..."
        )

    return LLM(
        model="groq/openai/gpt-oss-20b",   # LiteLLM prefix + Groq model ID
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=90,
    )


# ---------------------------------------------------------------------------
# 2.  Tool input schemas (Pydantic v2 — one per tool)
# ---------------------------------------------------------------------------

class FetchSchemaInput(BaseModel):
    """Input for FetchSchemaTool — no parameters needed; fetches all tables."""
    force_refresh: bool = Field(
        default=False,
        description="Set True to bypass the schema cache and re-read from the database.",
    )


class SampleTableInput(BaseModel):
    """Input for SampleTableTool — specifies which table to sample."""
    table_name: str = Field(
        description="Exact table name to sample. Must exist in the reflected schema.",
    )
    n_rows: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of sample rows to return (1–10).",
    )


class ValidateTablesInput(BaseModel):
    """Input for ValidateTablesTool — checks a list of table names for existence."""
    table_names: List[str] = Field(
        description="List of table names to verify against the reflected schema.",
    )


# ---------------------------------------------------------------------------
# 3.  BaseTool subclasses — thin wrappers around DatabaseManager
# ---------------------------------------------------------------------------

class FetchSchemaTool(BaseTool):
    """
    Reads the full database schema via SQLAlchemy reflection.

    Returns a structured JSON object with:
      - database_type: dialect string
      - tables: per-table column list, PK, FKs, indexes, row_count
      - foreign_key_strings: human-readable join paths
      - ambiguous_column_names: columns that are vague without domain context
      - raw_ddl: reconstructed CREATE TABLE statement per table
      - sample_rows: up to 3 sample rows per table for type inference
    """

    name: str = "fetch_database_schema"
    description: str = (
        "Reflects the FULL database schema — tables, columns, primary keys, "
        "foreign keys, indexes, row counts, and reconstructed DDL. "
        "Call this FIRST before any other tool. "
        "Returns a JSON object with everything needed to understand the database."
    )
    args_schema: Type[BaseModel] = FetchSchemaInput

    # DatabaseManager is injected at construction time and stored as a
    # plain attribute so CrewAI's Pydantic model validation doesn't try
    # to serialise it.
    _db: DatabaseManager

    def __init__(self, db: DatabaseManager, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Store as a private attribute to avoid Pydantic field validation
        object.__setattr__(self, "_db", db)

    def _run(self, force_refresh: bool = False) -> str:
        """
        Executes schema reflection and returns a JSON string.

        Error handling:
          - DBConnectionError     → tells agent the DB is unreachable
          - SchemaReflectionError → tells agent what went wrong with reflection
          - Empty schema          → warns agent; partial results still returned
          - Any other exception   → wrapped with full traceback context
        """
        try:
            meta: SchemaMetadata = self._db.get_schema_metadata(
                force_refresh=force_refresh
            )
        except DBConnectionError as exc:
            return json.dumps({
                "error": "connection_failed",
                "message": str(exc),
                "advice": (
                    "The database is unreachable. Verify the connection URL, "
                    "network access, and credentials before retrying."
                ),
            })
        except SchemaReflectionError as exc:
            return json.dumps({
                "error": "reflection_failed",
                "message": str(exc),
                "advice": (
                    "Schema reflection failed. The user may lack SELECT privilege "
                    "on information_schema / sqlite_master, or the database is empty."
                ),
            })
        except Exception as exc:
            logger.exception("Unexpected error in FetchSchemaTool")
            return json.dumps({
                "error": "unexpected_error",
                "message": str(exc),
                "type": type(exc).__name__,
            })

        if not meta.tables:
            return json.dumps({
                "warning": "empty_schema",
                "message": (
                    "The database returned zero tables. "
                    "It may be empty or the user may lack visibility."
                ),
                "database_type": meta.database_type,
                "tables": {},
                "foreign_key_strings": [],
                "ambiguous_column_names": [],
            })

        # Serialise SchemaMetadata → dict (dataclasses are not JSON-serialisable natively)
        tables_dict: Dict[str, Any] = {}
        for tname, tmeta in meta.tables.items():
            tables_dict[tname] = _table_meta_to_dict(tmeta)

        # Collect sample rows for each table so the agent can reason about
        # concrete values (e.g. "status" contains "active"|"inactive" not ints)
        samples: Dict[str, Any] = {}
        for tname in meta.tables:
            try:
                sample = self._db.get_table_sample(tname, n_rows=3)
                samples[tname] = sample.get("rows", [])
            except Exception as exc:
                samples[tname] = f"unavailable: {exc}"

        return json.dumps({
            "database_type": meta.database_type,
            "table_count": len(meta.tables),
            "tables": tables_dict,
            "foreign_key_strings": meta.foreign_key_strings,
            "ambiguous_column_names": meta.ambiguous_column_names,
            "raw_ddl": meta.raw_ddl,
            "sample_rows": samples,
        }, default=str)


class SampleTableTool(BaseTool):
    """
    Fetches a small number of real rows from a named table.

    Use this after FetchSchemaTool to inspect concrete column values — critical
    for columns like "status", "type", or "category" where the name alone
    does not reveal the domain vocabulary (e.g. status ∈ {'open','closed','pending'}).
    """

    name: str = "sample_table_rows"
    description: str = (
        "Returns a small number of real rows from a specific table. "
        "Use this to understand what values live in ambiguous columns "
        "(e.g. status, type, category) so you can write accurate table_summaries. "
        "Input: {table_name: str, n_rows: int (1-10)}."
    )
    args_schema: Type[BaseModel] = SampleTableInput

    _db: DatabaseManager

    def __init__(self, db: DatabaseManager, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_db", db)

    def _run(self, table_name: str, n_rows: int = 3) -> str:
        try:
            result = self._db.get_table_sample(table_name, n_rows=n_rows)
            return json.dumps(result, default=str)
        except SchemaReflectionError as exc:
            return json.dumps({
                "error": "table_not_found",
                "table_name": table_name,
                "message": str(exc),
            })
        except DatabaseManagerError as exc:
            return json.dumps({
                "error": "sample_failed",
                "table_name": table_name,
                "message": str(exc),
            })
        except Exception as exc:
            logger.exception("Unexpected error in SampleTableTool for table '%s'", table_name)
            return json.dumps({
                "error": "unexpected_error",
                "table_name": table_name,
                "message": str(exc),
            })


class ValidateTablesTool(BaseTool):
    """
    Verifies that a list of table names actually exist in the reflected schema.

    Use this as a self-check before finalising table_summaries or available_tables
    in the SchemaContext output. Returns any names not found so the agent can
    remove hallucinated tables from its output.
    """

    name: str = "validate_table_names"
    description: str = (
        "Checks a list of table names against the reflected schema. "
        "Returns the subset that do NOT exist. "
        "Use this before finalising your output to ensure available_tables "
        "contains only real table names and no hallucinations. "
        "Input: {table_names: list[str]}."
    )
    args_schema: Type[BaseModel] = ValidateTablesInput

    _db: DatabaseManager

    def __init__(self, db: DatabaseManager, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_db", db)

    def _run(self, table_names: List[str]) -> str:
        try:
            missing = self._db.validate_tables_exist(table_names)
            all_known = self._db.get_table_names()
            return json.dumps({
                "requested": table_names,
                "missing_tables": missing,
                "all_known_tables": all_known,
                "verdict": (
                    "all_valid" if not missing
                    else f"{len(missing)} table(s) not found in schema"
                ),
            })
        except DatabaseManagerError as exc:
            return json.dumps({
                "error": "validation_failed",
                "message": str(exc),
            })
        except Exception as exc:
            logger.exception("Unexpected error in ValidateTablesTool")
            return json.dumps({
                "error": "unexpected_error",
                "message": str(exc),
            })


# ---------------------------------------------------------------------------
# 4.  System prompt builder
# ---------------------------------------------------------------------------

def _build_system_prompt(schema_json_schema: str) -> str:
    """
    Construct the agent's backstory / system context.

    Embeds:
      - the SchemaContext JSON schema verbatim (so the LLM knows the exact
        output contract it must satisfy)
      - step-by-step reasoning instructions
      - explicit rules for ambiguity detection
      - SQL dialect awareness
    """
    return f"""You are an expert Database Schema Analyst with deep knowledge of
relational database design, SQL dialects (SQLite, PostgreSQL, MySQL, MS SQL,
BigQuery), and data modelling patterns.

YOUR SINGLE RESPONSIBILITY
--------------------------
Analyse the database schema provided by your tools and produce a validated
SchemaContext JSON object that downstream agents will use to generate
precise, unambiguous SQL queries.

MANDATORY REASONING STEPS
--------------------------
1. FETCH SCHEMA
   Call `fetch_database_schema` first. Read every table, column, type,
   primary key, foreign key, and index in the result.

2. SAMPLE AMBIGUOUS COLUMNS
   For any column whose name appears in the ambiguous_column_names list
   (e.g. "status", "type", "rank", "score"), call `sample_table_rows`
   to see real values. You cannot write a useful table_summary without
   knowing what "status" actually contains.

3. BUILD TABLE SUMMARIES
   Write one concrete, business-oriented sentence per table. Bad example:
   "This table stores data." Good example: "Each row is a completed
   customer order with a foreign key to customers and a total_amount in USD."

4. IDENTIFY FK JOIN PATHS
   List every foreign key as "child_table.child_col → parent_table.parent_col".
   This is the join map Agent 2 and Agent 3 will use.

5. FLAG AMBIGUOUS COLUMNS
   Add to ambiguous_columns any column name that could mean multiple things
   without domain context. Examples: "score", "rank", "best", "top", "flag",
   "type", "status", "value", "amount". If you sampled the column and its
   values are self-explanatory, you may exclude it.

6. SELF-VALIDATE
   Call `validate_table_names` with your proposed available_tables list.
   If any table comes back as missing, remove it before emitting output.

7. EMIT SCHEMA CONTEXT
   Output ONLY a valid JSON object matching this schema exactly:

{schema_json_schema}

RULES
-----
- available_tables MUST list ONLY tables that actually exist in the database.
  Never invent table names.
- table_summaries MUST have an entry for EVERY table in available_tables.
  A missing table_summary will cause a Pydantic ValidationError downstream.
- foreign_keys MUST use the format "child_table.child_col → parent_table.parent_col".
- row_count_estimates: use the values from fetch_database_schema. Use 0 if
  a table's row count was unavailable.
- ambiguous_columns: use table-qualified names ("orders.status", not just "status").
- confidence_score is NOT part of SchemaContext — do not invent extra fields.
- database_type MUST be one of: sqlite, postgresql, mysql, mssql, bigquery, other.
- Do NOT output markdown fences, prose, or any text outside the JSON object.
- If the database is empty (zero tables), set available_tables to an empty list
  and explain in the first table_summary entry why (use key "_empty_db").
  Note: SchemaContext requires min_length=1 on available_tables, so if the DB
  is genuinely empty, set database_type correctly and available_tables to
  ["_no_tables_found"] with a matching summary entry explaining the situation.

DIALECT-SPECIFIC NOTES
-----------------------
- SQLite  : no native BOOLEAN or DATETIME type — look for INTEGER/TEXT substitutes.
- MySQL   : TINYINT(1) is often used as BOOLEAN.
- MS SQL  : BIT = BOOLEAN, NVARCHAR = Unicode string.
- PostgreSQL : JSONB, ARRAY, and UUID types are common and worth noting.
- BigQuery   : STRUCT and REPEATED fields indicate nested / array data.

You are methodical, precise, and never guess. If a tool returns an error,
report what failed in your reasoning and emit the best partial SchemaContext
you can construct from available information.
"""


# ---------------------------------------------------------------------------
# 5.  Task description builder
# ---------------------------------------------------------------------------

def _build_task_description(user_question: str, database_url: str) -> str:
    """
    Build the task description that the agent sees at runtime.
    Embeds the user question so the agent can tailor summaries and
    ambiguity detection to what the downstream query will actually need.
    """
    # Redact credentials from the URL for safe logging in verbose mode
    safe_url = _redact_url(database_url)

    return f"""Analyse the database at: {safe_url}

The end-user will ask: "{user_question}"

Your task:
1. Call `fetch_database_schema` to read the full schema.
2. For any column flagged as ambiguous (especially columns relevant to the
   user's question), call `sample_table_rows` to inspect real values.
3. Pay special attention to columns or tables that could answer the question
   "{user_question}" — their semantics must be crystal-clear in table_summaries.
4. Call `validate_table_names` with your proposed available_tables to confirm
   none are hallucinated.
5. Return a single JSON object that strictly conforms to the SchemaContext schema.

The user's question may contain vague terms. Flag any schema columns that
could give rise to ambiguity (e.g. if the question says "best" and the schema
has both a revenue column and an orders_count column, both are candidates and
both must appear in ambiguous_columns).

Expected output: a valid SchemaContext JSON object — nothing else.
"""


# ---------------------------------------------------------------------------
# 6.  Crew builder — the public API
# ---------------------------------------------------------------------------

def build_schema_analyzer_crew(
    database_url: str,
    user_question: str,
    *,
    verbose: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    db_max_rows: int = 500,
    db_query_timeout: int = 30,
) -> tuple[Crew, Task]:
    """
    Construct and return a ready-to-kickoff Crew containing Agent 1.

    Parameters
    ----------
    database_url : str
        Any SQLAlchemy-compatible connection URL.
    user_question : str
        The raw natural language question from the user.
        Used to focus ambiguity detection on relevant columns.
    verbose : bool
        If True, CrewAI prints the agent's thought-action-observation loop.
    temperature : float
        LLM temperature. 0.0 = fully deterministic (recommended for schema tasks).
    max_tokens : int
        Maximum tokens for a single LLM response.
    db_max_rows : int
        Row cap passed to DatabaseManager.execute_query().
    db_query_timeout : int
        Per-statement timeout in seconds passed to DatabaseManager.

    Returns
    -------
    (crew, task) tuple
        Call crew.kickoff() to run.
        Access task.output.pydantic for the validated SchemaContext.

    Raises
    ------
    RuntimeError
        If GROQ_API_KEY is not set.
    DBConnectionError
        If the database_url is unreachable at construction time.
    ValueError
        If user_question is empty.
    """
    if not user_question or not user_question.strip():
        raise ValueError("user_question must be a non-empty string.")

    # --- 6a. Validate connectivity immediately (fail fast) ---
    logger.info("Initialising DatabaseManager for: %s", _redact_url(database_url))
    db = DatabaseManager(
        database_url,
        max_rows=db_max_rows,
        query_timeout=db_query_timeout,
        echo=False,
    )
    try:
        db.connect()
    except DBConnectionError as exc:
        raise DBConnectionError(
            f"Schema Analyzer cannot start — database unreachable.\n{exc}"
        ) from exc

    # --- 6b. Build tools (share the same DatabaseManager instance) ---
    fetch_tool     = FetchSchemaTool(db=db)
    sample_tool    = SampleTableTool(db=db)
    validate_tool  = ValidateTablesTool(db=db)

    # --- 6c. Build LLM ---
    llm = build_llm(temperature=temperature, max_tokens=max_tokens)

    # --- 6d. Embed SchemaContext JSON schema in the system prompt ---
    all_schemas   = export_schemas()
    schema_ctx_js = json.dumps(all_schemas["SchemaContext"], indent=2)
    system_prompt = _build_system_prompt(schema_ctx_js)

    # --- 6e. Define the Agent ---
    schema_agent = Agent(
        role="Database Schema Analyst",
        goal=(
            "Reflect and deeply understand the target database schema so that "
            "downstream agents can generate precise, unambiguous SQL queries "
            "without guessing at table names, column semantics, or join paths."
        ),
        backstory=system_prompt,
        tools=[fetch_tool, sample_tool, validate_tool],
        llm=llm,
        verbose=verbose,
        allow_delegation=False,     # Agent 1 never delegates — it owns schema analysis
        max_iter=8,                 # sufficient for: fetch → sample ambiguous cols → validate → emit
        max_retry_limit=2,          # retry on malformed JSON output before raising
        respect_context_window=True,
    )

    # --- 6f. Define the Task ---
    schema_task = Task(
        description=_build_task_description(user_question, database_url),
        expected_output=(
            "A valid JSON object that exactly matches the SchemaContext Pydantic schema. "
            "Must include: database_type, available_tables (non-empty list), "
            "table_summaries (one entry per table), foreign_keys (list of FK strings), "
            "ambiguous_columns (table-qualified), and row_count_estimates."
        ),
        agent=schema_agent,
        output_pydantic=SchemaContext,  # CrewAI validates the LLM's JSON against this model
    )

    # --- 6g. Assemble the Crew ---
    crew = Crew(
        agents=[schema_agent],
        tasks=[schema_task],
        process=Process.sequential,     # single-agent pipeline; no parallelism needed
        verbose=verbose,
        memory=False,                   # stateless per-question; no cross-question bleed
        full_output=True,
    )

    return crew, schema_task


# ---------------------------------------------------------------------------
# 7.  Low-level utility: run the agent and return a clean SchemaContext
# ---------------------------------------------------------------------------

def analyze_schema(
    database_url: str,
    user_question: str,
    *,
    verbose: bool = False,
) -> SchemaContext:
    """
    High-level convenience wrapper.

    Builds the crew, runs it, validates the output, and returns a
    SchemaContext ready for the next agent in the pipeline.

    Raises
    ------
    RuntimeError
        If GROQ_API_KEY is not set, or if the crew fails to produce a
        valid SchemaContext after max_iter attempts.
    DBConnectionError
        If the database is unreachable.
    ValidationError
        If the LLM's output does not conform to SchemaContext even after
        CrewAI's built-in retry mechanism exhausts max_retry_limit.
    """
    crew, task = build_schema_analyzer_crew(
        database_url=database_url,
        user_question=user_question,
        verbose=verbose,
    )

    try:
        crew.kickoff()
    except Exception as exc:
        raise RuntimeError(
            f"Schema Analyzer crew failed to complete.\n"
            f"Cause: {type(exc).__name__}: {exc}"
        ) from exc

    # CrewAI puts the validated Pydantic object in task.output.pydantic
    # when output_pydantic is set and the LLM's JSON is valid.
    output = task.output

    if output is None:
        raise RuntimeError(
            "Schema Analyzer task produced no output. "
            "Check verbose logs for the agent's reasoning trace."
        )

    if output.pydantic is not None:
        ctx: SchemaContext = output.pydantic
        logger.info(
            "SchemaContext produced: %d tables, %d FKs, %d ambiguous columns.",
            len(ctx.available_tables),
            len(ctx.foreign_keys),
            len(ctx.ambiguous_columns),
        )
        return ctx

    # Fallback: if output_pydantic validation failed but raw output exists,
    # attempt manual parse so we can surface a useful ValidationError.
    if output.raw:
        logger.warning(
            "CrewAI did not populate output.pydantic — attempting manual parse."
        )
        try:
            raw_text = output.raw.strip()
            # Strip markdown fences if the LLM wrapped the JSON
            if raw_text.startswith("```"):
                raw_text = "\n".join(
                    line for line in raw_text.splitlines()
                    if not line.strip().startswith("```")
                ).strip()
            data = json.loads(raw_text)
            return SchemaContext(**data)
        except (json.JSONDecodeError, ValidationError) as parse_exc:
            raise RuntimeError(
                f"Schema Analyzer output could not be parsed into SchemaContext.\n"
                f"Raw output (first 500 chars): {output.raw[:500]}\n"
                f"Parse error: {parse_exc}"
            ) from parse_exc

    raise RuntimeError(
        "Schema Analyzer produced neither a Pydantic model nor raw text output."
    )


# ---------------------------------------------------------------------------
# 8.  Helpers
# ---------------------------------------------------------------------------

def _table_meta_to_dict(tmeta: TableMeta) -> Dict[str, Any]:
    """Serialise a TableMeta dataclass to a plain dict for JSON output."""
    return {
        "columns": [
            {
                "name": col.name,
                "type": col.type,
                "nullable": col.nullable,
                "primary_key": col.primary_key,
                "default": col.default,
                "comment": col.comment,
            }
            for col in tmeta.columns
        ],
        "primary_keys": tmeta.primary_keys,
        "foreign_keys": [
            {
                "constrained_columns": fk.constrained_columns,
                "referred_table": fk.referred_table,
                "referred_columns": fk.referred_columns,
                "human_readable": str(fk),
            }
            for fk in tmeta.foreign_keys
        ],
        "indexes": [
            {
                "name": idx.name,
                "columns": idx.columns,
                "unique": idx.unique,
            }
            for idx in tmeta.indexes
        ],
        "row_count_estimate": tmeta.row_count_estimate,
        "comment": tmeta.comment,
    }


def _redact_url(url: str) -> str:
    """
    Replace user:password in a database URL with ***:*** for safe logging.

    Examples:
      postgresql+psycopg2://admin:s3cr3t@host/db  →  postgresql+psycopg2://***:***@host/db
      sqlite:///local.db                           →  sqlite:///local.db  (unchanged)
    """
    import re
    return re.sub(
        r"(://)[^:@/]+:[^@/]+(@)",
        r"\1***:***\2",
        url,
    )


# ---------------------------------------------------------------------------
# 9.  CLI / smoke-test  (run: python schema_analyzer_agent.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import datetime
    import os
    import tempfile
    import textwrap

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    print("\n" + "=" * 68)
    print("SMOKE TEST — schema_analyzer_agent.py")
    print("=" * 68)

    # ------------------------------------------------------------------
    # A.  Verify all imports and tool instantiation work without a real
    #     Groq API key (we mock the LLM call and only test plumbing).
    # ------------------------------------------------------------------

    # Build a temp SQLite database with realistic schema
    from sqlalchemy import (
        Column, DateTime, ForeignKey, Integer, MetaData as SAMeta,
        Numeric, String, Table as SATable, Boolean,
    )
    from sqlalchemy import create_engine

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
    os.close(tmp_fd)
    DEMO_URL = f"sqlite:///{tmp_path}"

    engine_s = create_engine(DEMO_URL, future=True)
    meta_s   = SAMeta()
    SATable("customers", meta_s,
        Column("customer_id", Integer, primary_key=True),
        Column("name", String(120), nullable=False),
        Column("email", String(255)),
        Column("status", String(30)),          # ambiguous
        Column("joined_at", DateTime),
    )
    SATable("orders", meta_s,
        Column("order_id", Integer, primary_key=True),
        Column("customer_id", Integer, ForeignKey("customers.customer_id")),
        Column("total_amount", Numeric(10, 2)),
        Column("status", String(30)),          # ambiguous
        Column("is_refunded", Boolean),
        Column("created_at", DateTime),
    )
    SATable("products", meta_s,
        Column("product_id", Integer, primary_key=True),
        Column("name", String(255)),           # ambiguous
        Column("category", String(100)),       # ambiguous
        Column("score", Numeric(5, 2)),        # ambiguous
        Column("price", Numeric(10, 2)),
    )
    SATable("order_items", meta_s,
        Column("item_id", Integer, primary_key=True),
        Column("order_id", Integer, ForeignKey("orders.order_id")),
        Column("product_id", Integer, ForeignKey("products.product_id")),
        Column("quantity", Integer),
        Column("unit_price", Numeric(10, 2)),
    )
    meta_s.create_all(engine_s)
    engine_s.dispose()

    print(f"\n[OK] Temp database created: {tmp_path}")

    # ------------------------------------------------------------------
    # B.  Test each tool independently (no LLM needed)
    # ------------------------------------------------------------------
    print("\n--- Testing tools independently (no LLM) ---")

    db_test = DatabaseManager(DEMO_URL, max_rows=50)
    db_test.connect()

    # FetchSchemaTool
    fetch_tool   = FetchSchemaTool(db=db_test)
    sample_tool  = SampleTableTool(db=db_test)
    validate_tool = ValidateTablesTool(db=db_test)

    result_raw = fetch_tool._run(force_refresh=False)
    schema_dict = json.loads(result_raw)
    assert "error" not in schema_dict, f"FetchSchemaTool error: {schema_dict}"
    assert schema_dict["table_count"] == 4, \
        f"Expected 4 tables, got {schema_dict['table_count']}"
    print(f"[OK] FetchSchemaTool: {schema_dict['table_count']} tables, "
          f"dialect={schema_dict['database_type']}")
    print(f"     FK strings: {schema_dict['foreign_key_strings']}")
    print(f"     Ambiguous cols: {schema_dict['ambiguous_column_names']}")

    # SampleTableTool
    sample_raw = sample_tool._run(table_name="customers", n_rows=3)
    sample_dict = json.loads(sample_raw)
    assert "error" not in sample_dict, f"SampleTableTool error: {sample_dict}"
    print(f"[OK] SampleTableTool (customers): {sample_dict['row_count']} rows")

    # SampleTableTool — ghost table
    ghost_raw = sample_tool._run(table_name="ghost_table", n_rows=2)
    ghost_dict = json.loads(ghost_raw)
    assert ghost_dict.get("error") == "table_not_found", \
        f"Expected table_not_found, got: {ghost_dict}"
    print(f"[OK] SampleTableTool (ghost table): error='{ghost_dict['error']}'")

    # ValidateTablesTool — all real
    val_raw = validate_tool._run(table_names=["customers", "orders", "products", "order_items"])
    val_dict = json.loads(val_raw)
    assert val_dict["missing_tables"] == [], \
        f"Expected no missing, got: {val_dict['missing_tables']}"
    print(f"[OK] ValidateTablesTool (all real): verdict='{val_dict['verdict']}'")

    # ValidateTablesTool — with a hallucinated table
    val_raw2 = validate_tool._run(table_names=["customers", "ghost_table", "phantom"])
    val_dict2 = json.loads(val_raw2)
    assert set(val_dict2["missing_tables"]) == {"ghost_table", "phantom"}, \
        f"Expected ghost_table + phantom missing, got: {val_dict2['missing_tables']}"
    print(f"[OK] ValidateTablesTool (hallucinated): missing={val_dict2['missing_tables']}")

    db_test.close()

    # ------------------------------------------------------------------
    # C.  Test crew construction (no LLM call — just structural check)
    # ------------------------------------------------------------------
    print("\n--- Testing crew construction ---")

    # Temporarily set a dummy API key so build_llm() doesn't raise
    os.environ.setdefault("GROQ_API_KEY", "gsk_test_dummy_key_for_smoke_test")

    try:
        crew, task = build_schema_analyzer_crew(
            database_url=DEMO_URL,
            user_question="Who is our best customer?",
            verbose=False,
        )
        print(f"[OK] Crew constructed: {len(crew.agents)} agent(s), "
              f"{len(crew.tasks)} task(s)")
        print(f"     Agent role: '{crew.agents[0].role}'")
        print(f"     Tools: {[t.name for t in crew.agents[0].tools]}")
        assert len(crew.agents[0].tools) == 3, "Expected exactly 3 tools"
        assert task.output_pydantic is SchemaContext, \
            "Task output_pydantic must be SchemaContext"
        print("[OK] Task output_pydantic = SchemaContext")
    except Exception as exc:
        print(f"[FAIL] Crew construction: {exc}")
        raise

    # ------------------------------------------------------------------
    # D.  Test URL redaction
    # ------------------------------------------------------------------
    print("\n--- Testing URL redaction ---")
    cases = [
        ("postgresql+psycopg2://admin:s3cr3t@localhost:5432/mydb",
         "postgresql+psycopg2://***:***@localhost:5432/mydb"),
        ("mysql+pymysql://root:password@host/db",
         "mysql+pymysql://***:***@host/db"),
        ("sqlite:///local.db", "sqlite:///local.db"),
    ]
    for raw_url, expected in cases:
        redacted = _redact_url(raw_url)
        assert redacted == expected, f"Expected '{expected}', got '{redacted}'"
        print(f"[OK] {raw_url[:40]!r:40s} → {redacted}")

    # ------------------------------------------------------------------
    # E.  SchemaContext direct construction from tool output (no LLM)
    # ------------------------------------------------------------------
    print("\n--- Testing SchemaContext construction from tool output ---")
    db_verify = DatabaseManager(DEMO_URL, max_rows=50)
    db_verify.connect()
    meta = db_verify.get_schema_metadata()

    # Simulate what the LLM would produce (ground-truth assembly)
    ctx = SchemaContext(
        database_type=meta.database_type,
        available_tables=list(meta.tables.keys()),
        table_summaries={
            t: f"Table '{t}': " + ", ".join(
                c.name for c in meta.tables[t].columns[:3]
            ) + "…"
            for t in meta.tables
        },
        foreign_keys=meta.foreign_key_strings,
        ambiguous_columns=meta.ambiguous_column_names,
        row_count_estimates={
            t: meta.tables[t].row_count_estimate or 0
            for t in meta.tables
        },
    )
    db_verify.close()

    print(f"[OK] SchemaContext valid:")
    print(f"     database_type     = {ctx.database_type}")
    print(f"     available_tables  = {ctx.available_tables}")
    print(f"     foreign_keys      = {ctx.foreign_keys}")
    print(f"     ambiguous_columns = {ctx.ambiguous_columns}")
    print(f"     row_count_estimates = {ctx.row_count_estimates}")
    print(f"     table_summaries keys = {list(ctx.table_summaries.keys())}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    print("\n" + "=" * 68)
    print("ALL CHECKS PASSED")
    print("=" * 68 + "\n")

    print(textwrap.dedent("""
    To run Agent 1 against a REAL database with a live Groq call:

        from schema_analyzer_agent import analyze_schema

        ctx = analyze_schema(
            database_url="sqlite:///your_db.db",
            user_question="Who is our best customer?",
            verbose=True,
        )
        print(ctx.model_dump_json(indent=2))
    """))