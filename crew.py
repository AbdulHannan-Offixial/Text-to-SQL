"""
crew.py — Text-to-SQL Multi-Agent Pipeline Orchestrator
=========================================================
Wires together the three agents and their tasks into a single, well-structured
CrewAI pipeline with full state management, routing logic, and error handling.

Folder layout this file assumes
---------------------------------
Text-to-SQL/
├── agents/
│   ├── schema_analyzer_agent.py    → Agent 1 builders
│   ├── Clarification_agent.py      → Agent 2 builders
│   └── SQLexecutor_agent.py        → Agent 3 builders
├── database/
│   ├── database_manager.py         → DatabaseManager + QueryResult
│   └── schemas.py                  → All Pydantic contracts
├── crew.py                         ← THIS FILE
├── main.py
├── .env
└── requirements.txt

Pipeline flow
-------------
  User question + database_url
        ↓
  [Agent 1 — SchemaAnalyzer]
        ↓  SchemaContext
  [Agent 2 — ClarificationAgent]    ← may pause for human input
        ↓  ClarificationOutput
        │
        ├── status = "ambiguous"  → pause, collect answer, re-run Agent 2
        ├── status = "impossible" → return PipelineResult (halted)
        └── status = "clear"
                    ↓
          [Agent 3 — SQLExecutor]
                    ↓
          SQLExecutorOutput + QueryResult
                    ↓
          PipelineResult (completed / failed)

Public surface
--------------
  TextToSQLCrew          — the main orchestration class
  TextToSQLCrew.run()    — executes the full pipeline, returns PipelineResult
  PipelineResult         — typed output consumed by main.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup — makes "agents/" and "database/" importable regardless of the
# working directory the user launches main.py from.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.resolve()
for _subdir in ("agents", "database"):
    _path = str(_PROJECT_ROOT / _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

# ---------------------------------------------------------------------------
# Project imports — exact filenames from the folder structure
# ---------------------------------------------------------------------------
from database.database_manager import (          # database/database_manager.py
    DatabaseManager,
    DatabaseManagerError,
    QueryResult,
    build_manager_from_url,
    ConnectionError as DBConnectionError,
)
from schemas import (                   # database/schemas.py
    SchemaContext,
    ClarificationOutput,
    SQLExecutorOutput,
    PipelineResult as SchemaPipelineResult,
    UserClarificationResponse,
)
from agents.schema_analyzer_agent import (     # agents/schema_analyzer_agent.py
    build_schema_analyzer_crew,
    analyze_schema,
)
from agents.Clarification_agent import (       # agents/Clarification_agent.py
    build_clarification_crew,
    clarify_question,
    classify_question,
    QuestionClass,
)
from agents.SQLexecutor_agent import (         # agents/SQLexecutor_agent.py
    build_sql_executor_crew,
    execute_query,
)

logger = logging.getLogger("text_to_sql.crew")


# ---------------------------------------------------------------------------
# 1.  PipelineResult — the single object returned to main.py
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """
    Complete, typed result of one end-to-end Text-to-SQL pipeline run.

    Attributes
    ----------
    session_id : str
        Unique identifier for this run (UUID4). Use for logging / tracing.
    original_question : str
        The raw user question exactly as supplied.
    pipeline_status : str
        One of:
          "completed"             — SQL executed, rows returned
          "completed_no_rows"     — SQL executed, zero rows (valid result)
          "awaiting_clarification"— pipeline paused mid-run (should not
                                    surface to users; resolved internally)
          "impossible"            — schema cannot answer the question
          "failed"                — executor error after valid clarification
          "error"                 — unexpected pipeline-level exception
    schema_context : Optional[SchemaContext]
        Output of Agent 1. None only on very early errors.
    clarification_output : Optional[ClarificationOutput]
        Output of Agent 2. None only if Agent 1 failed.
    executor_output : Optional[SQLExecutorOutput]
        Output of Agent 3. None when pipeline halted before execution.
    query_result : Optional[QueryResult]
        Live database result (rows, columns, timing). None when no SQL ran.
    error_message : Optional[str]
        Human-readable error for display when pipeline_status == "error".
    total_elapsed_ms : float
        Wall-clock time for the full pipeline run in milliseconds.
    clarification_rounds : int
        Number of times the user was asked a clarification question.
    """

    session_id: str
    original_question: str
    pipeline_status: str
    schema_context: Optional[SchemaContext] = None
    clarification_output: Optional[ClarificationOutput] = None
    executor_output: Optional[SQLExecutorOutput] = None
    query_result: Optional[QueryResult] = None
    error_message: Optional[str] = None
    total_elapsed_ms: float = 0.0
    clarification_rounds: int = 0

    # ---- Convenience properties ----

    @property
    def succeeded(self) -> bool:
        return self.pipeline_status in ("completed", "completed_no_rows")

    @property
    def sql_query(self) -> Optional[str]:
        return self.executor_output.sql_query if self.executor_output else None

    @property
    def rows(self) -> List[Dict[str, Any]]:
        return self.query_result.rows if self.query_result else []

    @property
    def column_names(self) -> List[str]:
        return self.query_result.columns if self.query_result else []

    @property
    def row_count(self) -> int:
        return self.query_result.row_count if self.query_result else 0

    @property
    def execution_warnings(self) -> List[str]:
        return self.executor_output.execution_warnings if self.executor_output else []

    def summary(self) -> str:
        """Return a one-paragraph human-readable summary for display."""
        lines = [
            f"Session  : {self.session_id}",
            f"Question : {self.original_question}",
            f"Status   : {self.pipeline_status}",
            f"Elapsed  : {self.total_elapsed_ms:.0f} ms",
        ]
        if self.sql_query:
            lines.append(f"SQL      : {self.sql_query}")
        if self.query_result:
            lines.append(
                f"Result   : {self.row_count} row(s)"
                + (" [truncated]" if self.query_result.truncated else "")
            )
        if self.error_message:
            lines.append(f"Error    : {self.error_message}")
        if self.execution_warnings:
            lines.append(f"Warnings : {'; '.join(self.execution_warnings)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2.  TextToSQLCrew — the orchestrator
# ---------------------------------------------------------------------------

class TextToSQLCrew:
    """
    Orchestrates the three-agent Text-to-SQL pipeline.

    Lifecycle
    ---------
    1.  Instantiate with a database URL.
    2.  Call run(question) — returns a PipelineResult.
    3.  Optionally call close() or use as a context manager.

    Parameters
    ----------
    database_url : str
        Any SQLAlchemy-compatible connection URL.
    verbose : bool
        If True, CrewAI agents print their reasoning loop.
    human_input : bool
        If True, Agent 2 pauses to collect user clarification when ambiguous.
        Set False for fully automated / batch mode.
    max_clarification_rounds : int
        Maximum times Agent 2 may ask the user a clarification question
        before the pipeline gives up. Default 3.
    db_max_rows : int
        Row cap passed to DatabaseManager. Default 500.
    db_query_timeout : int
        Per-statement timeout in seconds. Default 30.

    Usage
    -----
    # Context manager (recommended):
    with TextToSQLCrew("sqlite:///sales.db", verbose=True) as crew:
        result = crew.run("Who is our best customer?")
        print(result.summary())
        for row in result.rows:
            print(row)

    # Manual lifecycle:
    crew = TextToSQLCrew("sqlite:///sales.db")
    result = crew.run("Show me total revenue by product category")
    crew.close()
    """

    def __init__(
        self,
        database_url: str,
        *,
        verbose: bool = False,
        human_input: bool = True,
        max_clarification_rounds: int = 3,
        db_max_rows: int = 500,
        db_query_timeout: int = 30,
    ) -> None:
        if not database_url or not isinstance(database_url, str):
            raise ValueError("database_url must be a non-empty string.")

        self._database_url         = database_url
        self._verbose              = verbose
        self._human_input          = human_input
        self._max_clarification    = max_clarification_rounds
        self._db_max_rows          = db_max_rows
        self._db_query_timeout     = db_query_timeout
        self._db: Optional[DatabaseManager] = None

        self._validate_environment()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "TextToSQLCrew":
        self._ensure_connected()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # 2a.  Environment validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_environment() -> None:
        """
        Fail fast if required environment variables are missing.
        Reads from os.environ (populated by python-dotenv in main.py).
        """
        missing = []
        if not os.environ.get("GROQ_API_KEY", "").strip():
            missing.append("GROQ_API_KEY")
        if missing:
            raise EnvironmentError(
                f"Missing required environment variable(s): {missing}\n"
                "Add them to your .env file:\n"
                "  GROQ_API_KEY=gsk_...\n"
                "and ensure python-dotenv loads them before creating TextToSQLCrew."
            )

    # ------------------------------------------------------------------
    # 2b.  Database connection management
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> DatabaseManager:
        """Return the live DatabaseManager, connecting on first call."""
        if self._db is None or not self._db.is_connected:
            logger.info("Connecting to database: %s", self._redact(self._database_url))
            self._db = build_manager_from_url(
                self._database_url,
                max_rows=self._db_max_rows,
                query_timeout=self._db_query_timeout,
            )
        return self._db

    def close(self) -> None:
        """Release the connection pool. Safe to call multiple times."""
        if self._db is not None:
            self._db.close()
            self._db = None
            logger.info("TextToSQLCrew: database connection closed.")

    # ------------------------------------------------------------------
    # 2c.  Main entry point
    # ------------------------------------------------------------------

    def run(self, question: str) -> PipelineResult:
        """
        Execute the full three-agent pipeline for a single user question.

        Parameters
        ----------
        question : str
            The raw natural language question from the user.

        Returns
        -------
        PipelineResult
            Always returns — never raises. All exceptions are caught and
            surfaced as PipelineResult(pipeline_status="error").

        Raises
        ------
        Nothing — all exceptions produce a PipelineResult with status="error".
        """
        session_id = str(uuid.uuid4())
        t_start    = time.perf_counter()

        logger.info(
            "[%s] Pipeline starting — question: %s",
            session_id[:8], repr(question[:80])
        )

        result = self._run_pipeline(
            session_id=session_id,
            question=question,
            t_start=t_start,
        )

        result.total_elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "[%s] Pipeline finished — status=%s  elapsed=%.0f ms",
            session_id[:8], result.pipeline_status, result.total_elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # 2d.  Internal pipeline implementation
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        session_id: str,
        question: str,
        t_start: float,
    ) -> PipelineResult:
        """
        Core pipeline. Catches all exceptions per-stage so a failure in
        one agent does not crash the entire process.
        """

        # ---- Stage 0: Fast pre-flight checks ----
        preflight = self._preflight_check(question)
        if preflight is not None:
            return PipelineResult(
                session_id=session_id,
                original_question=question,
                pipeline_status="impossible",
                error_message=preflight,
            )

        db: Optional[DatabaseManager] = None

        try:
            # ---- Stage 1: Connect to database ----
            try:
                db = self._ensure_connected()
            except DBConnectionError as exc:
                return PipelineResult(
                    session_id=session_id,
                    original_question=question,
                    pipeline_status="error",
                    error_message=(
                        f"Cannot connect to the database.\n{exc}\n"
                        "Check your database URL, credentials, and network access."
                    ),
                )
            except DatabaseManagerError as exc:
                return PipelineResult(
                    session_id=session_id,
                    original_question=question,
                    pipeline_status="error",
                    error_message=f"Database initialisation failed: {exc}",
                )

            # ---- Stage 2: Agent 1 — Schema Analyzer ----
            schema_ctx = self._run_agent1(question, db, session_id)
            if isinstance(schema_ctx, str):
                # Error string returned
                return PipelineResult(
                    session_id=session_id,
                    original_question=question,
                    pipeline_status="error",
                    error_message=schema_ctx,
                )

            logger.info(
                "[%s] Agent 1 complete: %d tables, %d FKs",
                session_id[:8],
                len(schema_ctx.available_tables),
                len(schema_ctx.foreign_keys),
            )

            # ---- Stage 3: Agent 2 — Clarification (with retry loop) ----
            clarif_out, clarification_rounds = self._run_agent2_with_retry(
                question=question,
                schema_ctx=schema_ctx,
                session_id=session_id,
            )

            if isinstance(clarif_out, str):
                return PipelineResult(
                    session_id=session_id,
                    original_question=question,
                    pipeline_status="error",
                    schema_context=schema_ctx,
                    clarification_rounds=clarification_rounds,
                    error_message=clarif_out,
                )

            logger.info(
                "[%s] Agent 2 complete: status=%s  rounds=%d",
                session_id[:8], clarif_out.status, clarification_rounds,
            )

            # ---- Route on clarification status ----
            if clarif_out.status == "impossible":
                return PipelineResult(
                    session_id=session_id,
                    original_question=question,
                    pipeline_status="impossible",
                    schema_context=schema_ctx,
                    clarification_output=clarif_out,
                    clarification_rounds=clarification_rounds,
                    error_message=clarif_out.impossible_reason,
                )

            # If still ambiguous after max rounds, halt gracefully
            if clarif_out.status == "ambiguous":
                return PipelineResult(
                    session_id=session_id,
                    original_question=question,
                    pipeline_status="awaiting_clarification",
                    schema_context=schema_ctx,
                    clarification_output=clarif_out,
                    clarification_rounds=clarification_rounds,
                    error_message=(
                        f"Maximum clarification rounds ({self._max_clarification}) reached "
                        "without full resolution. Please rephrase your question more specifically."
                    ),
                )

            # ---- Stage 4: Agent 3 — SQL Executor ----
            exec_out, query_result = self._run_agent3(
                clarif_out=clarif_out,
                schema_ctx=schema_ctx,
                db=db,
                session_id=session_id,
            )

            if isinstance(exec_out, str):
                return PipelineResult(
                    session_id=session_id,
                    original_question=question,
                    pipeline_status="failed",
                    schema_context=schema_ctx,
                    clarification_output=clarif_out,
                    clarification_rounds=clarification_rounds,
                    error_message=exec_out,
                )

            logger.info(
                "[%s] Agent 3 complete: status=%s  rows=%d",
                session_id[:8], exec_out.status,
                query_result.row_count if query_result else 0,
            )

            # ---- Determine final pipeline status ----
            if exec_out.status == "error":
                final_status = "failed"
            elif query_result and query_result.row_count == 0:
                final_status = "completed_no_rows"
            else:
                final_status = "completed"

            return PipelineResult(
                session_id=session_id,
                original_question=question,
                pipeline_status=final_status,
                schema_context=schema_ctx,
                clarification_output=clarif_out,
                executor_output=exec_out,
                query_result=query_result,
                clarification_rounds=clarification_rounds,
                error_message=exec_out.error_detail if exec_out.status == "error" else None,
            )

        except Exception as exc:
            logger.exception("[%s] Unexpected pipeline exception", session_id[:8])
            return PipelineResult(
                session_id=session_id,
                original_question=question,
                pipeline_status="error",
                error_message=(
                    f"Unexpected error in pipeline: {type(exc).__name__}: {exc}\n"
                    "Check the logs for the full traceback."
                ),
            )

    # ------------------------------------------------------------------
    # 2e.  Per-stage methods (isolated error boundaries)
    # ------------------------------------------------------------------

    def _preflight_check(self, question: str) -> Optional[str]:
        """
        Pure-Python pre-flight: run the same fast-path classifier from
        Clarification_agent.py before spending a single token.

        Returns an error string if the question is non-recoverable,
        or None if it is safe to proceed.
        """
        cls_result = classify_question(question)

        NON_RECOVERABLE = {
            QuestionClass.EMPTY,
            QuestionClass.BINARY_GARBAGE,
            QuestionClass.INJECTION,
            QuestionClass.HARMFUL,
        }
        if cls_result.question_class in NON_RECOVERABLE:
            flags = "; ".join(cls_result.detected_flags) or cls_result.question_class.value
            return (
                f"Question rejected before pipeline start: {flags}. "
                "Please rephrase in plain English."
            )
        return None

    def _run_agent1(
        self,
        question: str,
        db: DatabaseManager,
        session_id: str,
    ) -> "SchemaContext | str":
        """
        Run Agent 1 (SchemaAnalyzer).

        Returns SchemaContext on success, or an error string on failure.
        Uses analyze_schema() from schema_analyzer_agent.py which internally
        builds the crew, kicks it off, and validates the Pydantic output.
        """
        logger.info("[%s] Agent 1 starting...", session_id[:8])
        try:
            return analyze_schema(
                self._database_url,
                question,
                verbose=self._verbose,
            )
        except EnvironmentError as exc:
            return f"Environment error in Agent 1: {exc}"
        except DBConnectionError as exc:
            return f"Database unreachable in Agent 1: {exc}"
        except RuntimeError as exc:
            return f"Agent 1 (SchemaAnalyzer) failed: {exc}"
        except Exception as exc:
            logger.exception("[%s] Unexpected Agent 1 error", session_id[:8])
            return f"Unexpected Agent 1 error: {type(exc).__name__}: {exc}"

    def _run_agent2_with_retry(
        self,
        question: str,
        schema_ctx: SchemaContext,
        session_id: str,
    ) -> "tuple[ClarificationOutput | str, int]":
        """
        Run Agent 2 (ClarificationAgent) with a clarification retry loop.

        If the agent returns status='ambiguous' and human_input=True, the
        user's answer is collected (via CrewAI's built-in human_input pause
        or programmatic input() if running in CLI mode) and Agent 2 re-runs
        with the updated question until status becomes 'clear' or 'impossible',
        or max_clarification_rounds is reached.

        Returns (ClarificationOutput | error_str, rounds_taken).
        """
        current_question = question
        rounds = 0

        for round_num in range(1, self._max_clarification + 1):
            logger.info(
                "[%s] Agent 2 round %d/%d...",
                session_id[:8], round_num, self._max_clarification,
            )
            try:
                clarif = clarify_question(
                    raw_question=current_question,
                    schema_context=schema_ctx,
                    verbose=self._verbose,
                    human_input=self._human_input,
                )
                rounds = round_num
            except EnvironmentError as exc:
                return f"Environment error in Agent 2: {exc}", rounds
            except RuntimeError as exc:
                return f"Agent 2 (ClarificationAgent) failed: {exc}", rounds
            except Exception as exc:
                logger.exception("[%s] Unexpected Agent 2 error", session_id[:8])
                return f"Unexpected Agent 2 error: {type(exc).__name__}: {exc}", rounds

            # Terminal states — stop the loop
            if clarif.status in ("clear", "impossible"):
                return clarif, rounds

            # Ambiguous: when human_input=True, CrewAI already paused for the
            # user reply inside clarify_question() via the human_input Task flag.
            # If human_input=False (batch mode), we cannot resolve — exit loop.
            if clarif.status == "ambiguous":
                if not self._human_input:
                    logger.warning(
                        "[%s] Ambiguous question in batch mode — cannot pause for input.",
                        session_id[:8],
                    )
                    return clarif, rounds

                # When human_input=True, CrewAI pauses mid-task and collects
                # the user's reply automatically. After kickoff() returns,
                # the task output already incorporates the answer.
                # If after that it is still ambiguous, the user gave an
                # unusable answer — try again with the same original question
                # so the agent can re-read the resolved intent from the
                # CrewAI context window of the previous turn.
                logger.info(
                    "[%s] Still ambiguous after round %d — retrying.",
                    session_id[:8], round_num,
                )
                # Keep current_question unchanged; CrewAI context carries forward

        # Exhausted max rounds
        return clarif, rounds  # type: ignore[possibly-undefined]

    def _run_agent3(
        self,
        clarif_out: ClarificationOutput,
        schema_ctx: SchemaContext,
        db: DatabaseManager,
        session_id: str,
    ) -> "tuple[SQLExecutorOutput | str, Optional[QueryResult]]":
        """
        Run Agent 3 (SQLExecutor).

        Returns (SQLExecutorOutput | error_str, QueryResult | None).
        """
        logger.info("[%s] Agent 3 starting...", session_id[:8])
        try:
            exec_out, query_result = execute_query(
                clarification_output=clarif_out,
                schema_context=schema_ctx,
                db=db,
                verbose=self._verbose,
            )
            return exec_out, query_result
        except ValueError as exc:
            # Routing error — status was not 'clear'
            return f"Agent 3 routing error: {exc}", None
        except RuntimeError as exc:
            return f"Agent 3 (SQLExecutor) failed: {exc}", None
        except DatabaseManagerError as exc:
            return f"Database error during SQL execution: {exc}", None
        except Exception as exc:
            logger.exception("[%s] Unexpected Agent 3 error", session_id[:8])
            return f"Unexpected Agent 3 error: {type(exc).__name__}: {exc}", None

    # ------------------------------------------------------------------
    # 2f.  Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _redact(url: str) -> str:
        """Redact credentials from a database URL for safe logging."""
        import re
        return re.sub(r"(://)[^:@/]+:[^@/]+(@)", r"\1***:***\2", url)

    def health_check(self) -> Dict[str, Any]:
        """
        Return a health-check dict for monitoring.
        Safe to call at any time — does not change pipeline state.
        """
        report: Dict[str, Any] = {
            "database_url": self._redact(self._database_url),
            "groq_api_key_set": bool(os.environ.get("GROQ_API_KEY", "").strip()),
            "db_connected": False,
            "table_count": None,
        }
        try:
            db = self._ensure_connected()
            hc = db.health_check()
            report["db_connected"]  = hc.get("status") == "healthy"
            report["table_count"]   = hc.get("table_count")
            report["dialect"]       = hc.get("dialect")
        except Exception as exc:
            report["db_error"] = str(exc)
        return report


# ---------------------------------------------------------------------------
# 3.  Module-level convenience function
# ---------------------------------------------------------------------------

def run_pipeline(
    question: str,
    database_url: str,
    *,
    verbose: bool = False,
    human_input: bool = True,
    max_clarification_rounds: int = 3,
    db_max_rows: int = 500,
    db_query_timeout: int = 30,
) -> PipelineResult:
    """
    One-call convenience wrapper — builds the crew, runs it, tears it down.

    Ideal for scripts, tests, and one-off queries where lifecycle management
    is not needed. For repeated queries against the same database, use the
    TextToSQLCrew class directly (reuses the connection pool across calls).

    Parameters
    ----------
    question : str
        Raw natural language question.
    database_url : str
        SQLAlchemy-compatible connection URL.
    verbose : bool
        Print agent reasoning loop.
    human_input : bool
        Pause for user clarification when ambiguous.
    max_clarification_rounds : int
        Max Agent 2 retry iterations.
    db_max_rows : int
        Row cap for query results.
    db_query_timeout : int
        Per-statement timeout in seconds.

    Returns
    -------
    PipelineResult
        Always returns — never raises.
    """
    with TextToSQLCrew(
        database_url=database_url,
        verbose=verbose,
        human_input=human_input,
        max_clarification_rounds=max_clarification_rounds,
        db_max_rows=db_max_rows,
        db_query_timeout=db_query_timeout,
    ) as crew:
        return crew.run(question)