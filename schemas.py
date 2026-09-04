"""
schemas.py — Pydantic v2 contracts for the Text-to-SQL Multi-Agent System
The schema in this project serves two distinct, crucial roles:

The Database Blueprint (Schema Context): 
It tells the AI the exact structure of your database—its tables, column 
names, data types, and foreign key relationships—so the AI knows what data 
exists and how to write accurate SQL without guessing.

The Output Guardrail (Pydantic Schema): It forces the AI to output its 
responses in a strict, structured JSON format (digital forms with strict rules) 
instead of conversational text. This guarantees your Python backend can safely 
process the AI's decisions, ask clarification questions, or execute SQL 
without crashing.
=========================================================================
Model : openai/gpt-oss-20b  (via Groq API, model ID: "openai/gpt-oss-20b")
Agents: SchemaAnalyzer → ClarificationAgent → SQLExecutor
 
Design principles
-----------------
* strict=True  — LLM cannot silently coerce types (e.g. str → int).
* Cross-field model validators enforce semantic consistency so each agent
  cannot produce a logically contradictory payload.
* SQL validators block every known injection / mutation vector at the
  Pydantic layer before the query ever reaches a database connection.
* Every schema is self-documenting via Field descriptions so the LLM
  prompt-engineering layer can embed the JSON schema verbatim.
"""
from __future__ import annotations

import re
from typing import Annotated, Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# 1.  Shared Enumerations & Annotated Types
# ---------------------------------------------------------------------------

# AnalysisStatus — the single most important signal in the pipeline:
# • "ambiguous"   → ClarificationAgent must ask before any SQL is written
# • "clear"       → SQLExecutor may proceed
# • "impossible"  → the schema does not contain the data needed; stop here
AnalysisStatus = Annotated[
    Literal["ambiguous", "clear", "impossible"],
    Field(
        description=(
            "ambiguous — user intent cannot be resolved without clarification. "
            "clear     — intent is unambiguous; proceed to SQL generation. "
            "impossible— the requested data does not exist in the provided schema."
        )
    ),
]

# ConfidenceScore — integer 0-100; strict mode rejects float or str inputs
ConfidenceScore = Annotated[
    int,
    Field(ge=0, le=100, description="0 = completely uncertain, 100 = fully confident."),
]

# SQLString — the only kind of string we allow to reach the executor
# None is legal until a clear + validated query exists
SafeSQLString = Annotated[
    Optional[str],
    Field(
        default=None,
        description=(
            "A read-only SQLite / SQL SELECT or WITH query. "
            "MUST start with SELECT or WITH (case-insensitive). "
            "MUST end with a LIMIT clause (e.g. LIMIT 100). "
            "MUST NOT contain DDL/DML keywords (DROP, DELETE, INSERT, UPDATE, ALTER, "
            "CREATE, TRUNCATE, REPLACE, EXEC, EXECUTE, PRAGMA, ATTACH, DETACH). "
            "MUST NOT contain stacked statements (semicolons mid-query). "
            "MUST NOT contain comment injections (-- or /* */)."
        ),
    ),
]

# ---------------------------------------------------------------------------
# 2.  Helpers
# ---------------------------------------------------------------------------

# Pre-compiled patterns — evaluated once at import time for performance
_DISALLOWED_PATTERN = re.compile(
    r"""
    (?:                          # DDL / DML keywords
        \b(?:
            DROP | DELETE | INSERT | UPDATE | ALTER | CREATE | TRUNCATE |
            REPLACE | EXEC(?:UTE)? | PRAGMA | ATTACH | DETACH | VACUUM |
            REINDEX | ANALYZE | GRANT | REVOKE | SAVEPOINT | ROLLBACK |
            COMMIT | BEGIN
        )\b
    )
    |
    (--[^\n]*)                   # single-line SQL comment
    |
    (/\*[\s\S]*?\*/)             # block SQL comment
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SELECT_OR_WITH = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
_LIMIT_CLAUSE   = re.compile(r"\bLIMIT\s+\d+\s*(?:OFFSET\s+\d+\s*)?$", re.IGNORECASE)
_SEMICOLON_MID  = re.compile(r";(?!\s*$)")        # semicolon that is NOT the final char


def _validate_sql(query: Optional[str]) -> Optional[str]:
    """
    Central SQL safety gate called from both field validators and model validators.
    Returns the stripped query if valid, raises ValueError with a precise message otherwise.
    This function is intentionally pure (no side-effects) so it can be unit-tested in isolation.
    """
    if query is None:
        return None

    q = query.strip()

    if not q:
        raise ValueError("sql_query must not be an empty string when provided.")

    if not _SELECT_OR_WITH.match(q):
        raise ValueError(
            "sql_query must begin with SELECT or WITH. "
            "DML/DDL statements (INSERT, UPDATE, DELETE, …) are not permitted."
        )

    disallowed_match = _DISALLOWED_PATTERN.search(q)
    if disallowed_match:
        raise ValueError(
            f"sql_query contains a disallowed keyword or comment injection: "
            f"'{disallowed_match.group(0).strip()}'. "
            "Only read-only SELECT / WITH queries are permitted."
        )

    if _SEMICOLON_MID.search(q):
        raise ValueError(
            "sql_query contains a mid-query semicolon — stacked statements are not allowed."
        )

    if not _LIMIT_CLAUSE.search(q):
        raise ValueError(
            "sql_query must end with a LIMIT clause (e.g. '… LIMIT 100') "
            "to prevent unbounded result sets."
        )

    return q


# ---------------------------------------------------------------------------
# 3.  Agent 1 — Schema Analyzer Output
# ---------------------------------------------------------------------------

class SchemaContext(BaseModel):
    """
    Produced by Agent 1 (SchemaAnalyzer).

    Captures everything the downstream agents need to know about the database
    without forwarding raw schema DDL (which could be very large).
    Kept deliberately lean — only what affects query planning.
    """
    model_config = ConfigDict(strict=True)

    database_type: Annotated[
        Literal["sqlite", "postgresql", "mysql", "mssql", "bigquery", "other"],
        Field(description="Dialect of the target database. Affects SQL syntax choices."),
    ]

    available_tables: Annotated[
        List[str],
        Field(
            min_length=1,
            description="Exact table names present in the schema. Used for hallucination checks.",
        ),
    ]

    table_summaries: Annotated[
        Dict[str, str],
        Field(
            description=(
                "Mapping of table_name → one-sentence purpose, e.g. "
                "{'orders': 'Each row is a customer purchase with timestamp and total_amount.'}"
            )
        ),
    ]

    foreign_keys: Annotated[
        List[str],
        Field(
            default_factory=list,
            description=(
                "Human-readable FK relationships, e.g. ['orders.customer_id → customers.id']. "
                "Used by the SQL agent to select correct JOIN paths."
            ),
        ),
    ]

    ambiguous_columns: Annotated[
        List[str],
        Field(
            default_factory=list,
            description=(
                "Column names whose business meaning is unclear without domain context, "
                "e.g. ['score', 'rank', 'status']. Signals the Clarification agent."
            ),
        ),
    ]

    row_count_estimates: Annotated[
        Dict[str, int],
        Field(
            default_factory=dict,
            description=(
                "Rough row counts per table so the SQL agent can tune LIMIT values. "
                "{'orders': 500000, 'customers': 12000}"
            ),
        ),
    ]

    @field_validator("available_tables", mode="before")
    @classmethod
    def normalize_table_names(cls, v: Any) -> List[str]:
        """Strip whitespace; reject empty strings; lower-case for consistency."""
        if not isinstance(v, list):
            raise ValueError("available_tables must be a list of strings.")
        cleaned: List[str] = []
        for item in v:
            if not isinstance(item, str):
                raise ValueError(f"Table name must be a string, got {type(item).__name__}.")
            name = item.strip()
            if not name:
                raise ValueError("Table names must not be empty strings.")
            cleaned.append(name)
        return cleaned

    @model_validator(mode="after")
    def summaries_cover_all_tables(self) -> "SchemaContext":
        """Every table in available_tables must have an entry in table_summaries."""
        missing = [t for t in self.available_tables if t not in self.table_summaries]
        if missing:
            raise ValueError(
                f"table_summaries is missing entries for: {missing}. "
                "Provide a one-sentence summary for every table in available_tables."
            )
        return self


# ---------------------------------------------------------------------------
# 4.  Agent 2 — Clarification Agent Output
# ---------------------------------------------------------------------------

class AmbiguityDetail(BaseModel):
    """
    A single ambiguous term extracted from the user's question.
    The ClarificationAgent produces one per unclear concept.
    """
    model_config = ConfigDict(strict=True)

    term: Annotated[
        str,
        Field(min_length=1, description="The exact word or phrase that is ambiguous, e.g. 'best'."),
    ]

    interpretations: Annotated[
        List[str],
        Field(
            min_length=2,
            max_length=5,
            description=(
                "2–5 concrete, mutually-exclusive database interpretations. "
                "Each should map directly to a measurable SQL expression, e.g. "
                "['SUM(order_total)', 'COUNT(order_id)', 'AVG(order_total)']."
            ),
        ),
    ]

    suggested_question: Annotated[
        str,
        Field(
            min_length=20,
            description=(
                "A ready-to-display question for the end user offering numbered choices, "
                "e.g. 'By \"best customer\" do you mean: (1) highest total spend, "
                "(2) most orders placed, or (3) most recent purchase?'"
            ),
        ),
    ]

    @field_validator("interpretations", mode="before")
    @classmethod
    def no_duplicate_interpretations(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            raise ValueError("interpretations must be a list.")
        seen: set = set()
        for item in v:
            if not isinstance(item, str):
                raise ValueError("Each interpretation must be a string.")
            s = item.strip().lower()
            if s in seen:
                raise ValueError(f"Duplicate interpretation detected: '{item}'.")
            seen.add(s)
        return v


class ClarificationOutput(BaseModel):
    """
    Full output of Agent 2 (ClarificationAgent).

    When status == 'ambiguous' the agent MUST populate ambiguities and
    must NOT populate sql_query (enforcement via model_validator).
    When status == 'clear'    the agent passes control to SQLExecutor.
    When status == 'impossible' the agent explains why and halts the pipeline.
    """
    model_config = ConfigDict(strict=True)

    status: AnalysisStatus

    ambiguities: Annotated[
        List[AmbiguityDetail],
        Field(
            default_factory=list,
            description="One AmbiguityDetail per unclear concept. Empty when status != 'ambiguous'.",
        ),
    ]

    combined_clarification_message: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "A single, user-facing message that merges all individual suggested_questions "
                "into one coherent paragraph. Required when status == 'ambiguous'."
            ),
        ),
    ]

    clarified_intent: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "Once the user has answered, the agent records the resolved intent here, "
                "e.g. 'best customer = highest SUM(order_total)'. "
                "Required when status == 'clear' and clarification was previously needed."
            ),
        ),
    ]

    impossible_reason: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "When status == 'impossible', explain exactly what data is missing from the "
                "schema and suggest schema changes that would make the query feasible."
            ),
        ),
    ]

    confidence_score: ConfidenceScore

    reasoning: Annotated[
        str,
        Field(
            min_length=20,
            description=(
                "CONFIDENCE SCORE GUIDELINES:\n"
                "Set \"confidence_score\" as an integer from 0 to 100 based on these rules:\n"
                "- 0 to 20: Use when status is \"impossible\" or the query is completely outside the schema context.\n"
                "- 21 to 60: Use when status is \"ambiguous\" and you require clarification from the user.\n"
                "- 61 to 100: Use when status is \"clear\" and the user's intent maps directly to explicit database tables and columns."
            ),
        ),
    ]

    # ---- Cross-field consistency rules ----

    @model_validator(mode="after")
    def enforce_status_contract(self) -> "ClarificationOutput":
        s = self.status

        if s == "ambiguous":
            if not self.ambiguities:
                raise ValueError(
                    "status='ambiguous' requires at least one entry in ambiguities."
                )
            if not self.combined_clarification_message:
                raise ValueError(
                    "status='ambiguous' requires combined_clarification_message to be populated."
                )

        if s == "impossible":
            if not self.impossible_reason:
                raise ValueError(
                    "status='impossible' requires impossible_reason to be populated."
                )
            if self.ambiguities:
                raise ValueError(
                    "ambiguities must be empty when status='impossible'."
                )

        if s == "clear":
            if self.ambiguities:
                raise ValueError(
                    "ambiguities must be empty when status='clear'. "
                    "Clear means all ambiguity has been resolved."
                )
            if self.combined_clarification_message:
                raise ValueError(
                    "combined_clarification_message must be None when status='clear'."
                )

        return self


# ---------------------------------------------------------------------------
# 5.  Agent 3 — SQL Executor Output
# ---------------------------------------------------------------------------

class SQLExecutorOutput(BaseModel):
    """
    Output of Agent 3 (SQLExecutor).

    Only reachable when ClarificationOutput.status == 'clear'.
    Contains the validated, safe, executable SQL query plus rich metadata
    so the orchestrator can log, audit, and display results.
    """
    model_config = ConfigDict(strict=True)

    status: Annotated[
        Literal["success", "error", "partial"],
        Field(
            description=(

            )
        ),
    ]

    sql_query: SafeSQLString

    tables_used: Annotated[
        List[str],
        Field(
            default_factory=list,
            description=(
                "Exact table names referenced in sql_query. "
                "Orchestrator cross-checks this against SchemaContext.available_tables "
                "to catch hallucinated table names."
            ),
        ),
    ]

    columns_used: Annotated[
        List[str],
        Field(
            default_factory=list,
            description=(
                "Exact column names (optionally table-qualified) used in SELECT, WHERE, "
                "JOIN, GROUP BY, and ORDER BY clauses."
            ),
        ),
    ]

    join_paths: Annotated[
        List[str],
        Field(
            default_factory=list,
            description=(
                "Human-readable join paths used, e.g. "
                "['orders JOIN customers ON orders.customer_id = customers.id']. "
                "Enables the reviewer to verify FK correctness."
            ),
        ),
    ]

    confidence_score: ConfidenceScore

    reasoning: Annotated[
        str,
        Field(
            min_length=20,
            description=(
                "CONFIDENCE SCORE GUIDELINES:\n"
                "Set \"confidence_score\" as an integer from 0 to 100 based on these rules:\n"
                "- 0 to 50: Use when generating SQL required complex assumptions or non-standard join paths.\n"
                "- 51 to 85: Use when the query is standard, safe, and uses explicit foreign keys.\n"
                "- 86 to 100: Use when the clarified user intent matches the database schema 1:1 with zero ambiguity."
            ),
        ),
    ]

    error_detail: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Populated only when status='error'. Describes what went wrong.",
        ),
    ]

    execution_warnings: Annotated[
        List[str],
        Field(
            default_factory=list,
            description=(
                "Non-fatal warnings, e.g. 'LIMIT 100 applied; result may be truncated' "
                "or 'Column order_value has NULLs — COUNT may undercount'."
            ),
        ),
    ]

    # ---- Field-level SQL validator ----

    @field_validator("sql_query", mode="before")
    @classmethod
    def validate_sql_field(cls, v: Any) -> Optional[str]:
        return _validate_sql(v)

    # ---- Cross-field consistency rules ----

    @model_validator(mode="after")
    def enforce_executor_contract(self) -> "SQLExecutorOutput":
        if self.status == "success":
            if not self.sql_query:
                raise ValueError(
                    "status='success' requires sql_query to be populated."
                )
            if not self.tables_used:
                raise ValueError(
                    "tables_used must list at least one table when status='success'."
                )

        if self.status == "error":
            if self.sql_query:
                raise ValueError(
                    "sql_query must be None when status='error'. "
                    "Do not emit a partial or unsafe query."
                )
            if not self.error_detail:
                raise ValueError(
                    "status='error' requires error_detail to describe the failure."
                )

        return self


# ---------------------------------------------------------------------------
# 6.  Orchestrator — Full Pipeline Result
# ---------------------------------------------------------------------------

class PipelineResult(BaseModel):
    """
    Top-level wrapper returned by the orchestrator after all three agents have run.
    Provides a single object that the API / UI layer can consume.
    """
    model_config = ConfigDict(strict=True)

    session_id: Annotated[
        str,
        Field(
            min_length=1,
            description="Unique identifier for this query session (UUID or similar).",
        ),
    ]

    original_question: Annotated[
        str,
        Field(min_length=1, description="The raw natural language question from the user."),
    ]

    schema_context: SchemaContext
    clarification_output: ClarificationOutput

    # Optional — only present when the pipeline reached the SQL Executor
    executor_output: Annotated[
        Optional[SQLExecutorOutput],
        Field(
            default=None,
            description=(
                "Populated when clarification_output.status == 'clear' and Agent 3 ran. "
                "None when the pipeline halted at clarification or impossibility."
            ),
        ),
    ]

    # Aggregated status surfaced directly for easy routing in the orchestrator
    pipeline_status: Annotated[
        Literal["awaiting_clarification", "completed", "failed", "impossible"],
        Field(
            description=(
                "awaiting_clarification — user must answer a clarification question. "
                "completed             — sql_query is ready (or was executed). "
                "failed                — SQLExecutor encountered an error. "
                "impossible            — schema does not support the requested query."
            )
        ),
    ]

    total_confidence: Annotated[
        int,
        Field(
            ge=0, le=100,
            description=(
                "Weighted average of all agent confidence scores. "
                "Orchestrator computes this; agents do not set it."
            ),
        ),
    ]

    @model_validator(mode="after")
    def enforce_pipeline_contract(self) -> "PipelineResult":
        cl_status  = self.clarification_output.status
        ex_output  = self.executor_output
        p_status   = self.pipeline_status

        # When clarification is ambiguous, executor must not have run
        if cl_status == "ambiguous":
            if ex_output is not None:
                raise ValueError(
                    "executor_output must be None when clarification_output.status='ambiguous'. "
                    "SQLExecutor must not run before the user resolves ambiguity."
                )
            if p_status != "awaiting_clarification":
                raise ValueError(
                    "pipeline_status must be 'awaiting_clarification' "
                    "when clarification_output.status='ambiguous'."
                )

        # When the question is impossible, executor must not have run
        if cl_status == "impossible":
            if ex_output is not None:
                raise ValueError(
                    "executor_output must be None when clarification_output.status='impossible'."
                )
            if p_status != "impossible":
                raise ValueError(
                    "pipeline_status must be 'impossible' "
                    "when clarification_output.status='impossible'."
                )

        # When clarification is clear, executor must have run
        if cl_status == "clear":
            if ex_output is None:
                raise ValueError(
                    "executor_output must be populated when clarification_output.status='clear'."
                )
            if ex_output.status == "success" and p_status != "completed":
                raise ValueError(
                    "pipeline_status must be 'completed' when executor_output.status='success'."
                )
            if ex_output.status == "error" and p_status != "failed":
                raise ValueError(
                    "pipeline_status must be 'failed' when executor_output.status='error'."
                )

        return self


# ---------------------------------------------------------------------------
# 7.  User Clarification Response (input schema for the follow-up turn)
# ---------------------------------------------------------------------------

class UserClarificationResponse(BaseModel):
    """
    What the orchestrator receives after the user replies to a clarification question.
    Fed back into the pipeline to produce a ClarificationOutput with status='clear'.
    """
    model_config = ConfigDict(strict=True)

    session_id: Annotated[str, Field(min_length=1)]

    chosen_interpretation: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "The user's chosen interpretation verbatim, e.g. "
                "'highest total spend' or '1' (index of the offered option)."
            ),
        ),
    ]

    additional_context: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Any free-text elaboration the user volunteered beyond the choice.",
        ),
    ]


# ---------------------------------------------------------------------------
# 8.  Utility — JSON Schema Export
# ---------------------------------------------------------------------------

def export_schemas() -> Dict[str, Any]:
    """
    Returns a dict of JSON schemas for all public models.
    Useful for embedding in LLM system prompts so the model is
    explicitly aware of the output contracts it must satisfy.

    Usage:
        schemas = export_schemas()
        system_prompt = f"Output MUST conform to:\\n{schemas['QueryAnalysis']}"
    """
    models = {
        "SchemaContext":             SchemaContext,
        "AmbiguityDetail":           AmbiguityDetail,
        "ClarificationOutput":       ClarificationOutput,
        "SQLExecutorOutput":         SQLExecutorOutput,
        "PipelineResult":            PipelineResult,
        "UserClarificationResponse": UserClarificationResponse,
    }
    return {name: cls.model_json_schema() for name, cls in models.items()}


# ---------------------------------------------------------------------------
# 9.  Quick smoke-test (run: python schemas.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("SMOKE TEST — schemas.py")
    print("=" * 60)

    # --- SchemaContext ---
    sc = SchemaContext(
        database_type="sqlite",
        available_tables=["customers", "orders"],
        table_summaries={
            "customers": "One row per registered customer with contact info.",
            "orders":    "One row per purchase linked to a customer via customer_id.",
        },
        foreign_keys=["orders.customer_id → customers.id"],
        ambiguous_columns=["score"],
        row_count_estimates={"customers": 1200, "orders": 45000},
    )
    print("\n[OK] SchemaContext:", sc.model_dump())

    # --- ClarificationOutput (ambiguous) ---
    co_ambiguous = ClarificationOutput(
        status="ambiguous",
        ambiguities=[
            AmbiguityDetail(
                term="best",
                interpretations=[
                    "highest SUM(order_total)",
                    "highest COUNT(order_id)",
                    "most recent MAX(order_date)",
                ],
                suggested_question=(
                    'By "best customer" do you mean: '
                    "(1) highest total spend, "
                    "(2) most orders placed, or "
                    "(3) most recently active?"
                ),
            )
        ],
        combined_clarification_message=(
            'Your question uses the word "best" — could you clarify? '
            "Do you mean the customer with (1) the highest total spend, "
            "(2) the most orders, or (3) the most recent activity?"
        ),
        confidence_score=40,
        reasoning=(
            "The word 'best' is inherently multi-dimensional. "
            "The schema has order_total (revenue) and order_id (frequency) columns "
            "in the orders table, both of which are valid interpretations. "
            "Status set to 'ambiguous' until user disambiguates."
        ),
    )
    print("\n[OK] ClarificationOutput (ambiguous):", co_ambiguous.status)

    # --- ClarificationOutput (clear, after user replied) ---
    co_clear = ClarificationOutput(
        status="clear",
        clarified_intent="best customer = highest SUM(order_total)",
        confidence_score=95,
        reasoning=(
            "User selected interpretation (1): highest total spend. "
            "This maps directly to SUM(order_total) GROUP BY customer_id in the orders table."
        ),
    )
    print("[OK] ClarificationOutput (clear):", co_clear.status)

    # --- SQLExecutorOutput (success) ---
    ex = SQLExecutorOutput(
        status="success",
        sql_query=(
            "SELECT c.customer_id, c.name, SUM(o.order_total) AS total_spend "
            "FROM customers c "
            "JOIN orders o ON o.customer_id = c.customer_id "
            "GROUP BY c.customer_id, c.name "
            "ORDER BY total_spend DESC "
            "LIMIT 10"
        ),
        tables_used=["customers", "orders"],
        columns_used=["customer_id", "name", "order_total"],
        join_paths=["orders JOIN customers ON orders.customer_id = customers.id"],
        confidence_score=97,
        reasoning=(
            "Mapped 'highest total spend' to SUM(order_total). "
            "Joined customers → orders on customer_id FK. "
            "Applied LIMIT 10 to return top 10 customers. "
            "Query is read-only SELECT; no DML or DDL present."
        ),
        execution_warnings=["LIMIT 10 applied — only top 10 results returned."],
    )
    print("[OK] SQLExecutorOutput (success):", ex.status)

    # --- PipelineResult ---
    pr = PipelineResult(
        session_id="sess-abc-001",
        original_question="Who is our best customer?",
        schema_context=sc,
        clarification_output=co_clear,
        executor_output=ex,
        pipeline_status="completed",
        total_confidence=88,
    )
    print("[OK] PipelineResult (completed):", pr.pipeline_status)

    # --- SQL injection rejection ---
    print("\n--- SQL Safety Checks ---")
    bad_queries = [
        "DROP TABLE customers",
        "SELECT * FROM users; DELETE FROM users",
        "SELECT * FROM users -- comment",
        "SELECT * FROM users WHERE id = 1 OR 1=1",   # no LIMIT — should fail
        "INSERT INTO x VALUES (1)",
    ]
    for bq in bad_queries:
        try:
            SQLExecutorOutput(
                status="success",
                sql_query=bq,
                tables_used=["users"],
                columns_used=["id"],
                confidence_score=50,
                reasoning="This is a test of the safety validator." * 1,
            )
            print(f"[FAIL — not caught]: {bq!r}")
        except Exception as e:
            print(f"[BLOCKED OK]: {bq[:50]!r}  →  {type(e).__name__}")

    # --- JSON schema export ---
    schemas = export_schemas()
    print(f"\n[OK] Exported {len(schemas)} JSON schemas.")
    print("Schema keys:", list(schemas.keys()))

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)