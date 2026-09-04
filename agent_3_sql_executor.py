import os
import json
from dotenv import load_dotenv
from groq import Groq
from schemas import SchemaContext, SQLExecutorOutput
from database_manager import DatabaseManager
from agent_1_schema_analyzer import SchemaAnalyzerAgent

load_dotenv()

class SQLExecutorAgent:
    """
    Agent 3: SQL Executor Agent
    Generates safe, read-only SQL queries from clarified user intent and schema context.
    Output is strictly validated against SQLExecutorOutput in schemas.py.
    """

    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY missing from environment!")
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def generate_sql(self, clarified_intent: str, schema_context: SchemaContext) -> SQLExecutorOutput:
        system_prompt = f"""
You are Agent 3 (SQL Executor Agent) in an enterprise Text-to-SQL pipeline.
Your job is to translate clarified user intent into a precise, valid, read-only SQLite query.

DATABASE SCHEMA CONTEXT:
Dialect: {schema_context.database_type}
Available Tables: {schema_context.available_tables}
Table Summaries: {json.dumps(schema_context.table_summaries)}
Foreign Keys: {schema_context.foreign_keys}

CRITICAL RULES & SAFETY CONSTRAINTS:
1. QUERY FORMAT:
   - Must be a read-only query starting with SELECT or WITH (case-insensitive).
   - MUST end with a LIMIT clause (e.g. LIMIT 100) to prevent memory issues.
   - Absolutely NO DDL/DML keywords (DROP, DELETE, INSERT, UPDATE, ALTER, CREATE, PRAGMA, etc.).
   - NO mid-query semicolons or comments (-- or /* */).

2. METADATA TRACKING:
   - 'tables_used': List exact table names referenced in the query (must match Available Tables).
   - 'columns_used': List column names used in SELECT, WHERE, JOIN, GROUP BY, or ORDER BY clauses.
   - 'join_paths': List explicit JOIN paths used (e.g., ['orders JOIN customers ON orders.customer_id = customers.id']).

3. CONFIDENCE SCORE (0 to 100):
   - 0–50: Complex query requiring unverified schema assumptions.
   - 51–85: Standard SQL query using verified foreign key joins.
   - 86–100: Exact 1:1 match with zero ambiguity.

4. STATUS:
   - If SQL generation succeeds: Set status to 'success', populate 'sql_query', 'tables_used', and set 'error_detail' to null.
   - If SQL generation fails: Set status to 'error', set 'sql_query' to null, and explain the failure in 'error_detail'.

Output MUST be strictly valid JSON conforming to the SQLExecutorOutput schema.
"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"RESOLVED INTENT: {clarified_intent}"}
            ],
            response_format={"type": "json_object"}
        )

        raw_json = response.choices[0].message.content
        
        # Enforces regex safety checks and cross-field Pydantic constraints
        validated_output = SQLExecutorOutput.model_validate_json(raw_json)
        return validated_output


# ---------------------------------------------------------------------------
# Execution Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING AGENT 3 (SQL EXECUTOR AGENT)")
    print("=" * 60)

    # 1. Initialize Database & Get Schema Context from Agent 1
    db_mgr = DatabaseManager("sample.db")
    agent1 = SchemaAnalyzerAgent()
    schema_context = agent1.analyze(db_mgr)

    # 2. Instantiate Agent 3
    agent3 = SQLExecutorAgent()

    # Simulated resolved intent from Agent 2
    clarified_intent = "best customer = customer with highest SUM(order_total)"

    print(f"\n[INPUT] Clarified Intent: '{clarified_intent}'")
    
    # 3. Generate SQL
    result = agent3.generate_sql(clarified_intent, schema_context)

    print("\n[OUTPUT] Validated SQLExecutorOutput:")
    print(f"Status: {result.status}")
    print(f"Generated SQL: {result.sql_query}")
    print(f"Tables Used: {result.tables_used}")
    print(f"Columns Used: {result.columns_used}")
    print(f"Join Paths: {result.join_paths}")
    print(f"Confidence Score: {result.confidence_score}")
    print(f"Reasoning: {result.reasoning}")

    # 4. Safely execute the validated query against SQLite via DatabaseManager
    if result.status == "success" and result.sql_query:
        print("\n--- Running Query on SQLite Database ---")
        rows = db_mgr.execute_read_only_query(result.sql_query)
        print(f"Query Execution Result: {rows}")

    print("\n" + "=" * 60)
    print("AGENT 3 COMPLETED SUCCESSFULLY")
    print("=" * 60)