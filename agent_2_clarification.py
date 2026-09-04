import os
import json
from dotenv import load_dotenv
from groq import Groq
from schemas import SchemaContext, ClarificationOutput
from database_manager import DatabaseManager
from agent_1_schema_analyzer import SchemaAnalyzerAgent

load_dotenv()

class ClarificationAgent:
    """
    Agent 2: Clarification Agent
    Analyzes the user's natural language request against the SchemaContext.
    
    Determines status:
    - 'ambiguous': Identifies unclear business terms and outputs multiple-choice questions.
    - 'impossible': Halts pipeline if requested data falls completely outside the schema.
    - 'clear': Passes control to Agent 3 (SQL Executor).
    """

    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY missing from environment!")
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def process_query(self, user_question: str, schema_context: SchemaContext) -> ClarificationOutput:
        system_prompt = f"""
You are Agent 2 (Clarification Agent) in an enterprise Text-to-SQL pipeline.
Your primary role is to intercept vague or unexecutable user prompts BEFORE any SQL is written.

DATABASE SCHEMA CONTEXT:
Database Dialect: {schema_context.database_type}
Available Tables: {schema_context.available_tables}
Table Summaries: {json.dumps(schema_context.table_summaries)}
Foreign Keys: {schema_context.foreign_keys}
Known Ambiguous Columns: {schema_context.ambiguous_columns}

INSTRUCTIONS & RULES:
1. Evaluate the user's natural language question against the database schema.

2. STATUS DETERMINATION:
   - 'impossible': If the user asks for entity data or metrics that DO NOT exist in the provided tables.
   - 'ambiguous': If the question uses subjective terms (e.g., "best", "top", "active", "recent") or column names that lack explicit mathematical formulas.
   - 'clear': If the user's request explicitly maps 1:1 to clear database tables and columns without subjective metrics.

3. CONFIDENCE SCORE (0 to 100):
   - 0–20: Set when status is 'impossible'.
   - 21–60: Set when status is 'ambiguous'.
   - 61–100: Set when status is 'clear'.

4. OUTPUT CONTRACT COMPLIANCE:
   - If status == 'ambiguous': You MUST populate 'ambiguities' (with term, 2-5 interpretations, and suggested_question) AND 'combined_clarification_message'. 'impossible_reason' MUST be null.
   - If status == 'impossible': You MUST populate 'impossible_reason'. 'ambiguities' MUST be an empty list [] and 'combined_clarification_message' MUST be null.
   - If status == 'clear': 'ambiguities' MUST be [], 'combined_clarification_message' MUST be null, and 'impossible_reason' MUST be null.

Output strictly valid JSON matching the ClarificationOutput schema.
"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"USER QUESTION: {user_question}"}
            ],
            response_format={"type": "json_object"}
        )

        raw_json = response.choices[0].message.content
        
        # Parse and enforce strict Pydantic cross-field validation rules from schemas.py
        validated_output = ClarificationOutput.model_validate_json(raw_json)
        return validated_output


# ---------------------------------------------------------------------------
# Execution Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING AGENT 2 (CLARIFICATION AGENT)")
    print("=" * 60)

    # 1. Initialize Database and Agent 1 to get SchemaContext
    db_mgr = DatabaseManager("sample.db")
    agent1 = SchemaAnalyzerAgent()
    schema_context = agent1.analyze(db_mgr)

    # 2. Instantiate Agent 2
    agent2 = ClarificationAgent()

    # TEST CASE 1: Ambiguous Query
    print("\n--- TEST 1: Ambiguous Query ---")
    question_1 = "Show me our best customers"
    res1 = agent2.process_query(question_1, schema_context)
    print(f"User Question: '{question_1}'")
    print(f"Status: {res1.status}")
    print(f"Confidence Score: {res1.confidence_score}")
    print(f"Clarification Message: {res1.combined_clarification_message}")
    print(f"Reasoning: {res1.reasoning}\n")

    # TEST CASE 2: Clear Query
    print("--- TEST 2: Clear Query ---")
    question_2 = "List all customer names and ids from the customers table"
    res2 = agent2.process_query(question_2, schema_context)
    print(f"User Question: '{question_2}'")
    print(f"Status: {res2.status}")
    print(f"Confidence Score: {res2.confidence_score}")
    print(f"Reasoning: {res2.reasoning}\n")

    # TEST CASE 3: Impossible Query
    print("--- TEST 3: Impossible Query ---")
    question_3 = "What is the weather forecast in Tokyo today?"
    res3 = agent2.process_query(question_3, schema_context)
    print(f"User Question: '{question_3}'")
    print(f"Status: {res3.status}")
    print(f"Confidence Score: {res3.confidence_score}")
    print(f"Impossible Reason: {res3.impossible_reason}")
    print(f"Reasoning: {res3.reasoning}")

    print("\n" + "=" * 60)
    print("AGENT 2 COMPLETED ALL TEST CASES SUCCESSFULLY")
    print("=" * 60)