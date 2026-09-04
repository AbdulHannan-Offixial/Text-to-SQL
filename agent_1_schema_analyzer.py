import os
import json
from dotenv import load_dotenv
from groq import Groq
from schemas import SchemaContext
from database_manager import DatabaseManager

load_dotenv()

class SchemaAnalyzerAgent:
    """
    Agent 1: Schema Analyzer
    Inspects the database using DatabaseManager, enriches the schema using LLM 
    reasoning, and outputs a strict SchemaContext matching schemas.py.
    """
    
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY missing from environment!")
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def analyze(self, db_manager: DatabaseManager) -> SchemaContext:
        # Step 1: Programmatically extract raw metadata from SQLite
        raw_context = db_manager.extract_schema_context()
        
        # Step 2: Use LLM to enrich table summaries and flag ambiguous columns
        system_prompt = f"""
You are Agent 1 (Schema Analyzer). Your job is to analyze database metadata and produce a rich, 
concise SchemaContext object.

RAW METADATA:
Available Tables: {raw_context.available_tables}
Foreign Keys: {raw_context.foreign_keys}
Row Counts: {raw_context.row_count_estimates}
Raw Table Summaries: {raw_context.table_summaries}

INSTRUCTIONS:
1. Provide a clear, one-sentence business summary for every table in 'table_summaries'.
2. Identify any column names across the database whose business meaning is inherently ambiguous (e.g. 'status', 'rank', 'type', 'score').
3. Output MUST strictly match the SchemaContext JSON schema.
"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Analyze the schema and output the SchemaContext JSON."}
            ],
            response_format={"type": "json_object"}
        )

        raw_json = response.choices[0].message.content
        
        # Step 3: Parse and validate using schemas.py
        validated_schema_context = SchemaContext.model_validate_json(raw_json)
        return validated_schema_context


# ---------------------------------------------------------------------------
# Execution Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING AGENT 1 (SCHEMA ANALYZER)")
    print("=" * 60)

    db_mgr = DatabaseManager("sample.db")
    agent1 = SchemaAnalyzerAgent()
    
    context = agent1.analyze(db_mgr)
    
    print("\n[OK] Agent 1 generated valid SchemaContext:")
    print(f"Database Type: {context.database_type}")
    print(f"Tables: {context.available_tables}")
    print(f"Table Summaries: {context.table_summaries}")
    print(f"Ambiguous Columns: {context.ambiguous_columns}")
    print(f"Foreign Keys: {context.foreign_keys}")
    print("\n" + "=" * 60)
    print("AGENT 1 COMPLETED SUCCESSFULLY — READY FOR AGENT 2")
    print("=" * 60)