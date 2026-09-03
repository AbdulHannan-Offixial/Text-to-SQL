import os
import json
from dotenv import load_dotenv
from groq import Groq
from schema import QueryAnalysis
from schema_extractor import extract_database_schema

# Automatically load environment variables from .env file
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found! Ensure it is set inside your .env file.")

# Initialize Groq client
client = Groq(api_key=api_key)

COGNITIVE_VERIFIER_SYSTEM_PROMPT = """
You are an expert SQL analysis agent. Your job is to analyze user prompts against a database schema.

SCHEMA:
{schema_text}

INSTRUCTIONS:
1. Examine the user's natural language request.
2. Check for ambiguous or subjective terms (e.g., "best", "top", "recent", "valuable") that lack explicit metrics.
3. If ambiguous, set "status" to "ambiguous", formulate a "clarification_question" asking the user to define the metric (e.g., total revenue vs total order count), and set "sql_query" to null.
4. If explicit and clear, set "status" to "clear", provide the valid SQLite query in "sql_query", and set "clarification_question" to null.
5. Return JSON strictly matching the QueryAnalysis schema.
"""

def test_query(user_prompt: str, schema_text: str):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": COGNITIVE_VERIFIER_SYSTEM_PROMPT.format(schema_text=schema_text)},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    raw_json = response.choices[0].message.content
    parsed_output = QueryAnalysis.model_validate_json(raw_json)
    return parsed_output

if __name__ == "__main__":
    schema = extract_database_schema("sample.db")
    
    # Test 1: Ambiguous Query
    print("--- Test 1: Ambiguous Query ---")
    res1 = test_query("Show me our best customers", schema)
    print(f"Status: {res1.status}")
    print(f"Question: {res1.clarification_question}")
    print(f"Reasoning: {res1.reasoning}\n")
    
    # Test 2: Clear Query
    print("--- Test 2: Clear Query ---")
    res2 = test_query("List all customers with total_spend greater than 500", schema)
    print(f"Status: {res2.status}")
    print(f"SQL: {res2.sql_query}")
    print(f"Reasoning: {res2.reasoning}")