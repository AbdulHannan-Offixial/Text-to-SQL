import uuid
import json
from typing import Optional
from schemas import (
    SchemaContext,
    ClarificationOutput,
    SQLExecutorOutput,
    PipelineResult,
    UserClarificationResponse
)
from database_manager import DatabaseManager
from agent_1_schema_analyzer import SchemaAnalyzerAgent
from agent_2_clarification import ClarificationAgent
from agent_3_sql_executor import SQLExecutorAgent


class TextToSQLOrchestrator:
    """
    Main Pipeline Orchestrator.
    Directs data flow across Agent 1 (Schema Analyzer), Agent 2 (Clarification), 
    and Agent 3 (SQL Executor) while enforcing strict state transitions.
    """

    def __init__(self, db_path: str, model_name: str = "openai/gpt-oss-20b"):
        self.db_manager = DatabaseManager(db_path)
        self.agent1 = SchemaAnalyzerAgent(model_name=model_name)
        self.agent2 = ClarificationAgent(model_name=model_name)
        self.agent3 = SQLExecutorAgent(model_name=model_name)

    def process_initial_question(
        self, 
        user_question: str, 
        session_id: Optional[str] = None
    ) -> PipelineResult:
        """
        Processes a fresh natural language query from the start of the pipeline.
        """
        if not session_id:
            session_id = f"sess-{uuid.uuid4().hex[:8]}"

        # Step 1: Agent 1 inspects the database and produces SchemaContext
        schema_context: SchemaContext = self.agent1.analyze(self.db_manager)

        # Step 2: Agent 2 checks for ambiguity or impossibility
        clarification_out: ClarificationOutput = self.agent2.process_query(
            user_question=user_question, 
            schema_context=schema_context
        )

        # Step 3: Conditional Branching based on Clarification Output Status
        if clarification_out.status == "ambiguous":
            total_confidence = clarification_out.confidence_score
            return PipelineResult(
                session_id=session_id,
                original_question=user_question,
                schema_context=schema_context,
                clarification_output=clarification_out,
                executor_output=None,
                pipeline_status="awaiting_clarification",
                total_confidence=total_confidence
            )

        elif clarification_out.status == "impossible":
            total_confidence = clarification_out.confidence_score
            return PipelineResult(
                session_id=session_id,
                original_question=user_question,
                schema_context=schema_context,
                clarification_output=clarification_out,
                executor_output=None,
                pipeline_status="impossible",
                total_confidence=total_confidence
            )

        elif clarification_out.status == "clear":
            # Direct execution path when query is explicit
            executor_out: SQLExecutorOutput = self.agent3.generate_sql(
                clarified_intent=user_question, 
                schema_context=schema_context
            )

            pipeline_status = "completed" if executor_out.status == "success" else "failed"
            total_confidence = int((clarification_out.confidence_score + executor_out.confidence_score) / 2)

            return PipelineResult(
                session_id=session_id,
                original_question=user_question,
                schema_context=schema_context,
                clarification_output=clarification_out,
                executor_output=executor_out,
                pipeline_status=pipeline_status,
                total_confidence=total_confidence
            )

    def resume_with_clarification(
        self, 
        previous_result: PipelineResult, 
        user_response: UserClarificationResponse
    ) -> PipelineResult:
        """
        Resumes a paused pipeline ('awaiting_clarification') using the user's choice.
        """
        if previous_result.pipeline_status != "awaiting_clarification":
            raise ValueError(
                f"Cannot resume pipeline session '{previous_result.session_id}'. "
                f"Current status is '{previous_result.pipeline_status}', expected 'awaiting_clarification'."
            )

        resolved_intent = (
            f"Original Request: {previous_result.original_question}. "
            f"User Selected Clarification: {user_response.chosen_interpretation}."
        )
        if user_response.additional_context:
            resolved_intent += f" Additional Context: {user_response.additional_context}"

        # Update Agent 2 state to 'clear'
        resolved_clarification = ClarificationOutput(
            status="clear",
            ambiguities=[],
            combined_clarification_message=None,
            clarified_intent=resolved_intent,
            impossible_reason=None,
            confidence_score=95,
            reasoning=f"Resolved via user clarification response choice: '{user_response.chosen_interpretation}'."
        )

        # Step 3: Run Agent 3 with the resolved intent
        executor_out: SQLExecutorOutput = self.agent3.generate_sql(
            clarified_intent=resolved_intent,
            schema_context=previous_result.schema_context
        )

        pipeline_status = "completed" if executor_out.status == "success" else "failed"
        total_confidence = int((resolved_clarification.confidence_score + executor_out.confidence_score) / 2)

        return PipelineResult(
            session_id=previous_result.session_id,
            original_question=previous_result.original_question,
            schema_context=previous_result.schema_context,
            clarification_output=resolved_clarification,
            executor_output=executor_out,
            pipeline_status=pipeline_status,
            total_confidence=total_confidence
        )


# ---------------------------------------------------------------------------
# End-to-End Orchestrator Pipeline Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING FULL AGENT ORCHESTRATOR PIPELINE")
    print("=" * 60)

    orchestrator = TextToSQLOrchestrator("sample.db")

    # --- FLOW 1: Ambiguous Query to User Clarification ---
    print("\n--- FLOW 1: Ambiguous Query Processing ---")
    question_1 = "Show me our best customers"
    result_1 = orchestrator.process_initial_question(question_1)

    print(f"Session ID: {result_1.session_id}")
    print(f"Pipeline Status: {result_1.pipeline_status}")
    print(f"Total Confidence: {result_1.total_confidence}")
    if result_1.pipeline_status == "awaiting_clarification":
        print(f"User Message: {result_1.clarification_output.combined_clarification_message}")

        # Simulate User Responding to Clarification Option
        print("\n--- FLOW 1 (Continued): Simulating User Response ---")
        user_reply = UserClarificationResponse(
            session_id=result_1.session_id,
            chosen_interpretation="highest total spend (SUM(order_total))",
            additional_context="Only include orders from 2026."
        )

        final_result_1 = orchestrator.resume_with_clarification(result_1, user_reply)
        print(f"Resumed Status: {final_result_1.pipeline_status}")
        print(f"Generated SQL: {final_result_1.executor_output.sql_query}")
        print(f"Tables Used: {final_result_1.executor_output.tables_used}")

        # Execute query if valid
        if final_result_1.executor_output.sql_query:
            db_rows = orchestrator.db_manager.execute_read_only_query(
                final_result_1.executor_output.sql_query
            )
            print(f"DB Execution Output: {db_rows}")

    # --- FLOW 2: Direct Clear Query ---
    print("\n--- FLOW 2: Clear Query Processing ---")
    question_2 = "List all customer names from the customers table"
    result_2 = orchestrator.process_initial_question(question_2)

    print(f"Pipeline Status: {result_2.pipeline_status}")
    print(f"Generated SQL: {result_2.executor_output.sql_query}")
    print(f"Total Confidence: {result_2.total_confidence}")

    print("\n" + "=" * 60)
    print("ORCHESTRATOR PIPELINE EXECUTED SUCCESSFULLY")
    print("=" * 60)