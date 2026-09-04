import os
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from schemas import PipelineResult, UserClarificationResponse
from orchestrator import TextToSQLOrchestrator

load_dotenv()

# Initialize FastAPI application
app = FastAPI(
    title="Enterprise Text-to-SQL Agentic Engine",
    description="Decoupled, multi-agent REST API pipeline for schema-aware, safe SQL generation.",
    version="1.0.0"
)

# Enable CORS for frontend integration (Streamlit / Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Orchestrator globally
DB_PATH = os.getenv("DATABASE_PATH", "sample.db")
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
orchestrator = TextToSQLOrchestrator(db_path=DB_PATH, model_name=MODEL_NAME)

# In-memory session cache for awaiting_clarification states
# Note: For production scaling across multiple workers, use Redis
SESSION_STORE: dict[str, PipelineResult] = {}


# --- Request DTOs ---

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural language database query.")
    session_id: Optional[str] = Field(default=None, description="Optional custom session identifier.")


class ClarificationRequest(BaseModel):
    session_id: str = Field(..., description="Active session ID awaiting clarification.")
    chosen_interpretation: str = Field(..., description="Selected clarification choice verbatim.")
    additional_context: Optional[str] = Field(default=None, description="Optional extra user notes.")


# --- API Endpoints ---

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """System health & connectivity verification."""
    return {"status": "online", "database": DB_PATH, "model": MODEL_NAME}


@app.post("/query", response_model=PipelineResult, status_code=status.HTTP_200_OK)
def submit_query(payload: QueryRequest):
    """
    Primary endpoint: Takes a natural language query, runs Agent 1 & Agent 2, 
    and either returns a SQL result or halts for clarification.
    """
    try:
        result = orchestrator.process_initial_question(
            user_question=payload.question,
            session_id=payload.session_id
        )
        
        # Cache pipeline result if pipeline halts for clarification
        if result.pipeline_status == "awaiting_clarification":
            SESSION_STORE[result.session_id] = result
            
        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline processing error: {str(e)}"
        )


@app.post("/clarify", response_model=PipelineResult, status_code=status.HTTP_200_OK)
def submit_clarification(payload: ClarificationRequest):
    """
    Resumes a paused session ('awaiting_clarification') using the user's selected choice.
    """
    session_id = payload.session_id
    if session_id not in SESSION_STORE:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session ID '{session_id}' not found or already completed."
        )

    previous_result = SESSION_STORE[session_id]

    try:
        user_response = UserClarificationResponse(
            session_id=session_id,
            chosen_interpretation=payload.chosen_interpretation,
            additional_context=payload.additional_context
        )

        final_result = orchestrator.resume_with_clarification(
            previous_result=previous_result,
            user_response=user_response
        )

        # Clean up session memory once execution finishes
        del SESSION_STORE[session_id]

        return final_result

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(ve)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clarification resume error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server on http://127.0.0.1:8000 ...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)