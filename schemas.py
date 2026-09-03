from pydantic import BaseModel, Field
from typing import Optional, Literal

class QueryAnalysis(BaseModel):
    status: Literal["ambiguous", "clear"] = Field(
        description="Must be 'ambiguous' if terms need user clarification, or 'clear' if ready for SQL execution."
    )
    clarification_question: Optional[str] = Field(
        default=None, 
        description="The precise question to ask the user if status is 'ambiguous'."
    )
    sql_query: Optional[str] = Field(
        default=None, 
        description="The generated SQLite query if status is 'clear'."
    )
    reasoning: Optional[str] = Field(
        default="No reasoning provided.",
        description="Brief explanation of why the query was marked clear or ambiguous based on the schema."
    )