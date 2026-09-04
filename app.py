import streamlit as st
import requests
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Text-to-SQL Enterprise Engine",
    page_icon="⚡",
    layout="wide"
)

API_BASE_URL = "http://127.0.0.1:8000"

st.title("⚡ Enterprise Text-to-SQL Agentic System")
st.caption("Decoupled Multi-Agent Architecture Powered by FastAPI & Groq")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "awaiting_clarification" not in st.session_state:
    st.session_state.awaiting_clarification = False
if "clarification_options" not in st.session_state:
    st.session_state.clarification_options = []

# Sidebar — Health Status & Metadata
with st.sidebar:
    st.header("⚙️ Engine Status")
    try:
        health_resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if health_resp.status_code == 200:
            st.success("Backend API: Online")
            data = health_resp.json()
            st.text(f"Database: {data['database']}")
            st.text(f"Model: {data['model']}")
        else:
            st.error("Backend API: Error")
    except Exception:
        st.error("Backend API: Offline")
        st.info("Start your backend with: `uvicorn main:app --reload`")

    if st.button("Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.session_state.active_session_id = None
        st.session_state.awaiting_clarification = False
        st.session_state.clarification_options = []
        st.rerun()

# Render Historic Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sql" in msg and msg["sql"]:
            st.code(msg["sql"], language="sql")
        if "data" in msg and msg["data"] is not None:
            st.dataframe(msg["data"])

# Clarification Widget Handler
if st.session_state.awaiting_clarification:
    st.warning("⚠️ Clarification Required Before Proceeding")
    selected_option = st.radio(
        "Please select your intended business logic:",
        options=st.session_state.clarification_options,
        key="radio_choice"
    )
    
    additional_notes = st.text_input("Additional filters or context (optional):", key="notes_input")
    
    if st.button("Submit Choice"):
        payload = {
            "session_id": st.session_state.active_session_id,
            "chosen_interpretation": selected_option,
            "additional_context": additional_notes if additional_notes else None
        }
        
        with st.spinner("Resuming pipeline with resolved intent..."):
            try:
                res = requests.post(f"{API_BASE_URL}/clarify", json=payload)
                if res.status_code == 200:
                    result_data = res.json()
                    
                    # Reset clarification lock
                    st.session_state.awaiting_clarification = False
                    
                    # Add clarification response & SQL output to chat history
                    executor_out = result_data["executor_output"]
                    sql_query = executor_out["sql_query"] if executor_out else None
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"**Intent Resolved:** {selected_option}\n\n**Confidence Score:** {result_data['total_confidence']}%",
                        "sql": sql_query,
                        "data": None
                    })
                    st.rerun()
                else:
                    st.error(f"Error resuming query: {res.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {str(e)}")

# Primary User Query Input
if not st.session_state.awaiting_clarification:
    if user_input := st.chat_input("Ask a question about your database..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Agent pipeline processing..."):
                try:
                    payload = {"question": user_input}
                    res = requests.post(f"{API_BASE_URL}/query", json=payload)
                    
                    if res.status_code == 200:
                        result_data = res.json()
                        st.session_state.active_session_id = result_data["session_id"]
                        status = result_data["pipeline_status"]
                        
                        if status == "awaiting_clarification":
                            st.session_state.awaiting_clarification = True
                            clarification_msg = result_data["clarification_output"]["combined_clarification_message"]
                            
                            # Extract radio options
                            ambiguity_details = result_data["clarification_output"]["ambiguities"]
                            options = []
                            if ambiguity_details:
                                options = ambiguity_details[0]["interpretations"]
                            
                            st.session_state.clarification_options = options
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": clarification_msg,
                                "sql": None,
                                "data": None
                            })
                            st.rerun()

                        elif status == "impossible":
                            reason = result_data["clarification_output"]["impossible_reason"]
                            msg = f"❌ **Query Not Supported by Schema:** {reason}"
                            st.markdown(msg)
                            st.session_state.messages.append({"role": "assistant", "content": msg, "sql": None, "data": None})

                        elif status == "completed":
                            executor_out = result_data["executor_output"]
                            sql_query = executor_out["sql_query"]
                            confidence = result_data["total_confidence"]
                            
                            st.markdown(f"**Confidence Score:** {confidence}%")
                            st.code(sql_query, language="sql")
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"**Query Executed Successfully** (Confidence: {confidence}%)",
                                "sql": sql_query,
                                "data": None
                            })

                    else:
                        st.error(f"Backend returned error code {res.status_code}: {res.text}")

                except Exception as e:
                    st.error(f"Connection failure: {str(e)}")