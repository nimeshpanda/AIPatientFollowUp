from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from src.rag_engine import ClinicalRAGEngine
from dotenv import load_dotenv
import os

# Global Memory State Interface
class PatientWorkflowState(TypedDict):
    patient_id: str
    patient_name: str
    scenario_type: str                  # Appointment, Discharge, Lab, Vaccine, Risk
    raw_input_data: Dict[str, Any]      
    extracted_profile: str              
    outbound_channel: str               # Enforced: SMS or Email
    generated_message: str              
    patient_reply: str                  
    is_escalated: bool                  
    escalation_reason: str              

# Initialize the Gemini LLM Layer
try:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.1, google_api_key=api_key)
except Exception:
    # Fallback mock LLM for local development when GOOGLE_API_KEY isn't set
    class _MockResp:
        def __init__(self, content: str):
            self.content = content

    class MockLLM:
        def invoke(self, prompt: str):
            return _MockResp("MOCKED RESPONSE")

    llm = MockLLM()
rag = ClinicalRAGEngine()

# Node 1: Patient Summary Agent
def patient_summary_node(state: PatientWorkflowState) -> Dict[str, Any]:
    raw_text = state["raw_input_data"].get("raw_discharge_summary", "")
    prompt = f"Extract key medical guidelines, drugs, and upcoming dates from this text:\n\n{raw_text}"
    response = llm.invoke(prompt)
    return {"extracted_profile": response.content}

# Node 2: Reminder Agent
def reminder_node(state: PatientWorkflowState) -> Dict[str, Any]:
    profile = state["extracted_profile"]
    channel = state["outbound_channel"]
    
    # Context Injection via our LangChain RAG pipeline
    context = rag.query_guidelines(state["scenario_type"])
    
    prompt = f"""
    Using this profile: {profile}
    And these rules: {context}
    Draft a personalized message for the patient via {channel}.
    Constraint: If channel is SMS, it MUST be brief and under 160 characters.
    """
    response = llm.invoke(prompt)
    return {"generated_message": response.content}

# Node 3: Escalation Agent (Incorporating Nikhila's safety metrics)
def escalation_node(state: PatientWorkflowState) -> Dict[str, Any]:
    reply = state["patient_reply"].lower()
    
    # Concrete high-risk keyword arrays mapped to the prompt execution
    prompt = f"""
    Analyze this patient reply: "{reply}"
    Does it indicate an emergency risk situation such as severe pain, bleeding, fever, or breathing issues? 
    Respond with exactly 'ESCALATE: <reason>' or 'NORMAL'.
    """
    response = llm.invoke(prompt).content.strip()
    
    if "ESCALATE" in response:
        return {"is_escalated": True, "escalation_reason": response}
    return {"is_escalated": False, "escalation_reason": "No anomalies flagged."}


# Minimal compiled engine wrapper used by the Streamlit app
class CompiledEngine:
    def __init__(self):
        # ensure RAG KB initialized if available
        try:
            rag.initialize_knowledge_base()
        except Exception:
            pass

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Ensure keys exist with defaults
        state.setdefault("extracted_profile", "")
        state.setdefault("generated_message", "")
        state.setdefault("patient_reply", "")
        state.setdefault("is_escalated", False)
        state.setdefault("escalation_reason", "")

        # Step 1: extract profile if missing
        if not state["extracted_profile"]:
            out = patient_summary_node(state)
            state.update(out)

        # Step 2: generate reminder if missing
        if not state["generated_message"]:
            out = reminder_node(state)
            state.update(out)

        # Step 3: analyze patient reply if provided
        if state.get("patient_reply"):
            out = escalation_node(state)
            state.update(out)
        else:
            state["is_escalated"] = False
            state["escalation_reason"] = state.get("escalation_reason", "No anomalies flagged.")

        return state


# Export a ready-to-use instance expected by `src/app.py`
compiled_engine = CompiledEngine()