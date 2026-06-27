import json
import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path so `from src.agents` works when running from `src/`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agents import compiled_engine

st.set_page_config(page_title="AI Patient Assistant Dashboard", layout="wide")
st.title("🏥 AI Patient Follow-Up Assistant Control Panel")

# Try multiple possible paths for patient_records.json
possible_paths = [
    Path(__file__).resolve().parents[1] / "data" / "patient_records.json",
    Path(__file__).resolve().parent / "data" / "patient_records.json",
    Path(__file__).resolve().parent / "patient_records.json",
    Path("data") / "patient_records.json",
    Path("patient_records.json"),
]
DATA_FILE_PATH = next((p for p in possible_paths if p.exists()), possible_paths[0])


def load_patient_records():
    if not DATA_FILE_PATH.exists():
        return []

    with DATA_FILE_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload
    return []


def build_initial_state(patient_data):
    return {
        "patient_id": patient_data.get("patient_id", "MOCK-ID"),
        "patient_name": patient_data.get("patient_name", "Jane Doe"),
        "scenario_type": patient_data.get("clinical_scenario_type", "Appointment Follow-Ups"),
        "raw_input_data": {"raw_discharge_summary": patient_data.get("dosage_instructions", "")},
        "outbound_channel": patient_data.get("preferred_channel", "SMS"),
        "patient_reply": "",
        "is_escalated": False,
        "escalation_reason": "",
    }


patient_records = load_patient_records()
selected_record = None
selected_index = 0

# Left Column: System Execution Input Console
with st.sidebar:
    st.header("Step 1: Load Patient Dataset")

    if not patient_records:
        st.error(f"No patient records found at {DATA_FILE_PATH.name}.")
    else:
        st.caption(f"Loaded {len(patient_records)} patient records from the local JSON file.")
        selected_index = st.selectbox(
            "Choose patient record",
            range(len(patient_records)),
            format_func=lambda idx: (
                f"{patient_records[idx].get('patient_id', 'N/A')} - "
                f"{patient_records[idx].get('patient_name', 'Unknown')} "
                f"({patient_records[idx].get('clinical_scenario_type', 'Unknown')})"
            ),
        )
        selected_record = patient_records[selected_index]

# Main Application Window Layout
col1, col2 = st.columns(2)

if selected_record is not None:
    record_key = selected_record.get("patient_id", "") or selected_record.get("patient_name", "")
    if st.session_state.get("active_patient_key") != record_key:
        initial_state = build_initial_state(selected_record)
        with st.spinner("Processing workflows..."):
            output = compiled_engine.invoke(initial_state)
            st.session_state["current_run_state"] = output
            st.session_state["active_patient_key"] = record_key

    with col1:
        st.subheader("📋 Selected Record Information")
        st.json(selected_record)

        if st.button("Re-run Agent Sequence"):
            initial_state = build_initial_state(selected_record)
            with st.spinner("Processing workflows..."):
                output = compiled_engine.invoke(initial_state)
                st.session_state["current_run_state"] = output
                st.session_state["active_patient_key"] = record_key
                st.success("Summary & Reminder Generated!")

    if "current_run_state" in st.session_state:
        current_state = st.session_state["current_run_state"]

        with col1:
            st.subheader("🧠 Extracted Summary")
            st.write(current_state.get("extracted_profile", "No summary available yet."))

            st.info(f"**Dispatched Reminder Output ({current_state['outbound_channel']}):**")
            st.write(current_state.get("generated_message", "No reminder generated yet."))

        with col2:
            st.subheader("📱 Patient Response Simulator")
            patient_text_input = st.text_input("Enter inbound simulated patient reply:")

            if st.button("Submit Reply to Escalation Agent"):
                current_state["patient_reply"] = patient_text_input
                updated_output = compiled_engine.invoke(current_state)
                st.session_state["current_run_state"] = updated_output

                if updated_output["is_escalated"]:
                    st.error(f"🚨 ALERT: System Escalated!\nReason: {updated_output['escalation_reason']}")
                else:
                    st.success("✅ Interaction Logged. Patient stable within threshold parameters.")