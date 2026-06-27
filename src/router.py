# Continuing inside src/agents.py
workflow = StateGraph(PatientWorkflowState)

# Injecting our execution blocks
workflow.add_node("summary_agent", patient_summary_node)
workflow.add_node("reminder_agent", reminder_node)
workflow.add_node("escalation_agent", escalation_node)

# Routing rules path construction
workflow.set_entry_point("summary_agent")
workflow.add_edge("summary_agent", "reminder_agent")
workflow.add_edge("reminder_agent", "escalation_agent")
workflow.add_edge("escalation_agent", END)

# Compile running app engine
compiled_engine = workflow.compile()