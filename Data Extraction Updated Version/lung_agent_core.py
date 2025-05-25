import streamlit as st

def run_lung_assessment(state):
    st.session_state["complaint_routed_to"] = "lung"
    st.session_state.complaint = state.get("complaint", "No complaint provided")
    st.rerun()