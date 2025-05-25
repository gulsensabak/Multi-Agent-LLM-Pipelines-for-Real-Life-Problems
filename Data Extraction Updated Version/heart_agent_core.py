import streamlit as st

def run_heart_assessment(state):
    st.session_state["complaint_routed_to"] = "heart"
    st.session_state.complaint = state.get("complaint", "No complaint provided")
    st.rerun()