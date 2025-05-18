import streamlit as st
from graph_builder import build_graph

st.set_page_config(page_title="Multi-Agent Health Checker", page_icon="🧠", layout="centered")
st.title("🧠 Multi-Agent Health Triage Chatbot")

if "complaint" not in st.session_state:
    complaint = st.text_input("Please describe your health concern (e.g., chest pain, shortness of breath)...")
    if complaint:
        st.session_state.complaint = complaint
        graph = build_graph()
        result = graph.invoke({"complaint": complaint})
        st.write("Assessment routed to agent.")
else:
    st.write(f"Complaint: {st.session_state.complaint}")
    st.success("✅ Already triaged.")

    # 👇 Heart agent routation
    if st.session_state.get("complaint_routed_to") == "heart":
        with open("heart_ui.py", "r", encoding="utf-8") as f:
            exec(f.read())
    
    # 👇 Lung agent routation
    if st.session_state.get("complaint_routed_to") == "lung":
        with open("lung_ui.py", "r", encoding="utf-8") as f:
            exec(f.read())
