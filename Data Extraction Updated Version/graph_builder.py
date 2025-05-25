from langgraph.graph import StateGraph, END
from supervisor_agent import detect_problem_area
from heart_agent_core import run_heart_assessment
from typing import TypedDict
from lung_agent_core import run_lung_assessment

# 👇 Bu state yapısını LangGraph için kullanıyoruz
class HealthState(TypedDict):
    complaint: str

# Supervisor routing function
def route(state: HealthState):
    complaint = state["complaint"]
    area = detect_problem_area(complaint)
    return area

# Supervisor node: just decides which agent to route to
def supervisor_node(state: HealthState):
    return state  # Supervisor veri değiştirmiyor, routing yapılıyor

def build_graph():
    builder = StateGraph(HealthState)

    # Supervisor node
    builder.add_node("supervisor", supervisor_node)

    # Agent nodes
    builder.add_node("heart", run_heart_assessment)
    builder.add_node("lung", run_lung_assessment)

    # routing function
    builder.add_conditional_edges("supervisor", route, {
        "heart": "heart",
        "lung": "lung"
    })

    # entry and finish points
    builder.set_entry_point("supervisor")
    builder.set_finish_point("heart")
    builder.set_finish_point("lung")

    return builder.compile()
