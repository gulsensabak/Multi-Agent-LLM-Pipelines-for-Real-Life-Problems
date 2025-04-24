import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict
from typing import Dict, Any, Sequence, Annotated
import operator

llm = ChatOllama(model="mistral")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    research_data: str
    summary_data: str
    analysis_data: str
    final_result: str


# Prompts
research_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a historian research expert. Gather the most up-to-date and accurate information about the user's query."),
    ("user", "{query}")
])

summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a master at summarizing. Summarize the given text concisely."),
    ("user", "{research_text}")
])

analyze_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a reliability analyst. Evaluate the sources, consistency and reliability of the information provided."),
    ("user", "{summary_text}")
])

result_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a results formatting expert. Convert summary and analysis results into a format that can be presented to the user."),
    ("user", "Özet: {summary_data}\nAnaliz: {analysis_data}")
])


# Araçlar (Tools) - Streamlit kodlarını kaldırdık
def research_tool(state: AgentState) -> Dict[str, Any]:
    """Conducts research on the user's query."""
    query = state["query"]
    prompt = research_prompt.format(query=query)
    response = llm.invoke(prompt)
    return {"research_data": response.content}


def summarize_tool(state: AgentState) -> Dict[str, Any]:
    """Summarizes the given text."""
    research_text = state["research_data"]
    prompt = summarize_prompt.format(research_text=research_text)
    response = llm.invoke(prompt)
    return {"summary_data": response.content}


def analyze_tool(state: AgentState) -> Dict[str, Any]:
    """Analyzes the reliability of the information given."""
    summary_text = state["summary_data"]
    prompt = analyze_prompt.format(summary_text=summary_text)
    response = llm.invoke(prompt)
    return {"analysis_data": response.content}


def result_tool(state: AgentState) -> Dict[str, Any]:
    """Converts the results into a format that can be presented to the user."""
    summary_data = state["summary_data"]
    analysis_data = state["analysis_data"]
    prompt = result_prompt.format(summary_data=summary_data, analysis_data=analysis_data)
    response = llm.invoke(prompt)
    return {"final_result": response.content}


# Supervisor Ajanı
def supervisor_agent(state: AgentState) -> Dict[str, Any]:
    """It decides which agent will work next."""
    if not state.get("research_data"):
        return {"next": "research"}
    elif not state.get("summary_data"):
        return {"next": "summarize"}
    elif not state.get("analysis_data"):
        return {"next": "analyze"}
    else:
        return {"next": "result"}


# Creating workflow
def create_workflow():
    workflow = StateGraph(AgentState)


    workflow.add_node("research", research_tool)
    workflow.add_node("summarize", summarize_tool)
    workflow.add_node("analyze", analyze_tool)
    workflow.add_node("result", result_tool)
    workflow.add_node("supervisor", supervisor_agent)


    workflow.set_entry_point("supervisor")
    workflow.add_edge("research", "supervisor")
    workflow.add_edge("summarize", "supervisor")
    workflow.add_edge("analyze", "supervisor")
    workflow.add_edge("result", END)

    conditional_map = {
        "research": "research",
        "summarize": "summarize",
        "analyze": "analyze",
        "result": "result"
    }
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

    return workflow.compile()


# Streamlit UI
st.title("📚 AI Research Assistant")
st.write("A multi-stage research AI assistant using LangGraph.")

# User input
query = st.text_input("Enter the topic you want to research")

if st.button("RESEARCH"):
    if query:
        try:
            # Create placeholders to display progress
            status_container = st.empty()
            research_container = st.container()
            summary_container = st.container()
            analysis_container = st.container()
            result_container = st.container()

            with status_container:
                st.info("Workflow is invoking")

            # a new workflow
            app = create_workflow()

            # Initial state
            inputs = {
                "messages": [HumanMessage(content=query)],
                "query": query,
                "research_data": "",
                "summary_data": "",
                "analysis_data": "",
                "final_result": ""
            }

            with status_container:
                st.info("İş akışı invoke ediliyor...")

                st.write("Supervisor is running. State: research_data=False, summary_data=False, analysis_data=False")
                st.write("Supervisor: directing to research tool")

            result = app.invoke(inputs)

            # Results
            status_container.success("Results are ready!")


            with research_container:
                st.subheader("📝 Reearch Results")
                with st.expander("Research Details", expanded=True):
                    st.write(result.get("research_data", "No research data generated."))

            # Summary
            with summary_container:
                st.subheader("📌 Summary")
                st.write(result.get("summary_data", "Result of summarization not generated."))

            # analysis
            with analysis_container:
                st.subheader("🔍 Analysis of Reliability")
                st.write(result.get("analysis_data", "No analysis output generated."))

            # final output
            with result_container:
                st.subheader("📊 Result")
                st.write(result.get("final_result", "No final output generated."))

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
    else:
        st.warning("Please tell me any historical figure that you have want to learn")