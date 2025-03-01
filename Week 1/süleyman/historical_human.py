import streamlit as st
from langchain.chat_models import ChatOllama
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import AIMessage, HumanMessage

st.title("Historical Human Chat")

llm_figure_selector = ChatOllama(model="gemma:2b")
llm_character = ChatOllama(model="mistral")

first_prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are a historian. Based on the given country, select one notable historical figure. Return only the name."),
    ("human", "Country: {country}"),
])
first_chain = first_prompt_template | llm_figure_selector | StrOutputParser()

second_prompt_template = ChatPromptTemplate.from_messages([
    (
    "system", "You are {figure}, a renowned historical figure. Answer the following question as if you were {figure}."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])
second_chain = second_prompt_template | llm_character | StrOutputParser()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = StreamlitChatMessageHistory()

if "figure" not in st.session_state:
    st.session_state.figure = None

country = st.text_input("Enter a country to select a historical figure:")

if country and st.button("Select Historical Figure"):
    figure = first_chain.invoke({"country": country})
    st.write(f"Selected Figure: {figure}")
    st.session_state.figure = figure
    st.session_state.chat_history.clear()

if st.session_state.figure:
    user_question = st.text_input(f"Ask a question to {st.session_state.figure}:")
    if user_question and st.button("Send"):
        payload = {
            "figure": st.session_state.figure,
            "input": user_question,
            "chat_history": st.session_state.chat_history.messages,
        }
        response = second_chain.invoke(payload)
        st.write(response)

        st.session_state.chat_history.add_message(HumanMessage(content=user_question))
        st.session_state.chat_history.add_message(AIMessage(content=response))
    print(st.session_state.chat_history.messages) # check the message history
