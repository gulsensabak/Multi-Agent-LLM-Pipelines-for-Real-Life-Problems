import streamlit as st
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA, create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.title("Ask a Historian")

llm = ChatOllama(model="mistral")

# Load historical figures' data and create a vector store
loader = TextLoader("historical_figures.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="mistral")
vector_store = FAISS.from_documents(split_docs, embeddings)
retriever = vector_store.as_retriever()

rag_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a historian with deep knowledge of historical figures. "
               "Answer questions using the provided information."
               "If the figure is not in the provided information, you should only say 'I don't know' and then do not give any information."
               "{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, rag_prompt_template)
qa_chain = create_stuff_documents_chain(llm, rag_prompt_template)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

history_for_chain = StreamlitChatMessageHistory()
chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history"
)

user_question = st.text_input("Ask about a historical figure:")


if user_question and st.button("Ask"):
    response = chain_with_history.invoke({"input": user_question}, {"configurable": {"session_id": "abc123"}})
    st.write(response["answer"])
    #print(response)

