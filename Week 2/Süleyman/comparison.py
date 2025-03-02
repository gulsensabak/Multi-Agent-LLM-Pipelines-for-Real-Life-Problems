import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_ollama import ChatOllama, OllamaEmbeddings
import time
import pandas as pd

# Load API Key

# Initialize Embeddings and Model
embeddings = OllamaEmbeddings(model="mistral")
llm = ChatOllama(model="mistral")

# Load and Split Document
document = TextLoader("historical_figures.txt").load()
text_splitters = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
chunks = text_splitters.split_documents(document)

# Create Vector Store
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever()

# Define Different Prompting Techniques
prompts = {
    "Zero-Shot": ChatPromptTemplate.from_messages([
        ("system", "You are a historian. Use the provided context to answer. If the document has no information about question. Say only 'I do not know'{context}"),
        ("human", "{input}")
    ]),
    "One-Shot": ChatPromptTemplate.from_messages([
        ("system",
         "Here is an example response: 'In legal terms, a contract is a mutual agreement...'.\nNow answer this query using the context. {context}"),
        ("human", "{input}")
    ]),
    "Few-Shot": ChatPromptTemplate.from_messages([
        ("system",
         "Examples:"
         "1. 'What is a contract?' - 'A contract is a legal agreement between...'\n"
         "2. 'What is a legal precedent?' - 'A legal precedent is...'\nNow, answer: {context}"),
        ("human", "{input}")
    ]),
    "Chain-of-Thought": ChatPromptTemplate.from_messages([
        ("system",
         "Think step-by-step before answering. Analyze legal principles first, then provide a final answer. {context}"),
        ("human", "{input}")
    ]),
    "Self-Consistency": ChatPromptTemplate.from_messages([
        ("system", "Generate multiple answers and return the most consistent one. {context}"),
        ("human", "{input}")
    ]),
    "ReAct": ChatPromptTemplate.from_messages([
        ("system", "1. Retrieve relevant legal information.\n2. Analyze facts.\n3. Answer logically. {context}"),
        ("human", "{input}")
    ]),
    "Contrastive": ChatPromptTemplate.from_messages([
        ("system", "Provide two opposing legal perspectives on this issue. {context}"),
        ("human", "{input}")
    ]),
    "Instruction-Tuned": ChatPromptTemplate.from_messages([
        ("system",
         "Follow these rules:\n- Use legal terminology.\n- Reference laws if applicable.\n- Be concise. {context}"),
        ("human", "{input}")
    ])
}

# Streamlit UI
st.title("RAG Prompting Techniques Comparison")
st.write("Compare different prompting techniques for legal document analysis.")

# User Input
user_question = st.text_input("Enter a question about historical figures:")

if user_question:
    # Create RAG Chain and get response for all prompting techniques
    responses = {}
    times = {}

    for prompt_name, prompt_template in prompts.items():
        start_time = time.time()

        # Create RAG Chain for each prompting technique
        qa_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, qa_chain)

        # Get Response
        response = rag_chain.invoke({"input": user_question})
        responses[prompt_name] = response['answer']

        end_time = time.time()
        times[prompt_name] = end_time - start_time  # Measure time taken

    # Convert times dictionary to a DataFrame
    df_time = pd.DataFrame({
        "Prompting Technique": list(responses.keys()),
        "Time (seconds)": [times[prompt] for prompt in responses.keys()]
    }).sort_values(by="Time (seconds)", ascending=True)  # Sort by time

    # Convert responses dictionary to a DataFrame
    df_responses = pd.DataFrame({
        "Prompting Technique": list(responses.keys()),
        "Generated Response": [responses[prompt] for prompt in responses.keys()]
    })

    # Calculate summary statistics
    min_time = df_time["Time (seconds)"].min()
    max_time = df_time["Time (seconds)"].max()
    avg_time = df_time["Time (seconds)"].mean()
    fastest_method = df_time[df_time["Time (seconds)"] == min_time]["Prompting Technique"].values[0]
    slowest_method = df_time[df_time["Time (seconds)"] == max_time]["Prompting Technique"].values[0]

    # Display Time Analysis Table
    st.subheader("⏳ Time Analysis of Prompting Techniques")
    st.dataframe(df_time)

    # Display Summary
    st.markdown(f"### 📊 Summary:")
    st.markdown(f"- ⏩ **Fastest Technique:** `{fastest_method}` (**{min_time:.4f} seconds**)")
    st.markdown(f"- 🕰️ **Slowest Technique:** `{slowest_method}` (**{max_time:.4f} seconds**)")
    st.markdown(f"- 📉 **Average Time:** `{avg_time:.4f} seconds`")

    # Display Response Table
    st.subheader("💬 Generated Responses by Each Prompting Technique")
    st.dataframe(df_responses)