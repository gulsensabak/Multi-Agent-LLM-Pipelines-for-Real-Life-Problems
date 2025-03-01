import streamlit as st
import pandas as pd
import time
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
import numpy as np
from datetime import datetime
import plotly.express as px

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Test cases with known categories
test_cases = [
    {
        "message": "I was charged twice for my last subscription payment.",
        "expected_category": "Billing",
    },
    {
        "message": "The app keeps crashing when I try to open it.",
        "expected_category": "Technical Issue",
    },
    {
        "message": "What are the differences between your Basic and Premium plans?",
        "expected_category": "Product Inquiry",
    },
    {
        "message": "I've been waiting for support for 3 days and no one has responded!",
        "expected_category": "Complaint",
    },
    {
        "message": "My account password isn't working after the recent update.",
        "expected_category": "Technical Issue",
    }
]

def create_chain(llm):
    """Create a LangChain pipeline for a given LLM."""
    analyze_prompt = PromptTemplate(
        input_variables=["message"],
        template="""You are a customer support AI.
        Categorize the following customer message into one of these categories:
        [Billing, Technical Issue, Product Inquiry, Complaint]
        Message: {message}
        Answer with only the category name."""
    )
    
    return analyze_prompt | llm | StrOutputParser()

def evaluate_model(llm, model_name, test_cases):
    """Evaluate a model's performance on test cases."""
    chain = create_chain(llm)
    results = []
    
    for test_case in test_cases:
        start_time = time.time()
        try:
            prediction = chain.invoke({"message": test_case["message"]}).strip()
            end_time = time.time()
            latency = end_time - start_time
            
            # Calculate accuracy (1 if correct, 0 if incorrect)
            accuracy = 1 if prediction == test_case["expected_category"] else 0
            
            results.append({
                "model": model_name,
                "message": test_case["message"],
                "expected": test_case["expected_category"],
                "predicted": prediction,
                "accuracy": accuracy,
                "latency": latency
            })
        except Exception as e:
            print(f"Error with {model_name}: {str(e)}")
            results.append({
                "model": model_name,
                "message": test_case["message"],
                "expected": test_case["expected_category"],
                "predicted": "ERROR",
                "accuracy": 0,
                "latency": 0
            })
    
    return results

def run_evaluation():
    """Run evaluation on all models and return results."""
    # Initialize models
    models = {
    "GPT-4": ChatOpenAI(model="gpt-4", api_key=OPENAI_API_KEY, temperature=0),
    "Mistral": ChatOllama(model="mistral"),
    "Gemma-2b": ChatOllama(model="gemma:2b"),  # Ensure this is correct
    "Gemma-2": ChatOllama(model="gemma2")  # Corrected model name for Gemma-2
    }

    
    all_results = []
    for model_name, model in models.items():
        results = evaluate_model(model, model_name, test_cases)
        all_results.extend(results)
    
    return pd.DataFrame(all_results)

# Run evaluation and create visualizations
results_df = run_evaluation()

# Calculate metrics by model
model_metrics = results_df.groupby('model').agg({
    'accuracy': 'mean',
    'latency': 'mean'
}).reset_index()

# Create accuracy visualization
fig_accuracy = px.bar(
    model_metrics,
    x='model',
    y='accuracy',
    title='Model Accuracy Comparison',
    labels={'accuracy': 'Accuracy', 'model': 'Model'},
    color='model'
)

# Create latency visualization
fig_latency = px.bar(
    model_metrics,
    x='model',
    y='latency',
    title='Model Latency Comparison',
    labels={'latency': 'Average Latency (seconds)', 'model': 'Model'},
    color='model'
)

# Display results in Streamlit
st.title("LLM Performance Evaluation")

st.subheader("Performance Metrics")
st.dataframe(model_metrics)

st.subheader("Accuracy Comparison")
st.plotly_chart(fig_accuracy)

st.subheader("Latency Comparison")
st.plotly_chart(fig_latency)

st.subheader("Detailed Results")
st.dataframe(results_df)