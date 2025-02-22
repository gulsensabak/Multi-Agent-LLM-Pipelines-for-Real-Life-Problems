import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# get the API key from environment
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# create the model
llm = ChatOpenAI(
    model = "gpt-4o",
    api_key = OPENAI_API_KEY,
    temperature = 0
)

# prompt for categorization of the message
analize_prompt = PromptTemplate(
    input_variables=["message"],
    template="""You are a customer support AI.
    Categorize the following customer message into one of these categories:
    [Billing, Technical Issue, Product Inquiry, Complaint]
    Message: {message}
    Answer with only the category name."""
)

# prompt for getting the solution
solution_prompt = PromptTemplate(
    input_variables=["category"],
    template="""You are an expert customer support agent.
    Provide an appropriate solution for a customer issue in the category: {category}.
    If the category is unclear, respond with 'Please provide more details'."""
)

# prompt for creating response
response_prompt = PromptTemplate(
    input_variables=["solution"],
    template="""You are a professional customer service agent.
    Craft a polite and professional response based on this solution:
    {solution}"""
)

# define chains
analize = analize_prompt | llm | StrOutputParser()
solution = solution_prompt | llm | StrOutputParser()
response = response_prompt | llm | StrOutputParser()

# finalize the chain
final_chain = (
    analize 
    | (lambda category: {"category": category}) 
    | solution 
    | (lambda solution: {"solution": solution}) 
    | response
)

# create a title for streamlite
st.title("AI Customer Support System")

# take input from user
message = st.text_area("Enter customer message:")

if message:
    # run the pipeline to get the final response
    final_response = final_chain.invoke({"message": message})
    
    # write the final_response to the webpage
    st.write("### Response to Customer:")
    st.write(final_response)