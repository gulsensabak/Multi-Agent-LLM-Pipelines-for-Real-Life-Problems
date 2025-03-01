# I will use Legal document analysis file

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


# get the api key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# create embeddings
embeddings = OpenAIEmbeddings(api_key = OPENAI_API_KEY)

# create model
llm = ChatOpenAI(model = "gpt-4o", api_key = OPENAI_API_KEY)

# get the document
document = TextLoader("Legal_Document_Analysis_Data.txt").load()

# create splitter
text_splitters = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 20)

# create chunks
chunks = text_splitters.split_documents(document)

# create vector store
vector_store = Chroma.from_documents(chunks, embeddings)

# create retriever
retriever = vector_store.as_retriever()

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an assistant for answering questions.
         Use the provided context to respond. If the answer
         isn't clear, acknowledge that you don't know.
         Limit your response to three concise sentences.
         {context}
         """ ),
         ("human", "{input}")
    ]
)

# The below 2 functions work together to complete the RAG process
# create_stuff_documents_chain is responsible for stuffing whatever data we get from the vector store that is relevant to this prompt, and then sending to the llm
qa_chain = create_stuff_documents_chain(llm, prompt_template)

# retriever retrieves the relevant data for the prompt that came in from the vector store and passes the data to the create_stuff_documents_chain
rag_chain = create_retrieval_chain(retriever, qa_chain)

print("Chat with Document")
question = input("Your question: ")

if question:
    response = rag_chain.invoke({"input": question})
    print(response['answer'])