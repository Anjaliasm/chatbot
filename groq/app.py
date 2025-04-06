import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

# Load the Groq API key
groq_api_key = os.environ.get("GROQ_API_KEY")

st.title("ChatGroq with Llama3")
st.write("Ask questions based on the uploaded documents.")

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# Define the prompt template
prompt_template = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Question: {input}
""")

def vector_embedding():
    """Loads PDFs, splits text, and creates vector embeddings."""
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="llama2")
        st.session_state.loader = PyPDFDirectoryLoader("./docs")  # Ensure this folder contains PDFs
        st.session_state.docs = st.session_state.loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
        st.session_state.final_documents = text_splitter.split_documents(st.session_state.docs[:10])
        
        # Create FAISS vector store
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.write("✅ Vector database is ready!")

# Button to initialize document embeddings
if st.button("Documents Embedding"):
    vector_embedding()

# User input
prompt_input = st.text_input("Enter your question here")

# Ensure vectors exist before running retrieval
if "vectors" in st.session_state:
    retriever = st.session_state.vectors.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    if prompt_input:
        start = time.process_time()
        response = retrieval_chain.invoke({"input": prompt_input})
        st.write("### Answer:")
        st.write(response["answer"])

        # Show document similarity search results
        with st.expander("🔍 Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(f"**Relevant Document {i+1}:**")
                st.write(doc.page_content)
                st.write("—" * 50)
else:
    st.warning("⚠️ Please click 'Documents Embedding' first to load PDFs.")
