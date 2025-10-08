import streamlit as st
from faker import Faker
import random
from datetime import datetime, timedelta
import pandas as pd
import os
import time

# --- LangChain & Vector Store Imports ---
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize Faker for synthetic data generation
fake = Faker()

# --- 1. Log Generation ---
def generate_synthetic_logs(num_logs=1000):
    """
    Generates a variety of synthetic security logs and saves them to a file.
    """
    log_types = ['SUCCESSFUL_LOGIN', 'FAILED_LOGIN', 'FILE_ACCESS', 'ERROR', 'BRUTE_FORCE_ATTEMPT']
    ip_addresses = [fake.ipv4() for _ in range(20)] + ['192.168.1.100'] # Add specific IP for targeted queries
    usernames = [fake.user_name() for _ in range(10)]
    filenames = ['/var/www/html/index.html', '/etc/shadow', '/home/user/data.csv', '/app/config.json']

    logs = []
    start_date = datetime.now() - timedelta(days=14) # Two weeks of logs

    for i in range(num_logs):
        timestamp = start_date + timedelta(seconds=random.randint(0, 14*24*60*60))
        log_type = random.choices(log_types, weights=[0.4, 0.2, 0.2, 0.1, 0.1], k=1)[0]
        ip = random.choice(ip_addresses)
        user = random.choice(usernames)
        
        log_entry = f"{timestamp.isoformat()} - {ip} - "
        
        if log_type == 'FAILED_LOGIN':
            log_entry += f"Failed login attempt for user {user}."
        elif log_type == 'SUCCESSFUL_LOGIN':
            log_entry += f"User {user} successfully logged in."
        elif log_type == 'FILE_ACCESS':
            file = random.choice(filenames)
            permission = random.choice(['GRANTED', 'DENIED'])
            log_entry += f"File access {permission} for user {user} on file {file}."
        elif log_type == 'ERROR':
            error_code = random.choice([500, 503, 404, 403])
            log_entry += f"Application error encountered: status_code={error_code}."
        elif log_type == 'BRUTE_FORCE_ATTEMPT':
            log_entry = f"{timestamp.isoformat()} - {ip} - "
            # Simulate multiple quick failed logins for a brute force attack
            for _ in range(random.randint(5, 15)):
                 log_entry += f"Failed login attempt for user {random.choice(usernames)}. "
            log_entry += "Potential brute-force attack detected."

        logs.append(log_entry)

    log_filename = "security_logs.log"
    with open(log_filename, "w") as f:
        for log in logs:
            f.write(log + "\n")
    return log_filename

# --- 2. RAG Pipeline Setup ---
@st.cache_resource
def build_rag_pipeline(log_file_path, google_api_key):
    """
    Builds the RAG pipeline: loads logs, chunks them, creates embeddings,
    and stores them in a FAISS vector store.
    
    Returns: A runnable RetrievalQA chain.
    """
    # Load the log data
    loader = TextLoader(log_file_path)
    documents = loader.load()

    # Chunk the documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Create embeddings (lightweight, runs locally)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create a FAISS vector store and save it locally
    db = FAISS.from_documents(docs, embeddings)
    
    # Define the prompt for the LLM
    # This prompt forces the LLM to cite log snippets.
    prompt_template = """
    You are a security analyst assistant. Your task is to answer questions based ONLY on the provided security log snippets.
    Do not use any external knowledge or make assumptions.
    If the answer is not found in the provided logs, you must state: "I cannot find the answer in the provided logs."

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER (provide a concise answer and cite the relevant log entries directly):
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Initialize the LLM
    # This uses Google's Gemini model. Ensure your API key is set.
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key,
                                 temperature=0.1, convert_system_message_to_human=True)

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 5}), # Retrieve top 5 relevant chunks
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    return qa_chain

# --- 3. Streamlit UI ---

# Page configuration
st.set_page_config(page_title="Security Log Copilot", layout="wide", initial_sidebar_state="expanded")

st.title("🔐 Security Log Copilot with RAG")
st.markdown("""
This tool allows you to ask natural language questions about security logs.
It uses a Retrieval-Augmented Generation (RAG) pipeline to find relevant log entries and generate accurate answers.
""")

# --- Sidebar Setup ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Get Google API Key
    google_api_key = st.text_input("Enter your Google API Key", type="password")
    
    st.info("You can get a free Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey).")
    
    if st.button("Generate New Logs", key="generate_logs"):
        with st.spinner("Generating 1000 synthetic security logs..."):
            generate_synthetic_logs(1000)
            st.success("New logs generated in `security_logs.log`")
            # Give a second to write file before UI might try to reload
            time.sleep(1)


# Check if log file exists, if not, create it
log_filename = "security_logs.log"
if not os.path.exists(log_filename):
    with st.spinner("No logs found. Generating initial synthetic logs..."):
        generate_synthetic_logs(1000)
    st.success(f"Created initial log file: `{log_filename}`")

# Display a sample of the logs
st.subheader("📜 Log File Sample")
try:
    with open(log_filename, "r") as f:
        log_sample = "".join(f.readlines(2000)) # read about 2KB
    st.code(log_sample, language="log", line_numbers=True)
except FileNotFoundError:
    st.error("Log file not found. Please generate logs using the button in the sidebar.")
    st.stop()


# --- Main Copilot Interface ---
st.subheader("🤖 Security Copilot")

# Check for API key before proceeding
if not google_api_key:
    st.warning("Please enter your Google API Key in the sidebar to activate the copilot.")
    st.stop()

# Build the RAG pipeline
try:
    with st.spinner("Initializing RAG Pipeline... (This may take a moment on first run)"):
        qa_chain = build_rag_pipeline(log_filename, google_api_key)
    st.success("RAG Pipeline is ready.")
except Exception as e:
    st.error(f"Failed to build RAG pipeline: {e}")
    st.stop()


# Pre-defined query buttons for easy testing
st.markdown("##### Try some example queries:")
example_queries = [
    "Show me all failed logins from IP 192.168.1.100.",
    "Were there any brute-force attempts yesterday?",
    "Which user had their file access denied?",
    "List all application errors with status code 500.",
    "Who successfully logged in most recently?"
]

query_cols = st.columns(len(example_queries))
for i, query in enumerate(example_queries):
    if query_cols[i].button(query, help=query):
        st.session_state.query = query
    
# User input
user_query = st.text_input("Ask a question about the logs:", key="query", placeholder="e.g., How many brute-force attempts yesterday?")

if user_query:
    with st.spinner("Searching logs and generating answer..."):
        try:
            # Run the query through the RAG chain
            result = qa_chain({"query": user_query})
            answer = result["result"]
            source_docs = result["source_documents"]

            # Display the answer
            st.markdown("### Answer")
            st.info(answer)

            # Display the sources (log snippets) used to generate the answer
            with st.expander("Show Cited Log Snippets"):
                for doc in source_docs:
                    st.code(doc.page_content, language="log")

        except Exception as e:
            st.error(f"An error occurred while processing your query: {e}")
