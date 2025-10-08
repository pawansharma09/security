import streamlit as st
from utils import parse_uploaded_log_file
from rag_pipeline import build_rag_pipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="Security Log Copilot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Title and Description ---
st.title("🔐 Security Log Copilot")
st.markdown("""
Welcome to your intelligent security assistant. 
Upload your security logs in `.txt`, `.log`, `.pdf`, or `.docx` format, and start asking questions.
""")

# --- Main Application Logic ---
google_api_key = st.secrets["GOOGLE_API_KEY"]

if not google_api_key:
    st.warning("Please add your Google API Key to the Streamlit secrets to activate the copilot.")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader(
    "Upload your log file",
    type=["txt", "log", "pdf", "docx"],
    help="Upload a file containing security logs to begin your analysis."
)

# --- Session State Management ---
# Reset session if a new file is uploaded
if uploaded_file and st.session_state.get("last_uploaded_filename") != uploaded_file.name:
    st.session_state["last_uploaded_filename"] = uploaded_file.name
    if "rag_pipeline" in st.session_state:
        del st.session_state["rag_pipeline"]
    if "log_content" in st.session_state:
        del st.session_state["log_content"]

if uploaded_file:
    # Process and display logs only if not already in session state
    if "log_content" not in st.session_state:
        with st.spinner(f"Parsing '{uploaded_file.name}'..."):
            log_content = parse_uploaded_log_file(uploaded_file)
            if log_content:
                st.session_state.log_content = log_content
                st.success("File parsed successfully!")
            else:
                st.error("Failed to parse the uploaded file. Please try another file.")
                st.stop()

    # Build RAG pipeline if it doesn't exist for the current logs
    if "rag_pipeline" not in st.session_state and "log_content" in st.session_state:
        with st.spinner("Indexing logs and building RAG pipeline... This may take a moment."):
            try:
                # The log content is used as a cache key for the pipeline
                st.session_state.rag_pipeline = build_rag_pipeline(
                    st.session_state.log_content,
                    google_api_key
                )
                st.success("RAG Pipeline is ready.")
            except Exception as e:
                st.error(f"Failed to build RAG pipeline: {e}")
                st.stop()
    
    # --- Interactive Copilot UI (only shows after successful pipeline build) ---
    if "rag_pipeline" in st.session_state:
        st.subheader("🤖 Security Copilot")
        st.markdown("##### Try some example queries:")
        
        example_queries = [
            "Show me all failed logins.",
            "Were there any brute-force attempts?",
            "Which user had their file access denied?",
            "List all application errors.",
        ]

        query_cols = st.columns(len(example_queries))
        for i, query in enumerate(example_queries):
            if query_cols[i].button(query, help=query):
                st.session_state.query = query

        user_query = st.text_input(
            "Ask a question about the logs:",
            key="query",
            placeholder="e.g., Show me failed logins from IP 192.168.1.100."
        )

        if user_query:
            with st.spinner("Searching logs and generating answer..."):
                try:
                    qa_chain = st.session_state.rag_pipeline
                    result = qa_chain.invoke({"query": user_query})
                    answer = result.get("result")
                    source_docs = result.get("source_documents")

                    st.markdown("### Answer")
                    st.info(answer)

                    if source_docs:
                        with st.expander("Show Cited Log Snippets"):
                            for doc in source_docs:
                                st.code(doc.page_content, language="log")

                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {e}")
else:
    st.info("Please upload a log file to get started.")

