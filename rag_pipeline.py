from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

@st.cache_resource
def build_rag_pipeline(_log_content, google_api_key):
    """
    Builds the RAG pipeline from a string of log content.
    The pipeline is cached to improve performance across Streamlit reruns.
    The underscore in _log_content indicates it's used as a cache key.
    
    Returns: A runnable RetrievalQA chain.
    """
    # Wrap the raw text in a LangChain Document object
    documents = [Document(page_content=_log_content)]

    # Chunk the documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Create embeddings (lightweight, runs locally)
    # This is cached by Streamlit after the first run.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create a FAISS vector store from the documents
    # This is the most computationally intensive part that benefits from caching
    db = FAISS.from_documents(docs, embeddings)
    
    # Define the prompt for the LLM
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

    # Initialize the LLM with a free-tier model
    llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", google_api_key=google_api_key,
                                 temperature=0.1, convert_system_message_to_human=True)

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    return qa_chain
