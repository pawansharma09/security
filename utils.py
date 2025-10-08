import streamlit as st
import pypdf
import docx
from io import StringIO

def parse_uploaded_log_file(uploaded_file):
    """
    Parses the content of an uploaded file based on its type.
    Supports .txt, .log, .pdf, and .docx files.

    Args:
        uploaded_file: The file object uploaded via Streamlit.

    Returns:
        A string containing the extracted text content of the file, or None if the file type is unsupported.
    """
    if uploaded_file is None:
        return None

    file_extension = uploaded_file.name.split('.')[-1].lower()

    try:
        if file_extension in ["txt", "log"]:
            # To convert to a string based IO object
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            return stringio.read()

        elif file_extension == "pdf":
            pdf_reader = pypdf.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text

        elif file_extension == "docx":
            doc = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text

        else:
            st.error(f"Unsupported file type: .{file_extension}")
            return None
            
    except Exception as e:
        st.error(f"Error parsing file '{uploaded_file.name}': {e}")
        return None
