import streamlit as st
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Modern 2026 standardized imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Fix for the ModuleNotFoundError (using the latest langchain-classic structure)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# 1. LOAD API KEY (Checks .env file first)
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

def get_pdf_text_and_metadata(pdf_docs):
    documents = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        doc = Document(page_content=text, metadata={"source": pdf.name})
        documents.append(doc)
    return documents

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def get_vector_store(text_chunks):
    # Free, local embeddings (no API key needed for this part)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    return vectorstore

def generate_section(vectorstore, theme, subtopic):
    if not GROQ_KEY:
        st.error("Missing Groq API Key! Add 'GROQ_API_KEY' to your .env file.")
        st.stop()

    llm = ChatGroq(
        temperature=0.3, 
        model_name="llama-3.3-70b-versatile",
        groq_api_key=GROQ_KEY
    )

    prompt_template = """
    You are an expert academic writer. 
    Theme: "{theme}"
    Section: "{subtopic}"

    Use ONLY the context below. Cite using [File_Name.pdf].
    Context: {context}

    Write the academic section:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "theme", "subtopic"])
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    response = retrieval_chain.invoke({"input": subtopic, "theme": theme, "subtopic": subtopic})
    return response['answer']

# --- UI ---
st.set_page_config(page_title="Thesis Draft Assistant", layout="wide")
st.title("🎓 Thesis Drafting Assistant (Groq Edition)")

if not GROQ_KEY:
    st.warning("⚠️ API Key not detected. Ensure you have a .env file with GROQ_API_KEY=gsk_...")

with st.sidebar:
    st.header("1. Upload Papers")
    pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
    if st.button("Process"):
        with st.spinner("Indexing..."):
            raw_docs = get_pdf_text_and_metadata(pdf_docs)
            chunks = get_text_chunks(raw_docs)
            st.session_state.vectorstore = get_vector_store(chunks)
            st.success("Done!")

st.header("2. Draft")
theme = st.text_input("Thesis Theme")
subtopics_input = st.text_area("Chapters (One per line)")

if st.button("Generate"):
    if "vectorstore" not in st.session_state:
        st.error("Upload papers first.")
    else:
        subtopics = [s.strip() for s in subtopics_input.split('\n') if s.strip()]
        for subtopic in subtopics:
            with st.spinner(f"Writing {subtopic}..."):
                text = generate_section(st.session_state.vectorstore, theme, subtopic)
                st.subheader(subtopic)
                st.write(text)
