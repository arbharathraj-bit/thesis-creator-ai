import streamlit as st
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Modern 2026 standardized imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# --- FINAL FIX FOR LINE 11 & 12 ---
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
def get_pdf_text_and_metadata(pdf_docs):
    """Extracts text from PDFs and attaches the filename as metadata for citations."""
    documents = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        
        # Store as a LangChain document with metadata
        doc = Document(page_content=text, metadata={"source": pdf.name})
        documents.append(doc)
    return documents

def get_text_chunks(documents):
    """Splits the documents into manageable chunks for the AI."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

def get_vector_store(text_chunks):
    """Creates a searchable database of the text chunks."""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    return vectorstore

def generate_section(vectorstore, theme, subtopic):
    """Generates a specific section of the thesis based on the subtopic."""
    # Use gpt-4o or gpt-3.5-turbo depending on your budget
    llm = ChatOpenAI(temperature=0.3, model_name="gpt-4o") 

    # Note the 'input' variable name; create_retrieval_chain expects 'input' by default
    prompt_template = """
    You are an expert European-level Master's Degree academic writer. 
    The overall theme of the thesis is: "{theme}".
    Your task is to write a detailed, highly academic section for the subtopic: "{subtopic}".

    Use ONLY the provided context below to write the section. 
    You MUST cite your sources using Harvard style in-text citations based on the 'source' metadata provided in the context (e.g., [Smith_2020.pdf]).
    If the context does not contain enough information to write a comprehensive section, state what is missing, but do not make up facts.

    Context:
    {context}

    Write the detailed academic section below:
    """

    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "theme", "subtopic"]
    )

    # Retrieval setup
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # Invoke the chain
    response = retrieval_chain.invoke({
        "input": subtopic, 
        "theme": theme, 
        "subtopic": subtopic
    })
    return response['answer']

# --- STREAMLIT UI ---

st.set_page_config(page_title="European Thesis Creator AI", page_icon="🎓", layout="wide")

st.title("🎓 European Master's Thesis Drafting Assistant")
st.markdown("Upload your research papers, define your theme and outline, and the AI will draft your thesis section-by-section with citations.")

with st.sidebar:
    st.header("1. Upload Literature")
    pdf_docs = st.file_uploader("Upload your PDF research papers here", accept_multiple_files=True, type=['pdf'])
    
    if st.button("Process Papers"):
        if pdf_docs:
            with st.spinner("Reading and indexing papers..."):
                raw_docs = get_pdf_text_and_metadata(pdf_docs)
                text_chunks = get_text_chunks(raw_docs)
                vectorstore = get_vector_store(text_chunks)
                # Store in session state so it persists across button clicks
                st.session_state.vectorstore = vectorstore
                st.success("Papers successfully indexed!")
        else:
            st.warning("Please upload PDFs first.")

st.header("2. Define Thesis Scope")
theme = st.text_input("Main Thesis Theme / Title")
subtopics_input = st.text_area("Enter your Subtopics/Chapters (One per line)", height=150)

if st.button("Generate Thesis Draft"):
    if "vectorstore" not in st.session_state:
        st.error("Please upload and process your papers first.")
    elif not theme or not subtopics_input:
        st.error("Please provide both a theme and subtopics.")
    else:
        subtopics = subtopics_input.split('\n')
        subtopics = [s.strip() for s in subtopics if s.strip()]

        st.header("3. Generated Draft")
        full_draft = f"# {theme}\n\n"

        # Generate the thesis section by section to avoid token limits
        for subtopic in subtopics:
            with st.spinner(f"Drafting section: {subtopic}..."):
                section_text = generate_section(st.session_state.vectorstore, theme, subtopic)
                
                # Display to UI
                st.subheader(subtopic)
                st.write(section_text)
                
                # Append to full draft string
                full_draft += f"## {subtopic}\n{section_text}\n\n"
        
        st.success("Drafting Complete!")
        
        # Download button
        st.download_button(
            label="Download Full Draft (.md)",
            data=full_draft,
            file_name="Thesis_Draft.md",
            mime="text/markdown"
        )