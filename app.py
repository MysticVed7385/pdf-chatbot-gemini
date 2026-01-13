import os
import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain.chains.retrieval_qa import RetrievalQA

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PDF Chatbot - Gemini",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Chat with PDF using LangChain & Google Gemini")

# ---------------- API KEY ----------------
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("❌ GOOGLE_API_KEY not found in Streamlit Secrets")
    st.stop()

# ---------------- FUNCTIONS ----------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


@st.cache_resource
def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    return FAISS.from_texts(chunks, embeddings)


def get_qa_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.2
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=False
    )

# ---------------- SESSION STATE ----------------
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.subheader("📂 Upload PDF")
    pdf_docs = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process PDF"):
        if not pdf_docs:
            st.warning("⚠️ Please upload at least one PDF.")
        else:
            with st.spinner("🔍 Processing PDF..."):
                raw_text = get_pdf_text(pdf_docs)

                if not raw_text.strip():
                    st.error("❌ Unable to extract text from PDF")
                    st.stop()

                chunks = get_text_chunks(raw_text)
                vector_store = create_vector_store(chunks)
                st.session_state.qa_chain = get_qa_chain(vector_store)

                st.success("✅ PDF processed successfully!")

# ---------------- CHAT ----------------
st.subheader("💬 Ask Questions")

question = st.chat_input("Ask something about your PDF")

if question:
    if st.session_state.qa_chain is None:
        st.warning("⚠️ Please upload and process a PDF first.")
    else:
        result = st.session_state.qa_chain.run(question)

        st.session_state.chat_history.append(
            (question, result)
        )

# ---------------- DISPLAY CHAT ----------------
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)