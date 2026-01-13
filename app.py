import os
import streamlit as st
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PDF Chatbot - Gemini",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Chat with PDF using Gemini AI")

# ---------------- API KEY ----------------
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("❌ GOOGLE_API_KEY missing in Streamlit secrets")
    st.stop()

# ---------------- FUNCTIONS ----------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


@st.cache_resource
def build_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings()
    return FAISS.from_texts(chunks, embeddings)


def build_chain(vstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.2
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vstore.as_retriever()
    )

# ---------------- SESSION ----------------
if "chain" not in st.session_state:
    st.session_state.chain = None

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.subheader("📂 Upload PDFs")
    pdfs = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process PDFs"):
        if not pdfs:
            st.warning("Upload at least one PDF")
        else:
            with st.spinner("Processing PDFs..."):
                text = get_pdf_text(pdfs)
                if not text.strip():
                    st.error("No text found in PDFs")
                    st.stop()

                chunks = split_text(text)
                vstore = build_vectorstore(chunks)
                st.session_state.chain = build_chain(vstore)
                st.success("PDFs processed successfully")

# ---------------- CHAT ----------------
st.subheader("💬 Ask Questions")

query = st.chat_input("Ask a question from the PDF")

if query:
    if st.session_state.chain is None:
        st.warning("Please upload and process PDFs first")
    else:
        result = st.session_state.chain({
            "question": query,
            "chat_history": st.session_state.history
        })

        st.session_state.history.append((query, result["answer"]))

# ---------------- DISPLAY ----------------
for q, a in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)