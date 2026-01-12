import os
import streamlit as st
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain

import google.generativeai as genai


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PDF Chatbot - Gemini",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Chat with PDF using LangChain & Google Gemini")


# ---------------- API KEY ----------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Add it to .env or Streamlit Secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)


# ---------------- FUNCTIONS ----------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
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
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    return FAISS.from_texts(chunks, embeddings)


def get_conversation_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever()
    )


# ---------------- SESSION STATE ----------------
if "conversation" not in st.session_state:
    st.session_state.conversation = None

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
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Processing PDF..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                vector_store = create_vector_store(chunks)
                st.session_state.conversation = get_conversation_chain(vector_store)
                st.success("PDF processed successfully!")


# ---------------- CHAT ----------------
st.subheader("💬 Ask Questions")

question = st.chat_input("Ask something about your PDF")

if question and st.session_state.conversation:
    response = st.session_state.conversation({
        "question": question,
        "chat_history": st.session_state.chat_history
    })
    st.session_state.chat_history.append((question, response["answer"]))


for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
