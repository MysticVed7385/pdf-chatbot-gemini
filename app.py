import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
import google.generativeai as genai

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="wide"
)

# ------------------ SESSION STATE ------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed" not in st.session_state:
    st.session_state.processed = False

# ------------------ FUNCTIONS ------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


@st.cache_resource
def get_vectorstore(chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    return FAISS.from_texts(chunks, embeddings)


def get_conversation_chain(vectorstore, api_key):
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        temperature=0.4
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )


def handle_user_input(question):
    response = st.session_state.conversation({"question": question})
    st.session_state.chat_history.append(
        {"question": question, "answer": response["answer"]}
    )


# ------------------ UI ------------------
def main():
    st.title("📄 Chat with PDF (Gemini)")
    st.markdown("Upload PDFs and chat with their content using **Google Gemini** 🚀")

    with st.sidebar:
        st.header("📁 Upload PDFs")

        try:
            api_key = st.secrets["GOOGLE_API_KEY"]
            genai.configure(api_key=api_key)
        except:
            st.error("❌ GOOGLE_API_KEY missing in Streamlit Secrets")
            st.stop()

        pdf_docs = st.file_uploader(
            "Upload PDFs",
            type="pdf",
            accept_multiple_files=True
        )

        if st.button("🔄 Process PDFs"):
            if not pdf_docs:
                st.warning("Upload at least one PDF")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(chunks, api_key)
                    st.session_state.conversation = get_conversation_chain(vectorstore, api_key)
                    st.session_state.processed = True
                    st.session_state.chat_history = []
                    st.success("✅ PDFs processed successfully!")

        if st.button("🗑 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    if st.session_state.processed:
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])

        user_question = st.chat_input("Ask something from the PDFs...")
        if user_question:
            with st.chat_message("user"):
                st.write(user_question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    handle_user_input(user_question)
                    st.write(st.session_state.chat_history[-1]["answer"])


if __name__ == "__main__":
    main()