import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- PAGE CONFIG ---
st.set_page_config(page_title="PDF Chat AI", layout="wide")

# --- INITIALIZE SESSION STATE ---
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                content = page.extract_text()
                if content:
                    text += content
        except Exception as e:
            st.error(f"Error reading file: {e}")
    return text

def get_vector_store(text_chunks):
    api_key = st.secrets["GOOGLE_API_KEY"]
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=api_key
    )
    # Direct FAISS instantiation
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

# --- SIDEBAR ---
with st.sidebar:
    st.title("📄 Document Center")
    pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
    
    if st.button("Process"):
        if pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=100
                )
                chunks = text_splitter.split_text(raw_text)
                
                vector_store = get_vector_store(chunks)
                st.session_state.conversation = get_conversation_chain(vector_store)
                st.success("Done!")
        else:
            st.warning("Please upload a file first.")

# --- MAIN CHAT UI ---
st.title("Chat with PDF")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if user_question := st.chat_input("Ask about your PDF:"):
    if st.session_state.conversation is None:
        st.warning("Please upload and process a PDF first.")
    else:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate response
        response = st.session_state.conversation({'question': user_question})
        answer = response['answer']

        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)