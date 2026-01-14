import streamlit as st
from PyPDF2 import PdfReader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

# --- PAGE CONFIG ---
st.set_page_config(page_title="PDF Chat AI", layout="wide")

# --- CUSTOM CSS FOR CHAT UI ---
st.markdown("""
    <style>
    .stChatMessage { border-radius: 10px; padding: 10px; margin: 5px 0; }
    </style>
""", unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {str(e)}")
    return text

def get_text_chunks(text):
    # Verified stable splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
        separators=["\n\n", "\n", ".", " "]
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    # Verify API Key from st.secrets
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Missing GOOGLE_API_KEY in Streamlit Secrets!")
        return None
        
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=api_key
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

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

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("⚠️ Please upload and process a PDF first.")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)

# --- SIDEBAR ---
with st.sidebar:
    st.title("📄 Document Center")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here", 
        accept_multiple_files=True, 
        type=['pdf']
    )
    
    if st.button("Process Documents"):
        if not pdf_docs:
            st.error("Please upload at least one PDF.")
        else:
            with st.spinner("Analyzing text..."):
                # 1. Extract Text
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.error("The uploaded PDFs contain no readable text.")
                else:
                    # 2. Chunking
                    text_chunks = get_text_chunks(raw_text)
                    # 3. Vectorization
                    vector_store = get_vector_store(text_chunks)
                    # 4. Chain Creation
                    st.session_state.conversation = get_conversation_chain(vector_store)
                    st.success("Ready to chat!")

# --- MAIN CHAT AREA ---
st.title("Chat with PDF (Gemini Pro)")

if "conversation" not in st.session_state:
    st.session_state.conversation = None

user_question = st.chat_input("Ask a question about your documents:")
if user_question:
    handle_userinput(user_question)