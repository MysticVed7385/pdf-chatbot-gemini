import streamlit as st
import time
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Pro PDF AI", layout="wide")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content.encode("utf-8", "ignore").decode("utf-8")
    return text

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    
    # POWERFUL FIX: Batch processing with a delay to avoid 429 Errors
    batch_size = 50 
    vectorstore = None
    
    progress_bar = st.progress(0)
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_texts(texts=batch, embedding=embeddings)
        else:
            vectorstore.add_texts(batch)
        
        # Update progress and wait to respect Rate Limits
        progress = (i + batch_size) / len(text_chunks)
        progress_bar.progress(min(progress, 1.0))
        time.sleep(1) # Safety pause for Free/Pro Tier limits
        
    return vectorstore

def get_conversational_rag_chain(vector_store):
    # Using Gemini 1.5 Pro for the most powerful reasoning
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro", 
        temperature=0.3,
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context:
    <context>
    {context}
    </context>
    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    return create_retrieval_chain(retriever, document_chain)

# --- APP FLOW ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.title("Settings")
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
    if st.button("Build Knowledge Base"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(uploaded_files)
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(raw_text)
            st.session_state.vector_store = get_vector_store(chunks)
            st.session_state.rag_chain = get_conversational_rag_chain(st.session_state.vector_store)
            st.success("Knowledge Base Ready!")

st.title("Gemini 1.5 Pro PDF Chat")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask about your documents..."):
    if "rag_chain" in st.session_state:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        response = st.session_state.rag_chain.invoke({"input": user_input})
        answer = response["answer"]
        
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
    else:
        st.warning("Please build the Knowledge Base first.")