import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import GooglePalm
import google.generativeai as genai
import os

# Page configuration
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed" not in st.session_state:
    st.session_state.processed = False

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files"""
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
    """Split text into chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource
def get_vectorstore(text_chunks, api_key):
    """Create FAISS vector store from text chunks"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, api_key):
    """Create conversational retrieval chain"""
    from langchain.chat_models import ChatGooglePalm
    
    llm = ChatGooglePalm(
        model_name="models/chat-bison-001",
        google_api_key=api_key,
        temperature=0.7
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    
    return conversation_chain

def handle_user_input(user_question):
    """Process user question and generate response"""
    if not st.session_state.processed:
        st.warning("⚠️ Please upload and process PDFs first before asking questions.")
        return
    
    if st.session_state.conversation is None:
        st.error("❌ Conversation chain not initialized. Please process PDFs again.")
        return
    
    try:
        response = st.session_state.conversation({
            "question": user_question
        })
        
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": response["answer"]
        })
        
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def main():
    st.title("📄 Chat with PDF")
    st.markdown("Upload your PDF documents and ask questions about their content!")
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📁 Document Upload")
        
        # Get API key from secrets
        try:
            api_key = st.secrets["GOOGLE_API_KEY"]
            genai.configure(api_key=api_key)
        except Exception as e:
            st.error("❌ GOOGLE_API_KEY not found in secrets. Please add it in Streamlit Cloud settings.")
            st.stop()
        
        pdf_docs = st.file_uploader(
            "Upload your PDFs here",
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if st.button("🔄 Process Documents", use_container_width=True):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing your documents..."):
                    try:
                        # Extract text
                        raw_text = get_pdf_text(pdf_docs)
                        
                        if not raw_text or len(raw_text.strip()) == 0:
                            st.error("❌ No text could be extracted from the PDFs. Please check if they contain readable text.")
                            st.session_state.processed = False
                            return
                        
                        # Create chunks
                        text_chunks = get_text_chunks(raw_text)
                        
                        if not text_chunks:
                            st.error("❌ Failed to create text chunks. Please try again.")
                            st.session_state.processed = False
                            return
                        
                        # Create vector store
                        vectorstore = get_vectorstore(text_chunks, api_key)
                        
                        # Create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore, api_key)
                        
                        st.session_state.processed = True
                        st.session_state.chat_history = []
                        
                        st.success(f"✅ Successfully processed {len(pdf_docs)} PDF(s)!")
                        st.info(f"📊 Created {len(text_chunks)} text chunks for analysis.")
                        
                    except Exception as e:
                        st.error(f"❌ Error during processing: {str(e)}")
                        st.session_state.processed = False
        
        # Display processing status
        st.divider()
        if st.session_state.processed:
            st.success("✅ Ready to answer questions")
        else:
            st.info("⏳ Upload and process PDFs to start")
        
        # Clear conversation button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat interface
    st.divider()
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(message["question"])
        with st.chat_message("assistant"):
            st.write(message["answer"])
    
    # Chat input
    user_question = st.chat_input("Ask a question about your documents...")
    
    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                handle_user_input(user_question)
                if st.session_state.chat_history:
                    st.write(st.session_state.chat_history[-1]["answer"])

if __name__ == "__main__":
    main()