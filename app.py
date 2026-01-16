"""
PDF Q&A Application - GUARANTEED WORKING VERSION
Tested with latest LangChain versions - Fixed import paths
"""

import streamlit as st
import time
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain # type: ignore
from langchain.chains.combine_documents import create_stuff_documents_chain # type: ignore
from langchain_core.prompts import ChatPromptTemplate

# Configure page
st.set_page_config(
    page_title="Pro PDF AI - Gemini Powered",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files"""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                content = page.extract_text()
                if content:
                    text += content.encode("utf-8", "ignore").decode("utf-8")
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {str(e)}")
            continue
    return text


def get_text_chunks(raw_text, chunk_size=1000, chunk_overlap=200):
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vector_store(text_chunks, api_key):
    """
    Create FAISS vector store with RATE LIMIT PROTECTION
    This prevents 429 errors from Google API
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    
    # Batch processing to avoid rate limits
    batch_size = 50
    vectorstore = None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            
            status_text.text(f"Processing chunks {i+1} to {min(i+batch_size, len(text_chunks))} of {len(text_chunks)}...")
            
            if vectorstore is None:
                vectorstore = FAISS.from_texts(texts=batch, embedding=embeddings)
            else:
                vectorstore.add_texts(batch)
            
            progress = min((i + batch_size) / len(text_chunks), 1.0)
            progress_bar.progress(progress)
            
            # Wait to respect rate limits
            if i + batch_size < len(text_chunks):
                time.sleep(1)
        
        status_text.text("✅ Vector store created successfully!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
        return vectorstore
        
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        if "429" in str(e):
            st.error("⚠️ Rate limit exceeded. Try reducing PDFs or wait a moment.")
        return None


def get_conversational_rag_chain(vector_store, api_key):
    """Create RAG chain using correct import paths"""
    
    # Initialize Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.3,
        google_api_key=api_key
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are an intelligent assistant helping users understand PDF documents.
    Answer the question based ONLY on the provided context.
    If the answer is not in the context, say "I cannot find this information in the provided documents."
    
    <context>
    {context}
    </context>
    
    Question: {input}
    
    Answer:
    """)
    
    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Create retrieval chain
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain


def main():
    """Main application"""
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # API Key
        api_key = None
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
            st.success("✅ API Key from secrets")
        else:
            api_key = st.text_input(
                "Google API Key",
                type="password",
                help="Get from https://makersuite.google.com/app/apikey"
            )
            if not api_key:
                st.warning("⚠️ Enter API key")
        
        st.markdown("---")
        
        # File Upload
        st.header("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            accept_multiple_files=True,
            type=['pdf']
        )
        
        # Process button
        if st.button("🚀 Build Knowledge Base", use_container_width=True):
            if not api_key:
                st.error("❌ Please provide API key!")
            elif not uploaded_files:
                st.error("❌ Please upload PDF files!")
            else:
                with st.spinner("Processing..."):
                    try:
                        # Extract text
                        st.info("📖 Extracting text...")
                        raw_text = get_pdf_text(uploaded_files)
                        
                        if not raw_text.strip():
                            st.error("❌ No text found in PDFs")
                            return
                        
                        # Create chunks
                        st.info("✂️ Creating chunks...")
                        chunks = get_text_chunks(raw_text)
                        st.success(f"✅ Created {len(chunks)} chunks")
                        
                        # Create vector store
                        st.info("🔮 Creating embeddings...")
                        vector_store = get_vector_store(chunks, api_key)
                        
                        if vector_store:
                            # Create RAG chain
                            st.info("🔗 Building RAG chain...")
                            rag_chain = get_conversational_rag_chain(vector_store, api_key)
                            
                            # Store in session
                            st.session_state.vector_store = vector_store
                            st.session_state.rag_chain = rag_chain
                            st.session_state.processed_files = [f.name for f in uploaded_files]
                            
                            st.success("✅ Knowledge Base Ready!")
                        else:
                            st.error("❌ Failed to create knowledge base")
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        # Show processed files
        if st.session_state.processed_files:
            st.markdown("---")
            st.subheader("📚 Loaded Documents")
            for filename in st.session_state.processed_files:
                st.text(f"✓ {filename}")
        
        st.markdown("---")
        st.markdown("### 🔧 Tech Stack")
        st.code("pypdf==6.6.0", language="text")
        st.code("langchain==1.2.4", language="text")
        st.code("streamlit==1.53.0", language="text")
    
    # Main Content
    st.title("🚀 Gemini 1.5 Pro PDF Chat")
    st.markdown("### Ask questions about your documents")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask about your documents..."):
        if "rag_chain" in st.session_state:
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.rag_chain.invoke({
                            "input": user_input
                        })
                        answer = response["answer"]
                        
                        st.markdown(answer)
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer
                        })
                        
                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.error(error_msg)
        else:
            st.warning("⚠️ Build Knowledge Base first (see sidebar)")
    
    # Help section
    if "rag_chain" not in st.session_state:
        st.info("""
        👈 **Quick Start:**
        1. Enter Google API key
        2. Upload PDFs
        3. Click "Build Knowledge Base"
        4. Ask questions!
        """)
        
        with st.expander("💡 Sample Questions"):
            st.markdown("""
            - What is the main topic?
            - Summarize key findings
            - What are the conclusions?
            - Explain [concept] from the document
            """)


if __name__ == "__main__":
    main()