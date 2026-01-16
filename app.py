"""
ULTIMATE PDF Q&A Application - Best of Both Worlds
Combines: Modern LangChain + Rate Limit Handling + Error Handling + Latest Libraries
"""

import streamlit as st
import time
from pypdf import PdfReader  # Latest pypdf (not PyPDF2)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Configure page
st.set_page_config(
    page_title="Pro PDF AI - Gemini Powered",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS for modern UI
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
    """
    Extract text from uploaded PDF files with error handling
    
    Args:
        pdf_docs: List of uploaded PDF files
        
    Returns:
        str: Extracted text from all PDFs
    """
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                content = page.extract_text()
                if content:
                    # Handle encoding issues
                    text += content.encode("utf-8", "ignore").decode("utf-8")
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {str(e)}")
            continue
    
    return text


def get_text_chunks(raw_text, chunk_size=1000, chunk_overlap=200):
    """
    Split text into chunks for processing
    
    Args:
        raw_text: Input text
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
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
    
    This is the CRITICAL improvement - handles Google API rate limits!
    
    Args:
        text_chunks: List of text chunks
        api_key: Google API key
        
    Returns:
        FAISS vector store
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    
    # POWERFUL FIX: Batch processing with delay to avoid 429 errors
    batch_size = 50  # Process 50 chunks at a time
    vectorstore = None
    
    # Show progress to user
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            
            # Update status
            status_text.text(f"Processing chunks {i+1} to {min(i+batch_size, len(text_chunks))} of {len(text_chunks)}...")
            
            # Create or add to vector store
            if vectorstore is None:
                vectorstore = FAISS.from_texts(texts=batch, embedding=embeddings)
            else:
                vectorstore.add_texts(batch)
            
            # Update progress
            progress = min((i + batch_size) / len(text_chunks), 1.0)
            progress_bar.progress(progress)
            
            # CRITICAL: Wait to respect rate limits
            if i + batch_size < len(text_chunks):
                time.sleep(1)  # 1 second delay between batches
        
        status_text.text("Vector store created successfully! ✅")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
        return vectorstore
        
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        if "429" in str(e):
            st.error("Rate limit exceeded. Try reducing the number of PDFs or wait a moment.")
        return None


def get_conversational_rag_chain(vector_store, api_key):
    """
    Create RAG chain using modern LangChain pattern
    
    Args:
        vector_store: FAISS vector store
        api_key: Google API key
        
    Returns:
        Retrieval chain
    """
    # Use Gemini 1.5 Pro for best reasoning
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.3,
        google_api_key=api_key
    )
    
    # Enhanced prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are an intelligent assistant helping users understand PDF documents.
    Answer the question based ONLY on the provided context. If the answer is not in the context,
    say "I cannot find this information in the provided documents."
    
    <context>
    {context}
    </context>
    
    Question: {input}
    
    Provide a clear, comprehensive answer:
    """)
    
    # Create chains using modern LangChain pattern
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain


def main():
    """Main application"""
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    
    # Sidebar - Configuration
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # API Key input with secrets fallback
        api_key = None
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
            st.success("✅ API Key loaded from secrets")
        else:
            api_key = st.text_input(
                "Google API Key",
                type="password",
                help="Get your key from https://makersuite.google.com/app/apikey"
            )
            if not api_key:
                st.warning("⚠️ Please provide your API key")
        
        st.markdown("---")
        
        # File Upload
        st.header("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            accept_multiple_files=True,
            type=['pdf'],
            help="Upload one or more PDF documents"
        )
        
        # Process button
        if st.button("🚀 Build Knowledge Base", use_container_width=True):
            if not api_key:
                st.error("Please provide an API key!")
            elif not uploaded_files:
                st.error("Please upload at least one PDF file!")
            else:
                with st.spinner("Processing your documents..."):
                    try:
                        # Extract text
                        st.info("📖 Extracting text from PDFs...")
                        raw_text = get_pdf_text(uploaded_files)
                        
                        if not raw_text.strip():
                            st.error("No text found in PDFs. Please check your files.")
                            return
                        
                        # Split into chunks
                        st.info("✂️ Splitting text into chunks...")
                        chunks = get_text_chunks(raw_text)
                        st.success(f"Created {len(chunks)} text chunks")
                        
                        # Create vector store with rate limit protection
                        st.info("🔮 Creating embeddings (this may take a moment)...")
                        vector_store = get_vector_store(chunks, api_key)
                        
                        if vector_store:
                            # Create RAG chain
                            st.info("🔗 Building RAG chain...")
                            rag_chain = get_conversational_rag_chain(vector_store, api_key)
                            
                            # Store in session state
                            st.session_state.vector_store = vector_store
                            st.session_state.rag_chain = rag_chain
                            st.session_state.processed_files = [f.name for f in uploaded_files]
                            
                            st.success("✅ Knowledge Base Ready! Start asking questions below.")
                        else:
                            st.error("Failed to create knowledge base. Please try again.")
                            
                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
        
        # Show processed files
        if st.session_state.processed_files:
            st.markdown("---")
            st.subheader("📚 Loaded Documents")
            for filename in st.session_state.processed_files:
                st.text(f"✓ {filename}")
        
        st.markdown("---")
        
        # Tech stack info
        with st.expander("🔧 Tech Stack"):
            st.code("pypdf==6.6.0")
            st.code("langchain==1.2.4")
            st.code("langchain-google-genai==4.2.0")
            st.code("faiss-cpu==1.13.2")
            st.code("streamlit==1.53.0")
    
    # Main Content - Chat Interface
    st.title("🚀 Gemini 1.5 Pro PDF Chat")
    st.markdown("### Ask questions about your PDF documents")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask about your documents..."):
        if "rag_chain" in st.session_state:
            # Add user message to chat
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
                        
                        # Add assistant response to chat
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer
                        })
                        
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg
                        })
        else:
            st.warning("⚠️ Please build the Knowledge Base first using the sidebar!")
    
    # Show helpful info when no knowledge base exists
    if "rag_chain" not in st.session_state:
        st.info("""
        👈 **Get Started:**
        1. Enter your Google API key (or add to secrets)
        2. Upload your PDF documents
        3. Click "Build Knowledge Base"
        4. Start asking questions!
        """)
        
        with st.expander("💡 Sample Questions"):
            st.markdown("""
            - What is the main topic of this document?
            - Summarize the key findings
            - What are the conclusions mentioned?
            - Explain [specific concept] from the document
            - List the important points discussed in section X
            """)


if __name__ == "__main__":
    main()