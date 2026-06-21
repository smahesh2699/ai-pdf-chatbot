import streamlit as st
import os
from dotenv import load_dotenv
import rag_helper

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="InsightDoc - AI-Powered PDF Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design and layout styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');
    
    /* Main body styles */
    .main-body {
        font-family: 'Inter', sans-serif;
    }
    
    /* Elegant Title Banner */
    .title-banner {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #9333ea 100%);
        padding: 2.2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(124, 58, 237, 0.15);
        color: white;
        text-align: center;
    }
    
    .title-banner h1 {
        margin: 0 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        font-size: 2.8rem !important;
        color: white !important;
        letter-spacing: -0.5px;
    }
    
    .title-banner p {
        margin: 0.6rem 0 0 0 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 300 !important;
        font-size: 1.15rem !important;
        opacity: 0.95 !important;
        color: #e9d5ff !important;
    }
    
    /* Welcome cards */
    .welcome-container {
        padding: 1.5rem;
        border-radius: 12px;
        background-color: white;
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
    }
    
    .welcome-card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.25rem;
        margin-top: 1.5rem;
    }
    
    .welcome-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: #f8fafc;
        border: 1px solid #f1f5f9;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .welcome-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.04);
        border-color: #e2e8f0;
    }
    
    .welcome-card h3 {
        margin-top: 0 !important;
        font-family: 'Outfit', sans-serif !important;
        color: #1e1b4b !important;
        font-size: 1.2rem !important;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .welcome-card p {
        font-size: 0.92rem !important;
        color: #475569 !important;
        line-height: 1.5;
        margin-bottom: 0 !important;
    }
    
    /* Sidebar styling refinements */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    
    [data-testid="stSidebar"] .stMarkdown h1, 
    [data-testid="stSidebar"] .stMarkdown h2, 
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #f8fafc !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    [data-testid="stSidebar"] label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
    }
    
    /* Custom divider line */
    .sidebar-divider {
        height: 1px;
        background-color: #334155;
        margin: 1.5rem 0;
    }
    
    /* Metadata and summaries container */
    .metadata-box {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.8rem;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .metadata-item {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 0.4rem;
    }
    
    .metadata-value {
        color: #38bdf8;
        font-weight: 600;
    }
    
    .doc-badge {
        display: inline-block;
        background-color: #334155;
        color: #cbd5e1;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        margin-bottom: 0.4rem;
        max-width: 100%;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    /* Citation styles */
    .citation-box {
        background-color: #f8fafc;
        border-left: 3px solid #6366f1;
        padding: 0.75rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.8rem;
        font-size: 0.88rem;
    }
    
    .citation-source {
        font-weight: 600;
        color: #4338ca;
        margin-bottom: 0.25rem;
    }
    
    .citation-content {
        color: #475569;
        font-style: italic;
    }
    
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # format: [{"role": "user"/"assistant", "content": str, "sources": list}]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "doc_summaries" not in st.session_state:
    st.session_state.doc_summaries = {}
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# --- SIDEBAR: Document Management & Configurations ---

# App Branding
st.sidebar.markdown("<h2 style='text-align: center; color: white; margin-top: 0;'>⚙️ InsightDoc Panel</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.85rem; margin-top: -0.5rem;'>Configure & Upload your PDFs</p>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

# 1. API Key Section
st.sidebar.markdown("### 🔑 Authentication")
env_key_exists = bool(os.getenv("GOOGLE_API_KEY"))

if env_key_exists:
    st.sidebar.caption("✅ Default API Key loaded from system environment")
    api_key_input = st.sidebar.text_input(
        "Override Google API Key",
        type="password",
        help="Input your own Google Gemini API key to override the default key."
    )
    api_key = api_key_input if api_key_input else os.getenv("GOOGLE_API_KEY")
else:
    st.sidebar.warning("⚠️ No environment API key found.")
    api_key_input = st.sidebar.text_input(
        "Enter Google API Key",
        type="password",
        help="Input your Google Gemini API key to run queries."
    )
    api_key = api_key_input

st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

# 2. Document Uploads
st.sidebar.markdown("### 📤 Document Upload")
uploaded_files = st.sidebar.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or multiple PDF documents to build your chatbot database."
)

process_btn = st.sidebar.button(
    "🔄 Process Documents", 
    use_container_width=True,
    disabled=not api_key or not uploaded_files
)

if not api_key:
    st.sidebar.info("👉 Enter an API key above to enable document processing.")
elif not uploaded_files:
    st.sidebar.caption("💡 Upload one or more PDFs to activate.")

# 3. Document Processing Logic
if process_btn:
    with st.spinner("Processing PDF text, generating embeddings, and building FAISS database..."):
        try:
            # Clean and run helper
            vector_store, total_pages, total_chunks, summaries = rag_helper.process_uploaded_files(
                uploaded_files, api_key
            )
            
            st.session_state.vector_store = vector_store
            st.session_state.total_pages = total_pages
            st.session_state.total_chunks = total_chunks
            st.session_state.doc_summaries = summaries
            st.session_state.processed_files = [f.name for f in uploaded_files]
            
            st.sidebar.success("🎉 Database built successfully!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error processing files: {str(e)}")

# 4. Display Processed File Info
if st.session_state.vector_store is not None:
    st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("### 📊 Document Intelligence")
    
    st.sidebar.markdown(f"""
    <div class='metadata-box'>
        <div class='metadata-item'>
            <span>Processed Files</span>
            <span class='metadata-value'>{len(st.session_state.processed_files)}</span>
        </div>
        <div class='metadata-item'>
            <span>Total Pages</span>
            <span class='metadata-value'>{st.session_state.total_pages}</span>
        </div>
        <div class='metadata-item'>
            <span>Vector Chunks</span>
            <span class='metadata-value'>{st.session_state.total_chunks}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Active Files Badges
    st.sidebar.caption("Active Files:")
    for f in st.session_state.processed_files:
        st.sidebar.markdown(f"<div class='doc-badge' title='{f}'>📄 {f}</div>", unsafe_allow_html=True)
    
    # Document Summaries Collapsible Pane
    if st.session_state.doc_summaries:
        st.sidebar.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        with st.sidebar.expander("📝 Document Summaries"):
            for filename, summary in st.session_state.doc_summaries.items():
                st.markdown(f"**{filename}**")
                st.markdown(summary)
                st.markdown("---")
                
    # Download Chat History Button
    if st.session_state.chat_history:
        chat_text = "InsightDoc Chat History\n" + "="*25 + "\n\n"
        for msg in st.session_state.chat_history:
            role_label = "USER" if msg["role"] == "user" else "ASSISTANT"
            chat_text += f"[{role_label}]: {msg['content']}\n"
            if msg.get("sources"):
                chat_text += "Sources Cited:\n"
                for idx, src in enumerate(msg["sources"]):
                    chat_text += f"  - Reference #{idx+1}: {src['source']} (Page {src['page']})\n"
            chat_text += "\n" + "-"*50 + "\n\n"
            
        st.sidebar.download_button(
            label="📥 Download Chat History",
            data=chat_text,
            file_name="chat_history.txt",
            mime="text/plain",
            use_container_width=True
        )
        st.sidebar.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)

    # Clear Session Button
    clear_btn = st.sidebar.button("🧹 Clear Chat & Index", use_container_width=True)
    if clear_btn:
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.session_state.total_pages = 0
        st.session_state.total_chunks = 0
        st.session_state.doc_summaries = {}
        st.session_state.processed_files = []
        rag_helper.clean_temp_directory()
        st.rerun()

# --- MAIN AREA: Conversational Interface ---

# Title Banner
st.markdown("""
<div class='title-banner'>
    <h1>InsightDoc</h1>
    <p>AI-Powered Conversational PDF Intelligence (RAG)</p>
</div>
""", unsafe_allow_html=True)

# Welcome Dashboard (Empty State)
if st.session_state.vector_store is None:
    st.markdown("""
    <div class='welcome-container'>
        <h2 style='margin-top: 0; font-family: "Outfit", sans-serif; color: #1e1b4b;'>Welcome to InsightDoc! 👋</h2>
        <p style='color: #475569;'>
            InsightDoc uses Retrieval-Augmented Generation (RAG) to scan your PDF documents, chunk their content, 
            and retrieve answers directly from the source pages. Get instant answers with page-level citations.
        </p>
        <div class='welcome-card-grid'>
            <div class='welcome-card'>
                <h3>📁 1. Upload Documents</h3>
                <p>Upload one or more PDF files (e.g. reports, textbooks, research papers) in the side panel.</p>
            </div>
            <div class='welcome-card'>
                <h3>⚡ 2. Index Embeddings</h3>
                <p>Click "Process Documents" to chunk the text and store them in the local FAISS vector store.</p>
            </div>
            <div class='welcome-card'>
                <h3>💬 3. Chat & Cite</h3>
                <p>Ask questions and see responses generated by Gemini, complete with exact page-level citations.</p>
            </div>
        </div>
        <p style='margin-top: 1.5rem; font-size: 0.9rem; color: #64748b; font-style: italic;'>
            Ready to start? Add your API key and upload your documents on the left.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Render Chat History
    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])
        
        with st.chat_message(role):
            st.write(content)
            
            # Show citations if present and role is assistant
            if role == "assistant" and sources:
                with st.expander("🔍 Citations & Sources"):
                    # Deduplicate sources for cleaner citation summary
                    seen_citations = set()
                    for idx, src in enumerate(sources):
                        cit_key = f"{src['source']}_page_{src['page']}"
                        if cit_key in seen_citations:
                            continue
                        seen_citations.add(cit_key)
                        
                        st.markdown(f"""
                        <div class='citation-box'>
                            <div class='citation-source'>Reference #{len(seen_citations)}: {src['source']} (Page {src['page']})</div>
                            <div class='citation-content'>\"{src['content'].strip()[:300]}...\"</div>
                        </div>
                        """, unsafe_allow_html=True)

    # Chat Input Box
    user_query = st.chat_input("Ask a question about your uploaded documents...")
    
    if user_query:
        if not api_key:
            st.error("Please enter a Google API Key in the sidebar to ask questions.")
            st.stop()
            
        # Display user query in chat
        with st.chat_message("user"):
            st.write(user_query)
            
        # Add to state chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Process and generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching document index and thinking..."):
                # 1. Similarity Search context
                context, sources = rag_helper.get_relevant_context_and_sources(
                    user_query, st.session_state.vector_store
                )
                
                # 2. Get LLM response
                # We feed the dialog list to keep conversational context
                chat_history_list = [(msg["role"], msg["content"]) for msg in st.session_state.chat_history[:-1]]
                response_text = rag_helper.get_rag_response(
                    user_query, chat_history_list, context, api_key
                )
                
                st.write(response_text)
                
                # Show citations
                if sources:
                    with st.expander("🔍 Citations & Sources"):
                        seen_citations = set()
                        for idx, src in enumerate(sources):
                            cit_key = f"{src['source']}_page_{src['page']}"
                            if cit_key in seen_citations:
                                continue
                            seen_citations.add(cit_key)
                            
                            st.markdown(f"""
                            <div class='citation-box'>
                                <div class='citation-source'>Reference #{len(seen_citations)}: {src['source']} (Page {src['page']})</div>
                                <div class='citation-content'>\"{src['content'].strip()[:300]}...\"</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
            # Add to state chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_text,
                "sources": sources
            })
            
            st.rerun()