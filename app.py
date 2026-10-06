"""
app.py - BuildMate AI Streamlit Web Application Interface
Part of BuildMate AI - Intelligent Construction Site Assistant
"""

import os
import streamlit as st
from rag_engine import (
    extract_text_from_file,
    chunk_text,
    generate_embeddings,
    generate_query_embedding,
    query_gemini_assistant
)
from vector_store import LocalVectorStore
from calculator import calculate_concrete_volume

# Page Configuration
st.set_page_config(
    page_title="BuildMate AI - Construction Site Assistant",
    page_icon="🏗️",
    layout="wide"
)

# Initialize Session State
if "vector_store" not in st.session_state:
    st.session_state.vector_store = LocalVectorStore()
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False
if "doc_name" not in st.session_state:
    st.session_state.doc_name = ""
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: API Key and Document Management
st.sidebar.title("🏗️ BuildMate AI")
st.sidebar.caption("Intelligent Construction Site Assistant v1.0")
st.sidebar.divider()

api_key = st.sidebar.text_input("Enter Google Gemini API Key:", type="password", help="Requires a valid Google AI Studio Gemini API Key")

st.sidebar.subheader("📄 Construction Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload PDF or TXT Document", type=["pdf", "txt"])

if uploaded_file and st.sidebar.button("Index Document", type="primary"):
    if not api_key:
        st.sidebar.error("Please enter a Google Gemini API Key first.")
    else:
        with st.sidebar.status("Processing document...", expanded=True) as status:
            st.write("Extracting text and page metadata...")
            pages = extract_text_from_file(uploaded_file.getvalue(), uploaded_file.name)
            
            st.write("Partitioning into semantic chunks...")
            chunks, metadata = chunk_text(pages)
            
            st.write("Generating Gemini embeddings (768-dim)...")
            embeds = generate_embeddings(chunks, api_key)
            
            st.session_state.vector_store.clear()
            st.session_state.vector_store.add_chunks(chunks, embeds, metadata)
            st.session_state.doc_loaded = True
            st.session_state.doc_name = uploaded_file.name
            st.session_state.chunk_count = len(chunks)
            status.update(label="Document Indexed Successfully!", state="complete", expanded=False)

if st.session_state.doc_loaded:
    st.sidebar.success(f"✓ Active File: **{st.session_state.doc_name}**")
    st.sidebar.info(f"Total Chunks: **{st.session_state.chunk_count}**")
    if st.sidebar.button("Clear Vector Store"):
        st.session_state.vector_store.clear()
        st.session_state.doc_loaded = False
        st.session_state.chat_history = []
        st.rerun()

# Main Application Views: Tabs
tab_chat, tab_calc, tab_about = st.tabs([
    "💬 RAG Construction Assistant",
    "📐 Concrete Quantity Calculator",
    "ℹ️ Architecture & System Info"
])

# TAB 1: RAG Assistant
with tab_chat:
    st.header("Site Engineering Q&A Assistant")
    st.caption("Ask questions grounded strictly in uploaded structural drawings, specifications, and safety codes.")

    if not st.session_state.doc_loaded:
        st.warning("⚠️ Please upload and index a construction document (PDF/TXT) from the sidebar to begin.")
    
    # Display Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Query Input
    user_query = st.chat_input("Ask about concrete grade, curing periods, waterproofing, safety codes...")
    if user_query:
        if not api_key:
            st.error("Please provide your Google Gemini API Key in the sidebar.")
        elif not st.session_state.doc_loaded:
            st.error("Please upload and index a document first.")
        else:
            # Display user message
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # RAG Search and Synthesis
            with st.chat_message("assistant"):
                with st.spinner("Searching document & formulating answer..."):
                    q_embed = generate_query_embedding(user_query, api_key)
                    retrieved = st.session_state.vector_store.search(q_embed, top_k=3, threshold=0.35)
                    answer = query_gemini_assistant(user_query, retrieved, api_key)
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# TAB 2: Quantity Calculator
with tab_calc:
    st.header("Construction Concrete Quantity Calculator")
    st.caption("Deterministic mathematical field utility for computing nominal concrete volume (m³).")

    col1, col2 = st.columns(2)
    with col1:
        elem_type = st.selectbox(
            "Structural Element Type",
            ["Column (Rectangular/Square)", "Beam", "Suspended Slab", "Isolated Footing"]
        )
        qty = st.number_input("Quantity (Number of identical units)", min_value=1, value=4, step=1)
    
    with col2:
        length = st.number_input("Length (L in metres)", min_value=0.01, value=0.30, step=0.05, format="%.2f")
        width = st.number_input("Width / Breadth (W in metres)", min_value=0.01, value=0.30, step=0.05, format="%.2f")
        depth = st.number_input("Depth / Height / Thickness (D in metres)", min_value=0.01, value=3.00, step=0.05, format="%.2f")

    st.divider()
    vol = calculate_concrete_volume(qty, length, width, depth)
    
    st.subheader("Calculation Result")
    res_col1, res_col2 = st.columns([2, 1])
    with res_col1:
        st.success(f"### Total Nominal Volume: **{vol:.3f} m³**")
        st.markdown(f"**Formula**: $\\text{{Volume}} = {qty} \\times {length:.2f} \\text{{ m}} \\times {width:.2f} \\text{{ m}} \\times {depth:.2f} \\text{{ m}} = {vol:.3f} \\text{{ m}}^3$")
    with res_col2:
        st.info("Ready for Ready-Mix Concrete (RMC) order verification.")

# TAB 3: About & Architecture
with tab_about:
    st.header("BuildMate AI Architecture")
    st.markdown("""
    **Core Technologies Used:**
    - **Language Model**: Google Gemini (gemini-1.5-flash) via Google GenAI SDK
    - **Vector Embeddings**: Google text-embedding-004 (768-dimensional float32)
    - **Vector Store**: In-Memory NumPy matrix with Cosine Similarity search
    - **Text Parsing**: PyPDF page-by-page extraction
    - **Interface**: Streamlit
    - **Math Engine**: Deterministic Python volume module (zero LLM reliance)
    """)
