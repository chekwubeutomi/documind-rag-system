"""
Streamlit frontend application for DocMind RAG system.

This module provides a web-based user interface for the DocMind RAG system.
Users can:
1. Upload documents to build the knowledge base
2. Ask questions and get AI-generated answers
3. Monitor system health and status

What is Streamlit?
- Python library for building data apps quickly
- Writes Python, Streamlit renders to web UI automatically
- No JavaScript/HTML needed (Python only!)
- Hot-reload: code changes update instantly
- Great for prototyping and dashboards

Architecture:
- Streamlit renders a web page when script runs
- st.* functions render UI elements
- User interactions trigger script re-runs
- Each re-run regenerates the entire page
- Session state persists data across reruns

How It Works:
1. User opens browser to http://localhost:8501
2. Streamlit server runs streamlit_app.py
3. st.* functions render HTML/CSS/JavaScript
4. User sees interactive UI
5. User clicks button → Page re-runs script
6. Script detects button click via st.button()
7. Code in if block executes
8. Page updates with results

Components in This App:
1. Header: Title and description
2. Sidebar: Configuration (API URL)
3. Tabs: Different pages (Query, Upload, Status)
4. Query Tab: Ask questions and get answers
5. Upload Tab: Add documents to RAG system
6. Status Tab: Monitor system health

API Communication:
- Uses requests library to call FastAPI backend
- Endpoints:
  - POST /api/v1/query → Get answer
  - POST /documents/upload → Upload file
  - GET /api/v1/health → Check status
- Error handling: Connection errors, timeouts, HTTP errors

Session State:
- Streamlit has built-in session state (like local storage)
- st.session_state: Access key-value store
- Persists across page reruns
- Example: st.session_state.user_name = "Alice"
"""
import streamlit as st
import requests
import json
from typing import Optional
import time

# ======================================================================
# PAGE CONFIGURATION
# ======================================================================
# Configure the Streamlit page settings (tab title, layout, etc)

st.set_page_config(
    page_title="DocMind RAG",  # Browser tab title
    page_icon="📚",  # Browser tab icon (emoji)
    layout="wide",  # Wide layout (full width) instead of default narrow
    initial_sidebar_state="expanded"  # Sidebar starts open
)

# ======================================================================
# STYLING
# ======================================================================
# Custom CSS to style the Streamlit app

st.markdown("""
    <style>
    .main {
        padding: 2rem;  /* Add spacing around main content */
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;  /* Larger tab labels */
    }
    </style>
""", unsafe_allow_html=True)

# ======================================================================
# SIDEBAR - Configuration
# ======================================================================
# Sidebar is the left panel with settings and info

with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Text input for API URL
    # Users can change this if API runs on different host/port
    api_url = st.text_input(
        "API URL",
        value="http://localhost:8000",  # Default: localhost
        help="Base URL of the DocMind RAG API (e.g., http://localhost:8000)"
    )
    
    # Separator
    st.markdown("---")
    
    # About section
    st.markdown("### About")
    st.markdown("""
    **DocMind RAG** is a Retrieval-Augmented Generation system
    that combines document retrieval with LLM generation to
    provide accurate, sourced answers to your questions.
    
    - **Retrieve:** Find relevant documents
    - **Generate:** Create answers using LLM
    - **Cite:** Show sources for transparency
    """)


# ======================================================================
# MAIN PAGE HEADER
# ======================================================================

st.title("📚 DocMind RAG System")
st.markdown("Document Retrieval-Augmented Generation Interface")

# ======================================================================
# TAB NAVIGATION
# ======================================================================
# Three tabs: Query, Upload, Status
# Users click tabs to switch between pages

tab1, tab2, tab3 = st.tabs(["💬 Query", "📤 Upload Documents", "📊 System Status"])

# ======================================================================
# TAB 1: QUERY
# ======================================================================
# Users ask questions here

with tab1:
    st.header("Ask Questions About Your Documents")
    st.markdown("Type a question and the RAG system will find relevant documents and generate an answer.")
    
    # Layout: question input on left, search button on right
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Text area for user question
        query = st.text_area(
            "Enter your question:",
            placeholder="What is FastAPI?",
            height=100,  # 100 pixels tall
            key="query_input"  # Unique identifier for this widget
        )
    
    with col2:
        # Add spacing to align button with text area
        st.markdown("")
        st.markdown("")
        # Search button
        submit_button = st.button("🔍 Search", key="submit_query")
    
    # When user clicks search button AND has entered text
    if submit_button and query.strip():
        try:
            # Show loading spinner while waiting for API
            with st.spinner("🔄 Processing your query..."):
                # Call FastAPI backend
                response = requests.post(
                    f"{api_url}/api/v1/query",  # API endpoint
                    json={
                        "query": query,  # User's question
                        "top_k": st.slider("Number of sources", 1, 10, 5),  # How many docs to retrieve
                        "include_sources": True  # Return source documents
                    },
                    timeout=30  # Wait max 30 seconds
                )
            
            # If API returned successfully (status 200)
            if response.status_code == 200:
                data = response.json()  # Parse JSON response
                
                # ============================================================
                # Display Answer
                # ============================================================
                st.markdown("### 🤖 Answer")
                st.markdown(data.get("answer", "No answer generated"))
                
                # ============================================================
                # Display Source Documents
                # ============================================================
                if data.get("sources"):
                    st.markdown("### 📚 Sources")
                    # Show each source in an expandable container
                    for i, source in enumerate(data["sources"], 1):
                        # Similarity score as percentage (0.89 = 89%)
                        confidence = source['score'] * 100
                        with st.expander(f"Source {i} (Confidence: {confidence:.0f}%)"):
                            # Show file name
                            st.markdown(f"**File:** {source['metadata'].get('filename', 'Unknown')}")
                            # Show document content
                            st.markdown(f"**Content:** {source['content']}")
                
                # ============================================================
                # Display Metrics
                # ============================================================
                col1, col2 = st.columns(2)
                with col1:
                    # Show how long query took
                    st.metric("Processing Time", f"{data.get('processing_time', 0):.2f}s")
                with col2:
                    # Show which LLM was used
                    st.metric("Model Used", data.get("model", "Unknown"))
            else:
                # API returned error
                st.error(f"Error: {response.status_code} - {response.text}")
        
        except requests.exceptions.ConnectionError:
            # Can't connect to API
            st.error("❌ Could not connect to API. Make sure it's running at " + api_url)
        except Exception as e:
            # Other errors
            st.error(f"Error: {str(e)}")


# ======================================================================
# TAB 2: UPLOAD DOCUMENTS
# ======================================================================
# Users upload files here

with tab2:
    st.header("Upload Documents")
    st.markdown("Upload PDF, Markdown, or Text files to index them in the RAG system.")
    
    # File uploader widget
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "md", "markdown"],  # Allowed file types
        help="Supported formats: PDF, TXT, Markdown"
    )
    
    # When file is uploaded
    if uploaded_file is not None:
        # Layout: file info on left, button on right
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Show file name
            st.markdown(f"**File:** {uploaded_file.name}")
            # Show file size
            st.markdown(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
        
        with col2:
            # Add spacing
            st.markdown("")
            # Upload button
            upload_button = st.button("📤 Upload", key="upload_button")
        
        # When user clicks upload button
        if upload_button:
            try:
                # Show loading spinner
                with st.spinner("📤 Uploading and processing..."):
                    # Prepare file for upload
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    # Call upload endpoint
                    response = requests.post(
                        f"{api_url}/documents/upload",
                        files=files,
                        timeout=60  # Allow up to 60 seconds (file processing takes time)
                    )
                
                # If upload successful
                if response.status_code == 200:
                    data = response.json()
                    
                    # Show success message
                    st.success("✅ Document uploaded successfully!")
                    
                    # Show metrics: document ID, number of chunks, status
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        # Show first 8 characters of UUID (enough to identify)
                        st.metric("Document ID", data["document_id"][:8] + "...")
                    with col2:
                        # How many text chunks were created
                        st.metric("Chunks Created", data["num_chunks"])
                    with col3:
                        # Processing status
                        st.metric("Status", data["status"])
                    
                    # Show status message
                    st.info(data.get("message", "Processing complete"))
                else:
                    # Upload failed
                    st.error(f"Upload failed: {response.status_code} - {response.text}")
            
            except requests.exceptions.ConnectionError:
                # Can't reach API
                st.error("❌ Could not connect to API")
            except Exception as e:
                # Other errors
                st.error(f"Error: {str(e)}")


# ======================================================================
# TAB 3: SYSTEM STATUS
# ======================================================================
# Show system health and service status

with tab3:
    st.header("System Status")
    st.markdown("Monitor the health of DocMind services.")
    
    # Refresh button to check status
    if st.button("🔄 Refresh Status"):
        try:
            # Show loading spinner
            with st.spinner("Checking system status..."):
                # Call health check endpoint
                response = requests.get(
                    f"{api_url}/api/v1/health",
                    timeout=5
                )
            
            # If health check successful
            if response.status_code == 200:
                data = response.json()  # Parse response
                
                # ============================================================
                # Overall Status Display
                # ============================================================
                # Green dot (🟢) for healthy, yellow (🟡) for degraded
                status_color = "🟢" if data["status"] == "healthy" else "🟡"
                st.markdown(f"## {status_color} Status: {data['status'].upper()}")
                st.markdown(f"**Version:** {data.get('version', 'Unknown')}")
                
                # ============================================================
                # Individual Service Status
                # ============================================================
                st.markdown("### Services")
                services = data.get("services", {})
                
                # Show each service in its own column
                # Services: embedding_service, llm_service, vector_db_service
                cols = st.columns(len(services))
                for col, (service, status) in zip(cols, services.items()):
                    with col:
                        # Green checkmark (✅) for ok, red X (❌) for error
                        status_icon = "✅" if status == "ok" else "❌"
                        # Format service name: "embedding_service" → "Embedding Service"
                        service_name = service.replace('_', ' ').title()
                        st.markdown(f"{status_icon} **{service_name}**")
            else:
                # Health check failed
                st.error(f"Failed to get status: {response.status_code}")
        
        except requests.exceptions.ConnectionError:
            # Can't reach API
            st.error(f"❌ Cannot connect to API at {api_url}")
        except Exception as e:
            # Other errors
            st.error(f"Error: {str(e)}")
    else:
        # Before user clicks button
        st.info("Click 'Refresh Status' to check system health")
