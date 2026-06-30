"""
Streamlit frontend application for DocMind RAG system.
"""
import streamlit as st
import requests
import json
from typing import Optional
import time

# Configure Streamlit page
st.set_page_config(
    page_title="DocMind RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    api_url = st.text_input(
        "API URL",
        value="http://localhost:8000",
        help="Base URL of the DocMind RAG API"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **DocMind RAG** is a Retrieval-Augmented Generation system
    that combines document retrieval with LLM generation to
    provide accurate, sourced answers to your questions.
    """)


# Main content
st.title("📚 DocMind RAG System")
st.markdown("Document Retrieval-Augmented Generation Interface")

# Create tabs
tab1, tab2, tab3 = st.tabs(["💬 Query", "📤 Upload Documents", "📊 System Status"])

# ============= Query Tab =============
with tab1:
    st.header("Ask Questions About Your Documents")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_area(
            "Enter your question:",
            placeholder="What is FastAPI?",
            height=100,
            key="query_input"
        )
    
    with col2:
        st.markdown("")
        st.markdown("")
        submit_button = st.button("🔍 Search", key="submit_query")
    
    if submit_button and query.strip():
        try:
            with st.spinner("🔄 Processing your query..."):
                response = requests.post(
                    f"{api_url}/api/v1/query",
                    json={
                        "query": query,
                        "top_k": st.slider("Number of sources", 1, 10, 5),
                        "include_sources": True
                    },
                    timeout=30
                )
            
            if response.status_code == 200:
                data = response.json()
                
                # Display answer
                st.markdown("### 🤖 Answer")
                st.markdown(data.get("answer", "No answer generated"))
                
                # Display sources
                if data.get("sources"):
                    st.markdown("### 📚 Sources")
                    for i, source in enumerate(data["sources"], 1):
                        with st.expander(f"Source {i} (Confidence: {source['score']:.2%})"):
                            st.markdown(f"**File:** {source['metadata'].get('filename', 'Unknown')}")
                            st.markdown(f"**Content:** {source['content']}")
                
                # Display metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Processing Time", f"{data.get('processing_time', 0):.2f}s")
                with col2:
                    st.metric("Model Used", data.get("model", "Unknown"))
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to API. Make sure it's running at " + api_url)
        except Exception as e:
            st.error(f"Error: {str(e)}")


# ============= Upload Tab =============
with tab2:
    st.header("Upload Documents")
    st.markdown("Upload PDF, Markdown, or Text files to index them in the RAG system.")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "md", "markdown"],
        help="Supported formats: PDF, TXT, Markdown"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**File:** {uploaded_file.name}")
            st.markdown(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
        
        with col2:
            st.markdown("")
            upload_button = st.button("📤 Upload", key="upload_button")
        
        if upload_button:
            try:
                with st.spinner("📤 Uploading and processing..."):
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    response = requests.post(
                        f"{api_url}/documents/upload",
                        files=files,
                        timeout=60
                    )
                
                if response.status_code == 200:
                    data = response.json()
                    st.success("✅ Document uploaded successfully!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Document ID", data["document_id"][:8] + "...")
                    with col2:
                        st.metric("Chunks Created", data["num_chunks"])
                    with col3:
                        st.metric("Status", data["status"])
                    
                    st.info(data.get("message", "Processing complete"))
                else:
                    st.error(f"Upload failed: {response.status_code} - {response.text}")
            
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to API")
            except Exception as e:
                st.error(f"Error: {str(e)}")


# ============= System Status Tab =============
with tab3:
    st.header("System Status")
    
    if st.button("🔄 Refresh Status"):
        try:
            with st.spinner("Checking system status..."):
                response = requests.get(
                    f"{api_url}/api/v1/health",
                    timeout=5
                )
            
            if response.status_code == 200:
                data = response.json()
                
                # Overall status
                status_color = "🟢" if data["status"] == "healthy" else "🟡"
                st.markdown(f"## {status_color} Status: {data['status'].upper()}")
                st.markdown(f"**Version:** {data.get('version', 'Unknown')}")
                
                # Services status
                st.markdown("### Services")
                services = data.get("services", {})
                
                cols = st.columns(len(services))
                for col, (service, status) in zip(cols, services.items()):
                    with col:
                        status_icon = "✅" if status == "ok" else "❌"
                        st.markdown(f"{status_icon} **{service.replace('_', ' ').title()}**")
            else:
                st.error(f"Failed to get status: {response.status_code}")
        
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Cannot connect to API at {api_url}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.info("Click 'Refresh Status' to check system health")
