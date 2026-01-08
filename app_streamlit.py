import streamlit as st
import os
import tempfile
import time
import hashlib

from rag_core import (
    process_pdf_and_create_vector_store, 
    multimodal_pipeline_pdf_rag_pipeline,
    save_vector_store,
    load_vector_store,
    list_saved_documents
)

st.set_page_config(
    page_title="RAG AI Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #FAFAFA;
    }
    
    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #262730;
    }

    /* Chat Input Styling */
    .stChatInput {
        border-radius: 20px;
    }

    /* Header Styling */
    h1 {
        font-family: 'Helvetica', sans-serif;
        font-weight: 700;
        color: white;
    }
    
    /* Custom button styling to look like 'New User' pill */
    .user-pill {
        display: inline-block;
        background-color: #a3b8cc; 
        color: #0e1117;
        padding: 5px 15px;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.9em;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_data" not in st.session_state:
    st.session_state.rag_data = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None

if "document_id" not in st.session_state:
    st.session_state.document_id = None

def generate_document_id(filename: str) -> str:
    """Generate a unique document ID based on filename and timestamp."""
    unique_string = f"{filename}_{time.time()}"
    return hashlib.md5(unique_string.encode()).hexdigest()

with st.sidebar:
    st.header("📚 Document Management")
    
    # Show saved documents
    saved_docs = list_saved_documents()
    if saved_docs:
        st.subheader("💾 Saved Documents")
        selected_doc = st.selectbox(
            "Load a saved document:",
            options=["-- Select --"] + saved_docs,
            key="saved_doc_selector"
        )
        
        if selected_doc != "-- Select --" and selected_doc != st.session_state.document_id:
            with st.spinner("Loading saved document..."):
                loaded_data = load_vector_store(selected_doc)
                if loaded_data:
                    st.session_state.rag_data = loaded_data
                    st.session_state.document_id = selected_doc
                    st.session_state.current_file = selected_doc
                    st.success(f"✅ Loaded: {selected_doc}")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"I have loaded the saved document '{selected_doc}'. Ask me anything about it!"
                    })
                    st.rerun()
                else:
                    st.error("Failed to load document.")
    
    st.markdown("---")
    st.subheader("📤 Upload New Document")
    
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

    if uploaded_file and uploaded_file.name != st.session_state.current_file:
        with st.spinner("Processing Document..."):
            bytes_data = uploaded_file.read()
            
            # Generate document ID
            doc_id = generate_document_id(uploaded_file.name)
            
            # Process PDF
            rag_output = process_pdf_and_create_vector_store(bytes_data)
            
            if rag_output and rag_output['vector_store']:
                st.session_state.rag_data = rag_output
                st.session_state.current_file = uploaded_file.name
                st.session_state.document_id = doc_id
                
                # Save to disk for persistence
                try:
                    save_vector_store(
                        rag_output['vector_store'],
                        rag_output['image_data_store'],
                        doc_id
                    )
                    st.success(f"✅ Document Ready & Saved! (ID: {doc_id[:8]}...)")
                except Exception as e:
                    st.warning(f"Document processed but save failed: {e}")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"I have processed '{uploaded_file.name}'. Ask me anything about it!"
                })
            else:
                st.error("Failed to process document.")

    st.markdown("---")
    st.subheader("💬 Chat History")
    
    if st.session_state.messages:
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                display_text = (msg["content"][:40] + '..') if len(msg["content"]) > 40 else msg["content"]
                st.caption(f"Q: {display_text}")

    if st.button("Clear Chat", type="primary"):
        st.session_state.messages = []
        st.rerun()




st.title("RAG based AI Chat Model")
st.markdown('<div class="user-pill">Hey How Can I Help You?</div>', unsafe_allow_html=True)

for message in st.session_state.messages:
    if message["role"] == "user":
        avatar = "🔴" 
    else:
        avatar = "🟡" 
        
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

prompt = st.chat_input("What is your question?")

if prompt:
    with st.chat_message("user", avatar="🔴"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.rag_data is None:
        response = "Please upload a PDF document in the sidebar to start chatting."
        with st.chat_message("assistant", avatar="🟡"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        with st.chat_message("assistant", avatar="🟡"):
            with st.spinner("Thinking..."):
                try:
                    result = multimodal_pipeline_pdf_rag_pipeline(
                        query=prompt,
                        vector_store=st.session_state.rag_data['vector_store'],
                        image_data_store=st.session_state.rag_data['image_data_store']
                    )
                    response = result['answer']
                    st.markdown(response)
                except ValueError as e:
                    response = f"⚠️ Processing Error: {str(e)}"
                    st.error(response)
                except ConnectionError as e:
                    response = "🔌 Connection Error: Unable to reach the AI service. Please check your internet connection and try again."
                    st.error(response)
                except Exception as e:
                    response = f"❌ Unexpected Error: {str(e)}. Please try again or contact support if the issue persists."
                    st.error(response)
                    logging.exception("Unexpected error in query processing")
        
        st.session_state.messages.append({"role": "assistant", "content": response})