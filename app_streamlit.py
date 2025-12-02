import streamlit as st
import os
import tempfile
import time

from rag_core import process_pdf_and_create_vector_store, multimodal_pipeline_pdf_rag_pipeline

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

with st.sidebar:
    st.header("Chat History")
    
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

    if uploaded_file and uploaded_file.name != st.session_state.current_file:
        with st.spinner("Processing Document..."):
            bytes_data = uploaded_file.read()
            rag_output = process_pdf_and_create_vector_store(bytes_data)
            
            if rag_output and rag_output['vector_store']:
                st.session_state.rag_data = rag_output
                st.session_state.current_file = uploaded_file.name
                st.success("Document Ready!")
                st.session_state.messages.append({"role": "assistant", "content": f"I have processed '{uploaded_file.name}'. Ask me anything about it!"})
            else:
                st.error("Failed to process document.")

    st.markdown("---")
    
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
                except Exception as e:
                    response = f"An error occurred: {e}"
                    st.error(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})