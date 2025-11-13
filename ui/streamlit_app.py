"""
Simple Streamlit UI for interactive demonstration.
Run: streamlit run ui/streamlit_app.py
"""

import streamlit as st
from uuid import uuid4
from pipeline import HybridRetrievalPipeline

# Page config
st.set_page_config(
    page_title="Hybrid Retrieval System",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = HybridRetrievalPipeline(db_name="hyper_kb_ui")
    st.session_state.session_id = str(uuid4())
    st.session_state.messages = []

pipeline = st.session_state.pipeline
session_id = st.session_state.session_id

# Title
st.title("🔍 Hybrid Retrieval System")
st.caption("Conversational AI with Drift Handling")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    top_k = st.slider("Retrieved Contexts", 1, 10, 5)
    use_adaptive = st.checkbox("Adaptive Retrieval", value=True)
    
    st.divider()
    st.header("Session Info")
    st.text(f"Session: {session_id[:8]}...")
    
    stats = pipeline.get_statistics()
    st.metric("Total Interactions", stats['kb_stats']['total_interactions'])
    st.metric("Active Sessions", stats['active_sessions'])
    
    if st.button("New Session"):
        st.session_state.session_id = str(uuid4())
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.header("Chat")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        if msg["role"] == "assistant" and "metadata" in msg:
            with st.expander("View Context Details"):
                st.json(msg["metadata"])

# Chat input
if query := st.chat_input("Ask a question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.write(query)
    
    # Get enhanced prompt
    with st.spinner("Retrieving context..."):
        result = pipeline.query(
            query,
            session_id,
            top_k=top_k,
            use_adaptive=use_adaptive
        )
    
    # Simulate response
    response = f"Retrieved {result['retrieved_contexts']} contexts. In production, this would call the LLM with the enhanced prompt."
    
    # Store interaction
    pipeline.process_interaction(query, response, session_id)
    
    # Add assistant message
    metadata = {
        "retrieved_contexts": result['retrieved_contexts'],
        "session_turn": result['session_turn'],
        "shift_detected": result['drift_state']['state']['last_shift_turn'] == result['session_turn'] if result['drift_state']['state'] else False
    }
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "metadata": metadata
    })
    
    with st.chat_message("assistant"):
        st.write(response)
        
        with st.expander("View Context Details"):
            st.json(metadata)
            
            st.subheader("Enhanced Prompt")
            st.text_area("Prompt", result['enhanced_prompt'], height=300)

# Footer
st.divider()
st.caption("Hybrid Retrieval System - Research Project")