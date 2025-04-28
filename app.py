import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import VectorStoreRetrieverMemory
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from datetime import datetime
import re

from chatbot import memory_system, chain

st.title("Bioengineering Lab Chatbot")
st.caption("Enhanced with timestamped memory and metadata")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you today?", "timestamp": datetime.now().isoformat()}]

# Display messages with timestamps
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        st.caption(f"Time: {msg['timestamp']}")

if prompt := st.chat_input():
    current_time = datetime.now().isoformat()
    
    # Add user message
    st.session_state["messages"].append({
        "role": "user", 
        "content": prompt,
        "timestamp": current_time
    })
    with st.chat_message("user"):
        st.write(prompt)
        st.caption(f"Time: {current_time}")

    # Get relevant history with metadata
    history = memory_system.get_relevant_history(prompt)
    formatted_history = "\n".join([
        f"[{h.get('timestamp', 'No Timestamp')}] User: {h.get('user', 'Unknown')}\nAI: {h.get('ai', 'No Response')}"
        for h in history
    ])

    # Generate response with enhanced context
    result = chain.invoke({
        "history": formatted_history,
        "current_time": current_time,
        "question": prompt
    })
    ai_response = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()

    # Add AI response
    st.session_state["messages"].append({
        "role": "assistant",
        "content": ai_response,
        "timestamp": datetime.now().isoformat()
    })
    with st.chat_message("assistant"):
        st.write(ai_response)
        st.caption(f"Time: {datetime.now().isoformat()}")

    # Store interaction with metadata
    memory_system.store_interaction(prompt, ai_response)


