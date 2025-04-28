import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import VectorStoreRetrieverMemory
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

from chatbot import memory, chain, store_interaction


# ... (Your existing code for initializing model, memory, etc.) ...

st.title("Bioengineering Lab Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    retrieved_memory = memory.load_memory_variables({"input": prompt})["history"]
    result = chain.invoke({"history": retrieved_memory, "question": prompt})
    ai_response = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()

    st.session_state["messages"].append({"role": "assistant", "content": ai_response})
    st.chat_message("assistant").write(ai_response)

    store_interaction(prompt, ai_response)


