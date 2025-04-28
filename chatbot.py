from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

import re
import os

# Initialize ChromaDB for long-term memory storage
memory_path = "./chroma_db"
vectorstore = Chroma(
    persist_directory=memory_path,
    embedding_function=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"}  # Changed from "cuda" to "cpu"
    )
)

# Retrieve top 3 relevant memories using vector similarity
memory = VectorStoreRetrieverMemory(retriever=vectorstore.as_retriever(search_kwargs={"k": 3}))

# Define Ollama LLM model (make sure Ollama is running locally with this model)
model = OllamaLLM(model="deepseek-r1:7b")

# Prompt template includes memory history + user input
template = """
You are a helpful AI assistant. Use the relevant past conversations retrieved from memory to provide context-aware responses.

Relevant conversation history:
{history}

User: {question}
AI:
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def store_interaction(user_input, ai_response):
    """Stores each user/AI interaction into ChromaDB."""
    document = Document(page_content=f"User: {user_input}\nAI: {ai_response}")
    vectorstore.add_documents([document])

def handle_conversation():
    print("🤖 Chatbot with Long-Term Memory is ready. Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Retrieve relevant memory
        retrieved_memory = memory.load_memory_variables({"input": user_input})["history"]

        # Generate response with context
        result = chain.invoke({"history": retrieved_memory, "question": user_input})
        ai_response = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()

        print("Bot:", ai_response)

        # Store the current exchange in memory
        store_interaction(user_input, ai_response)

if __name__ == "__main__":
    handle_conversation()

# Export important objects to be reused
__all__ = ["memory", "chain", "store_interaction"]


