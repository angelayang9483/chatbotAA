from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from datetime import datetime
import json
import re
import os

class EnhancedMemory:
    def __init__(self, persist_directory="./chroma_db"):
        self.memory_path = persist_directory
        
        # Initialize ChromaDB with enhanced embeddings
        self.vectorstore = Chroma(
            persist_directory=self.memory_path,
            embedding_function=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={"device": "cpu"}
            )
        )
        
        # Initialize memory retriever with more context
        self.memory = VectorStoreRetrieverMemory(
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 5}  # Number of relevant memories to retrieve
            )
        )

    def store_interaction(self, user_input, ai_response, category="general"):
        """
        Stores interaction with enhanced metadata including timestamp and category.
        
        Args:
            user_input (str): User's question or input
            ai_response (str): AI's response
            category (str): Category of the conversation (default: "general")
        """
        timestamp = datetime.now().isoformat()
        
        # Create metadata
        metadata = {
            "timestamp": timestamp,
            "category": category,
            "interaction_type": "conversation",
            "length": len(user_input) + len(ai_response)
        }
        
        # Create structured content
        content = {
            "user_input": user_input,
            "ai_response": ai_response,
            "timestamp": timestamp,
            "category": category
        }
        
        # Store as document with metadata
        document = Document(
            page_content=json.dumps(content),
            metadata=metadata
        )
        
        self.vectorstore.add_documents([document])

    def get_relevant_history(self, query, k=5):
        """
        Retrieves relevant conversation history with metadata.
        
        Args:
            query (str): The current user query
            k (int): Number of relevant memories to retrieve
        
        Returns:
            dict: Contains formatted history and metadata
        """
        raw_history = self.memory.load_memory_variables({"input": query})["history"]
        
        # Parse and format the history
        formatted_history = []
        try:
            for entry in raw_history.split('\n\n'):
                if entry.strip():
                    memory_data = json.loads(entry)
                    formatted_history.append({
                        "user": memory_data["user_input"],
                        "ai": memory_data["ai_response"],
                        "timestamp": memory_data["timestamp"],
                        "category": memory_data["category"]
                    })
        except json.JSONDecodeError:
            # Handle legacy format if any
            formatted_history = [{"content": raw_history, "timestamp": "unknown"}]
            
        return formatted_history

# Initialize enhanced memory system
memory_system = EnhancedMemory()

# Define Ollama LLM model
model = OllamaLLM(model="gemma3:1b")

# Enhanced prompt template that includes structured history
template = """
You are a helpful AI assistant for a bioengineering lab. You should maintain context from previous conversations and remember important details about the user.

Previous conversations:
{history}

Current timestamp: {current_time}
User: {question}
AI: """

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversation():
    print("🤖 Enhanced Chatbot with Structured Memory is ready. Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Get current timestamp
        current_time = datetime.now().isoformat()

        # Retrieve relevant memory with metadata
        history = memory_system.get_relevant_history(user_input)
        
        # Format history for the prompt
        formatted_history = "\n".join([
            f"[{h.get('timestamp', 'unknown')}] User: {h.get('user', 'unknown')}\nAI: {h.get('ai', 'unknown')}"
            for h in history
        ])

        # Generate response with enhanced context
        result = chain.invoke({
            "history": formatted_history,
            "current_time": current_time,
            "question": user_input
        })
        
        ai_response = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
        print("Bot:", ai_response)

        # Store interaction with metadata
        memory_system.store_interaction(user_input, ai_response)

if __name__ == "__main__":
    handle_conversation()

# Export important objects to be reused
__all__ = ["memory_system", "chain"]


