from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from datetime import datetime
import json
import re

class EnhancedMemory:
    def __init__(self, persist_directory="./chroma_db"):
        self.memory_path = persist_directory
        self.recent_log = []

        self.vectorstore = Chroma(
            persist_directory=self.memory_path,
            embedding_function=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={"device": "cpu"}
            )
        )

    def store_interaction(self, user_input, ai_response, category="general"):
        timestamp = datetime.now().isoformat()

        content = {
            "user_input": user_input,
            "ai_response": ai_response,
            "timestamp": timestamp,
            "category": category
        }

        metadata = {
            "timestamp": timestamp,
            "category": category,
            "length": len(user_input) + len(ai_response)
        }

        document = Document(page_content=json.dumps(content), metadata=metadata)
        self.vectorstore.add_documents([document])

        self.recent_log.append(content)
        if len(self.recent_log) > 10:
            self.recent_log.pop(0)

    def get_facts_and_history(self, query, k=5):
        docs = self.vectorstore.similarity_search(query, k=k)
        relevant = []
        for doc in docs:
            try:
                parsed = json.loads(doc.page_content)
                relevant.append(parsed)
            except:
                continue

        all_history = {d["timestamp"]: d for d in self.recent_log + relevant}
        sorted_history = sorted(all_history.values(), key=lambda x: x["timestamp"])

        fact_statements = []
        history_lines = []
        for h in sorted_history:
            user_input = h["user_input"].strip().lower()
            history_lines.append(f"[{h['timestamp']}]\nUser: {h['user_input']}\nAI: {h['ai_response']}")
            if "my name is" in user_input:
                name = user_input.split("my name is")[-1].strip().capitalize()
                fact_statements.append(f"The user's name is {name}.")
            if "i like" in user_input:
                item = user_input.split("i like")[-1].strip()
                fact_statements.append(f"The user likes {item}.")

        return fact_statements, history_lines

memory_system = EnhancedMemory()
model = OllamaLLM(model="gemma3:1b")

template = """
You are a helpful AI assistant in a bioengineering lab. You remember what the user tells you and use it to answer fact-based questions accurately.

Facts the user has told you:
{facts}

Conversation history:
{history}

Current time: {current_time}
User: {question}
AI:"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversation():
    print("\U0001F916 Enhanced Chatbot with Light RAG is ready. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        current_time = datetime.now().isoformat()

        facts, history_lines = memory_system.get_facts_and_history(user_input)
        formatted_facts = "\n".join(facts)
        formatted_history = "\n".join(history_lines)

        result = chain.invoke({
            "facts": formatted_facts,
            "history": formatted_history,
            "current_time": current_time,
            "question": user_input
        })

        ai_response = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
        print("Bot:", ai_response)

        memory_system.store_interaction(user_input, ai_response)

if __name__ == "__main__":
    handle_conversation()

__all__ = ["memory_system", "chain"]
