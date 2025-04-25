import streamlit as st
import requests
import os

# Long-term memory file
MEMORY_FILE = "memory.txt"
SYSTEM_PROMPT = "You are a helpful, friendly assistant with long-term memory. Be clear and concise."

# Load memory from file
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r") as f:
        history = f.read().splitlines()
else:
    history = []

# Streamlit UI
st.title("💬 Chat with Ollama (with Memory)")
user_input = st.text_input("You:")

if user_input:
    if user_input.lower() == "clear":
        history = []
        open(MEMORY_FILE, "w").close()
        st.success("🧠 Memory cleared!")
    else:
        history.append(f"You: {user_input}")
        full_prompt = f"{SYSTEM_PROMPT}\n\n" + "\n".join(history)

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama2", "prompt": full_prompt, "stream": False}
        )

        reply = response.json()["response"].strip()
        history.append(f"Bot: {reply}")

        # Save memory
        with open(MEMORY_FILE, "w") as f:
            f.write("\n".join(history))

# Display chat history
for msg in history[::-1]:
    st.markdown(msg)
