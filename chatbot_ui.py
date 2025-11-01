import streamlit as st
import requests

# Streamlit page setup
st.set_page_config(page_title="💬 Dynamic AI Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Dynamic AI Chatbot")
st.caption("Powered by FastAPI + NLP + Machine Learning")

# Sidebar
st.sidebar.header("⚙️ Configuration")
backend_url = st.sidebar.text_input("Backend API URL", "http://127.0.0.1:8000/chat")

# Session state to store conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input box
user_input = st.chat_input("Type your message...")

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Send to FastAPI backend
    try:
        response = requests.post(
            backend_url,
            json={"message": user_input, "session_id": "user123"},
            timeout=10
        )

        if response.status_code == 200:
            bot_reply = response.json().get("response", "⚠️ No response from chatbot.")
        else:
            bot_reply = f"⚠️ Error {response.status_code}: {response.text}"

    except requests.exceptions.RequestException as e:
        bot_reply = f"❌ Connection error: {e}"

    # Add bot response
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
# 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