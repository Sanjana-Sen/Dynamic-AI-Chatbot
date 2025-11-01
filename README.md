💬 Dynamic AI Chatbot

An intelligent, hybrid AI chatbot built using FastAPI, NLP, and Machine Learning, with a Streamlit frontend for real-time conversations.

This chatbot combines intent recognition, entity extraction, contextual memory, and a GPT-based fallback, making it capable of handling both rule-based and open-ended queries.

🚀 Features

✅ Intent Recognition — TF-IDF + Logistic Regression model for classifying user queries
✅ Named Entity Recognition (NER) — Extracts entities using spaCy (en_core_web_sm)
✅ Contextual Memory — Keeps track of previous conversations in a session
✅ Rule-based + Generative Responses — Uses pre-defined replies or OpenAI GPT fallback
✅ FastAPI Backend — Lightweight and fast REST API for integration
✅ Streamlit Frontend — Simple, interactive chat interface
✅ Extensible Architecture — Can connect to web, Slack, WhatsApp, or mobile apps

🧠 Tech Stack
| Component         | Technology            |
| ----------------- | --------------------- |
| **Backend API**   | FastAPI               |
| **Frontend UI**   | Streamlit             |
| **ML/NLP**        | Scikit-learn, spaCy   |
| **Generative AI** | OpenAI GPT (optional) |
| **Language**      | Python 3.8+           |

1️⃣ Clone the Repository
git clone https://github.com/<your-username>/dynamic-ai-chatbot.git
cd dynamic-ai-chatbot

2️⃣ Create and Activate Virtual Environment
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On macOS/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Download spaCy Model
python -m spacy download en_core_web_sm

5️⃣ Set Your OpenAI API Key (Optional but recommended)

Do not hardcode it in the code!
Instead, set it in your environment:

# Windows
setx OPENAI_API_KEY "your_openai_api_key"

# macOS/Linux
export OPENAI_API_KEY="your_openai_api_key"

🧩 Run the Application
▶️ Start the FastAPI Backend
uvicorn app:app --reload --port 8000
It will start your backend at http://127.0.0.1:8000

🗣️ Usage

Type a message in the chat input box.

The backend detects intent and entities using ML and spaCy.

If confidence is high, it responds using rule-based logic.

If confidence is low, it uses GPT (OpenAI) as a fallback for natural replies.

The conversation context is preserved per session.

🧩 Future Enhancements

🧮 Integrate PostgreSQL or Redis for session persistence

🧠 Expand intent dataset for better classification

🗂️ Add user authentication for personalized chats

💬 Integrate WhatsApp / Slack bot connector

⚡ Add streaming GPT responses for real-time typing effect
