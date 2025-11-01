# app.py
"""
Dynamic AI Chatbot
------------------
An intelligent conversational AI system built with NLP, ML, and Deep Learning principles.
Features:
- Intent Recognition (TF-IDF + Logistic Regression)
- Named Entity Recognition (NER) with spaCy
- Contextual Memory (in-memory session)
- Rule-based + Generative (GPT-based) Responses
- API-based structure (for Web, WhatsApp, Slack, etc.)
"""

import os
import time
import uuid
import joblib
import asyncio
import aiohttp
import spacy
os.environ["OPENAI_API_KEY"] = "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"
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Optional: enable OpenAI or generative API fallback (env var OPENAI_API_KEY)
# import openai

# -------------------------------------------------------------------
# Train small demo intent classifier (you can replace with larger data)
# -------------------------------------------------------------------
def train_intent_model():
    import pandas as pd

    data = [
        ("What is my account balance?", "balance"),
        ("How much did I spend last month?", "transactions"),
        ("I need help with a disputed charge", "dispute"),
        ("Block my credit card", "block_card"),
        ("I lost my credit card", "block_card"),
        ("Check fraud for transaction id 123", "fraud_check"),
        ("How do I update my address?", "update_profile"),
        ("Hello", "greeting"),
        ("Hi there", "greeting"),
        ("Thanks, bye", "goodbye"),
        ("I want to talk to customer support", "connect_agent"),
    ]

    df = pd.DataFrame(data, columns=["text", "intent"])
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["intent"], test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=2000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    print("\n--- Intent Model Trained ---")
    print(classification_report(y_test, preds))
    joblib.dump(pipeline, "intent_pipeline.joblib")
    print("✅ Model saved as 'intent_pipeline.joblib'")
    return pipeline


# Load or train model
if os.path.exists("intent_pipeline.joblib"):
    intent_pipeline: Pipeline = joblib.load("intent_pipeline.joblib")
else:
    intent_pipeline = train_intent_model()

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Initialize FastAPI app
app = FastAPI(title="Dynamic AI Chatbot")

# -------------------------------------------------------------------
# Context memory store
# -------------------------------------------------------------------
CONTEXT_STORE: Dict[str, Dict[str, Any]] = {}

# -------------------------------------------------------------------
# Pydantic Schemas
# -------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    user_id: Optional[str] = None
    metadata: Optional[dict] = None

class ChatResponse(BaseModel):
    session_id: str
    intent: Optional[str]
    entities: Dict[str, Any]
    response: str
    confidence: Optional[float] = None
    timestamp: float

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def get_or_create_session(session_id: Optional[str]) -> str:
    if session_id and session_id in CONTEXT_STORE:
        return session_id
    new_sid = session_id or str(uuid.uuid4())
    CONTEXT_STORE[new_sid] = {"history": [], "created_at": time.time(), "metadata": {}}
    return new_sid

def store_message(session_id: str, user_msg: str, bot_msg: str, meta: Optional[dict] = None):
    entry = {"user": user_msg, "bot": bot_msg, "timestamp": time.time(), "meta": meta or {}}
    CONTEXT_STORE[session_id]["history"].append(entry)

def predict_intent(text: str):
    try:
        probs = intent_pipeline.predict_proba([text])[0]
        idx = probs.argmax()
        intent = intent_pipeline.classes_[idx]
        confidence = float(probs[idx])
        return {"intent": intent, "confidence": confidence}
    except Exception:
        intent = intent_pipeline.predict([text])[0]
        return {"intent": intent, "confidence": 1.0}

def extract_entities(text: str):
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, []).append(ent.text)
    return entities

# -------------------------------------------------------------------
# Rule-based response mapping
# -------------------------------------------------------------------
RULE_RESPONSES = {
    "greeting": "Hello! 👋 How can I assist you today?",
    "goodbye": "Goodbye! 😊 Have a great day ahead.",
    "balance": "I can check your balance — please provide the last 4 digits of your account.",
    "transactions": "I can show your recent transactions. Which date range should I check?",
    "block_card": "I’ll start the card block process. Please confirm your card’s last 4 digits.",
    "dispute": "I can help with your dispute. Please share the transaction ID and short details.",
    "fraud_check": "Let me check the fraud risk for that transaction — could you share the ID?",
    "update_profile": "Sure! What details would you like to update?",
    "connect_agent": "I'll connect you with a support agent right away. 🔄",
}

# -------------------------------------------------------------------
# Optional: Generative fallback (uses OpenAI API if available)
# -------------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "AI Chatbot Backend Running"}

# -------------------------------------------------------------------
# Optional: Generative fallback (uses OpenAI API if available)
# -------------------------------------------------------------------
import openai

async def generate_with_openai(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "🤖 I’m currently using my rule-based brain. (Generative AI not active)"
    
    openai.api_key = api_key

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        return response.choices[0].message["content"]
    except Exception as e:
        print("❌ OpenAI API Error:", e)
        return "Sorry, AI service is temporarily unavailable."


# -------------------------------------------------------------------
# Main Chat Endpoint
# -------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    sid = get_or_create_session(req.session_id)
    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty message")

    entities = extract_entities(text)
    prediction = predict_intent(text)
    intent = prediction["intent"]
    confidence = prediction["confidence"]

    # Choose response
    if intent in RULE_RESPONSES and confidence >= 0.4:
        response_text = RULE_RESPONSES[intent]
    else:
        # Fallback to GPT
        context_snippet = str(CONTEXT_STORE[sid]["history"][-3:]) if CONTEXT_STORE[sid]["history"] else ""
        prompt = f"User: {text}\nContext: {context_snippet}\nRespond like an assistant helping with finance queries."
        response_text = await generate_with_openai(prompt)
        intent = intent or "openai_response"

    store_message(sid, text, response_text, {"intent": intent, "confidence": confidence})

    return ChatResponse(
        session_id=sid,
        intent=intent,
        entities=entities,
        response=response_text,
        confidence=confidence,
        timestamp=time.time()
    )

# -------------------------------------------------------------------
# Health & Session endpoints
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(intent_pipeline)}

@app.get("/session/{session_id}")
def session_info(session_id: str):
    if session_id not in CONTEXT_STORE:
        raise HTTPException(404, "Session not found")
    return CONTEXT_STORE[session_id]

# -------------------------------------------------------------------
# Run command: uvicorn app:app --reload --port 8000
# Then POST to /chat or test via Swagger at http://127.0.0.1:8000/docs
# -------------------------------------------------------------------

