import os
import json
import math
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
import requests
from dotenv import load_dotenv

# Load environment variables from .env file (if exists)
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Initialize Groq client
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

# Data Store (Global state)
KNOWLEDGE_BASE: List[Dict] = []
# Use environment variable for DATA_PATH if set (e.g., in vercel.json)
DATA_PATH = os.getenv("DATA_PATH")
if not DATA_PATH:
    # Default relative path from api/ folder
    DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/embedding.json")
elif not os.path.isabs(DATA_PATH):
    # If it's a relative path in vercel.json, make it relative to the project root
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", DATA_PATH)

def load_knowledge_base():
    global KNOWLEDGE_BASE
    if not KNOWLEDGE_BASE:
        if os.path.exists(DATA_PATH):
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                KNOWLEDGE_BASE = json.load(f)
        else:
            print(f"Warning: Knowledge base not found at {DATA_PATH}")

# Cosine Similarity implementation
def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot_product = sum(x * y for x, y in zip(v1, v2))
    magnitude1 = math.sqrt(sum(x * x for x in v1))
    magnitude2 = math.sqrt(sum(x * x for x in v2))
    if not magnitude1 or not magnitude2:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

def get_query_embedding(text: str) -> List[float]:
    """
    Get embedding for query.
    Prioritizes 384-dim sources to match 'all-MiniLM-L6-v2' knowledge base.
    """
    # 1. Try Hugging Face (Correct 384 dimensions)
    HF_TOKEN = os.getenv("HF_TOKEN")
    if HF_TOKEN:
        try:
            hf_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
            hf_headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            response = requests.post(hf_url, headers=hf_headers, json={"inputs": text}, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"HF Error: {e}")

    # 2. Try OpenAI (Warning: Returns 1536 dims, which will fail similarity comparison)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        try:
            url = "https://api.openai.com/v1/embeddings"
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"}
            data = {"input": text, "model": "text-embedding-3-small"}
            response = requests.post(url, headers=headers, json=data, timeout=10)
            if response.status_code == 200:
                # NOTE: This returns 1536 dimensions. 
                # To fix this properly, we'd need to truncate or re-embed the KB.
                return response.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"OpenAI Error: {e}")

    # 3. Fallback for Local Testing (Mock 384-dim vector if no keys present)
    # This prevents the app from crashing during development if keys aren't set yet.
    if os.getenv("VERCEL") != "1":
        print("Warning: No Embedding keys found. Using zero-vector for local testing.")
        return [0.0] * 384
    
    raise HTTPException(status_code=500, detail="HF_TOKEN or OPENAI_API_KEY required for production embeddings.")

class ChatRequest(BaseModel):
    message: str
    use_rag: bool = True

@app.post("/chat")
@app.post("/api/chat")
async def chat(request: ChatRequest):
    load_knowledge_base()
    
    context = ""
    sources = []
    
    if request.use_rag and KNOWLEDGE_BASE:
        # 1. Get Query Embedding
        query_vec = get_query_embedding(request.message)
        
        # 2. Vector Search (Simulated RAG)
        similarities = []
        for item in KNOWLEDGE_BASE:
            score = cosine_similarity(query_vec, item["embedding"])
            similarities.append((score, item))
        
        # Sort and take top 5
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_k = similarities[:5]
        
        formatted_chunks = []
        for i, (score, item) in enumerate(top_k, 1):
            source_name = item["metadata"].get("source", "Unknown Policy")
            formatted_chunks.append(f"SOURCE {i} ({source_name}):\n{item['text']}")
            sources.append({
                "id": i,
                "source": source_name,
                "preview": item["text"][:150] + "..."
            })
        
        context = "\n\n".join(formatted_chunks)

    # 3. Prompt Construction
    if context:
        prompt = f"""You are a helpful policy assistant for NovaTech Solutions Pvt. Ltd.
Answer the employee's question based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't have that information in our policy documents."

CONTEXT:
{context}

QUESTION: {request.message}

ANSWER:"""
    else:
        prompt = f"""You are a helpful policy assistant for NovaTech Solutions Pvt. Ltd.
Answer the employee's question about company policies.

QUESTION: {request.message}

ANSWER:"""

    # 4. LLM Call
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "answer": answer,
        "sources": sources,
        "rag_used": bool(context)
    }

# For Vercel: Need to expose the app
# (Optional) Add a simple health check
@app.get("/health")
@app.get("/api/health")
def health():
    load_knowledge_base()
    return {"status": "ok", "kb_size": len(KNOWLEDGE_BASE)}
