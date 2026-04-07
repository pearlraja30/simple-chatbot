import os
import json
import math
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
import requests

# Initialize FastAPI
app = FastAPI()

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Data Store (Global state)
KNOWLEDGE_BASE: List[Dict] = []
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/embedding.json")

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
    Get embedding for query using OpenAI API.
    To stay < 50MB, we avoid loading the 90MB local model on Vercel.
    """
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    data = {
        "input": text,
        "model": "text-embedding-3-small" # Efficient and high quality
    }
    
    # Check if dimensions match (all-MiniLM-L6-v2 is 384, text-embedding-3-small is 1536)
    # WAIT: If the knowledge base was built with all-MiniLM-L6-v2, 
    # we MUST use an API that returns 384 dimensions OR re-embed.
    # SINCE we exported 384-dim vectors, we need a 384-dim API or use a tiny local model.
    # ACTUALLY, for a 50MB limit, we can use 'sentence-transformers' but it might be tight.
    # ALTERNATIVE: Use a free Inference API (like Hugging Face) for query embedding.
    
    HF_TOKEN = os.getenv("HF_TOKEN")
    if HF_TOKEN:
        hf_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        hf_headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        response = requests.post(hf_url, headers=hf_headers, json={"inputs": text})
        if response.status_code == 200:
            return response.json()
    
    # Fallback/Default if no token: We'll instruct the user to set HF_TOKEN or use a lightweight lib.
    # To be safe for this demo, I'll use a mock if no API key, but warn.
    raise HTTPException(status_code=500, detail="HF_TOKEN (Hugging Face) required for query embedding to stay under 50MB limit.")

class ChatRequest(BaseModel):
    message: str
    use_rag: bool = True

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
@app.get("/api/health")
def health():
    return {"status": "ok", "kb_size": len(KNOWLEDGE_BASE)}
