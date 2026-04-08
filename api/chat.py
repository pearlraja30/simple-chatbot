import os
import json
import math
from typing import List
from groq import Groq
from dotenv import load_dotenv

# Load env early
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "nomic-embed-text-v1.5"
DATA_PATH = os.getenv("DATA_PATH", "api/data/embedding_groq.json")

# Shared clients
_groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
KNOWLEDGE_BASE = []

def _normalize(vector: List[float]) -> List[float]:
    """Ensures vectors are unit length for consistent cosine similarity."""
    if not vector:
        return []
    mag = math.sqrt(sum(x * x for x in vector))
    if mag > 0:
        return [x / mag for x in vector]
    return vector

def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Robust cosine similarity between two vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    return sum(x * y for x, y in zip(v1, v2))

def _get_embedding(text: str) -> List[float]:
    """Retrieves embedding for the query from Groq (nomic-embed-text-v1.5)."""
    if not text or not _groq_client:
        return []

    try:
        response = _groq_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return _normalize(response.data[0].embedding)
    except Exception as e:
        print(f"Embedding Search Error: {e}")
        return []

def _initialize():
    """Load the knowledge base once into memory."""
    global KNOWLEDGE_BASE
    if KNOWLEDGE_BASE:
        return

    # Look in possible paths for Vercel vs Local
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "data/embedding_groq.json"),
        "api/data/embedding_groq.json",
        "vercelapp/api/data/embedding_groq.json",
        "/var/task/api/data/embedding_groq.json"
    ]
    
    target_path = None
    for p in possible_paths:
        if os.path.exists(p):
            target_path = p
            break
            
    if target_path:
        try:
            print(f"Loading knowledge base from: {target_path}")
            with open(target_path, "r", encoding="utf-8") as f:
                KNOWLEDGE_BASE = json.load(f)
        except Exception as e:
            print(f"Data Load Error: {e}")
            # Don't crash here, just log
    else:
        print(f"Warning: Data file not found in any of {possible_paths}")

def _get_answer(question: str, use_rag: bool) -> dict:
    """Optimized lightweight search + LLM generation."""
    _initialize()
    
    context = ""
    sources = []
    top_score = 0.0
    
    if use_rag and KNOWLEDGE_BASE:
        try:
            query_vec = _get_embedding(question)
            
            if query_vec:
                scores = []
                for item in KNOWLEDGE_BASE:
                    score = _cosine_similarity(query_vec, item["embedding"])
                    scores.append((score, item))
                
                scores.sort(key=lambda x: x[0], reverse=True)
                top_k = scores[:10]
                
                if top_k:
                    top_score = top_k[0][0]
                
                formatted_chunks = []
                for i, (score, item) in enumerate(top_k, 1):
                    # Filter for relevance but keep it loose for diagnostic
                    if score > 0.1:
                        source_name = item.get("metadata", {}).get("source", "Policy Document")
                        formatted_chunks.append(f"SOURCE {i} ({source_name}):\n{item['text']}")
                        sources.append({
                            "id": i,
                            "source": source_name,
                            "score": round(score, 3),
                            "preview": item["text"][:150] + "..."
                        })
                
                context = "\n\n".join(formatted_chunks)
        except Exception as e:
            print(f"Search Error: {e}")

    # Synchronized system prompt from local demo
    if context:
        prompt = f"""You are a helpful policy assistant for NovaTech Solutions Pvt. Ltd.
Answer the employee's question based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't have that information in our policy documents."
If the answer can be inferred from the context, compute it and answer directly.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    else:
        prompt = f"""You are a helpful policy assistant for NovaTech Solutions Pvt. Ltd.
Answer the employee's question about company policies.

QUESTION: {question}

ANSWER:"""

    if not _groq_client:
        return {"answer": "GROQ_API_KEY missing.", "sources": [], "rag_used": False}

    completion = _groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    
    return {
        "answer": completion.choices[0].message.content,
        "sources": sources,
        "rag_used": bool(context),
        "debug": {
            "kb_size": len(KNOWLEDGE_BASE),
            "context_len": len(context),
            "source_count": len(sources),
            "kb_loaded": len(KNOWLEDGE_BASE) > 0,
            "top_score": round(top_score, 3)
        }
    }

# Handler remains similar but calls _get_answer directly
from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            question = data.get('question', '')
            use_rag = data.get('use_rag', True)
            
            result = _get_answer(question, use_rag)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == '/api/health':
            _initialize()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "kb_size": len(KNOWLEDGE_BASE)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
