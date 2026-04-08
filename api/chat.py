import os
import sys
import json
import math
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Optional
from groq import Groq
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "embedding.json")

# Singletons for reuse across warm invocations (Vercel optimization)
KNOWLEDGE_BASE: List[Dict] = []
_groq_client = None

def _initialize():
    """Lightweight initialization: loads JSON knowledge base once into memory."""
    global KNOWLEDGE_BASE, _groq_client
    
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise ValueError("Configuration Error: GROQ_API_KEY not found in Vercel settings.")
        _groq_client = Groq(api_key=GROQ_API_KEY)
        
    if not KNOWLEDGE_BASE:
        # Check multiple potential data paths for Vercel vs Local
        possible_paths = [
            DATA_PATH,
            os.path.join(os.getcwd(), "data", "embedding.json"),
            os.path.join(os.getcwd(), "api", "..", "data", "embedding.json")
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
                raise ValueError(f"Could not load knowledge base: {str(e)}")
        else:
            print(f"Warning: Data file not found in any of {possible_paths}")

def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Manual cosine similarity calculation."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot_product = sum(x * y for x, y in zip(v1, v2))
    mag1 = math.sqrt(sum(x * x for x in v1))
    mag2 = math.sqrt(sum(x * x for x in v2))
    if not mag1 or not mag2:
        return 0.0
    return dot_product / (mag1 * mag2)

def _get_embedding(text: str) -> List[float]:
    """Retrieves embedding for the query from Hugging Face (384-dims)."""
    if HF_TOKEN:
        try:
            url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            response = requests.post(url, headers=headers, json={"inputs": text}, timeout=8)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"HF API returned {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Embedding Fetch Error: {e}")
            
    # Mock fallback for local testing
    if os.getenv("VERCEL") != "1":
        return [0.0] * 384
    
    # If in Vercel and search key is missing
    raise ValueError("HF_TOKEN required for Policy Search. Please add it to Vercel environment variables.")

def _get_answer(question: str, use_rag: bool) -> dict:
    """Optimized lightweight search + LLM generation."""
    _initialize()
    
    context = ""
    sources = []
    
    if use_rag and KNOWLEDGE_BASE:
        try:
            # 1. Get embedding for the question
            query_vec = _get_embedding(question)
            
            # 2. Sequential Similarity Search
            scores = []
            for item in KNOWLEDGE_BASE:
                score = _cosine_similarity(query_vec, item["embedding"])
                scores.append((score, item))
            
            # 3. Sort and take top 10 (Remove threshold to ensure context is always provided)
            scores.sort(key=lambda x: x[0], reverse=True)
            top_k = scores[:10]
            
            formatted_chunks = []
            for i, (score, item) in enumerate(top_k, 1):
                source_name = item.get("metadata", {}).get("source", "Policy Document")
                formatted_chunks.append(f"SOURCE {i} ({source_name}):\n{item['text']}")
                sources.append({
                    "id": i,
                    "source": source_name,
                    "score": round(score, 3),
                    "preview": item["text"][:150] + "..."
                })
            
            if formatted_chunks:
                context = "\n\n".join(formatted_chunks)
            else:
                print("Warning: No context chunks were formatted.")
        except Exception as e:
            print(f"Search Error: {e}")

    # Prompt logic
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
        # Fallback if no context found or search failed
        prompt = f"""You are a helpful policy assistant for NovaTech Solutions Pvt. Ltd.
Answer the employee's question about company policies.

QUESTION: {question}

ANSWER:"""

    completion = _groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    
    return {
        "answer": completion.choices[0].message.content,
        "sources": sources,
        "rag_used": bool(context)
    }

class handler(BaseHTTPRequestHandler):
    """Vercel-optimized request handler."""
    
    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))
            
            message = body.get("message", body.get("question", ""))
            use_rag = bool(body.get("use_rag", True))
            
            if not message:
                self._respond(400, {"error": "Message is required"})
                return

            result = _get_answer(message, use_rag)
            self._respond(200, result)

        except Exception as e:
            error_msg = str(e)
            print(f"Server Error: {error_msg}")
            # Send detailed error back to frontend for debugging
            self._respond(500, {"error": f"Backend Error: {error_msg}"})

    def do_GET(self):
        """Health check support."""
        if self.path == "/api/health" or self.path == "/health":
            try:
                _initialize()
                kb_size = len(KNOWLEDGE_BASE)
                self._respond(200, {"status": "ok", "kb_size": kb_size})
            except Exception as e:
                self._respond(500, {"error": str(e)})
        else:
            self._respond(404, {"error": "Not Found"})

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_headers()
        self.end_headers()

    def _respond(self, status_code: int, data: dict):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self._set_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _set_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format, *args):
        return

if __name__ == "__main__":
    from http.server import HTTPServer
    server = HTTPServer(("127.0.0.1", 8001), handler)
    print("🚀 Local Lightweight server running on http://127.0.0.1:8001")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
