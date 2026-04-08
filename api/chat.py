import os
import sys
import json
import math
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Optional

# --- VERCEL SQLITE FIX ---
# ChromaDB requires sqlite3 >= 3.35.0. 
# We use pysqlite3-binary to override the system version on Vercel (Linux).
if sys.platform.startswith("linux"):
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    except ImportError:
        print("Warning: pysqlite3-binary not found. Fallback to system sqlite3.")

from groq import Groq
import requests
from dotenv import load_dotenv
import chromadb

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "embedding.json")
COLLECTION_NAME = "novatech_policies"

# Singletons for reuse across warm invocations (Vercel optimization)
_client = None
_collection = None
_groq_client = None

def _initialize():
    """Initializes the in-memory ChromaDB and Groq client."""
    global _client, _collection, _groq_client
    
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise ValueError("Configuration Error: GROQ_API_KEY not found in environment. Please add it to Vercel settings.")
        _groq_client = Groq(api_key=GROQ_API_KEY)
        
    if _collection is None:
        # 1. Start in-memory ChromaDB client
        _client = chromadb.Client()
        _collection = _client.create_collection(name=COLLECTION_NAME)
        
        # 2. Load the embedding.json
        if os.path.exists(DATA_PATH):
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                kb_data = json.load(f)
            
            # 3. Index data in batches for speed
            BATCH_SIZE = 100
            for i in range(0, len(kb_data), BATCH_SIZE):
                batch = kb_data[i : i + BATCH_SIZE]
                _collection.add(
                    ids=[d.get("id", str(idx + i)) for idx, d in enumerate(batch)],
                    embeddings=[d["embedding"] for d in batch],
                    documents=[d["text"] for d in batch],
                    metadatas=[{"source": d.get("metadata", {}).get("source", "Unknown")} for d in batch]
                )
        else:
            print(f"Warning: Data file not found at {DATA_PATH}")

def _get_embedding(text: str) -> List[float]:
    """Retrieves embedding for the query (384-dims)."""
    if HF_TOKEN:
        try:
            url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            response = requests.post(url, headers=headers, json={"inputs": text}, timeout=10)
            if response.status_code == 200:
                return response.json()
        except:
            pass
            
    # Mock fallback for local testing if no HF token
    if os.getenv("VERCEL") != "1":
        return [0.0] * 384
    raise ValueError("HF_TOKEN required for production RAG search.")

def _get_answer(question: str, use_rag: bool) -> dict:
    """Core RAG logic matching the reference repo style."""
    _initialize()
    
    context = ""
    sources = []
    
    if use_rag and _collection and _collection.count() > 0:
        try:
            query_vec = _get_embedding(question)
            results = _collection.query(query_embeddings=[query_vec], n_results=5)
            
            formatted_chunks = []
            for i, (text, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
                source_name = meta.get("source", "Unknown Policy")
                formatted_chunks.append(f"SOURCE {i} ({source_name}):\n{text}")
                sources.append({
                    "id": i,
                    "source": source_name,
                    "preview": text[:150] + "..."
                })
            context = "\n\n".join(formatted_chunks)
        except Exception as e:
            print(f"RAG Error: {e}")

    # Build prompt
    if context:
        prompt = f"""You are a helpful policy assistant for NovaTech Solutions Pvt. Ltd.
Answer the employee's question based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't have that information in our policy documents."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    else:
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
    """The Vercel-compatible request handler (Classic Style)."""
    
    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))
            
            # Match frontend field names (the user's index.html uses 'message')
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
            self._respond(500, {"error": f"Internal Server Error: {error_msg}"})

    def do_GET(self):
        """Health check support."""
        if self.path == "/api/health" or self.path == "/health":
            _initialize()
            kb_size = _collection.count() if _collection else 0
            self._respond(200, {"status": "ok", "kb_size": kb_size})
        else:
            self._respond(404, {"error": "Not Found"})

    def do_OPTIONS(self):
        """CORS support."""
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
        # Suppress logging for cleaner Vercel logs
        return

if __name__ == "__main__":
    from http.server import HTTPServer
    server = HTTPServer(("127.0.0.1", 8001), handler)
    print("🚀 Local test server running on http://127.0.0.1:8001")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
