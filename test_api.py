import requests
import json
import time
import subprocess
import os
import signal

def test_backend():
    print("🚀 Starting local test...")
    
    # 1. Start the server
    # We use uvicorn to run the FastAPI app
    # Cwd should be the vercelapp folder
    cwd = "/Users/rajasekaran/Projects/course_gen_agen_ai/python_ai/rag/vercelapp"
    
    print("📦 Starting Uvicorn server...")
    server_process = subprocess.Popen(
        ["uvicorn", "api.index:app", "--port", "8001"],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    
    # Wait for server to start
    time.sleep(3)
    
    try:
        # 2. Test Health Check
        print("🔍 Testing /api/health...")
        response = requests.get("http://127.0.0.1:8001/api/health")
        print(f"Health Response: {response.status_code} - {response.json()}")
        assert response.status_code == 200
        
        # 3. Test Chat Endpoint (Using Mocked Embeddings)
        print("💬 Testing /api/chat...")
        chat_payload = {
            "message": "What is the leaf policy?",
            "use_rag": True
        }
        response = requests.post(
            "http://127.0.0.1:8001/api/chat", 
            json=chat_payload
        )
        print(f"Chat Response Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Chat Response Answer: {data.get('answer')[:100]}...")
            print(f"RAG Used: {data.get('rag_used')}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Test Failed: {e}")
    finally:
        # Cleanup
        print("🛑 Shutting down server...")
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        print("✅ Test complete.")

if __name__ == "__main__":
    test_backend()
