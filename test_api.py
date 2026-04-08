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
    
    print("📦 Starting Standalone server...")
    server_process = subprocess.Popen(
        ["python3", "api/chat.py"],
        cwd=cwd,
        preexec_fn=os.setsid
    )
    
    # Wait for server to start
    time.sleep(3)
    
    try:
        # 0. Test Site Root (Should now return HTML)
        print("🏠 Testing Site Root (/)...")
        response = requests.get("http://127.0.0.1:8001/")
        print(f"Root Response: {response.status_code}")
        if response.status_code == 200 and "text/html" in response.headers.get("Content-Type", ""):
            print("✅ Root successfully serves HTML!")
        else:
            print(f"❌ Root failed to serve HTML: {response.headers.get('Content-Type')}")

        # 1. Test Health Check
        print("🔍 Testing /api/health...")
        response = requests.get("http://127.0.0.1:8001/api/health")
        print(f"Health Response: {response.status_code} - {response.json()}")
        assert response.status_code == 200
        
        # 3. Test Chat Endpoint (Using Mocked Embeddings)
        print("💬 Testing /api/chat...")
        test_query = {"message": "How many days of earned leave do I get per year?", "use_rag": True}
        response = requests.post("http://127.0.0.1:8001/api/chat", json=test_query)
        print(f"Chat Response Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Chat Response Answer: {data['answer'][:100]}...")
            print(f"RAG Used: {data.get('rag_used')}")
            if "18" in data['answer']:
                print("✅ TEST PASSED: AI correctly identified 18 days.")
            else:
                print(f"❌ TEST FAILED: AI did not mention 18 days. Answer was: {data['answer']}")
        else:
            print(f"❌ Chat failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Test Failed: {e}")
    finally:
        # Cleanup
        print("🛑 Shutting down server...")
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        print("✅ Test complete.")

if __name__ == "__main__":
    test_backend()
