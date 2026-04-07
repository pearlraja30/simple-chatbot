# NovaTech Policy Assistant - Vercel Deployment

This is a production-ready, lightweight RAG application designed for Vercel.

## 🚀 Deployment Instructions

### 1. Push to GitHub
Run these commands from your terminal (make sure you are inside the `vercelapp` folder or create a new repo from it):

```bash
cd vercelapp
git init
git add .
git commit -m "Deploy: NovaTech RAG Assistant"
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 2. Connect to Vercel
1. Log in to [Vercel](https://vercel.com).
2. Click **Add New** -> **Project**.
3. Import your GitHub repository.
4. **Environment Variables**: Add the following:
   - `GROQ_API_KEY`: (From Groq Console)
   - `HF_TOKEN`: (From Hugging Face Settings) or `OPENAI_API_KEY` (if modified)
5. Click **Deploy**.

## 📂 Project Structure
- `api/chat.py`: FastAPI serverless function (Llama 3.3 + Vector Similarity).
- `public/index.html`: Premium Dark-Mode frontend.
- `data/embedding.json`: Pre-calculated 384-dim knowledge base (~17MB).
- `vercel.json`: Deployment/Routing configuration.
