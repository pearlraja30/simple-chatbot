# Vercel Deployment for NovaTech Policy Assistant

This folder contains a production-ready RAG chatbot built for deployment on Vercel.

## What is included
- `api/chat.py` — FastAPI-backed serverless endpoint for policy question answering
- `index.html` — lightweight browser chat UI that calls `/api/chat`
- `vercel.json` — Vercel routing and Python build configuration
- `.vercelignore` — excludes local cache files from deployment
- `requirements.txt` — Python dependencies for production

## How it works
- Documents are loaded from `sample_docs/*.docx`
- Content is chunked, embedded, and indexed using Chroma
- User questions are answered by retrieving relevant chunks and then using a language model to respond from context

## Environment variables
Set one of these in Vercel:
- `OPENAI_API_KEY` — preferred
- `GROQ_API_KEY` — alternate provider
- `POLICY_DOCS_PATH` — optional path to the `sample_docs` folder if you keep it outside the Vercel root

## Deploy steps
1. In Vercel, create a new project and point the root to this folder.
2. Ensure `requirements.txt` is installed by Vercel.
3. Add the required environment variable.
4. Deploy.

## Endpoints
- `GET /health` — simple health check
- `POST /api/chat` — request body: `{ "question": "..." }`

## Notes
- The first cold start may take a bit longer while the embedding model downloads and the vector store is built.
- The frontend loads from `index.html` and sends requests to the serverless chat endpoint.
