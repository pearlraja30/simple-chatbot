import os
from functools import lru_cache
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    ChatOpenAI = None
    OpenAIEmbeddings = None

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "technova_policies"


def get_sample_docs_dir() -> Path:
    env_path = os.getenv("POLICY_DOCS_PATH")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            BASE_DIR / "sample_docs",
            BASE_DIR.parent / "sample_docs",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise RuntimeError(
        "No sample_docs directory found. Create a 'sample_docs' folder in the Vercel root or set POLICY_DOCS_PATH."
    )


SAMPLE_DOCS_DIR = get_sample_docs_dir()

app = FastAPI(
    title="NovaTech Policy Assistant",
    description="A lightweight production-ready RAG chatbot backend for policy question answering.",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]


def get_openai_api_key():
    return os.getenv("OPENAI_API_KEY")


def get_groq_api_key():
    return os.getenv("GROQ_API_KEY")


@lru_cache(maxsize=1)
def get_embedding_model():
    if get_openai_api_key() and OpenAIEmbeddings is not None:
        return OpenAIEmbeddings(model="text-embedding-3-small")

    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


@lru_cache(maxsize=1)
def get_llm():
    if get_openai_api_key() and ChatOpenAI is not None:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    if get_groq_api_key() and ChatGroq is not None:
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

    raise RuntimeError(
        "No LLM provider configured. Set OPENAI_API_KEY or GROQ_API_KEY in environment variables."
    )


@lru_cache(maxsize=1)
def build_vectorstore():
    embedding_model = get_embedding_model()

    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        try:
            return Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=embedding_model,
                collection_name=COLLECTION_NAME,
            )
        except Exception:
            pass

    if not SAMPLE_DOCS_DIR.exists():
        raise RuntimeError(f"Document directory not found: {SAMPLE_DOCS_DIR}")

    loader = DirectoryLoader(
        str(SAMPLE_DOCS_DIR),
        glob="*.docx",
        loader_cls=Docx2txtLoader,
    )
    raw_documents = loader.load()
    if not raw_documents:
        raise RuntimeError("No documents loaded from sample_docs directory.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    documents = splitter.split_documents(raw_documents)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )
    return vectorstore


@lru_cache(maxsize=1)
def get_retriever():
    vectorstore = build_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": 20})


def dedupe_docs(docs):
    seen = set()
    unique = []
    for doc in docs:
        key = (doc.metadata.get("source"), doc.page_content)
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def format_docs(docs):
    formatted = []
    for doc in docs:
        source_name = Path(doc.metadata.get("source", "unknown")).name
        formatted.append(f"SOURCE: {source_name}\n{doc.page_content}")
    return "\n\n".join(formatted)


RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful policy assistant for NovaTech Solutions Pvt. Ltd.
Answer the employee's question based ONLY on the provided context.
If the context doesn't contain the answer, say \"I don't have that information in our policy documents.\"
If the answer can be inferred from the context, answer directly.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
)


@app.get("/health")
def health_check():
    return {"status": "ok", "source": "NovaTech Policy Assistant"}


@app.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        retriever = get_retriever()
        docs = retriever.invoke(request.question)
        docs = dedupe_docs(docs)
        if not docs:
            raise ValueError("No documents retrieved for the question.")

        context = format_docs(docs)
        llm = get_llm()
        chain = RAG_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": request.question})

        sources = []
        for idx, doc in enumerate(docs, start=1):
            source_name = Path(doc.metadata.get("source", "unknown")).name
            sources.append(f"{idx}. {source_name}")

        return ChatResponse(answer=answer.strip(), sources=sources)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {exc}")
