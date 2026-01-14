import os
import hashlib
import redis
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ---------------- CONFIG ---------------- #

DB_CONNECTION = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")

app = FastAPI(title="Document Intelligence System")
redis_client = redis.Redis.from_url(REDIS_URL)

# ---------------- CORPUS VERSION ---------------- #

def get_corpus_version():
    v = redis_client.get("corpus_version")
    if not v:
        redis_client.set("corpus_version", 1)
        return "1"
    return v.decode()

def bump_corpus_version():
    redis_client.incr("corpus_version")

# ---------------- VECTOR STORE ---------------- #

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="pdf_collection",
    connection=DB_CONNECTION,
    use_jsonb=True,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ---------------- LLM ---------------- #

llm = ChatOllama(model="llama3.2", temperature=0)

# ---------------- RAG PIPELINE ---------------- #

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# ---------------- CACHE (VERSIONED) ---------------- #

def cache_key(query: str):
    version = get_corpus_version()
    raw = f"{version}:{query}"
    return hashlib.sha256(raw.encode()).hexdigest()

def get_cached_response(query: str):
    key = cache_key(query)
    cached = redis_client.get(key)
    return cached.decode() if cached else None

def save_to_cache(query: str, response: str):
    key = cache_key(query)
    redis_client.setex(key, 3600, response)

# ---------------- API ---------------- #

class QueryRequest(BaseModel):
    question: str
    session_id: str

@app.post("/chat")
async def chat(request: QueryRequest):
    cached = get_cached_response(request.question)
    if cached:
        return {"answer": cached, "source": "cache"}

    rag_chain = get_rag_chain()
    answer = rag_chain.invoke(request.question)

    save_to_cache(request.question, answer)

    return {"answer": answer, "source": "llm"}

@app.post("/upload")
async def upload_document(file: UploadFile):
    path = f"temp_{file.filename}"

    with open(path, "wb") as f:
        f.write(await file.read())

    from ingest import ingest_pdf
    ingest_pdf(path)

    # 🔥 Invalidate all previous cache
    bump_corpus_version()

    os.remove(path)
    return {"message": "File processed and indexed successfully"}
