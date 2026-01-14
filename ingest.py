import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

DB_CONNECTION = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")
print("DB_CONNECTION =", DB_CONNECTION)

def ingest_pdf(file_path: str):
    # 1. Load the PDF
    print("Loading PDF...")
    loader = PyPDFLoader(file_path)
    raw_documents = loader.load()

    # 2. Chunking (The Context Window optimization)
    # We chop text into 500-token blocks. Overlap ensures sentences aren't cut in half awkwardly.
    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    documents = text_splitter.split_documents(raw_documents)

    # 3. Embed and Store in Postgres
    print("Embedding and storing...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # This automatically converts text -> numbers and inserts into the DB
    PGVector.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="pdf_collection",
        connection=DB_CONNECTION,
        use_jsonb=True,
    )
    print("Done!")

# Run this once to test
if __name__ == "__main__":
    # Create a dummy PDF or use a real one to test
    ingest_pdf("sample.pdf")