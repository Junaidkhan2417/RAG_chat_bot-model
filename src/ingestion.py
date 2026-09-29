import os
import json
import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

from src.config import (
    PDF_PATH,
    DATA_DIR,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    has_live_credentials,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LOCAL_CHUNKS_PATH = DATA_DIR / "processed_chunks.json"


def load_and_split_pdf(pdf_path: str = str(PDF_PATH)) -> List[Document]:
    """
    Load document using PyPDFLoader and divide into manageable chunks
    using RecursiveCharacterTextSplitter.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    logger.info(f"Loading document from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} pages from PDF.")

    logger.info(f"Chunking with chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Generated {len(chunks)} total text chunks.")

    # Save local cache of chunks for transparency and local retrieval fallback
    save_chunks_locally(chunks, LOCAL_CHUNKS_PATH)
    return chunks


def save_chunks_locally(chunks: List[Document], dest_path: Path):
    """Save extracted chunks to JSON for auditability and local fallback."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [
        {
            "id": f"chunk-{i}",
            "text": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for i, chunk in enumerate(chunks)
    ]
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(serialized)} chunks to local cache: {dest_path}")


def setup_pinecone_index(index_name: str = PINECONE_INDEX_NAME) -> Optional[object]:
    """
    Ensure the target Pinecone serverless index exists with cosine distance and dimension 1536.
    """
    if not PINECONE_AVAILABLE or not PINECONE_API_KEY:
        logger.warning("Pinecone API key not configured. Skipping remote index creation.")
        return None

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [idx["name"] if isinstance(idx, dict) else idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        logger.info(f"Creating Pinecone index '{index_name}' (dimension={EMBEDDING_DIMENSION}, metric=cosine)...")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        logger.info(f"Index '{index_name}' successfully created.")
    else:
        logger.info(f"Pinecone index '{index_name}' already exists.")

    return pc.Index(index_name)


def run_ingestion(pdf_path: str = str(PDF_PATH), index_name: str = PINECONE_INDEX_NAME):
    """
    Complete ingestion pipeline:
    1. Load PDF document
    2. Chunk document using RecursiveCharacterTextSplitter
    3. Initialize Pinecone index (dimension 1536, cosine metric)
    4. Upsert embeddings into Pinecone vector store
    """
    logger.info("=" * 60)
    logger.info("STARTING RAG DOCUMENT INGESTION PIPELINE")
    logger.info("=" * 60)

    # 1. Load and chunk document
    chunks = load_and_split_pdf(pdf_path)

    # 2. Check credentials
    if not has_live_credentials():
        logger.warning(
            "⚠️ Live OpenAI / Pinecone credentials are not configured in .env.\n"
            "   Chunks have been successfully parsed and stored in data/processed_chunks.json.\n"
            "   When credentials are provided in .env, run this script again to index into Pinecone."
        )
        return {
            "status": "cached_locally",
            "chunks_count": len(chunks),
            "message": "Chunks parsed and cached locally. Configure .env with valid keys for live Pinecone upsert.",
        }

    # 3. Setup Pinecone index
    index = setup_pinecone_index(index_name)

    # 4. Generate embeddings and upsert
    logger.info(f"Initializing OpenAI embeddings with model: {OPENAI_EMBEDDING_MODEL}")
    embeddings = OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )

    logger.info(f"Upserting {len(chunks)} chunks into Pinecone index '{index_name}'...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    idx = pc.Index(index_name)

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        
        # Embed texts
        vector_embeddings = embeddings.embed_documents(texts)
        
        # Prepare vectors for Pinecone upsert
        vectors_to_upsert = []
        for j, (text, meta, emb) in enumerate(zip(texts, metadatas, vector_embeddings)):
            meta_clean = {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}
            meta_clean["text"] = text
            vectors_to_upsert.append({
                "id": f"doc-{i + j}",
                "values": emb,
                "metadata": meta_clean,
            })
        
        idx.upsert(vectors=vectors_to_upsert)
        logger.info(f"Upserted vectors {i + 1} to {min(i + len(batch), len(chunks))} of {len(chunks)}")

    logger.info("✅ Ingestion and vector indexing completed successfully!")
    return {
        "status": "success",
        "chunks_count": len(chunks),
        "index_name": index_name,
    }


if __name__ == "__main__":
    run_ingestion()
