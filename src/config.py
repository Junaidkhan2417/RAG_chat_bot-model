import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_PATH = DATA_DIR / "Ebook-Agentic-AI.pdf"

# Load .env file from project root or parent workspace
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = 1536

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Chunking & Retrieval parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "3"))

# Server configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

def has_live_credentials() -> bool:
    """Check if live API credentials for OpenAI and Pinecone are configured."""
    return bool(OPENAI_API_KEY and PINECONE_API_KEY and PINECONE_API_KEY != "your_pinecone_api_key_here")

def get_config_summary() -> dict:
    """Return non-sensitive configuration parameters for logging and inspection."""
    return {
        "project_root": str(PROJECT_ROOT),
        "pdf_path": str(PDF_PATH),
        "pdf_exists": PDF_PATH.exists(),
        "openai_model": OPENAI_MODEL,
        "openai_embedding_model": OPENAI_EMBEDDING_MODEL,
        "pinecone_index_name": PINECONE_INDEX_NAME,
        "pinecone_cloud": PINECONE_CLOUD,
        "pinecone_region": PINECONE_REGION,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "has_live_credentials": has_live_credentials(),
    }
