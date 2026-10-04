"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'gemini')
LLM_MODEL = os.getenv('LLM_MODEL', 'gemini-3.5-flash-lite')
EVAL_MODEL = os.getenv('EVAL_MODEL', LLM_MODEL)
ENABLE_LLM_EVAL = os.getenv('ENABLE_LLM_EVAL', '0') == '1'
ENABLE_LLM_ENRICHMENT = os.getenv('ENABLE_LLM_ENRICHMENT', '0') == '1'

# --- Qdrant ---
QDRANT_HOST = os.getenv('QDRANT_HOST', '127.0.0.1')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', '6333'))
COLLECTION_NAME = 'lab18_duy_production'
NAIVE_COLLECTION = 'lab18_duy_naive'

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
