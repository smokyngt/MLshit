import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "startups")
DENSE_MODEL = os.getenv("DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "prithivida/Splade_PP_en_v1")
DATA_PATH = os.getenv("DATA_PATH", "data/startups_demo.json")

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"