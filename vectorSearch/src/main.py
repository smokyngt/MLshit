from fastapi import FastAPI
from hybrid_searcher import HybridSearcher

app = FastAPI()
searcher = HybridSearcher()

@app.get("/")
def root():
    return {"message": "Hybrid Search API"}

@app.get("/search")
def search(q: str):
    return {"results": searcher.search(q)}