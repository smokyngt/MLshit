from fastapi import FastAPI, Query

from src.config import COLLECTION_NAME
from src.hybrid_searcher import HybridSearcher

app = FastAPI(title="Hybrid Search API")
searcher = HybridSearcher(collection_name=COLLECTION_NAME)


@app.get("/api/search")
def search_startup(
    q: str = Query(..., description="Search query"),
    limit: int = Query(5, ge=1, le=50),
    city: str | None = Query(None, description="Optional city filter"),
):
    return {
        "query": q,
        "results": searcher.search(text=q, limit=limit, city=city),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)