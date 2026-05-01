from qdrant_client import QdrantClient, models

from src.config import (
    QDRANT_URL,
    DENSE_MODEL,
    SPARSE_MODEL,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
)


class HybridSearcher:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.client = QdrantClient(url=QDRANT_URL)

    def search(
        self,
        text: str,
        limit: int = 5,
        city: str | None = None,
    ) -> list[dict]:
        query_filter = None
        if city:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="city",
                        match=models.MatchValue(value=city),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.Document(text=text, model=DENSE_MODEL),
                    using=DENSE_VECTOR_NAME,
                    limit=20,
                ),
                models.Prefetch(
                    query=models.Document(text=text, model=SPARSE_MODEL),
                    using=SPARSE_VECTOR_NAME,
                    limit=20,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=limit,
        ).points

        return [point.payload for point in results]