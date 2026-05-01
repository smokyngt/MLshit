import json
from pathlib import Path

from qdrant_client import QdrantClient, models
from tqdm import tqdm

from src.config import (
    QDRANT_URL,
    COLLECTION_NAME,
    DENSE_MODEL,
    SPARSE_MODEL,
    DATA_PATH,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
)

# Set to None to index the full dataset, or an int to limit (e.g. 1000)
SAMPLE_SIZE: int | None = 1000


def create_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists. Skipping creation.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=client.get_embedding_size(DENSE_MODEL),
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams()
        },
    )
    print(f"Collection '{COLLECTION_NAME}' created.")


def load_data(path: str, sample_size: int | None = None) -> tuple[list[dict], list[dict]]:
    documents = []
    metadata = []

    with open(path) as fd:
        for i, line in enumerate(fd):
            if sample_size is not None and i >= sample_size:
                break

            obj = json.loads(line)
            description = obj["description"]
            documents.append({
                DENSE_VECTOR_NAME: models.Document(text=description, model=DENSE_MODEL),
                SPARSE_VECTOR_NAME: models.Document(text=description, model=SPARSE_MODEL),
            })
            metadata.append(obj)

    return documents, metadata


def main() -> None:
    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            f"Download it with: wget https://storage.googleapis.com/generall-shared-data/startups_demo.json -O {DATA_PATH}"
        )

    client = QdrantClient(url=QDRANT_URL)
    create_collection(client)

    if SAMPLE_SIZE is not None:
        print(f"Loading data (sampling first {SAMPLE_SIZE} documents)...")
    else:
        print("Loading data (full dataset)...")

    documents, metadata = load_data(DATA_PATH, sample_size=SAMPLE_SIZE)
    print(f"Loaded {len(documents)} documents.")

    print("Encoding and uploading (this may take a while)...")
    client.upload_collection(
        collection_name=COLLECTION_NAME,
        vectors=tqdm(documents),
        payload=metadata,
    )
    print("Done.")


if __name__ == "__main__":
    main()