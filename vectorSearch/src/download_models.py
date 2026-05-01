from fastembed import TextEmbedding, SparseTextEmbedding
from src.config import DENSE_MODEL, SPARSE_MODEL


def main():
    print(f"Downloading dense model: {DENSE_MODEL}...")
    TextEmbedding(model_name=DENSE_MODEL)
    print("Dense model ready.")

    print(f"Downloading sparse model: {SPARSE_MODEL}...")
    SparseTextEmbedding(model_name=SPARSE_MODEL)
    print("Sparse model ready.")

    print("All models downloaded.")


if __name__ == "__main__":
    main()