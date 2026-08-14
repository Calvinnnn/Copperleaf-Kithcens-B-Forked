from sentence_transformers import SentenceTransformer

from rag.chunking import create_chunks


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def create_embeddings(chunks):
    """
    Convert text chunks into numerical vector embeddings.
    """

    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return embeddings


if __name__ == "__main__":

    print("Loading documents and creating chunks...")

    chunks = create_chunks()

    print(f"Total chunks: {len(chunks)}")
    print()

    print("Creating embeddings...")

    embeddings = create_embeddings(chunks)

    print()
    print("Embeddings created successfully!")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")