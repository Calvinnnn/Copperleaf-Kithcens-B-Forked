"""Retriever module for Copperleaf Kitchens RAG pipeline.

Provides vector-similarity retrieval from ChromaDB with optional metadata
pre-filtering support (source, page, category) and configurable top-k.
"""

from typing import Any, Dict, List, Optional


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "copperleaf_documents"
DB_PATH = "rag/chroma_db"


def retrieve(
    query: str,
    top_k: int = 5,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve the most relevant document chunks for a user query.

    Args:
        query: Natural language search query.
        top_k: Maximum number of chunks to return (default 5).
        where: Optional ChromaDB metadata filter dict, e.g. {'source': 'menu.pdf'}.

    Returns:
        List of chunk dicts with keys: text, metadata, distance.
    """
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            f"RAG retrieval requires chromadb and sentence-transformers: {e}"
        ) from e

    model = SentenceTransformer(MODEL_NAME)

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0].tolist()

    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_collection(name=COLLECTION_NAME)

    query_kwargs: Dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
    }
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    retrieved_chunks: List[Dict[str, Any]] = []

    for i in range(len(results["documents"][0])):
        retrieved_chunks.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return retrieved_chunks


if __name__ == "__main__":
    query = input("Enter your question: ")
    results = retrieve(query)

    print("\nRetrieved chunks:\n")

    for index, result in enumerate(results, start=1):
        print(f"--- Result {index} ---")
        print(f"Source: {result['metadata']['source']}")
        print(f"Page: {result['metadata']['page']}")
        print(f"Distance: {result['distance']:.4f}")
        print(result["text"])
        print()