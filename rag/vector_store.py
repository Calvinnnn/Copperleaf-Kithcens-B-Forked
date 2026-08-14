"""Vector Store Module for Copperleaf Kitchens RAG Pipeline.

Uses ChromaDB as the persistent vector database with:
- HNSW (Hierarchical Navigable Small World) ANN index for approximate nearest-neighbour search
- Metadata payload store (source document, page number, chunk ID, document type)
- Metadata index enabling pre-search and mid-search filtering by document type or source
  before similarity scores are computed, not just as a post-retrieval filter.
"""

import chromadb
from chromadb.config import Settings

from rag.chunking import create_chunks
from rag.embeddings import create_embeddings


COLLECTION_NAME = "copperleaf_documents"
DB_PATH = "rag/chroma_db"

# ---------------------------------------------------------------------------
# HNSW index configuration (passed as collection metadata to ChromaDB).
# ChromaDB exposes these directly as hnsw: prefixed keys.
# M        — number of bi-directional links per node (higher = better recall,
#            more memory). 16 is the ChromaDB default; 32 improves recall on
#            larger corpora at modest memory cost.
# ef_construction — size of the dynamic candidate list during index build.
#                   200 gives a good recall/build-time trade-off.
# ef_search — candidate list size at query time. 100 is a safe default that
#             beats the ChromaDB default (10) significantly on recall.
# space     — distance metric. "cosine" matches sentence-transformer embeddings.
# ---------------------------------------------------------------------------
HNSW_CONFIG = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 200,
    "hnsw:M": 32,
    "hnsw:search_ef": 100,
}

# Metadata index fields — ChromaDB builds an inverted index over these fields
# so WHERE clause filters on them execute before (or during) the ANN search,
# not as a post-retrieval pass over the full result set.
METADATA_INDEX_FIELDS = ["source", "doc_type", "page"]


def create_vector_store():
    """Create a persistent ChromaDB collection with HNSW index and metadata indexing.

    Stores document chunks with their embeddings and rich metadata payloads.
    The HNSW configuration is embedded in the collection metadata so it
    persists across restarts without re-configuration.
    """

    print("Loading documents and creating chunks...")
    chunks = create_chunks()
    print(f"Total chunks: {len(chunks)}")

    print("Creating embeddings...")
    embeddings = create_embeddings(chunks)

    print("Connecting to ChromaDB (HNSW index, cosine space)...")
    client = chromadb.PersistentClient(path=DB_PATH)

    # Create or retrieve the collection.
    # The HNSW_CONFIG keys are only honoured on first creation; subsequent
    # calls to get_or_create_collection with an existing name are no-ops for
    # the config but still return the correctly configured collection.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Copperleaf Kitchens RAG knowledge base — operational policy documents",
            **HNSW_CONFIG,
        },
    )

    ids = []
    documents = []
    metadatas = []
    vectors = []

    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        ids.append(f"chunk_{index}")
        documents.append(chunk["text"])

        # Rich metadata payload — each field is indexed for pre-search filtering.
        source = chunk["metadata"]["source"]
        metadatas.append({
            "source": source,
            "page": chunk["metadata"]["page"],
            "chunk_id": chunk["metadata"]["chunk_id"],
            # Derive a doc_type tag from the filename for coarse-grained filtering.
            # e.g. "Food_Safety_Manual.pdf" → "food_safety"
            "doc_type": _derive_doc_type(source),
        })

        vectors.append(embedding.tolist())

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=vectors,
    )

    print()
    print("Vector store created successfully!")
    print(f"Collection      : {COLLECTION_NAME}")
    print(f"HNSW space      : {HNSW_CONFIG['hnsw:space']}")
    print(f"HNSW M          : {HNSW_CONFIG['hnsw:M']}")
    print(f"HNSW ef_search  : {HNSW_CONFIG['hnsw:search_ef']}")
    print(f"Documents stored: {collection.count()}")
    return collection


def get_collection():
    """Return the existing ChromaDB collection (raises if not yet created)."""
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(name=COLLECTION_NAME)


def query_vector_store(
    query_embedding: list,
    top_k: int = 5,
    where: dict | None = None,
):
    """Query the vector store with optional pre-search metadata filtering.

    The `where` clause is applied by ChromaDB's metadata index BEFORE the
    ANN similarity search runs — not as a post-retrieval filter. This means
    only chunks matching the metadata predicate are candidates for the HNSW
    search, reducing both latency and irrelevant results.

    Args:
        query_embedding: Dense embedding vector for the query.
        top_k:           Number of nearest neighbours to return.
        where:           Optional ChromaDB metadata filter dict, e.g.
                         {"doc_type": "food_safety"} or
                         {"$or": [{"doc_type": "food_safety"},
                                  {"doc_type": "waste_management"}]}

    Returns:
        List of dicts with keys: text, metadata, distance.

    Example — filter to food-safety documents only:
        results = query_vector_store(
            query_embedding=embed("fasting window before sedation"),
            top_k=5,
            where={"doc_type": "food_safety"},
        )
    """
    collection = get_collection()

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    # Pass the metadata filter into ChromaDB — executed pre-ANN-search.
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({"text": doc, "metadata": meta, "distance": dist})

    return chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _derive_doc_type(source: str) -> str:
    """Map a PDF filename to a short doc_type tag used in metadata filtering."""
    source_lower = source.lower()
    if "food_safety" in source_lower:
        return "food_safety"
    if "waste" in source_lower:
        return "waste_management"
    if "supplier" in source_lower or "procurement" in source_lower:
        return "procurement"
    if "compliance" in source_lower:
        return "compliance"
    if "employee" in source_lower or "handbook" in source_lower:
        return "hr"
    if "branch" in source_lower or "operations" in source_lower:
        return "operations"
    if "casebook" in source_lower:
        return "casebook"
    return "general"


if __name__ == "__main__":
    create_vector_store()