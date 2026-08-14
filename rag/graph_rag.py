"""
Graph RAG Module for Copperleaf Kitchens.

Why Graph RAG fits Copperleaf's data:
Copperleaf's documents have real entity relationships worth modeling as a graph:
- Suppliers (e.g. APX-9982, GRW-4477) are linked to specific branches and products
- Policy codes (BO-101, WM-3, FS-2) are linked to document sections and compliance requirements  
- Branches are linked to their compliance status, write-off thresholds, and preferred suppliers
- Products/ingredients are linked to storage requirements and waste policies

These relationships span multiple documents and cannot be retrieved by a single vector
similarity query. Graph RAG traverses these entity relationships to surface connected
evidence from multiple chunks.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False


@dataclass
class GraphRAGResult:
    query: str
    answer: str
    retrieved_chunks: List[Dict[str, Any]]
    entities_found: List[str]
    graph_hops: int
    relevant_chunks: List[Dict[str, Any]] = field(default_factory=list)


# Entity patterns specific to Copperleaf Kitchens documents
ENTITY_PATTERNS = [
    (r'\b(APX-\d+|GRW-\d+|FSP-\d+|CPL-\d+)\b', 'supplier_code'),
    (r'\b(BO-\d+|WM-\d+|FS-\d+|HR-\d+|CC-\d+|SC-\d+)\b', 'policy_code'),
    (r'\b[Bb]ranch\s*\d+\b', 'branch'),
    (r'\b(dairy|produce|seafood|meat|poultry|bakery|frozen)\b', 'product_category'),
    (r'\b(write-off|write off|spoilage|waste|threshold|compliance|procurement)\b', 'concept'),
]


class EntityExtractor:
    """Extract named entities from chunk text using domain-specific regex patterns."""

    def extract(self, text: str) -> List[Tuple[str, str]]:
        """Return list of (entity_text, entity_type) tuples."""
        entities = []
        for pattern, etype in ENTITY_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append((match.group(0).lower().strip(), etype))
        return entities


class CopperleafKnowledgeGraph:
    """In-memory entity graph built from RAG document chunks."""

    def __init__(self):
        self._nodes: Dict[str, Dict[str, Any]] = {}  # entity -> {type, chunks}
        self._edges: Dict[str, Set[str]] = defaultdict(set)  # entity -> {related_entities}
        self._chunk_index: List[Dict[str, Any]] = []
        self._extractor = EntityExtractor()

    def build_from_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Build graph from a list of chunk dicts (keys: text, metadata)."""
        self._chunk_index = chunks
        for chunk_idx, chunk in enumerate(chunks):
            text = chunk.get('text', '')
            entities = self._extractor.extract(text)
            chunk_entities = []
            for ent_text, ent_type in entities:
                if ent_text not in self._nodes:
                    self._nodes[ent_text] = {'type': ent_type, 'chunk_indices': []}
                self._nodes[ent_text]['chunk_indices'].append(chunk_idx)
                chunk_entities.append(ent_text)
            # Co-occurrence edges: entities in same chunk are connected
            for i, e1 in enumerate(chunk_entities):
                for e2 in chunk_entities[i+1:]:
                    if e1 != e2:
                        self._edges[e1].add(e2)
                        self._edges[e2].add(e1)

    def find_relevant_chunks(
        self,
        query_entities: List[str],
        hop_depth: int = 2,
        max_chunks: int = 8,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Traverse graph from query entities to collect relevant chunks."""
        visited_entities: Set[str] = set()
        frontier = set(query_entities)
        collected_chunk_indices: Set[int] = set()
        total_hops = 0

        for hop in range(hop_depth):
            if not frontier:
                break
            next_frontier: Set[str] = set()
            for entity in frontier:
                if entity in self._nodes:
                    visited_entities.add(entity)
                    collected_chunk_indices.update(self._nodes[entity]['chunk_indices'])
                    # Expand to neighbours
                    for neighbour in self._edges.get(entity, set()):
                        if neighbour not in visited_entities:
                            next_frontier.add(neighbour)
            frontier = next_frontier
            total_hops = hop + 1
            if len(collected_chunk_indices) >= max_chunks:
                break

        retrieved = [
            self._chunk_index[i]
            for i in sorted(collected_chunk_indices)[:max_chunks]
            if i < len(self._chunk_index)
        ]
        return retrieved, total_hops


class GraphRAGOrchestrator:
    """
    Graph RAG orchestrator for Copperleaf Kitchens.

    Retrieval flow:
    1. Extract entities from query (supplier codes, policy codes, branches, concepts)
    2. Look up entities in the knowledge graph
    3. Traverse graph edges (co-occurrence links) to collect related chunks
    4. Generate answer from multi-hop evidence
    """

    def __init__(
        self,
        chunks: Optional[List[Dict[str, Any]]] = None,
        hop_depth: int = 2,
        max_chunks: int = 8,
    ):
        self._hop_depth = hop_depth
        self._max_chunks = max_chunks
        self._extractor = EntityExtractor()
        self._graph = CopperleafKnowledgeGraph()
        if chunks:
            self._graph.build_from_chunks(chunks)

    def build_graph(self, chunks: List[Dict[str, Any]]) -> None:
        """Build or rebuild the knowledge graph from document chunks."""
        self._graph.build_from_chunks(chunks)

    def run(self, query: str) -> GraphRAGResult:
        """
        Execute graph-based retrieval for the given query.

        Args:
            query: Natural language query string.

        Returns:
            GraphRAGResult with retrieved chunks, entities found, and hop count.
        """
        # Step 1: Extract entities from query
        query_entity_pairs = self._extractor.extract(query)
        query_entities = [ent for ent, _ in query_entity_pairs]

        # Step 2: Graph traversal to find relevant chunks
        retrieved_chunks, graph_hops = self._graph.find_relevant_chunks(
            query_entities,
            hop_depth=self._hop_depth,
            max_chunks=self._max_chunks,
        )

        # Step 3: Build answer summary from retrieved evidence
        if retrieved_chunks:
            evidence_snippets = [
                f"[Chunk from {c.get('metadata', {}).get('source', 'unknown')}]: "
                f"{c.get('text', '')[:200]}"
                for c in retrieved_chunks[:3]
            ]
            answer = (
                f"Graph RAG retrieved {len(retrieved_chunks)} chunks via "
                f"{graph_hops}-hop graph traversal from entities: {query_entities}. "
                f"Evidence: " + " | ".join(evidence_snippets)
            )
        else:
            answer = (
                f"Graph RAG found no direct entity matches for: {query_entities}. "
                "Falling back to all-chunk fallback — consider expanding entity patterns."
            )

        return GraphRAGResult(
            query=query,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            relevant_chunks=retrieved_chunks,
            entities_found=list(set(query_entities)),
            graph_hops=graph_hops,
        )
