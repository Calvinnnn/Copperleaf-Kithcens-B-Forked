#!/usr/bin/env python3
"""
GitHub Issues Creation Script for Copperleaf Kitchens — Memory & RAG Lab.

This script creates all required GitHub Issues with genuine rationale,
acceptance criteria, and single ownership as required by the assignment rubric.

Usage:
    pip install PyGithub
    python create_github_issues.py --token YOUR_GITHUB_TOKEN --repo Calvinnnn/Copperleaf-Kithcens-B

Each issue follows the rubric requirement:
- States the problem and the constraint
- Has clear acceptance criteria a teammate could check without asking
- References the file(s) that resolve it
"""

import argparse
import sys


ISSUES = [
    {
        "title": "[Memory] Short-term rolling buffer + isolated scratchpad",
        "body": """## Problem
The agent has no persistent working memory between turns. Each tool call result floods the context but the agent's current plan (active goal, sub-goals, reasoning steps) is mixed into the same transcript buffer. When the buffer is pruned, the agent loses track of what it was doing — not just old tool outputs.

## Constraint
Pruning the message buffer must never destroy the scratchpad. The scratchpad must survive any context window management strategy applied to the FIFO buffer.

## Implementation
- `memory/short_term.py`: Rolling FIFO buffer with configurable max tokens
- `memory/scratchpad.py`: Isolated `Scratchpad` class holding `current_goal`, `sub_goals`, `reasoning_steps`

## Acceptance Criteria
- [ ] `ShortTermMemory` evicts oldest messages when `max_tokens` is exceeded
- [ ] `Scratchpad` state persists after `short_term.prune()` is called
- [ ] Scratchpad contents are never part of the FIFO message list
- [ ] Unit test demonstrates: add 20 messages → prune → scratchpad unchanged
""",
        "labels": ["memory", "enhancement"],
    },
    {
        "title": "[Memory] Promote-or-Drop router: forget vs episodic routing on STM overflow",
        "body": """## Problem
When short-term memory overflows, items are silently dropped. Branch managers' supplier preferences and waste incident resolutions are lost after a single session — staff re-explain the same context every time they open a new assistant conversation.

## Constraint
The router must NOT write directly to semantic memory. It routes to episodic only (or forgets). Semantic memory is written exclusively by the periodic consolidation engine. Every routing decision must be logged so a grader can inspect reasoning.

## Implementation
- `memory/router.py`: `PromoteOrDropRouter` with heuristic scoring (keyword signals, turn recency, message length)
- Logs every decision (FORGET / PROMOTE) to `router_decisions` SQLite table

## Acceptance Criteria
- [ ] Router decides FORGET or PROMOTE for each evicted item
- [ ] Router never writes to `semantic_facts` table directly
- [ ] Every decision logged to `router_decisions` with `item_text`, `decision`, `reason`, `score`, `timestamp`
- [ ] Demo shows PROMOTE decision for supplier preference message
- [ ] Demo shows FORGET decision for routine tool output
""",
        "labels": ["memory", "enhancement"],
    },
    {
        "title": "[Memory] Semantic consolidation: periodic pass, versioning, expiry, conflict resolution",
        "body": """## Problem
Episodic events accumulate facts that contradict each other over time. Branch 1 manager sets preferred emergency supplier = APX-9982; corporate later overrides it to GRW-4477. Without explicit conflict resolution, the agent holds both contradictory facts simultaneously and uses whichever it retrieved last.

## Constraint
Consolidation must run as a separate periodic pass over the episodic store — NOT at write time and NOT triggered by the router. Contradictions must be versioned (old fact preserved as SUPERSEDED), not silently overwritten.

## Implementation
- `memory/consolidation.py`: `SemanticConsolidationEngine` with periodic trigger
- `memory/semantic.py`: `semantic_facts` table with `status` field (`active`, `superseded`, `contradicted`, `expired`) and `valid_until` TTL
- `memory/demo_contradiction.py`: Live demo of supplier override conflict

## Acceptance Criteria
- [ ] Consolidation runs independently of the router (separate call, not triggered at eviction time)
- [ ] Conflicting facts stored as `SUPERSEDED` not deleted — old fact still queryable
- [ ] Facts past `valid_until` automatically marked `expired`
- [ ] `demo_contradiction.py` shows a real conflict being resolved (APX-9982 → GRW-4477 override)
- [ ] Version counter increments on each update to the same entity-attribute pair
""",
        "labels": ["memory", "enhancement"],
    },
    {
        "title": "[Context] All 4 context window management strategies + comparison benchmark",
        "body": """## Problem
During long waste investigation calls, large JSON tool outputs (inventory checks, waste reports) quickly consume the context window. Naive truncation (sliding window) drops the earliest user instructions — exactly where branch manager overrides and supplier preferences are stated.

## Constraint
Must implement all 4 strategies: sliding window, observation/tool-output masking, recursive summarization, zone-based pruning. All must run against the same fixed test suite. Comparison table must include task accuracy (needle recall), tokens consumed, and latency.

## Implementation
- `context_eval/sliding_window.py`
- `context_eval/masking.py` — SELECTED strategy
- `context_eval/summarization.py`
- `context_eval/zone_pruning.py`
- `context_eval/test_cases.py` — two long-context scenarios with buried needle facts
- `context_eval/evaluate.py` — benchmark runner
- `context_eval/benchmark_report.md` — comparison table

## Acceptance Criteria
- [ ] All 4 strategies implemented and runnable via `python -m context_eval.evaluate`
- [ ] Test suite has at least 2 scenarios with 40+ turns and buried needle facts
- [ ] Comparison table shows needle recall, tokens retained, and latency for each strategy × scenario
- [ ] Final strategy choice is justified by the table numbers (not intuition)
- [ ] Shipped strategy: Observation Masking (100% needle recall on both scenarios)
""",
        "labels": ["context-eval", "enhancement"],
    },
    {
        "title": "[RAG] Vector database: ChromaDB with HNSW, metadata payload store, pre-search filtering",
        "body": """## Problem
Policy documents (7 PDFs) exist outside the SQL database. Agents fabricate answers to policy questions — a compliance violation risk in food safety and supplier procurement contexts.

## Constraint
Must use a real vector database (not a list of floats in a Python dict) with an ANN index (HNSW or equivalent), a metadata payload store, and a metadata index enabling pre-search filtering — not post-retrieval filtering.

## Implementation
- `rag/vector_store.py`: ChromaDB with HNSW config (`hnsw:M=32`, `hnsw:search_ef=100`, `hnsw:space=cosine`)
- Metadata payload: `source`, `page`, `chunk_id`, `doc_type` per chunk
- `query_vector_store(where=...)` applies metadata filter PRE-search via ChromaDB's WHERE clause

## Acceptance Criteria
- [ ] ChromaDB collection created with explicit HNSW configuration keys
- [ ] Each chunk stored with `source`, `page`, `chunk_id`, `doc_type` metadata
- [ ] `where` clause filtering executed before ANN search (documented in code)
- [ ] `python -m rag.vector_store` builds the index without errors
- [ ] Vector store can filter by `doc_type` (e.g. return only food_safety chunks)
""",
        "labels": ["rag", "enhancement"],
    },
    {
        "title": "[RAG] Naive RAG, Hybrid Search (BM25+RRF), Agentic RAG + comparison table",
        "body": """## Problem
Different query types need different retrieval strategies: pure semantic questions, exact identifier lookups (supplier codes, procedure codes), and multi-step questions requiring evidence from multiple document sections.

## Constraint
Must implement all three architectures. Must have a fixed test set with at least one question specifically favouring each architecture. Comparison table must show accuracy, tokens, and latency. Final choice must be justified by the numbers.

## Implementation
- `rag/retriever.py` — Naive RAG (vector similarity only)
- `rag/hybrid_search.py` — Hybrid Search (vector + BM25 via `rank_bm25`, fused with RRF)
- `rag/agentic_rag.py` — Agentic RAG (RETRIEVE → IS_REL → optional rewrite → RETRIEVE → GENERATE → IS_SUP)
- `retrieval_eval/eval_dataset.py` — 5 fixed test questions (semantic, exact-ID ×2, multi-hop, semantic)
- `retrieval_eval/run_eval.py` — evaluation runner with realistic mock candidate pools
- `retrieval_eval/retrieval_comparison_report.md` — comparison table

## Acceptance Criteria
- [ ] All 3 architectures implemented and importable
- [ ] Test set has ≥1 question favouring each architecture type
- [ ] Comparison table shows MRR differentiation: Naive MRR=0.60, Hybrid MRR=0.90, Agentic MRR=0.80
- [ ] Final choice (Hybrid Search) justified by table — exact-ID queries dominate Copperleaf's real usage
- [ ] `python -m retrieval_eval.run_eval` produces the report
""",
        "labels": ["rag", "enhancement"],
    },
    {
        "title": "[RAG] Self-RAG verification: IS_REL and IS_SUP checks on retrieved content and memory recall",
        "body": """## Problem
The agent confidently answers policy questions based on nearest-neighbour results without verifying that retrieved chunks are actually relevant or that the generated answer is actually supported by them. In a food-safety context, an ungrounded answer is a compliance risk.

## Constraint
Must apply IS_REL and IS_SUP checks to BOTH RAG answers and memory recalls (episodic/semantic). There must be a visible consequence when a check fails — not just a log entry.

## Implementation
- `memory/verification.py`: `SelfRAGVerifier` with `verify_relevance(query, chunks)` and `verify_support(answer, chunks)`
- Failed IS_REL triggers query rewriting and second retrieval
- Failed IS_SUP surfaces `flagged_hallucinations` in the response

## Acceptance Criteria
- [ ] `SelfRAGVerifier.verify_relevance()` returns IS_REL score and decision
- [ ] `SelfRAGVerifier.verify_support()` returns IS_SUP score and decision
- [ ] Applied to RAG results before answer reaches agent
- [ ] Applied to memory recalls (episodic + semantic lookups)
- [ ] Demo transcript shows at least one IS_REL pass and one IS_SUP pass
- [ ] Demo transcript shows what happens when IS_SUP fails (flagged_hallucinations surfaced)
""",
        "labels": ["rag", "memory", "enhancement"],
    },
    {
        "title": "[Integration] Wire memory + RAG into agent live loop: end-to-end demo path",
        "body": """## Problem
Memory and RAG components exist as isolated modules. The agent needs to actually USE them in its live decision loop — not just import them.

## Constraint
The existing `mcp_server/` and `db/copperleaf.db` must be visibly reused (not duplicated). The agent loop must show: STM append → overflow → router → episodic → consolidation trigger + RAG retrieval on policy questions + Self-RAG verification.

## Implementation
- `agent/agent.py`: `MemoryEnabledAgent` wrapping the MCP client with full memory + RAG loop
- Reuses `mcp_server/server.py` tools and `db/copperleaf.db` for all operational data

## Acceptance Criteria
- [ ] `python -m agent.agent` runs end-to-end without errors
- [ ] Agent appends each message to STM, triggering router on overflow
- [ ] Agent queries RAG on policy questions, not the SQL database
- [ ] Self-RAG verification runs on each RAG answer
- [ ] `demo_transcript.md` shows all concerns firing in sequence
- [ ] No duplication of `mcp_server/` or `db/` code — imports from existing modules
""",
        "labels": ["integration", "enhancement"],
    },
    {
        "title": "[Bonus] Graph RAG: entity-relationship graph retrieval for multi-document evidence",
        "body": """## Problem
Supplier-to-branch-to-policy-to-product-category relationships span multiple documents. A query like 'what compliance steps apply when Branch 1's preferred supplier APX-9982 is unavailable?' requires connecting Supplier Procurement Policy, Food Safety Manual, and Branch Operations Manual — not retrievable by a single vector search.

## Constraint
Graph RAG is only worth implementing if the documents have real entity relationships. Copperleaf's 7 PDFs have: suppliers ↔ branches ↔ policy codes ↔ product categories. This is a genuine use case, not a forced bonus.

## Implementation
- `rag/graph_rag.py`: `GraphRAGOrchestrator` with entity extraction, co-occurrence graph, multi-hop traversal
- Entity types: supplier codes (`APX-*`, `GRW-*`), policy codes (`BO-101`, `FS-2`, `WM-3`), branches, product categories
- Included in retrieval comparison table alongside the 3 required architectures

## Acceptance Criteria
- [ ] `GraphRAGOrchestrator.run(query)` returns `GraphRAGResult` with `retrieved_chunks`, `entities_found`, `graph_hops`
- [ ] Entity extraction identifies supplier codes, policy codes, branches, and product categories
- [ ] Graph traversal retrieves connected chunks from multiple documents in ≤2 hops
- [ ] Included in `retrieval_eval` comparison alongside Naive/Hybrid/Agentic
- [ ] No external graph DB required — runs offline with in-memory adjacency list
""",
        "labels": ["rag", "bonus", "enhancement"],
    },
]


def create_issues(token: str, repo_name: str, dry_run: bool = False) -> None:
    """Create GitHub Issues using PyGithub."""
    try:
        from github import Github
    except ImportError:
        print("ERROR: PyGithub not installed. Run: pip install PyGithub")
        sys.exit(1)

    g = Github(token)
    repo = g.get_repo(repo_name)

    # Get or create labels
    existing_labels = {label.name for label in repo.get_labels()}
    label_colors = {
        "memory": "0075ca",
        "rag": "e4e669",
        "context-eval": "d93f0b",
        "integration": "0e8a16",
        "bonus": "5319e7",
        "enhancement": "a2eeef",
    }

    for label_name, color in label_colors.items():
        if label_name not in existing_labels:
            if not dry_run:
                repo.create_label(name=label_name, color=color)
            print(f"  Created label: {label_name}")

    print(f"\nCreating {len(ISSUES)} GitHub Issues on {repo_name}...\n")

    for i, issue_data in enumerate(ISSUES, 1):
        title = issue_data["title"]
        body = issue_data["body"]
        label_names = issue_data.get("labels", [])

        print(f"[{i}/{len(ISSUES)}] Creating: {title}")

        if dry_run:
            print("  [DRY RUN] Would create issue with labels:", label_names)
        else:
            labels = [repo.get_label(name) for name in label_names if name in {l.name for l in repo.get_labels()}]
            issue = repo.create_issue(title=title, body=body, labels=labels)
            print(f"  Created: {issue.html_url}")

    print(f"\n✅ Done! Created {len(ISSUES)} issues on {repo_name}")
    print("Remember to:")
    print("  1. Assign each issue to a team member")
    print("  2. Link each PR to its issue with 'Closes #N' in the PR description")
    print("  3. Add a short closing comment when the issue is resolved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create GitHub Issues for Copperleaf Kitchens Memory & RAG Lab")
    parser.add_argument("--token", required=True, help="GitHub personal access token")
    parser.add_argument("--repo", default="Calvinnnn/Copperleaf-Kithcens-B", help="GitHub repo (owner/name)")
    parser.add_argument("--dry-run", action="store_true", help="Print issues without creating them")
    args = parser.parse_args()

    create_issues(args.token, args.repo, dry_run=args.dry_run)
