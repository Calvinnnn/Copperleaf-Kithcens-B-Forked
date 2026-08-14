# Context Window Management Strategy Comparison Report

**Evaluation Timestamp**: 2026-08-08T07:19:42.077857+00:00

**Max Token Budget Limit**: 1200 tokens

**Test Suite**: Two long-context scenarios simulating real Copperleaf Kitchens inventory workflows:
- **Scenario 1 — Inventory Waste Investigation Benchmark**: A multi-turn investigation where a branch manager and agent work through a waste report. A critical policy override (preferred emergency supplier preference) is buried early in turn 3 and must survive to turn 40+ when the final write-off approval is requested.
- **Scenario 2 — 50-Turn Extreme Scale Benchmark**: A 50-turn conversation dominated by large JSON tool outputs (inventory checks, waste logs, SQL tables). A supplier preference decision from turn 5 must remain accessible at the final decision turn.

---

## Comparison Table — All 4 Required Strategies

| Strategy | Scenario | Orig Tokens | Retained | Saved | Reduction | Needle Recall | Latency |
|---|---|---|---|---|---|---|---|
| **Sliding Window** | Inventory Waste Investigation | 1522 | 1196 | 326 | 21.4% | **0.0%** | 0.06ms |
| **Observation Masking** | Inventory Waste Investigation | 1522 | 1082 | 440 | 28.9% | **100.0%** | 0.09ms |
| **Recursive Summarization** | Inventory Waste Investigation | 1522 | 263 | 1259 | 82.7% | **0.0%** | 0.05ms |
| **Zone-Based Pruning** | Inventory Waste Investigation | 1522 | 1196 | 326 | 21.4% | **0.0%** | 0.03ms |
| **Sliding Window** | 50-Turn Extreme Scale | 1204 | 1189 | 15 | 1.2% | **0.0%** | 0.03ms |
| **Observation Masking** | 50-Turn Extreme Scale | 1204 | 937 | 267 | 22.2% | **100.0%** | 0.06ms |
| **Recursive Summarization** | 50-Turn Extreme Scale | 1204 | 234 | 970 | 80.6% | **0.0%** | 0.04ms |
| **Zone-Based Pruning** | 50-Turn Extreme Scale | 1204 | 1193 | 11 | 0.9% | **0.0%** | 0.05ms |

> **Needle Recall** = % of buried critical facts (e.g. emergency supplier overrides, manager decisions) that survived context pruning and remained detectable in the final context payload.

---

## Strategy Analysis

### Sliding Window
Drops the oldest turns entirely once the token budget fills up. Since the needle fact is buried in an early turn (turn 3), it is the **first item to be dropped** when the window rolls forward. Needle Recall: **0.0% across both scenarios** — the worst result for this class of problem where early decisions must survive to the end of the conversation.

### Observation Masking ✅ SELECTED
Keeps all dialogue turns but replaces large raw tool output payloads with compact metadata placeholders (e.g. `[TOOL OUTPUT MASKED: 1,250 tokens | tool=get_inventory_report]`). The user message in turn 3 containing the supplier preference is **never masked** — only tool JSON outputs are replaced. This means the needle survives intact. Needle Recall: **100.0% across both scenarios**. Token reduction: 28.9% (Scenario 1) and 22.2% (Scenario 2) at minimal latency (0.06–0.09ms).

### Recursive Summarization
Compresses older turns into a rolling `[RECURSIVE CONVERSATION SUMMARY]` block. The summarizer collapses turn 3's supplier preference into an abstracted phrase that loses the exact detail needed to answer "which supplier did the manager specify?". Token reduction is the highest (82.7%) but Needle Recall is **0.0%** — the exact fact is destroyed in compression. Unsuitable for this domain where precise operational details (supplier account numbers, write-off thresholds) must survive verbatim.

### Zone-Based Pruning
Divides the context into System Zone, Scratchpad Zone, Recent Dialogue Zone, and Middle History Zone, then aggressively prunes the middle zone. Needle facts buried in the middle zone are pruned. Needle Recall: **0.0%**. Token reduction is the same as Sliding Window (21.4%) with no accuracy advantage. The zone boundaries as configured are not optimized for early-turn needle preservation.

---

## Final Strategy Choice: **Observation Masking**

**Justification based on the comparison table above:**

Copperleaf Kitchens' real failure mode is **context inflation from tool JSON outputs** — each inventory check, waste report generation, and supplier lookup returns large structured payloads (typically 200–800 tokens of JSON). These payloads become irrelevant once the agent has acted on them, but the critical user instructions and manager decisions that arrived as short natural-language messages remain necessary throughout the session.

Observation Masking is the only strategy that:
1. **Achieves 100% needle recall** on both test scenarios (the only strategy to do so)
2. **Reduces token usage by 22–29%** without any additional LLM calls
3. **Runs at sub-millisecond latency** (0.06–0.09ms) — making it safe for real-time use
4. **Targets the actual source of bloat** (tool JSON outputs) rather than indiscriminately dropping conversation turns

Sliding Window and Zone-Based Pruning both fail on needle recall (0.0%) because they drop the earliest turns where critical decisions are made. Recursive Summarization achieves higher token savings but destroys exact facts through abstraction — unacceptable in a restaurant operations context where exact supplier codes, write-off thresholds, and manager authorization IDs must be preserved verbatim for compliance.

**Shipped strategy: Observation Masking** (`context_eval/masking.py`, `ObservationMaskingStrategy`)