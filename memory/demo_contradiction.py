"""demo_contradiction.py — Concrete Contradiction Resolution Demonstration.

This script shows that the Semantic Consolidation Engine correctly detects
and resolves REAL contradictions arising from conflicting episodic events.

Copperleaf Kitchens Scenario
------------------------------
Branch 1 currently sources emergency produce from 'Apex Fresh Logistics'
(Account #APX-9982), a preference logged by Manager Mona during a session.
Later, a corporate policy feed episode arrives declaring that all Branch 1
emergency orders must route through 'GreenRoute Wholesale' (Account #GRW-4477).

These two episodes produce a direct contradiction on the same semantic fact:
  subject   = "branch_1_supplier"
  predicate = "preferred_emergency_supplier"

Three behaviours are demonstrated in sequence:

  Demo 1  SUPERSEDE      — newer episode wins; old value is archived in
                           fact history (no silent overwrite).
  Demo 2  MARK_CONTRADICTION — conflict is flagged; manual resolution via
                               resolve_contradiction() closes the loop.
  Demo 3  Expiration     — a time-limited fact (yesterday's flash discount)
                           is automatically expired by the next consolidation
                           run without any manual call.

Run:
    python memory/demo_contradiction.py
"""

from datetime import datetime, timedelta, timezone

from memory.consolidation import (
    ConflictResolutionStrategy,
    SemanticConsolidationEngine,
)
from memory.episodic import EpisodicMemory, EventType
from memory.semantic import FactStatus, SemanticMemory

_SEP = "=" * 70


def _banner(title: str) -> None:
    print(f"\n{_SEP}\n  {title}\n{_SEP}")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _store_conflicting_episodes(
    episodic: EpisodicMemory,
) -> tuple[str, str]:
    """Store Episode A (Mona's preference) then Episode B (corporate override)."""
    event_a = episodic.store_event(
        event_type=EventType.PREFERENCE,
        summary="Preferred emergency supplier: Apex Fresh Logistics (Account #APX-9982)",
        details={
            "subject": "branch_1_supplier",
            "preferred_emergency_supplier": "Apex Fresh Logistics (Account #APX-9982)",
            "set_by": "Manager Mona Farid",
            "branch_id": 1,
        },
        importance_score=0.82,
        tags=["supplier", "preference", "branch_1"],
        source="agent_session_mona",
    )
    event_b = episodic.store_event(
        event_type=EventType.BUSINESS_EVENT,
        summary="Preferred emergency supplier: GreenRoute Wholesale (Account #GRW-4477)",
        details={
            "subject": "branch_1_supplier",
            "preferred_emergency_supplier": "GreenRoute Wholesale (Account #GRW-4477)",
            "set_by": "Corporate Policy Override — Regional Director Karim Al-Hassan",
            "branch_id": 1,
        },
        importance_score=0.95,
        tags=["supplier", "corporate_override", "branch_1"],
        source="corporate_policy_feed",
    )
    print(f"  [Episode A] ID={event_a.event_id[:8]}  ->  {event_a.summary}")
    print(f"  [Episode B] ID={event_b.event_id[:8]}  ->  {event_b.summary}")
    return event_a.event_id, event_b.event_id


# ─────────────────────────────────────────────────────────────────────────────
# Demo 1: SUPERSEDE
# ─────────────────────────────────────────────────────────────────────────────

def demo_supersede() -> None:
    _banner("DEMO 1 — SUPERSEDE: Newer Episode Wins, Old Value Archived")

    episodic = EpisodicMemory()
    semantic = SemanticMemory()
    engine = SemanticConsolidationEngine(
        episodic_memory=episodic,
        semantic_memory=semantic,
        default_strategy=ConflictResolutionStrategy.SUPERSEDE,
    )

    print("\nStep 1 — Storing two conflicting episodes:")
    _store_conflicting_episodes(episodic)

    print("\nStep 2 — Running consolidation (SUPERSEDE strategy):")
    result = engine.run_consolidation()

    print(f"  Events processed : {len(result.processed_event_ids)}")
    print(f"  Facts created    : {result.created_facts_count}")
    print(f"  Facts superseded : {result.updated_facts_count}")
    print(f"  Contradictions   : {result.contradictions_count}")

    print("\nStep 3 — Resulting semantic fact:")
    for fact in semantic.get_all_active_facts():
        print(f"  subject    : {fact.subject}")
        print(f"  predicate  : {fact.predicate}")
        print(f"  value      : {fact.value!r}   (was) GreenRoute wins (Episode B)")
        print(f"  version    : {fact.version}")
        print(f"  history    : {len(fact.history)} archived version(s)")
        for h in fact.history:
            print(f"    v{h.version}: {h.value!r}   (was) Apex (Episode A, now archived)")

    print("\nStep 4 — Audit log entries:")
    for log in result.logs:
        print(f"  [{log.action_taken:20}]  {log.subject}.{log.predicate}")
        print(f"    old={log.old_value!r}  new={log.new_value!r}")
        print(f"    reason: {log.reason}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 2: MARK_CONTRADICTION + resolve_contradiction()
# ─────────────────────────────────────────────────────────────────────────────

def demo_mark_and_resolve() -> None:
    _banner("DEMO 2 — MARK_CONTRADICTION: Flag Then Resolve")

    episodic = EpisodicMemory()
    semantic = SemanticMemory()
    engine = SemanticConsolidationEngine(
        episodic_memory=episodic,
        semantic_memory=semantic,
        default_strategy=ConflictResolutionStrategy.MARK_CONTRADICTION,
    )

    print("\nStep 1 — Storing two conflicting episodes:")
    _store_conflicting_episodes(episodic)

    print("\nStep 2 — Running consolidation (MARK_CONTRADICTION strategy):")
    result = engine.run_consolidation()

    print(f"  Events processed : {len(result.processed_event_ids)}")
    print(f"  Contradictions   : {result.contradictions_count}   (was) CONTRADICTION FLAGGED")

    print("\nStep 3 — Contradicted fact state:")
    contradicted = [
        f for f in semantic._facts.values()
        if f.status == FactStatus.CONTRADICTED
    ]
    print(f"  Contradicted facts: {len(contradicted)}")
    for fact in contradicted:
        print(f"  fact_id   : {fact.fact_id[:8]}")
        print(f"  value     : {fact.value!r}   (was) UNCHANGED (pending resolution)")
        print(f"  status    : {fact.status}")
        print(f"  history   : {len(fact.history)} entries")

        print("\nStep 4 — Calling resolve_contradiction() with authoritative value:")
        resolved = engine.resolve_contradiction(
            fact_id=fact.fact_id,
            resolved_value="GreenRoute Wholesale (Account #GRW-4477)",
            justification=(
                "Corporate policy (Regional Director Karim Al-Hassan, 2026-08-07) "
                "supersedes local manager preference."
            ),
        )
        if resolved:
            print(f"  value     : {resolved.value!r}   (was) RESOLVED")
            print(f"  status    : {resolved.status}   (was) ACTIVE")
            print(f"  version   : {resolved.version}")
            for h in resolved.history:
                print(f"  history   : v{h.version} — {h.reason[:70]}")

    print("\nStep 5 — Contradiction audit logs:")
    for log in result.logs:
        print(f"  [{log.action_taken:20}]  {log.subject}.{log.predicate}")
        print(f"    reason: {log.reason}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 3: Automatic fact expiration
# ─────────────────────────────────────────────────────────────────────────────

def demo_expiration() -> None:
    _banner("DEMO 3 — Auto-Expiration via valid_until TTL")

    episodic = EpisodicMemory()
    semantic = SemanticMemory()
    engine = SemanticConsolidationEngine(episodic_memory=episodic, semantic_memory=semantic)

    # Add a fact that expired yesterday
    past_ts = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    fact = semantic.add_fact(
        subject="branch_1_discount",
        predicate="flash_discount_active",
        value="15% off all produce (Weekend Sale)",
        confidence=0.9,
        valid_until=past_ts,
    )
    print(f"\n  Fact added     : '{fact.value}'")
    print(f"  valid_until    : {fact.valid_until}   (was) already in the past")
    print(f"  Status (before): {fact.status}")

    print("\n  Running consolidation — _expire_stale_facts() fires at Step 0 ...")
    engine.run_consolidation()

    print(f"  Status (after) : {fact.status}   (was) EXPIRED")
    assert fact.status == FactStatus.EXPIRED, "Expiration did not trigger!"
    print("  [PASS] Fact automatically expired during consolidation run.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(_SEP)
    print("  COPPERLEAF KITCHENS — MEMORY CONTRADICTION & EXPIRATION DEMO")
    print(f"  Run time: {datetime.now(timezone.utc).isoformat()}")
    print(_SEP)

    demo_supersede()
    demo_mark_and_resolve()
    demo_expiration()

    print(f"\n{_SEP}")
    print("  ALL 3 DEMOS COMPLETED SUCCESSFULLY")
    print(_SEP)
