# Person 1 Planning & Implementation Roadmap

## 1. Project Overview & Branch Setup
- **Role**: PERSON 1
- **Target Branch**: `person1` (Do NOT commit/push directly to `main`)
- **Repository Location**: `C:\Users\DELL\Desktop\amr project\Copperleaf-Kithcens-B-Forked`

## 2. File Ownership & Boundaries
- **Owned Files**:
  - `planning_lab/algorithms/decomposition.py`
  - `planning_lab/algorithms/dynamic_decomposition.py`
  - `planning_lab/algorithms/plan_and_solve.py`
  - `planning_lab/algorithms/tree_of_thoughts.py`
  - `planning_lab/algorithms/lats.py`
  - `agent/planning_agent.py`
- **Person 2 Files (DO NOT MODIFY)**:
  - `planning_lab/algorithms/self_refine.py`
  - `planning_lab/algorithms/reflexion.py`
  - `planning_lab/algorithms/environment.py`
  - `planning_eval/test_cases.py`
  - `planning_eval/evaluate.py`

## 3. Issue Breakdown & Implementation Steps

### Phase 1: Issue #7 — [DECOMPOSITION] DAG Planning & Dynamic Planning
1. **Decomposition-First (`planning_lab/algorithms/decomposition.py`)**:
   - Complete plan generation prior to execution.
   - Enforce DAG validation & acyclicity (NetworkX dependency graph). Reject cyclic graphs (A → B → C → A).
   - Topological/dependency-safe execution.
   - *Commit*: `git commit -m "feat(decomposition): implement decomposition-first DAG planning (#7)"`
2. **Dynamic Decomposition (`planning_lab/algorithms/dynamic_decomposition.py`)**:
   - Interleaved planning and step-by-step execution.
   - Re-plan dynamically based on observation/environment feedback or state change.
   - Implement real domain divergence case (e.g. Copperleaf inventory discrepancy causing dynamic path change).
   - *Commit*: `git commit -m "feat(decomposition): implement dynamic DAG planning (#7)"`

### Phase 2: Issue #3 — [PLANNING] Algorithmic Integration & Task Routing
1. **Plan-and-Solve (`planning_lab/algorithms/plan_and_solve.py`)**:
   - Sequential, predictable sub-task resolution.
   - *Commit*: `git commit -m "feat(planning): integrate Plan-and-Solve (#3)"`
2. **Tree of Thoughts (`planning_lab/algorithms/tree_of_thoughts.py`)**:
   - Multi-candidate step generation, evaluation, and beam search/pruning.
   - *Commit*: `git commit -m "feat(planning): integrate Tree of Thoughts (#3)"`
3. **LATS (`planning_lab/algorithms/lats.py`)**:
   - Monte Carlo Tree Search loop (Select, Expand, Evaluate, Backpropagate).
   - *Commit*: `git commit -m "feat(planning): integrate LATS (#3)"`
4. **Task Router**:
   - Route sub-tasks dynamically to PS, ToT, or LATS depending on problem complexity and branching requirements.

### Phase 3: Issue #6 — [INTEGRATION] Planning Agent & Project Systems
1. **Planning Agent (`agent/planning_agent.py`)**:
   - New `PlanningAgent` class coexisting alongside `MemoryEnabledAgent`.
   - Connect decomposition, planning routing, and execution.
   - Call existing `mcp_server` tools (`get_inventory`, `get_low_stock_items`, `get_supplier_orders`, etc.) and database queries.
   - *Commit*: `git commit -m "feat(integration): integrate planning agent (#6)"`

## 4. Execution & Commit Protocol
- Test each file individually before staging.
- Stage and commit strictly one file at a time on `person1`.
- Verify `git status` and `git diff` before every commit.
- Push exclusively to `origin/person1`.
