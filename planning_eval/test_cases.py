# Fixed Benchmark Test Cases for Planning Agent Evaluation

TEST_CASES = [
    {
        "id": "A",
        "name": "Decomposition-First Case",
        "description": "A request that favors Decomposition-First because it has independent sub-tasks that can be scheduled in parallel.",
        "goal": "Check the low stock items at Branch 1 (Downtown) and Branch 2 (Harbor) separately, look up active supplier orders for each branch, and compile a single consolidated audit report.",
    },
    {
        "id": "B",
        "name": "Dynamic Decomposition Case",
        "description": "A request that favors Dynamic Decomposition because an intermediate result (e.g. pending order status) changes the next step.",
        "goal": "Check the status of supplier orders for Branch 1. If there is a pending order, look up the current stock for item_id 1. If the current quantity is below 5 units, draft an escalation warning; otherwise, draft a standard status report.",
    },
    {
        "id": "C",
        "name": "Search/Lookahead Case",
        "description": "A request requiring meaningful lookahead or search (e.g. Tree of Thoughts or LATS) to find an optimal solution path under constraints.",
        "goal": "Determine the optimal restock quantities for low stock items at Branch 1 (Roma Tomatoes, Chicken Breast) under a strict total cost budget of $50.00, maximizing quantity within budget.",
    },
    {
        "id": "D",
        "name": "Self-Refine Case",
        "description": "A request where Self-Refine improves a first attempt by catching simple formatting, reason, or schema errors.",
        "goal": "Write off 15 units of Yellow Onions (item_id 2) at Branch 1 (Downtown) using Mona Farid's manager token 'tok_mona_mgr_9f2a' for reason spoiled_before_use.",
    },
    {
        "id": "E",
        "name": "Reflexion Case",
        "description": "A request where a simple retry is not enough, and Reflexion's cross-trial episodic memory helps resolve multi-layered failures (such as auth role mismatch and quantity stock limit).",
        "goal": "Submit an inventory write-off for Roma Tomatoes (item_id 1) at Branch 1. You want to write off 10 units of Roma Tomatoes, but you must find and use the correct api_token and ensure the quantity is valid.",
    },
    {
        "id": "F",
        "name": "Grounded Feedback Case",
        "description": "A request where grounded feedback catches a real database constraint violation (writing off more than is in stock) that ungrounded critique misses.",
        "goal": "Submit an inventory write-off of 10.0 units of Roma Tomatoes (item_id 1) at Branch 1 using Mona Farid's manager token 'tok_mona_mgr_9f2a' for reason spoiled_before_use.",
    },
]
