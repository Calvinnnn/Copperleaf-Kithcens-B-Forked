import re
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel


import json
from pathlib import Path
import sqlite3

DB_PATH = Path(__file__).resolve().parents[2] / "db" / "copperleaf.db"


def parse_write_off_details(text: str) -> dict:
    details = {}
    
    # Try parsing JSON first
    json_match = re.search(r"\{.*?\}", text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    key = k.lower()
                    if "item" in key and "id" in key:
                        details["item_id"] = int(v)
                    elif "qty" in key or "quantity" in key:
                        details["quantity"] = float(v)
                    elif "reason" in key:
                        details["reason"] = str(v)
                    elif "token" in key:
                        details["api_token"] = str(v)
        except Exception:
            pass
            
    # Fallback to Regex patterns
    if "item_id" not in details:
        item_id_match = re.search(r"(?:item_id|item id|item)[^\d]*(\d+)", text, re.IGNORECASE)
        if item_id_match:
            details["item_id"] = int(item_id_match.group(1))
            
    if "quantity" not in details:
        qty_match = re.search(r"(?:quantity|qty)[^\d]*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if qty_match:
            details["quantity"] = float(qty_match.group(1))
            
    if "reason" not in details:
        reason_match = re.search(r"reason[ :='\"]+([a-zA-Z0-9_-]+)", text, re.IGNORECASE)
        if reason_match:
            details["reason"] = reason_match.group(1).strip("'\"")
            
    if "api_token" not in details:
        token_match = re.search(r"(?:api_token|api token|token)[ :='\"]+([a-zA-Z0-9_-]+)", text, re.IGNORECASE)
        if token_match:
            details["api_token"] = token_match.group(1).strip("'\"")
            
    return details


def validate_against_db(details: dict, text: str) -> list[str]:
    issues = []
    item_id = details.get("item_id")
    quantity = details.get("quantity")
    reason = details.get("reason")
    api_token = details.get("api_token")
    
    if item_id is None:
        issues.append("Missing or invalid item_id in the write-off details.")
    if quantity is None or quantity <= 0:
        issues.append(f"Quantity must be a positive number, got {quantity}.")
    if reason is None:
        issues.append("Missing reason for write-off.")
    elif reason not in {"spoiled_before_use", "past_expiry", "damaged_in_delivery", "prep_error", "other"}:
        issues.append(f"reason '{reason}' is not recognized. Must be one of: ['damaged_in_delivery', 'other', 'past_expiry', 'prep_error', 'spoiled_before_use'].")
        
    if not DB_PATH.exists():
        issues.append(f"Database file not found at {DB_PATH}.")
        return issues
        
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        # Check API token
        if api_token is None:
            issues.append("Missing API token (api_token) for authentication.")
            return issues
            
        staff_row = conn.execute(
            "SELECT staff_id, branch_id, full_name, role, active FROM staff WHERE api_token = ?",
            (api_token,),
        ).fetchone()
        
        if staff_row is None:
            issues.append("Invalid API token. Access denied.")
            return issues
            
        if staff_row["active"] != 1:
            issues.append(f"Staff account '{staff_row['full_name']}' is deactivated.")
            return issues
            
        if staff_row["role"] != "manager":
            issues.append(f"'{staff_row['full_name']}' has role '{staff_row['role']}' — only managers can write off inventory.")
            return issues
            
        # Check item
        if item_id is not None:
            item_row = conn.execute(
                "SELECT item_id, branch_id, current_quantity, unit_cost, name FROM inventory_items WHERE item_id = ?",
                (item_id,),
            ).fetchone()
            
            if item_row is None:
                issues.append(f"No inventory item with item_id={item_id}.")
            else:
                # Check branch scope
                if item_row["branch_id"] != staff_row["branch_id"]:
                    issues.append(
                        f"'{staff_row['full_name']}' manages branch_id={staff_row['branch_id']}, "
                        f"but item_id={item_id} belongs to branch_id={item_row['branch_id']}."
                    )
                # Check quantity ceiling
                if quantity is not None:
                    if quantity > 500.0:
                        issues.append(f"quantity {quantity} exceeds the maximum allowed single write-off (500.0).")
                    if quantity > item_row["current_quantity"]:
                        issues.append(
                            f"Cannot write off {quantity} units of item {item_id} — only "
                            f"{item_row['current_quantity']} currently in stock."
                        )
                    # Check high risk (elicitation)
                    unit_cost = item_row["unit_cost"]
                    total_cost = quantity * unit_cost
                    if total_cost >= 100.0 or unit_cost >= 50.0:
                        has_approval = re.search(r"approved|approval|sign-off|signed off", text, re.IGNORECASE)
                        if not has_approval:
                            issues.append("High financial risk transaction. Supervisor sign-off/approval is required.")
    except Exception as e:
        issues.append(f"Database error during validation: {str(e)}")
    finally:
        conn.close()
        
    return issues


def deterministic_checks(goal: str, draft: str) -> list[str]:
    issues: list[str] = []
    
    is_write_off = any(
        kw in goal.lower() or kw in draft.lower()
        for kw in ["write_off", "write-off", "writeoff", "item_id", "token"]
    )
    
    if not is_write_off:
        if len(draft.split()) < 80:
            issues.append("The deliverable is under 80 words and is probably incomplete.")
        goal_terms = {
            word.lower()
            for word in re.findall(r"[A-Za-z]{5,}", goal)
            if word.lower() not in {"create", "design", "write", "build", "about", "using"}
        }
        represented = [term for term in goal_terms if term in draft.lower()]
        if goal_terms and not represented:
            issues.append("The output contains none of the goal's significant terms.")
        if not re.search(r"(^|\n)(#{1,3}\s+|\d+[.)]\s+|[-*]\s+)", draft):
            issues.append("The deliverable has no visible structure (headings or list items).")
    else:
        # Extract details from draft, and fallback to goal if missing
        details = parse_write_off_details(draft)
        goal_details = parse_write_off_details(goal)
        for k, v in goal_details.items():
            if k not in details:
                details[k] = v
        db_issues = validate_against_db(details, draft)
        issues.extend(db_issues)
        
    return issues


@dataclass
class ReflectionResult:
    draft: str
    critique: str
    revised: str
    grounded_issues: list[str]


def reflect_and_refine(goal: str, draft: str, llm: BaseChatModel) -> ReflectionResult:
    grounded = deterministic_checks(goal, draft)
    grounded_report = "\n".join(f"- {issue}" for issue in grounded) or "- Deterministic checks passed."
    # This can be done better, how should it be done?
    critique_response = llm.invoke([
        ("system", "You are a separate critic. Judge against the rubric; do not rewrite the draft."),
        ("human", f"""Goal: {goal}
Rubric: correctness, completeness, internal consistency, and instruction adherence.
External deterministic checks:
{grounded_report}

Draft:
{draft}

List concrete issues. If there are none, respond exactly PASS."""),
    ], temperature=0.2)
    critique = critique_response.content
    if not isinstance(critique, str) or not critique.strip():
        raise RuntimeError("The chat model returned an empty or unsupported response")
    critique = critique.strip()
    if critique.strip().upper() == "PASS" and not grounded:
        revised = draft
    else:
        response = llm.invoke([
            ("system", "Revise a deliverable using both external checks and an independent critique."),
            ("human", f"Goal: {goal}\n\nDraft:\n{draft}\n\nGrounded checks:\n{grounded_report}\n\nCritique:\n{critique}\n\nReturn only the improved deliverable."),
        ], temperature=0.2)
        revised = response.content
        if not isinstance(revised, str) or not revised.strip():
            raise RuntimeError("The chat model returned an empty or unsupported response")
        revised = revised.strip()
    return ReflectionResult(draft, critique, revised, grounded)
