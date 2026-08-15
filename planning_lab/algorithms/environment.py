import json
import re
import sqlite3
from pathlib import Path

from ..models import EnvironmentFeedback

DB_PATH = Path(__file__).resolve().parents[2] / "db" / "copperleaf.db"


class Environment:
    """A real database-grounded evaluator for Copperleaf Kitchens."""

    def __init__(
        self,
        success_threshold: float = 0.6,
        rng: None = None,  # kept for signature compatibility
    ):
        self.success_threshold = success_threshold

    def evaluate(self, state: str) -> EnvironmentFeedback:
        # Check if the state involves write-off or database operations
        is_write_off = any(
            kw in state.lower()
            for kw in ["write_off", "write-off", "writeoff", "item_id", "token"]
        )
        
        if not is_write_off:
            # Basic fallback validation
            details = []
            if len(state.split()) < 10:
                details.append("State content is too short to be valid.")
            if not details:
                return EnvironmentFeedback(success=True, score=1.0, details=["Pass basic text checks."])
            else:
                return EnvironmentFeedback(success=False, score=0.0, details=details)
                
        # Parse details
        details_dict = self._parse_write_off_details(state)
        issues = self._validate_against_db(details_dict, state)
        
        if issues:
            return EnvironmentFeedback(success=False, score=0.0, details=issues)
        else:
            return EnvironmentFeedback(
                success=True,
                score=1.0,
                details=["All database and authorization constraints passed."],
            )

    def _parse_write_off_details(self, text: str) -> dict:
        details = {}
        # Try JSON
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
        
        # Regex fallbacks
        if "item_id" not in details:
            m = re.search(r"(?:item_id|item id|item)[^\d]*(\d+)", text, re.IGNORECASE)
            if m:
                details["item_id"] = int(m.group(1))
        if "quantity" not in details:
            m = re.search(r"(?:quantity|qty)[^\d]*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
            if m:
                details["quantity"] = float(m.group(1))
        if "reason" not in details:
            m = re.search(r"reason[ :='\"]+([a-zA-Z0-9_-]+)", text, re.IGNORECASE)
            if m:
                details["reason"] = m.group(1).strip("'\"")
        if "api_token" not in details:
            m = re.search(r"(?:api_token|api token|token)[ :='\"]+([a-zA-Z0-9_-]+)", text, re.IGNORECASE)
            if m:
                details["api_token"] = m.group(1).strip("'\"")
        return details

    def _validate_against_db(self, details: dict, text: str) -> list[str]:
        issues = []
        item_id = details.get("item_id")
        quantity = details.get("quantity")
        reason = details.get("reason")
        api_token = details.get("api_token")
        
        if item_id is None:
            issues.append("Missing or invalid item_id.")
        if quantity is None or quantity <= 0:
            issues.append(f"Quantity must be positive, got {quantity}.")
        if reason is None:
            issues.append("Missing reason.")
        elif reason not in {"spoiled_before_use", "past_expiry", "damaged_in_delivery", "prep_error", "other"}:
            issues.append(f"reason '{reason}' is invalid.")
            
        if not DB_PATH.exists():
            issues.append("Database file missing.")
            return issues
            
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            if api_token is None:
                issues.append("Missing API token.")
                return issues
                
            staff = conn.execute(
                "SELECT staff_id, branch_id, full_name, role, active FROM staff WHERE api_token = ?",
                (api_token,),
            ).fetchone()
            
            if staff is None:
                issues.append("Invalid API token.")
                return issues
            if staff["active"] != 1:
                issues.append("Inactive staff account.")
                return issues
            if staff["role"] != "manager":
                issues.append("Only managers can write off inventory.")
                return issues
                
            if item_id is not None:
                item = conn.execute(
                    "SELECT item_id, branch_id, current_quantity, unit_cost, name FROM inventory_items WHERE item_id = ?",
                    (item_id,),
                ).fetchone()
                
                if item is None:
                    issues.append(f"No item with item_id={item_id}.")
                else:
                    if item["branch_id"] != staff["branch_id"]:
                        issues.append(
                            f"Branch mismatch: item belongs to branch {item['branch_id']}, staff belongs to branch {staff['branch_id']}."
                        )
                    if quantity is not None:
                        if quantity > 500.0:
                            issues.append("Quantity exceeds single write-off limit of 500.")
                        if quantity > item["current_quantity"]:
                            issues.append(
                                f"Cannot write off {quantity} units of item {item_id} — only {item['current_quantity']} currently in stock."
                            )
                        
                        # High risk (elicitation) validation
                        unit_cost = item["unit_cost"]
                        total_cost = quantity * unit_cost
                        if total_cost >= 100.0 or unit_cost >= 50.0:
                            has_approval = re.search(r"approved|approval|sign-off|signed off", text, re.IGNORECASE)
                            if not has_approval:
                                issues.append("High financial risk transaction. Supervisor sign-off/approval is required.")
        except Exception as e:
            issues.append(f"DB error: {str(e)}")
        finally:
            conn.close()
        return issues
