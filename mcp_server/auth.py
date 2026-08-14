"""
auth.py — Session-based authentication for the Copperleaf Kitchens MCP server.

Design decision (see README "Deliberate Scope & Design Decisions"):
No tool accepts staff identity as an argument. A model, or an injected
prompt, could simply assert any identity it wanted if identity were a plain
parameter. Instead, the CLIENT authenticates once when a session starts
(with staff.api_token), and every tool handler resolves identity/role from
that already-authenticated session — never from anything the model typed.

This module only answers one question: given a token, who is calling, and
are they allowed to be here at all (active account)? It knows nothing about
which tools exist or what a given tool requires — that's handler-level
authorization, done in server.py, on top of this.

TODO before the Streamable HTTP transport transition: this is currently
called ONCE per connection (stdio: one process = one client). A single
process under Streamable HTTP can serve MULTIPLE staff concurrently — do
not keep a single module-level SESSION global once that transition happens.
resolve_staff() must be called per-request at that point (e.g. from a
per-request Authorization header), not once at process startup.
"""
from dataclasses import dataclass
from typing import Optional

try:
    from mcp_server.db import get_connection
except ImportError:
    from db import get_connection


class AuthError(Exception):
    """Raised when a session cannot be tied to a valid, active staff record."""


@dataclass(frozen=True)
class Session:
    staff_id: int
    branch_id: int
    full_name: str
    role: str  # 'staff' | 'manager'


def resolve_staff(api_token: Optional[str]) -> Session:
    """Resolve an api_token to an authenticated Session, or raise AuthError.

    Called once per connection at startup (stdio) — see server.py's
    module-level SESSION. Kept as a standalone function (not inlined) so the
    same logic can later be called per-request when the transport moves to
    Streamable HTTP, where a single server process serves multiple sessions.
    """
    if not api_token:
        raise AuthError(
            "No API token provided. This server requires an authenticated "
            "session — set COPPERLEAF_API_TOKEN before connecting."
        )

    with get_connection() as conn:
        row = conn.execute(
            "SELECT staff_id, branch_id, full_name, role, active "
            "FROM staff WHERE api_token = ?",
            (api_token,),
        ).fetchone()

    if row is None:
        raise AuthError("Invalid API token.")

    if row["active"] != 1:
        raise AuthError(
            f"Staff account '{row['full_name']}' is deactivated and cannot "
            "authenticate."
        )

    return Session(
        staff_id=row["staff_id"],
        branch_id=row["branch_id"],
        full_name=row["full_name"],
        role=row["role"],
    )
