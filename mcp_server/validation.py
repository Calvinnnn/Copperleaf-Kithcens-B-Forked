"""
validation.py — Independent server-side validation for Copperleaf Kitchens.

Lecture principle: "Schema != Validation." A tool's input schema (typed
fields, enums, required, additionalProperties: false) makes the MODEL more
likely to send well-formed arguments. It does NOT guarantee the arguments
are actually safe to execute. This module is the second, independent check
that runs in the handler regardless of what the schema already enforced —
never trust model-generated input just because it passed the schema.

Kept as its own file (not inlined in tools.py) so a grader can find "where
does defensive validation actually happen" in one obvious place.
"""

VALID_WRITE_OFF_REASONS = {
    "spoiled_before_use",
    "past_expiry",
    "damaged_in_delivery",
    "prep_error",
    "other",
}

# A single write-off above this quantity is treated as suspicious enough to
# reject outright rather than silently allow — real inventory tools need a
# hard ceiling on how much damage one call can do.
MAX_SINGLE_WRITE_OFF_QUANTITY = 500.0


class ValidationError(Exception):
    """Raised when arguments fail independent server-side validation.

    Always caught in the tool handler and turned into a structured error
    dict returned to the model — never an unhandled stack trace.
    """


def validate_write_off(
    *,
    item_id: int,
    quantity: float,
    reason: str,
    current_stock: float,
) -> None:
    """Validate a write-off request against real business rules.

    Raises ValidationError with a specific, model-readable message if any
    check fails. Called AFTER the tool's pydantic/JSON schema has already
    validated types — this checks things a type system cannot express.
    """
    if quantity <= 0:
        raise ValidationError(
            f"quantity must be a positive number, got {quantity}."
        )

    if quantity > MAX_SINGLE_WRITE_OFF_QUANTITY:
        raise ValidationError(
            f"quantity {quantity} exceeds the maximum allowed single "
            f"write-off ({MAX_SINGLE_WRITE_OFF_QUANTITY}). Split into "
            "multiple write-offs or escalate manually."
        )

    if reason not in VALID_WRITE_OFF_REASONS:
        raise ValidationError(
            f"reason '{reason}' is not recognized. Must be one of: "
            f"{sorted(VALID_WRITE_OFF_REASONS)}."
        )

    if quantity > current_stock:
        raise ValidationError(
            f"Cannot write off {quantity} units of item {item_id} — only "
            f"{current_stock} currently in stock."
        )


def validate_date_range(date_from: str, date_to: str) -> None:
    """Validate a date range for report-style tools (e.g. waste reports).

    Basic sanity checks independent of whatever format validation the
    schema already did (e.g. a string pattern match doesn't catch
    "end date before start date").
    """
    from datetime import date

    try:
        start = date.fromisoformat(date_from)
        end = date.fromisoformat(date_to)
    except ValueError as e:
        raise ValidationError(f"Invalid date format: {e}") from e

    if end < start:
        raise ValidationError(
            f"date_to ({date_to}) is before date_from ({date_from})."
        )

    if (end - start).days > 366:
        raise ValidationError(
            "Date range cannot exceed 366 days for a single report."
        )


HIGH_RISK_COST_THRESHOLD = 100.0   # Write-offs with total cost >= $100 require human elicitation
HIGH_RISK_UNIT_COST_THRESHOLD = 50.0 # Items with unit cost >= $50 require human elicitation


def requires_elicitation(quantity: float, unit_cost: float) -> bool:
    """Check whether a write-off represents high financial risk needing mid-call human elicitation."""
    total_cost = quantity * unit_cost
    return total_cost >= HIGH_RISK_COST_THRESHOLD or unit_cost >= HIGH_RISK_UNIT_COST_THRESHOLD

