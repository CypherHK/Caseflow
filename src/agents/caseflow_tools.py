import json
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "caseflow"


@lru_cache
def _load_json(name: str) -> list[dict[str, Any]]:
    with (DATA_DIR / name).open(encoding="utf-8") as f:
        return json.load(f)


def _score(query: str, keywords: list[str]) -> int:
    normalized = query.lower()
    return sum(1 for keyword in keywords if keyword.lower() in normalized)


def _best_match(query: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        rows,
        key=lambda row: (_score(query, row.get("keywords", [])), row.get("case_id", "")),
        reverse=True,
    )
    return ranked[0]


def search_kb(query: str) -> dict[str, Any]:
    """Search local FAQ, SOP, and policy snippets for a support query."""

    row = _best_match(query, _load_json("kb.json"))
    return {
        "source": row["source"],
        "title": row["title"],
        "summary": row["summary"],
    }


def search_case_history(query: str) -> dict[str, Any]:
    """Search local historical case summaries for a similar support case."""

    row = _best_match(query, _load_json("case_history.json"))
    return {
        "case_id": row["case_id"],
        "summary": row["summary"],
    }


def get_customer_profile(customer_id: str | None) -> dict[str, Any]:
    """Return a mock customer profile used by the CaseFlow agent."""

    requested_id = customer_id or "cust-003"
    customers = _load_json("customers.json")
    for row in customers:
        if row["customer_id"] == requested_id:
            return row
    return next(row for row in customers if row["customer_id"] == "cust-003")


def create_ticket(thread_id: str, customer_id: str, intent: str) -> dict[str, Any]:
    """Create a deterministic mock ticket record."""

    suffix = abs(hash((thread_id, customer_id, intent))) % 100000
    return {
        "ticket_id": f"TCK-{suffix:05d}",
        "status": "created",
        "thread_id": thread_id,
        "customer_id": customer_id,
        "intent": intent,
    }


def escalate_ticket(ticket_id: str, reason: str) -> dict[str, Any]:
    """Escalate a mock ticket to a human queue."""

    return {
        "ticket_id": ticket_id,
        "status": "escalated",
        "queue": reason,
    }


def save_case_note(thread_id: str, case_result: dict[str, Any]) -> dict[str, Any]:
    """Return a mock saved-note confirmation without writing external systems."""

    return {
        "status": "saved",
        "thread_id": thread_id,
        "intent": case_result.get("intent"),
        "priority": case_result.get("priority"),
    }
