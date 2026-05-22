import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "caseflow"


class DemoCase(BaseModel):
    case_id: str
    title: str
    category: str
    user_query: str
    customer_id: str
    order_summary: str
    product_summary: str
    customer_history_summary: str
    expected_key_nodes: list[str] = Field(default_factory=list)
    expected_signal: str


@lru_cache
def load_demo_cases() -> tuple[DemoCase, ...]:
    with (DATA_DIR / "demo_cases.json").open(encoding="utf-8") as file:
        payload = json.load(file)
    return tuple(DemoCase.model_validate(item) for item in payload)


def demo_cases_payload() -> list[dict[str, Any]]:
    return [case.model_dump(mode="json") for case in load_demo_cases()]


def get_demo_case(case_id: str | None) -> DemoCase:
    cases = load_demo_cases()
    for case in cases:
        if case.case_id == case_id:
            return case
    return cases[0]
