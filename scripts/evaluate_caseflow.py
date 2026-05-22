import argparse
import asyncio
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agents.caseflow_agent import analyze_case, caseflow_agent  # noqa: E402

EVAL_PATH = ROOT / "data" / "caseflow" / "eval_cases.json"
REQUIRED_RESULT_FIELDS = {
    "intent",
    "priority",
    "needs_human_approval",
    "evidence",
    "draft_response",
}


class EvaluationError(RuntimeError):
    """Raised when evaluation cannot extract a valid CaseFlow result."""


def _hit_any(text: str, keywords: list[str]) -> bool:
    return any(keyword.lower() in text.lower() for keyword in keywords)


def _score_cases(cases: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(cases)
    if total == 0:
        raise EvaluationError("Evaluation requires at least one case.")
    if len(results) != total:
        raise EvaluationError(
            f"Evaluation returned {len(results)} results for {total} cases."
        )

    intent_hits = 0
    approval_hits = 0
    evidence_hits = 0
    draft_hits = 0
    priority_hits = 0

    for case, result in zip(cases, results, strict=True):
        intent_hits += result["intent"] == case["expected_intent"]
        approval_hits += (
            result["needs_human_approval"] == case["expected_needs_human_approval"]
        )
        evidence_text = "\n".join(result.get("evidence", []))
        evidence_hits += _hit_any(evidence_text, case["expected_evidence_keywords"])
        draft_hits += all(
            keyword.lower() in result.get("draft_response", "").lower()
            for keyword in case["draft_response_must_include"]
        )
        priority_hits += result["priority"] == case["expected_priority"]

    metrics = {
        "total_cases": total,
        "intent_accuracy": intent_hits / total,
        "approval_accuracy": approval_hits / total,
        "evidence_hit_rate": evidence_hits / total,
        "draft_groundedness_rate": draft_hits / total,
        "priority_accuracy": priority_hits / total,
    }
    return metrics


def _validate_result(case_id: str, result: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(REQUIRED_RESULT_FIELDS - result.keys())
    if missing:
        raise EvaluationError(
            f"Live evaluation case {case_id} returned custom_data missing fields: {missing}."
        )
    return result


def _validate_results(
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(results) != len(cases):
        raise EvaluationError(
            f"Evaluation returned {len(results)} results for {len(cases)} cases."
        )
    return [_validate_result(case["id"], result) for case, result in zip(cases, results)]


def _is_fallback_model(model_used: Any) -> bool:
    model_label = str(model_used or "")
    return model_label == "deterministic-fallback" or model_label.endswith(":fallback")


def _truncate(value: Any, limit: int = 180) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _case_mismatches(case: dict[str, Any], result: dict[str, Any]) -> list[str]:
    mismatches = []
    if result.get("intent") != case["expected_intent"]:
        mismatches.append("intent")
    if result.get("priority") != case["expected_priority"]:
        mismatches.append("priority")
    if result.get("needs_human_approval") != case["expected_needs_human_approval"]:
        mismatches.append("approval")
    evidence_text = "\n".join(result.get("evidence", []))
    if not _hit_any(evidence_text, case["expected_evidence_keywords"]):
        mismatches.append("evidence")
    draft_response = str(result.get("draft_response", ""))
    if not all(
        keyword.lower() in draft_response.lower()
        for keyword in case["draft_response_must_include"]
    ):
        mismatches.append("draft")
    return mismatches


def _qualitative_example(
    case: dict[str, Any],
    result: dict[str, Any],
    live_record: dict[str, Any],
) -> dict[str, Any]:
    evidence = list(result.get("evidence", []))
    return {
        "id": case["id"],
        "user_query": case["user_query"],
        "expected": {
            "intent": case["expected_intent"],
            "priority": case["expected_priority"],
            "needs_human_approval": case["expected_needs_human_approval"],
        },
        "actual": {
            "intent": result.get("intent"),
            "priority": result.get("priority"),
            "needs_human_approval": result.get("needs_human_approval"),
        },
        "model_used": result.get("model_used", "unknown-model"),
        "fallback_used": _is_fallback_model(result.get("model_used")),
        "interrupt_count": live_record.get("interrupt_count", 0),
        "approval_resume": live_record.get("approval_resume"),
        "evidence_excerpt": [_truncate(item, 140) for item in evidence[:2]],
        "draft_excerpt": _truncate(result.get("draft_response"), 180),
        "mismatches": _case_mismatches(case, result),
    }


def _select_qualitative_examples(
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
    live_records: list[dict[str, Any]],
    examples_limit: int,
) -> list[dict[str, Any]]:
    if examples_limit < 0:
        raise EvaluationError("examples_limit must be greater than or equal to 0.")

    examples = [
        _qualitative_example(case, result, live_record)
        for case, result, live_record in zip(cases, results, live_records, strict=True)
    ]

    def sort_key(item: dict[str, Any]) -> tuple[bool, bool, bool, str]:
        high_risk = item["expected"]["needs_human_approval"] or item["actual"][
            "needs_human_approval"
        ]
        return (
            not bool(item["mismatches"]),
            not high_risk,
            not item["fallback_used"],
            item["id"],
        )

    if examples_limit == 0:
        return []

    selected = sorted(examples, key=sort_key)[:examples_limit]
    has_human_review_example = any(
        item["interrupt_count"] > 0 or item["expected"]["needs_human_approval"]
        for item in selected
    )
    human_review_candidates = [
        item
        for item in sorted(examples, key=lambda item: (item["id"]))
        if item["interrupt_count"] > 0 or item["expected"]["needs_human_approval"]
    ]
    if human_review_candidates and not has_human_review_example:
        selected[-1] = human_review_candidates[0]
    return selected


def _live_model_verification(results: list[dict[str, Any]]) -> dict[str, Any]:
    fallback_count = sum(_is_fallback_model(result.get("model_used")) for result in results)
    non_fallback_count = len(results) - fallback_count
    status = "verified" if non_fallback_count else "fallback-only"
    diagnostic = (
        "At least one live eval case used non-fallback model output."
        if non_fallback_count
        else (
            "All live eval cases executed through the graph but used fallback outputs. "
            "Verify model credentials, endpoint/model configuration, and JSON schema compliance "
            "before marking Story 5.2 done."
        )
    )
    return {
        "status": status,
        "verified": bool(non_fallback_count),
        "fallback_cases": fallback_count,
        "non_fallback_cases": non_fallback_count,
        "diagnostic": diagnostic,
    }


def _build_evaluation_report(
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    mode: str,
    live_records: list[dict[str, Any]] | None = None,
    examples_limit: int = 5,
    output_path: Path | None = None,
) -> dict[str, Any]:
    results = _validate_results(cases, results)
    report = _score_cases(cases, results)
    report["mode"] = mode

    if mode == "live-llm":
        records = live_records or [
            {"interrupt_count": 0, "approval_resume": None} for _ in results
        ]
        fallback_count = sum(_is_fallback_model(result.get("model_used")) for result in results)
        report["fallback_rate"] = fallback_count / len(results)
        report["models_used"] = sorted(
            {str(result.get("model_used", "unknown-model")) for result in results}
        )
        report["live_model_verification"] = _live_model_verification(results)
        report["qualitative_examples"] = _select_qualitative_examples(
            cases,
            results,
            records,
            examples_limit,
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return report


def _coerce_stream_events(response: Any) -> list[tuple[str, Any]]:
    if isinstance(response, list):
        return response
    if isinstance(response, dict):
        return [("values", response)]
    raise EvaluationError(f"Unexpected live evaluation response type: {type(response).__name__}")


def _extract_interrupt(events: list[tuple[str, Any]]) -> str | None:
    for response_type, response in reversed(events):
        if response_type == "updates" and "__interrupt__" in response:
            interrupts = response["__interrupt__"]
            if interrupts:
                return str(getattr(interrupts[0], "value", interrupts[0]))
    return None


def _extract_custom_data(events: list[tuple[str, Any]], case_id: str) -> dict[str, Any]:
    for response_type, response in reversed(events):
        if response_type != "values":
            continue
        messages = response.get("messages", [])
        if not messages:
            continue
        for message in reversed(messages):
            additional_kwargs = getattr(message, "additional_kwargs", None)
            if additional_kwargs is None and isinstance(message, dict):
                additional_kwargs = message.get("additional_kwargs")
            if not isinstance(additional_kwargs, dict):
                continue
            custom_data = additional_kwargs.get("custom_data")
            if isinstance(custom_data, dict):
                return dict(custom_data)
    raise EvaluationError(f"Live evaluation case {case_id} did not return final custom_data.")


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


@contextmanager
def _temporary_checkpointer(graph: Any) -> Iterator[None]:
    existing_checkpointer = getattr(graph, "checkpointer", None)
    if existing_checkpointer is not None:
        yield
        return

    graph.checkpointer = MemorySaver()
    try:
        yield
    finally:
        graph.checkpointer = existing_checkpointer


async def _evaluate_live_case(
    case: dict[str, Any],
    *,
    graph: Any,
    approval_response: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    thread_id = f"eval-{case['id']}"
    config = RunnableConfig(
        configurable={
            "thread_id": thread_id,
            "user_id": "cust-001",
        }
    )
    input_payload: Command | dict[str, Any] = {
        "messages": [HumanMessage(content=case["user_query"])]
    }
    all_events: list[tuple[str, Any]] = []
    interrupt_payloads: list[str] = []

    with _temporary_checkpointer(graph):
        for _ in range(4):
            events = _coerce_stream_events(
                await graph.ainvoke(
                    input_payload,
                    config=config,
                    stream_mode=["updates", "values"],
                )
            )
            all_events.extend(events)
            interrupt_payload = _extract_interrupt(events)
            if interrupt_payload is None:
                break
            interrupt_payloads.append(interrupt_payload)
            input_payload = Command(resume=approval_response)
        else:
            raise EvaluationError(
                f"Live evaluation case {case['id']} exceeded interrupt resume limit."
            )

    result = _extract_custom_data(all_events, case["id"])
    record = {
        "id": case["id"],
        "thread_id": thread_id,
        "interrupt_count": len(interrupt_payloads),
        "interrupt_payloads": interrupt_payloads,
        "approval_resume": approval_response if interrupt_payloads else None,
    }
    return result, record


async def _evaluate_live_cases(
    cases: list[dict[str, Any]],
    *,
    graph: Any = caseflow_agent,
    approval_response: str = "approve",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results = []
    records = []
    for case in cases:
        result, record = await _evaluate_live_case(
            case,
            graph=graph,
            approval_response=approval_response,
        )
        results.append(result)
        records.append(record)
    return results, records


def evaluate_cases(
    live_llm: bool = False,
    *,
    examples_limit: int = 5,
    approval_response: str = "approve",
    output_path: Path | None = None,
) -> dict[str, Any]:
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    if live_llm:
        results, live_records = asyncio.run(
            _evaluate_live_cases(cases, approval_response=approval_response)
        )
        return _build_evaluation_report(
            cases,
            results,
            mode="live-llm",
            live_records=live_records,
            examples_limit=examples_limit,
            output_path=output_path,
        )
    else:
        results = [
            analyze_case(
                user_query=case["user_query"],
                thread_id=f"eval-{case['id']}",
                user_id="cust-001",
            )
            for case in cases
        ]
        return _build_evaluation_report(
            cases,
            results,
            mode="deterministic",
            output_path=output_path,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CaseFlow Agent scenarios.")
    parser.add_argument(
        "--live-llm",
        action="store_true",
        help="Run the graph with the configured live model. Default uses deterministic fallback.",
    )
    parser.add_argument(
        "--examples-limit",
        type=_non_negative_int,
        default=5,
        help="Maximum qualitative examples to include in live LLM reports.",
    )
    parser.add_argument(
        "--approval-response",
        default="approve",
        help="Response used to resume live evaluation cases that trigger human approval.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON evaluation report.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_cases(
                live_llm=args.live_llm,
                examples_limit=args.examples_limit,
                approval_response=args.approval_response,
                output_path=args.output,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
