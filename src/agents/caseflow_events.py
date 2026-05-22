from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

WorkflowNodeStatus = Literal["pending", "running", "succeeded", "blocked", "failed", "skipped"]


class WorkflowEvidence(BaseModel):
    source: str
    title: str
    snippet: str


class WorkflowRiskFlag(BaseModel):
    type: str
    severity: Literal["low", "medium", "high"]
    reason: str


class WorkflowNodeDefinition(BaseModel):
    node_id: str
    node_label: str
    goal: str
    graph_node: str | None = None
    is_risk_control: bool = False


class WorkflowEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = ""
    case_id: str = ""
    node_id: str
    node_label: str
    status: WorkflowNodeStatus
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    latency_ms: int | None = None
    input_summary: str = ""
    output_summary: str = ""
    evidence: list[WorkflowEvidence] = Field(default_factory=list)
    risk_flags: list[WorkflowRiskFlag] = Field(default_factory=list)
    approval_required: bool = False
    error: str | None = None


WORKFLOW_NODES: tuple[WorkflowNodeDefinition, ...] = (
    WorkflowNodeDefinition(
        node_id="intent_recognition",
        node_label="Intent Recognition",
        graph_node="intake_and_classify",
        goal="Capture the customer issue and runtime identity for the workflow.",
    ),
    WorkflowNodeDefinition(
        node_id="evidence_retrieval",
        node_label="Evidence Retrieval",
        graph_node="retrieve_evidence",
        goal="Retrieve local policy, SOP, order-like context, and customer history evidence.",
    ),
    WorkflowNodeDefinition(
        node_id="action_planning",
        node_label="Action Planning",
        graph_node="plan_actions",
        goal="Classify intent, assign priority, decide approval need, and create the action plan.",
    ),
    WorkflowNodeDefinition(
        node_id="draft_response",
        node_label="Draft Response",
        graph_node="draft_resolution",
        goal="Create an evidence-grounded customer-facing draft response.",
    ),
    WorkflowNodeDefinition(
        node_id="risk_review",
        node_label="Risk Review",
        graph_node="reflect_and_risk_check",
        goal="Check evidence sufficiency and enforce deterministic guardrails.",
        is_risk_control=True,
    ),
    WorkflowNodeDefinition(
        node_id="human_approval",
        node_label="Human Approval",
        graph_node="request_human_approval_if_needed",
        goal="Pause high-risk execution until a human approves, rejects, or requests changes.",
        is_risk_control=True,
    ),
    WorkflowNodeDefinition(
        node_id="ticket_execution",
        node_label="Ticket Execution",
        graph_node="execute_action",
        goal="Create or escalate deterministic mock tickets only after policy allows execution.",
    ),
    WorkflowNodeDefinition(
        node_id="result_persistence",
        node_label="Result Persistence",
        graph_node="finalize_and_persist",
        goal="Persist the auditable result payload and compact case summary.",
    ),
)

NODE_BY_ID = {node.node_id: node for node in WORKFLOW_NODES}
NODE_BY_GRAPH_NAME = {
    node.graph_node: node for node in WORKFLOW_NODES if node.graph_node is not None
}


def workflow_node_definitions() -> list[dict[str, Any]]:
    return [node.model_dump(mode="json") for node in WORKFLOW_NODES]


def _coerce_result(update: Any) -> dict[str, Any]:
    if not isinstance(update, dict):
        return {}
    for key in ("caseflow_result", "typed_result"):
        value = update.get(key)
        if isinstance(value, dict):
            return value
    return update


def _truncate(value: Any, max_length: int = 190) -> str:
    text = str(value or "").strip()
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 1]}..."


def parse_workflow_evidence_item(item: object) -> WorkflowEvidence:
    if isinstance(item, dict):
        source = str(item.get("source") or item.get("type") or "evidence")
        title = str(item.get("title") or item.get("case_id") or item.get("name") or source)
        snippet = str(item.get("snippet") or item.get("summary") or item.get("text") or "")
        return WorkflowEvidence(source=source, title=title, snippet=_truncate(snippet, 240))

    text = str(item or "").strip()
    upper = text.upper()
    if upper.startswith(("FAQ -", "SOP -", "POLICY -", "KB -")):
        source, rest = text.split(" - ", 1)
        title, _, snippet = rest.partition(":")
        return WorkflowEvidence(
            source=source.lower(),
            title=title.strip() or source,
            snippet=_truncate(snippet or rest, 240),
        )
    if upper.startswith("CASE-"):
        title, _, snippet = text.partition(":")
        return WorkflowEvidence(
            source="customer_history",
            title=title.strip(),
            snippet=_truncate(snippet or text, 240),
        )
    if upper.startswith("CUSTOMER PROFILE:"):
        return WorkflowEvidence(
            source="customer",
            title="Customer profile",
            snippet=_truncate(text.removeprefix("Customer profile:").strip() or text, 240),
        )
    if text.startswith("历史案例 / 近期记忆:"):
        return WorkflowEvidence(
            source="case_memory",
            title="Recent case memory",
            snippet=_truncate(text, 240),
        )
    return WorkflowEvidence(source="evidence", title="Evidence", snippet=_truncate(text, 240))


def evidence_from_result(result: dict[str, Any]) -> list[WorkflowEvidence]:
    raw_evidence = result.get("evidence")
    if not isinstance(raw_evidence, list):
        raw_evidence = [raw_evidence] if raw_evidence else []
    return [parse_workflow_evidence_item(item) for item in raw_evidence if item not in (None, "")]


def risk_flags_from_result(result: dict[str, Any]) -> list[WorkflowRiskFlag]:
    query = str(result.get("user_query") or "")
    intent = str(result.get("intent") or "")
    raw_reflection = result.get("reflection")
    reflection: dict[str, Any] = raw_reflection if isinstance(raw_reflection, dict) else {}
    flags: list[WorkflowRiskFlag] = []

    def add_flag(flag_type: str, severity: Literal["low", "medium", "high"], reason: str) -> None:
        if not any(flag.type == flag_type for flag in flags):
            flags.append(WorkflowRiskFlag(type=flag_type, severity=severity, reason=reason))

    if "退款" in query or "退费" in query or "扣费" in query or intent == "退款 / 升级类":
        add_flag("refund", "high", "涉及退款、退费、扣费或资金类动作。")
    if "赔偿" in query or "补偿" in query:
        add_flag("compensation", "high", "客户要求赔偿或补偿，不能由模型直接承诺。")
    if "投诉" in query or intent == "投诉类":
        add_flag("complaint", "high", "客户明确投诉，需要人工确认处理口径。")
    if "升级" in query or "主管" in query:
        add_flag("escalation", "high", "涉及主管介入或升级队列。")

    approval_reason = str(reflection.get("approval_reason") or "").strip()
    if approval_reason and not flags:
        severity: Literal["low", "medium", "high"] = (
            "high" if result.get("needs_human_approval") else "medium"
        )
        add_flag("policy_guardrail", severity, approval_reason)
    return flags


def _execution_status(result: dict[str, Any]) -> str:
    execution = result.get("execution_result")
    if not isinstance(execution, dict):
        return ""
    return str(execution.get("status") or "").strip().lower()


def _ticket_created(result: dict[str, Any]) -> bool:
    execution = result.get("execution_result")
    if not isinstance(execution, dict):
        return False
    return bool(execution.get("ticket") or execution.get("escalation"))


def _approval_required(result: dict[str, Any]) -> bool:
    return bool(result.get("needs_human_approval"))


def _node_event(
    node_id: str,
    status: WorkflowNodeStatus,
    result: dict[str, Any],
    *,
    run_id: str = "",
    case_id: str = "",
    input_summary: str = "",
    output_summary: str = "",
    error: str | None = None,
) -> WorkflowEvent:
    node = NODE_BY_ID[node_id]
    return WorkflowEvent(
        run_id=run_id,
        case_id=case_id,
        node_id=node.node_id,
        node_label=node.node_label,
        status=status,
        input_summary=input_summary,
        output_summary=output_summary,
        evidence=evidence_from_result(result) if node_id == "evidence_retrieval" else [],
        risk_flags=risk_flags_from_result(result) if node_id in {"risk_review", "human_approval"} else [],
        approval_required=_approval_required(result),
        error=error,
    )


def completed_status_for_node(node_id: str, result: dict[str, Any]) -> WorkflowNodeStatus:
    execution_status = _execution_status(result)
    approval_required = _approval_required(result)

    if node_id == "risk_review":
        return "blocked" if approval_required and execution_status == "pending_approval" else "succeeded"
    if node_id == "human_approval":
        if not approval_required:
            return "skipped"
        if execution_status == "pending_approval":
            return "running"
        if execution_status in {"rejected", "modification_requested"}:
            return "blocked"
        return "succeeded"
    if node_id == "ticket_execution":
        if execution_status == "pending_approval":
            return "pending"
        if execution_status in {"rejected", "modification_requested"}:
            return "skipped"
        return "succeeded" if _ticket_created(result) else "skipped"
    return "succeeded"


def summaries_for_node(node_id: str, result: dict[str, Any]) -> tuple[str, str]:
    query = _truncate(result.get("user_query"), 180)
    intent = str(result.get("intent") or "unknown")
    priority = str(result.get("priority") or "unknown")
    execution_status = _execution_status(result) or "not_started"
    if node_id == "intent_recognition":
        return query, f"Captured issue for workflow. Current intent: {intent}, priority: {priority}."
    if node_id == "evidence_retrieval":
        return query, f"Retrieved {len(evidence_from_result(result))} evidence item(s)."
    if node_id == "action_planning":
        plan = result.get("proposed_action_plan") or []
        return f"{intent} / {priority}", f"Planned {len(plan) if isinstance(plan, list) else 0} action step(s)."
    if node_id == "draft_response":
        return f"{intent} with evidence context", _truncate(result.get("draft_response"), 180)
    if node_id == "risk_review":
        raw_reflection = result.get("reflection")
        reflection: dict[str, Any] = raw_reflection if isinstance(raw_reflection, dict) else {}
        return f"{intent} / approval_required={_approval_required(result)}", _truncate(
            reflection.get("approval_reason") or reflection.get("reason"), 220
        )
    if node_id == "human_approval":
        return f"Guarded execution status: {execution_status}", _truncate(
            result.get("approved_action") or "Waiting for approve / reject / modify", 180
        )
    if node_id == "ticket_execution":
        return f"Approved action: {result.get('approved_action')}", f"Execution status: {execution_status}."
    return "Final typed payload", f"Result persisted with execution status: {execution_status}."


def build_workflow_events(
    result: dict[str, Any],
    *,
    run_id: str = "",
    case_id: str = "",
) -> list[dict[str, Any]]:
    events: list[WorkflowEvent] = []
    for node in WORKFLOW_NODES:
        status = completed_status_for_node(node.node_id, result)
        input_summary, output_summary = summaries_for_node(node.node_id, result)
        events.append(
            _node_event(
                node.node_id,
                status,
                result,
                run_id=run_id,
                case_id=case_id,
                input_summary=input_summary,
                output_summary=output_summary,
            )
        )
    return [event.model_dump(mode="json") for event in events]


def running_event_for_graph_node(
    graph_node: str,
    *,
    run_id: str = "",
    case_id: str = "",
) -> dict[str, Any] | None:
    node = NODE_BY_GRAPH_NAME.get(graph_node)
    if not node:
        return None
    event = WorkflowEvent(
        run_id=run_id,
        case_id=case_id,
        node_id=node.node_id,
        node_label=node.node_label,
        status="running",
        input_summary="Node execution started.",
        output_summary="Waiting for node output.",
        approval_required=False,
    )
    return event.model_dump(mode="json")


def workflow_events_from_graph_update(
    graph_node: str,
    update: Any,
    *,
    run_id: str = "",
    case_id: str = "",
) -> list[dict[str, Any]]:
    if graph_node == "__interrupt__":
        interrupt_node = NODE_BY_ID["human_approval"]
        event = WorkflowEvent(
            run_id=run_id,
            case_id=case_id,
            node_id=interrupt_node.node_id,
            node_label=interrupt_node.node_label,
            status="running",
            input_summary="High-risk action is paused by LangGraph interrupt().",
            output_summary="Waiting for human approve / reject / modify.",
            approval_required=True,
            risk_flags=[
                WorkflowRiskFlag(
                    type="human_in_the_loop",
                    severity="high",
                    reason="Graph execution is paused before ticket/refund/escalation execution.",
                )
            ],
        )
        return [event.model_dump(mode="json")]

    graph_node_definition = NODE_BY_GRAPH_NAME.get(graph_node)
    if not graph_node_definition:
        return []

    result = _coerce_result(update)
    if not result:
        result = {"user_query": "", "execution_result": {"status": "not_started"}}
    status = completed_status_for_node(graph_node_definition.node_id, result)
    input_summary, output_summary = summaries_for_node(graph_node_definition.node_id, result)
    events = [
        _node_event(
            graph_node_definition.node_id,
            status,
            result,
            run_id=run_id,
            case_id=case_id,
            input_summary=input_summary,
            output_summary=output_summary,
        )
    ]

    if graph_node_definition.node_id == "risk_review" and not _approval_required(result):
        approval_input, approval_output = summaries_for_node("human_approval", result)
        events.append(
            _node_event(
                "human_approval",
                "skipped",
                result,
                run_id=run_id,
                case_id=case_id,
                input_summary=approval_input,
                output_summary=approval_output,
            )
        )
    return [event.model_dump(mode="json") for event in events]
