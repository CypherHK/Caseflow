import inspect
import json
import re
from typing import Any, Literal, TypeVar, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agents.caseflow_events import build_workflow_events, workflow_node_definitions
from agents.caseflow_tools import (
    create_ticket,
    escalate_ticket,
    get_customer_profile,
    save_case_note,
    search_case_history,
    search_kb,
)
from core import get_model, settings

Intent = Literal["咨询类", "投诉类", "异常类", "退款 / 升级类", "信息不全待补充类"]
Priority = Literal["low", "medium", "high"]
RiskLevel = Literal["normal", "medium", "high"]
ApprovalDecisionValue = Literal["approve", "reject", "modify"]
SchemaT = TypeVar("SchemaT", bound=BaseModel)
CASEFLOW_SUMMARY_NAMESPACE_PREFIX = ("caseflow", "case_summaries")


class CaseAnalysis(BaseModel):
    intent: Intent
    priority: Priority
    needs_human_approval: bool
    reasoning: str
    missing_information: list[str] = Field(default_factory=list)


class DraftResolution(BaseModel):
    proposed_action_plan: list[str] = Field(min_length=1)
    draft_response: str = Field(min_length=1)
    next_step: str = Field(min_length=1)
    reasoning: str


class RiskReflection(BaseModel):
    evidence_sufficient: bool
    risk: RiskLevel
    reason: str
    approval_reason: str | None = None


class CaseApprovalDecision(BaseModel):
    decision: ApprovalDecisionValue
    reason: str | None = None
    modification_notes: str | None = None


class CustomerProfile(BaseModel):
    model_config = ConfigDict(extra="allow")

    customer_id: str
    name: str
    tier: str
    preferences: list[str] = Field(default_factory=list)


class RetrievedEvidence(BaseModel):
    customer_profile: CustomerProfile
    evidence: list[str] = Field(min_length=1)


class CaseExecutionResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    ticket: dict[str, Any] | None = None
    escalation: dict[str, Any] | None = None
    reason: str | None = None
    approval_note: str | None = None


class CaseFlowReasoning(BaseModel):
    analysis: str | None = None
    resolution: str | None = None


class CaseFlowResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    user_query: str | None = None
    thread_id: str | None = None
    user_id: str | None = None
    customer_profile: CustomerProfile | None = None
    intent: Intent
    priority: Priority
    evidence: list[str]
    proposed_action_plan: list[str] = Field(min_length=1)
    draft_response: str = Field(min_length=1)
    needs_human_approval: bool
    approved_action: str | None = None
    approval_decision: CaseApprovalDecision | None = None
    next_step: str = Field(min_length=1)
    execution_result: CaseExecutionResult
    model_used: str = "deterministic-fallback"
    llm_reasoning: CaseFlowReasoning = Field(default_factory=CaseFlowReasoning)
    reflection: RiskReflection
    case_note: dict[str, Any] | None = None
    missing_information: list[str] = Field(default_factory=list)


class CaseFlowState(MessagesState, total=False):
    user_query: str
    thread_id: str
    user_id: str
    retrieved_context: dict[str, Any]
    analysis: dict[str, Any]
    draft: dict[str, Any]
    reflection: dict[str, Any]
    execution_result: dict[str, Any]
    typed_result: dict[str, Any]
    caseflow_result: dict[str, Any]
    approved_action: str | None


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _blank_to_none(value: Any) -> str | None:
    text = str(value).strip() if value not in (None, "") else ""
    return text or None


def parse_caseflow_approval_decision(value: object) -> CaseApprovalDecision:
    """Parse structured and legacy HITL approval resume payloads."""

    if isinstance(value, CaseApprovalDecision):
        return value

    if isinstance(value, dict):
        return CaseApprovalDecision.model_validate(
            {
                "decision": str(value.get("decision", "")).strip().lower(),
                "reason": _blank_to_none(value.get("reason")),
                "modification_notes": _blank_to_none(value.get("modification_notes")),
            }
        )

    raw_text = str(value).strip()
    payload: Any = None
    if raw_text.startswith("{") and raw_text.endswith("}"):
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            payload = None
    if isinstance(payload, dict):
        try:
            return parse_caseflow_approval_decision(payload)
        except (ValidationError, TypeError, ValueError):
            pass

    normalized = raw_text.lower()
    if normalized in {"approve", "approved"} or raw_text in {"批准", "批准执行"}:
        return CaseApprovalDecision(decision="approve")
    if "reject" in normalized or "拒绝" in raw_text:
        return CaseApprovalDecision(decision="reject", reason=_blank_to_none(raw_text))
    if "modify" in normalized or "修改" in raw_text or "调整" in raw_text:
        return CaseApprovalDecision(
            decision="modify",
            reason=None,
            modification_notes=_blank_to_none(raw_text),
        )
    return CaseApprovalDecision(
        decision="modify",
        reason=None,
        modification_notes=_blank_to_none(raw_text),
    )


def classify_intent(user_query: str) -> Intent:
    normalized = user_query.lower()
    if _contains_any(normalized, ["退款", "退费", "扣费", "账单", "赔偿", "补偿", "升级", "主管"]):
        return "退款 / 升级类"
    if _contains_any(normalized, ["投诉", "不满意", "差评", "举报"]):
        return "投诉类"
    if _contains_any(normalized, ["异常", "失败", "报错", "无法使用", "订单"]):
        if not _contains_any(normalized, ["订单号", "截图", "错误码", "#"]):
            return "信息不全待补充类"
        return "异常类"
    if _contains_any(
        normalized,
        ["密码", "登录", "怎么", "如何", "faq", "帮助", "政策", "售后", "退换货", "无理由"],
    ):
        return "咨询类"
    return "信息不全待补充类"


def infer_priority(intent: Intent, user_query: str, customer_tier: str) -> Priority:
    if intent == "退款 / 升级类":
        return "high"
    if intent == "投诉类" and customer_tier == "enterprise":
        return "high"
    if intent in {"投诉类", "异常类", "信息不全待补充类"}:
        return "medium"
    return "low"


def requires_human_approval(intent: Intent, priority: Priority) -> bool:
    return intent in {"退款 / 升级类", "投诉类"} or priority == "high"


def _approval_reason(intent: Intent, priority: Priority) -> str | None:
    if intent == "退款 / 升级类":
        return "涉及退款、补偿、账单升级或主管介入，必须强制人工审批。"
    if intent == "投诉类":
        return "涉及投诉处理或人工升级，必须强制人工审批。"
    if priority == "high":
        return "高优先级 case 执行动作前必须人工确认。"
    return None


def _action_plan(intent: Intent, approval_required: bool) -> list[str]:
    if intent == "咨询类":
        return [
            "确认客户问题属于标准咨询。",
            "引用 FAQ/SOP 证据生成回复草稿。",
            "保存本次处理记录，必要时继续跟进。",
        ]
    if intent == "信息不全待补充类":
        return [
            "先向客户补充索要订单号、发生时间、截图或错误码。",
            "暂不承诺退款、补偿或升级。",
            "客户补齐信息后重新检索 SOP 和历史案例。",
        ]
    if intent == "异常类":
        return [
            "根据 SOP 创建异常工单。",
            "把客户提供的订单号、截图或错误码写入工单。",
            "如影响范围扩大，再升级给二线支持。",
        ]
    if intent == "投诉类":
        return [
            "整理客户投诉事实和时间线。",
            "生成安抚回复草稿。",
            "人工确认后升级主管或二线支持。",
        ]
    return [
        "核对退款或升级诉求是否符合政策。",
        "生成退款/升级建议和客户回复草稿。",
        "等待人工审批后再创建或升级工单。",
    ]


def _draft_response(intent: Intent, evidence: list[str]) -> str:
    if intent == "咨询类":
        evidence_text = " ".join(evidence)
        if _contains_any(evidence_text, ["售后", "退换货", "无理由", "拆封", "二次销售"]):
            return "您好，关于 7 天无理由退换货，需要结合商品品类、签收时间、包装状态以及是否影响二次销售来判断。您当前是政策咨询，我会先按售后政策解释规则；如后续要实际退货或退款，再补充订单号后进入对应审核流程。"
        return "您好，您可以通过登录页的“忘记密码/重置密码”入口处理。请确认绑定邮箱或手机号可用，如仍无法登录，我可以继续为您记录并跟进。"
    if intent == "信息不全待补充类":
        return "您好，为了继续处理，请补充订单号、问题发生时间、相关截图或错误码。收到后我会根据 SOP 判断是否需要建单或升级。"
    if intent == "异常类":
        return "您好，我会先根据异常订单 SOP 为您创建工单，并记录订单号、错误截图和发生时间，后续由支持团队继续排查。"
    if intent == "投诉类":
        return "您好，非常抱歉给您带来不好的体验。我会先整理本次问题的时间线和证据，并在人工确认后为您升级处理。"
    return "您好，退款或升级类事项需要人工审批。我会先整理政策依据、历史案例和处理建议，待审批通过后再执行对应工单动作。"


def _fallback_reflection(
    intent: Intent,
    priority: Priority,
    approval_required: bool,
    evidence: list[str],
) -> dict[str, Any]:
    approval_reason = _approval_reason(intent, priority)
    return {
        "evidence_sufficient": bool(evidence),
        "risk": "high" if approval_required else "normal",
        "reason": "涉及退款、投诉、升级或高优先级动作，执行前需要人工确认。"
        if approval_required
        else "当前动作仅生成草稿或补充信息请求，可直接输出。",
        "approval_reason": approval_reason,
    }


def _validate_llm_output(schema: type[SchemaT], value: Any) -> SchemaT | None:
    if value is None:
        return None
    if isinstance(value, schema):
        return value
    try:
        return schema.model_validate(value)  # type: ignore[attr-defined]
    except (ValidationError, TypeError, ValueError):
        return None


def _dump_model(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


def apply_policy_guardrails(result: dict[str, Any]) -> dict[str, Any]:
    guarded = dict(result)
    query = str(guarded.get("user_query", "")).lower()
    intent = guarded.get("intent")
    priority = guarded.get("priority")
    force_keywords = ["退款", "退费", "扣费", "赔偿", "补偿", "升级", "主管", "投诉"]
    force_approval = intent in {"退款 / 升级类", "投诉类"} or _contains_any(query, force_keywords)

    if force_approval:
        if intent == "退款 / 升级类" or _contains_any(query, ["升级", "主管"]):
            guarded["priority"] = "high"
            priority = "high"
        guarded["needs_human_approval"] = True
        guarded["next_step"] = "await_human_approval"
        guarded["execution_result"] = {"status": "pending_approval"}
        reflection = dict(guarded.get("reflection") or {})
        reflection["risk"] = "high"
        reason = (
            _approval_reason(cast(Intent, intent), cast(Priority, priority))
            if intent is not None and priority is not None
            else None
        ) or (
            "命中退款、补偿、投诉、升级或主管介入关键词，系统策略强制人工审批。"
        )
        reflection["approval_reason"] = f"强制人工审批：{reason}"
        guarded["reflection"] = reflection
    return guarded


def _retrieve_case_context(user_query: str, user_id: str) -> tuple[dict[str, Any], list[str]]:
    profile = get_customer_profile(user_id)
    kb = search_kb(user_query)
    history = search_case_history(user_query)
    evidence = [
        f"{kb['source'].upper()} - {kb['title']}: {kb['summary']}",
        f"{history['case_id']}: {history['summary']}",
        f"Customer profile: {profile['name']} ({profile['tier']})",
    ]
    return profile, evidence


def _analysis_from_result(result: dict[str, Any]) -> CaseAnalysis | None:
    if not {"intent", "priority", "needs_human_approval"}.issubset(result):
        return None
    return _validate_llm_output(
        CaseAnalysis,
        {
            "intent": result["intent"],
            "priority": result["priority"],
            "needs_human_approval": result["needs_human_approval"],
            "reasoning": result.get("llm_reasoning", {}).get("analysis")
            or "Deterministic case analysis.",
            "missing_information": result.get("missing_information", []),
        },
    )


def _resolution_from_result(result: dict[str, Any]) -> DraftResolution | None:
    if not {"proposed_action_plan", "draft_response", "next_step"}.issubset(result):
        return None
    return _validate_llm_output(
        DraftResolution,
        {
            "proposed_action_plan": result["proposed_action_plan"],
            "draft_response": result["draft_response"],
            "next_step": result["next_step"],
            "reasoning": result.get("llm_reasoning", {}).get("resolution")
            or "Deterministic resolution drafting.",
        },
    )


def _mark_model_fallback(result: dict[str, Any]) -> None:
    model_used = str(result.get("model_used", "deterministic-fallback"))
    if model_used != "deterministic-fallback" and not model_used.endswith(":fallback"):
        result["model_used"] = f"{model_used}:fallback"


def _safe_evidence(result: dict[str, Any]) -> list[str]:
    evidence = result.get("evidence") or []
    if isinstance(evidence, list):
        return [str(item) for item in evidence]
    return [str(evidence)]


def _safe_customer_profile(result: dict[str, Any], user_id: str) -> dict[str, Any]:
    raw_profile = result.get("customer_profile")
    profile = dict(raw_profile) if isinstance(raw_profile, dict) else {}
    profile.setdefault("customer_id", user_id)
    profile.setdefault("name", "Unknown Customer")
    profile.setdefault("tier", "standard")
    profile.setdefault("preferences", [])
    return profile


def caseflow_case_summary(result: dict[str, Any]) -> dict[str, Any]:
    execution = result.get("execution_result")
    execution_result = execution if isinstance(execution, dict) else {}
    return {
        "thread_id": str(result.get("thread_id") or ""),
        "user_id": str(result.get("user_id") or ""),
        "intent": result.get("intent"),
        "priority": result.get("priority"),
        "next_step": result.get("next_step"),
        "execution_status": execution_result.get("status", "unknown"),
        "draft_response": result.get("draft_response"),
        "evidence": _safe_evidence(result),
        "case_note": result.get("case_note"),
    }


async def persist_case_summary(store: Any, result: dict[str, Any]) -> dict[str, Any]:
    thread_id = str(result.get("thread_id") or "")
    user_id = str(result.get("user_id") or "")
    if not store:
        return {"status": "skipped", "reason": "store_unavailable"}
    if not thread_id or not user_id:
        return {"status": "skipped", "reason": "missing_thread_or_user"}

    namespace = (*CASEFLOW_SUMMARY_NAMESPACE_PREFIX, user_id)
    try:
        await store.aput(namespace, thread_id, caseflow_case_summary(result))
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}
    return {"status": "saved", "namespace": list(namespace), "key": thread_id}


async def _maybe_await_store_result(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _call_store_list(store: Any, namespace: tuple[str, ...], limit: int) -> Any:
    for method_name in ("alist", "list"):
        method = getattr(store, method_name, None)
        if not method:
            continue
        try:
            return await _maybe_await_store_result(method(namespace, limit=limit))
        except TypeError:
            return await _maybe_await_store_result(method(namespace))
    return []


async def _load_recent_case_summary_items(
    store: Any,
    namespace: tuple[str, ...],
    query: str,
    limit: int,
) -> Any:
    search = getattr(store, "asearch", None)
    if search:
        try:
            return await _maybe_await_store_result(search(namespace, query=query, limit=limit))
        except TypeError:
            try:
                return await _maybe_await_store_result(search(namespace, limit=limit))
            except TypeError:
                return await _maybe_await_store_result(search(namespace))
            except (AttributeError, NotImplementedError):
                pass
        except (AttributeError, NotImplementedError):
            pass
    return await _call_store_list(store, namespace, limit)


def _memory_item_value(item: Any) -> dict[str, Any] | None:
    value = getattr(item, "value", item)
    if isinstance(value, dict) and "value" in value and isinstance(value["value"], dict):
        value = value["value"]
    if not isinstance(value, dict):
        return None
    return value


def _truncate_memory_text(value: Any, max_length: int = 140) -> str:
    text = str(value or "").strip()
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 1]}..."


def _format_recent_case_memory(item: Any, user_id: str) -> str | None:
    value = _memory_item_value(item)
    if not value:
        return None

    item_user_id = str(value.get("user_id") or user_id)
    if item_user_id != user_id:
        return None

    thread_id = str(value.get("thread_id") or "").strip()
    intent = str(value.get("intent") or "未知意图").strip()
    status = str(value.get("execution_status") or "unknown").strip()
    draft_response = _truncate_memory_text(value.get("draft_response") or "无回复摘要")
    if not thread_id and draft_response == "无回复摘要":
        return None

    thread_label = thread_id or "unknown-thread"
    return f"历史案例 / 近期记忆: {thread_label} / {intent} / {status}: {draft_response}"


async def retrieve_recent_case_memory(
    store: Any,
    user_id: str,
    query: str,
    limit: int = 3,
) -> list[str]:
    """Retrieve compact customer-scoped case summaries as display-safe evidence."""

    if not store or not user_id:
        return []

    namespace = (*CASEFLOW_SUMMARY_NAMESPACE_PREFIX, user_id)
    try:
        items = await _load_recent_case_summary_items(store, namespace, query, limit)
    except Exception:
        return []

    memory_evidence: list[str] = []
    for item in items or []:
        formatted = _format_recent_case_memory(item, user_id)
        if formatted:
            memory_evidence.append(formatted)
        if len(memory_evidence) >= limit:
            break
    return memory_evidence


def _intent_from_result_or_query(result: dict[str, Any], user_query: str) -> Intent:
    intent = result.get("intent")
    if intent in {"咨询类", "投诉类", "异常类", "退款 / 升级类", "信息不全待补充类"}:
        return cast(Intent, intent)
    return classify_intent(user_query)


def _priority_from_result_or_query(
    result: dict[str, Any],
    intent: Intent,
    user_query: str,
    profile: dict[str, Any],
) -> Priority:
    priority = result.get("priority")
    if priority in {"low", "medium", "high"}:
        return cast(Priority, priority)
    return infer_priority(intent, user_query, str(profile.get("tier", "standard")))


def _approval_from_result_or_policy(
    result: dict[str, Any],
    intent: Intent,
    priority: Priority,
) -> bool:
    approval_required = result.get("needs_human_approval")
    if isinstance(approval_required, bool):
        return approval_required
    return requires_human_approval(intent, priority)


def _next_step_for(intent: Intent, approval_required: bool) -> str:
    if intent == "信息不全待补充类":
        return "ask_for_missing_information"
    if approval_required:
        return "await_human_approval"
    return "send_draft_response"


def _execution_result_for(approval_required: bool) -> dict[str, str]:
    return {"status": "pending_approval" if approval_required else "ready_to_send"}


def _analysis_from_fields(
    intent: Intent,
    priority: Priority,
    approval_required: bool,
    result: dict[str, Any],
    analysis: CaseAnalysis | None = None,
) -> CaseAnalysis:
    if analysis:
        return analysis
    return CaseAnalysis(
        intent=intent,
        priority=priority,
        needs_human_approval=approval_required,
        reasoning=dict(result.get("llm_reasoning") or {}).get("analysis")
        or "Deterministic case analysis.",
        missing_information=list(result.get("missing_information") or []),
    )


def _draft_from_fields(
    intent: Intent,
    approval_required: bool,
    evidence: list[str],
    result: dict[str, Any],
    resolution: DraftResolution | None = None,
) -> DraftResolution:
    action_plan = list(result.get("proposed_action_plan") or _action_plan(intent, approval_required))
    draft_response = str(
        result.get("draft_response")
        or (resolution.draft_response if resolution else _draft_response(intent, evidence))
    )
    next_step = str(
        result.get("next_step")
        or (resolution.next_step if resolution else _next_step_for(intent, approval_required))
    )
    return DraftResolution(
        proposed_action_plan=action_plan,
        draft_response=draft_response,
        next_step=next_step,
        reasoning=(resolution.reasoning if resolution else None)
        or dict(result.get("llm_reasoning") or {}).get("resolution")
        or "Deterministic resolution drafting.",
    )


def _legacy_caseflow_result(result: dict[str, Any]) -> dict[str, Any]:
    legacy = dict(result)
    user_query = str(legacy.get("user_query") or "")
    user_id = str(legacy.get("user_id") or "cust-003")
    profile = _safe_customer_profile(legacy, user_id)
    evidence = _safe_evidence(legacy)
    intent = _intent_from_result_or_query(legacy, user_query)
    priority = _priority_from_result_or_query(legacy, intent, user_query, profile)
    approval_required = _approval_from_result_or_policy(legacy, intent, priority)

    legacy.setdefault("user_query", user_query)
    legacy.setdefault("user_id", user_id)
    legacy["customer_profile"] = _dump_model(CustomerProfile.model_validate(profile))
    legacy["evidence"] = evidence
    legacy["intent"] = intent
    legacy["priority"] = priority
    legacy["needs_human_approval"] = approval_required
    legacy.setdefault("approved_action", None)
    legacy.setdefault("proposed_action_plan", _action_plan(intent, approval_required))
    legacy.setdefault("draft_response", _draft_response(intent, evidence))
    legacy.setdefault("next_step", _next_step_for(intent, approval_required))
    legacy.setdefault("execution_result", _execution_result_for(approval_required))
    legacy.setdefault("model_used", "deterministic-fallback")
    legacy.setdefault("llm_reasoning", {"analysis": None, "resolution": None})
    legacy.setdefault(
        "reflection",
        _fallback_reflection(intent, priority, approval_required, evidence),
    )
    return _dump_model(CaseFlowResult.model_validate(legacy))


def analyze_case(
    user_query: str,
    thread_id: str,
    user_id: str,
    *,
    llm_analysis: CaseAnalysis | dict[str, Any] | None = None,
    llm_resolution: DraftResolution | dict[str, Any] | None = None,
    llm_reflection: RiskReflection | dict[str, Any] | None = None,
    model_used: str = "deterministic-fallback",
) -> dict[str, Any]:
    profile, evidence = _retrieve_case_context(user_query, user_id)
    analysis = _validate_llm_output(CaseAnalysis, llm_analysis)
    resolution = _validate_llm_output(DraftResolution, llm_resolution)
    reflection = _validate_llm_output(RiskReflection, llm_reflection)
    used_fallback = bool(model_used != "deterministic-fallback" and not (analysis and resolution and reflection))

    intent = analysis.intent if analysis else classify_intent(user_query)
    priority = analysis.priority if analysis else infer_priority(intent, user_query, profile["tier"])
    approval_required = analysis.needs_human_approval if analysis else requires_human_approval(intent, priority)
    next_step = "await_human_approval" if approval_required else "send_draft_response"
    if intent == "信息不全待补充类":
        next_step = "ask_for_missing_information"
    execution_status = "pending_approval" if approval_required else "ready_to_send"

    result = {
        "user_query": user_query,
        "thread_id": thread_id,
        "user_id": user_id,
        "customer_profile": profile,
        "intent": intent,
        "priority": priority,
        "evidence": evidence,
        "proposed_action_plan": resolution.proposed_action_plan
        if resolution
        else _action_plan(intent, approval_required),
        "draft_response": resolution.draft_response if resolution else _draft_response(intent, evidence),
        "needs_human_approval": approval_required,
        "approved_action": None,
        "next_step": resolution.next_step if resolution else next_step,
        "execution_result": {"status": execution_status},
        "model_used": f"{model_used}:fallback" if used_fallback else model_used,
        "llm_reasoning": {
            "analysis": analysis.reasoning if analysis else None,
            "resolution": resolution.reasoning if resolution else None,
        },
        "reflection": reflection.model_dump()
        if reflection
        else _fallback_reflection(intent, priority, approval_required, evidence),
    }
    result = apply_policy_guardrails(result)
    if result["intent"] == "信息不全待补充类" and not result["needs_human_approval"]:
        result["next_step"] = "ask_for_missing_information"
        result["execution_result"] = {"status": "ready_to_send"}
    return _legacy_caseflow_result(result)


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "\n".join(part for part in parts if part)
    return str(content)


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    elif "{" in stripped and "}" in stripped:
        stripped = stripped[stripped.find("{") : stripped.rfind("}") + 1]
    return json.loads(stripped)


def _normalize_intent(value: Any) -> Any:
    text = str(value)
    if text in {"咨询类", "投诉类", "异常类", "退款 / 升级类", "信息不全待补充类"}:
        return text
    if _contains_any(text.lower(), ["refund", "billing", "escalat", "退款", "扣费", "补偿", "升级"]):
        return "退款 / 升级类"
    if _contains_any(text.lower(), ["complaint", "投诉", "不满意"]):
        return "投诉类"
    if _contains_any(text.lower(), ["incident", "error", "异常", "报错", "失败"]):
        return "异常类"
    if _contains_any(text.lower(), ["missing", "insufficient", "缺", "不全"]):
        return "信息不全待补充类"
    if _contains_any(text.lower(), ["faq", "question", "咨询", "密码", "登录"]):
        return "咨询类"
    return value


def _normalize_priority(value: Any) -> Any:
    text = str(value).lower()
    if text in {"low", "medium", "high"}:
        return text
    if text in {"低", "低优先级", "normal", "普通"}:
        return "low"
    if text in {"中", "中等", "中优先级", "moderate"}:
        return "medium"
    if text in {"高", "高优先级", "urgent", "critical"}:
        return "high"
    return value


def _normalize_risk(value: Any) -> Any:
    text = str(value).lower()
    if text in {"normal", "medium", "high"}:
        return text
    if text in {"low", "低", "普通", "正常"}:
        return "normal"
    if text in {"中", "中等", "moderate"}:
        return "medium"
    if text in {"高", "urgent", "critical"}:
        return "high"
    return value


def _normalize_action_plan(value: Any) -> Any:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        items = [item.strip(" -0123456789.、") for item in re.split(r"[\n;；]", value)]
        return [item for item in items if item]
    return value


def _normalize_next_step(value: Any, needs_approval: bool, intent: Intent | None = None) -> str:
    text = str(value).lower()
    if text in {"send_draft_response", "ask_for_missing_information", "await_human_approval"}:
        return text
    if needs_approval or _contains_any(text, ["approval", "approve", "审批", "人工"]):
        return "await_human_approval"
    if intent == "信息不全待补充类" or _contains_any(text, ["missing", "补充", "信息"]):
        return "ask_for_missing_information"
    return "send_draft_response"


def _coerce_llm_payload(schema: type[BaseModel], data: dict[str, Any]) -> dict[str, Any]:
    coerced = dict(data)
    if schema is CaseAnalysis:
        coerced["intent"] = _normalize_intent(coerced.get("intent"))
        coerced["priority"] = _normalize_priority(coerced.get("priority"))
        if "needs_human_approval" not in coerced:
            coerced["needs_human_approval"] = requires_human_approval(
                coerced["intent"], coerced["priority"]
            )
        coerced.setdefault("reasoning", "LLM completed case analysis.")
        if "missing_information" not in coerced or coerced["missing_information"] in (None, ""):
            coerced["missing_information"] = []
        elif isinstance(coerced["missing_information"], str):
            coerced["missing_information"] = [coerced["missing_information"]]
    elif schema is DraftResolution:
        coerced["proposed_action_plan"] = _normalize_action_plan(
            coerced.get("proposed_action_plan") or coerced.get("action_plan")
        )
        needs_approval = bool(coerced.get("needs_human_approval"))
        coerced["next_step"] = _normalize_next_step(coerced.get("next_step"), needs_approval)
        coerced.setdefault("reasoning", "LLM completed resolution drafting.")
    elif schema is RiskReflection:
        coerced["risk"] = _normalize_risk(coerced.get("risk"))
        coerced.setdefault("evidence_sufficient", True)
        coerced.setdefault("reason", "LLM completed risk reflection.")
        if str(coerced.get("approval_reason", "")).lower() in {"", "none", "null", "n/a"}:
            coerced["approval_reason"] = None
    return coerced


def _configured_model_label(config) -> str:
    model_name = config["configurable"].get("model", settings.DEFAULT_MODEL)
    return getattr(model_name, "value", str(model_name))


async def _call_llm_json(schema: type[BaseModel], system_prompt: str, payload: dict[str, Any], config):
    model_name = config["configurable"].get("model", settings.DEFAULT_MODEL)
    model = get_model(model_name)
    prompt = (
        "Return only one valid JSON object. Do not include markdown or extra text.\n"
        f"JSON schema fields: {list(schema.model_fields.keys())}\n\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    response = await model.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]
    )
    data = _coerce_llm_payload(
        schema,
        _extract_json_object(_message_content_to_text(response.content)),
    )
    return schema.model_validate(data)


async def call_llm_case_analysis(
    user_query: str,
    customer_profile: dict[str, Any],
    evidence: list[str],
    config,
) -> CaseAnalysis | None:
    try:
        return await _call_llm_json(
            CaseAnalysis,
            (
                "You are CaseFlow Agent's triage node. Classify an enterprise customer "
                "support case into exactly one Chinese intent label and priority. "
                "Use needs_human_approval=true for refund, compensation, complaint, "
                "escalation, supervisor handoff, or high-risk actions."
            ),
            {
                "user_query": user_query,
                "customer_profile": customer_profile,
                "evidence": evidence,
                "allowed_intents": [
                    "咨询类",
                    "投诉类",
                    "异常类",
                    "退款 / 升级类",
                    "信息不全待补充类",
                ],
                "allowed_priorities": ["low", "medium", "high"],
            },
            config,
        )
    except Exception:
        return None


async def call_llm_resolution(
    user_query: str,
    analysis: CaseAnalysis | None,
    evidence: list[str],
    config,
) -> DraftResolution | None:
    try:
        return await _call_llm_json(
            DraftResolution,
            (
                "You are CaseFlow Agent's resolution planner. Generate a concise "
                "business action plan and a customer-facing Chinese draft response. "
                "Stay grounded in the evidence. Do not promise refunds or escalation "
                "execution before approval."
            ),
            {
                "user_query": user_query,
                "analysis": analysis.model_dump() if analysis else None,
                "evidence": evidence,
                "next_step_options": [
                    "send_draft_response",
                    "ask_for_missing_information",
                    "await_human_approval",
                ],
            },
            config,
        )
    except Exception:
        return None


async def call_llm_reflection(
    user_query: str,
    analysis: CaseAnalysis | None,
    resolution: DraftResolution | None,
    evidence: list[str],
    config,
) -> RiskReflection | None:
    try:
        return await _call_llm_json(
            RiskReflection,
            (
                "You are CaseFlow Agent's risk reflection node. Check whether the "
                "evidence is sufficient, identify risk, and explain whether human "
                "approval is required before any ticket, refund, compensation, "
                "complaint escalation, or supervisor handoff."
            ),
            {
                "user_query": user_query,
                "analysis": analysis.model_dump() if analysis else None,
                "resolution": resolution.model_dump() if resolution else None,
                "evidence": evidence,
                "allowed_risk": ["normal", "medium", "high"],
            },
            config,
        )
    except Exception:
        return None


def _latest_user_query(state: CaseFlowState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


async def intake_and_classify(state: CaseFlowState, config) -> CaseFlowState:
    thread_id = config["configurable"].get("thread_id", "local-thread")
    user_id = config["configurable"].get("user_id", "cust-003")
    user_query = _latest_user_query(state)
    return cast(CaseFlowState, {
        "user_query": user_query,
        "thread_id": thread_id,
        "user_id": user_id,
        "caseflow_result": {
            "user_query": user_query,
            "thread_id": thread_id,
            "user_id": user_id,
            "approved_action": None,
            "model_used": _configured_model_label(config),
        },
    })


async def retrieve_evidence(
    state: CaseFlowState, config, store: BaseStore | None = None
) -> CaseFlowState:
    user_query = state.get("user_query") or _latest_user_query(state)
    thread_id = state.get("thread_id") or config["configurable"].get("thread_id", "local-thread")
    user_id = state.get("user_id") or config["configurable"].get("user_id", "cust-003")
    profile, evidence = _retrieve_case_context(user_query, user_id)
    evidence = list(evidence)
    evidence.extend(await retrieve_recent_case_memory(store, user_id, user_query))
    retrieved_context = RetrievedEvidence.model_validate(
        {"customer_profile": profile, "evidence": evidence}
    )
    result = dict(state.get("caseflow_result") or {})
    result.update(
        {
            "user_query": user_query,
            "thread_id": thread_id,
            "user_id": user_id,
            "customer_profile": _dump_model(retrieved_context.customer_profile),
            "evidence": list(retrieved_context.evidence),
        }
    )
    return cast(CaseFlowState, {
        "user_query": user_query,
        "thread_id": thread_id,
        "user_id": user_id,
        "retrieved_context": _dump_model(retrieved_context),
        "caseflow_result": result,
    })


async def plan_actions(state: CaseFlowState, config) -> CaseFlowState:
    result = dict(state.get("caseflow_result") or {})
    user_query = str(result.get("user_query") or state.get("user_query") or _latest_user_query(state))
    user_id = str(
        result.get("user_id")
        or state.get("user_id")
        or config["configurable"].get("user_id", "cust-003")
    )
    profile = _safe_customer_profile(result, user_id)
    evidence = _safe_evidence(result)
    result.setdefault("user_query", user_query)
    result.setdefault("user_id", user_id)
    result.setdefault("customer_profile", profile)
    result.setdefault("evidence", evidence)
    llm_analysis = await call_llm_case_analysis(user_query, profile, evidence, config)
    analysis = _validate_llm_output(CaseAnalysis, llm_analysis)

    intent = analysis.intent if analysis else classify_intent(user_query)
    priority = analysis.priority if analysis else infer_priority(intent, user_query, profile["tier"])
    approval_required = analysis.needs_human_approval if analysis else requires_human_approval(
        intent, priority
    )
    next_step = _next_step_for(intent, approval_required)
    typed_analysis = _analysis_from_fields(intent, priority, approval_required, result, analysis)

    result.update(
        {
            "intent": intent,
            "priority": priority,
            "needs_human_approval": approval_required,
            "approved_action": result.get("approved_action"),
            "proposed_action_plan": _action_plan(intent, approval_required),
            "next_step": next_step,
            "execution_result": _execution_result_for(approval_required),
            "llm_reasoning": {
                **dict(result.get("llm_reasoning") or {}),
                "analysis": analysis.reasoning if analysis else None,
            },
        }
    )
    if analysis is None:
        _mark_model_fallback(result)
    typed_result = _legacy_caseflow_result(result)
    return cast(
        CaseFlowState,
        {
            "analysis": _dump_model(typed_analysis),
            "typed_result": typed_result,
            "caseflow_result": result,
        },
    )


async def draft_resolution(state: CaseFlowState, config) -> CaseFlowState:
    result = dict(state.get("caseflow_result") or {})
    user_query = str(result.get("user_query") or state.get("user_query") or _latest_user_query(state))
    user_id = str(
        result.get("user_id")
        or state.get("user_id")
        or config["configurable"].get("user_id", "cust-003")
    )
    profile = _safe_customer_profile(result, user_id)
    evidence = _safe_evidence(result)
    intent = _intent_from_result_or_query(result, user_query)
    priority = _priority_from_result_or_query(result, intent, user_query, profile)
    approval_required = _approval_from_result_or_policy(result, intent, priority)
    result.setdefault("user_query", user_query)
    result.setdefault("user_id", user_id)
    result.setdefault("customer_profile", profile)
    result.setdefault("evidence", evidence)
    result.setdefault("intent", intent)
    result.setdefault("priority", priority)
    result.setdefault("needs_human_approval", approval_required)
    result.setdefault("proposed_action_plan", _action_plan(intent, approval_required))
    result.setdefault("next_step", _next_step_for(intent, approval_required))
    result.setdefault("execution_result", _execution_result_for(approval_required))
    analysis = _analysis_from_result(result)
    resolution = await call_llm_resolution(user_query, analysis, evidence, config)
    validated_resolution = _validate_llm_output(DraftResolution, resolution)
    if validated_resolution:
        result["draft_response"] = validated_resolution.draft_response
        result["llm_reasoning"] = {
            **dict(result.get("llm_reasoning") or {}),
            "resolution": validated_resolution.reasoning,
        }
    else:
        result["draft_response"] = _draft_response(intent, evidence)
        result["llm_reasoning"] = {
            **dict(result.get("llm_reasoning") or {}),
            "resolution": None,
        }
        _mark_model_fallback(result)
    typed_draft = _draft_from_fields(intent, approval_required, evidence, result, validated_resolution)
    typed_result = _legacy_caseflow_result(result)
    return cast(
        CaseFlowState,
        {
            "draft": _dump_model(typed_draft),
            "typed_result": typed_result,
            "caseflow_result": result,
        },
    )


async def reflect_and_risk_check(state: CaseFlowState, config) -> CaseFlowState:
    result = dict(state.get("caseflow_result") or {})
    user_query = str(result.get("user_query") or state.get("user_query") or _latest_user_query(state))
    user_id = str(
        result.get("user_id")
        or state.get("user_id")
        or config["configurable"].get("user_id", "cust-003")
    )
    profile = _safe_customer_profile(result, user_id)
    evidence = _safe_evidence(result)
    intent = _intent_from_result_or_query(result, user_query)
    priority = _priority_from_result_or_query(result, intent, user_query, profile)
    approval_required = _approval_from_result_or_policy(result, intent, priority)
    result.setdefault("user_query", user_query)
    result.setdefault("user_id", user_id)
    result.setdefault("customer_profile", profile)
    result.setdefault("evidence", evidence)
    result.setdefault("intent", intent)
    result.setdefault("priority", priority)
    result.setdefault("needs_human_approval", approval_required)
    result.setdefault("proposed_action_plan", _action_plan(intent, approval_required))
    result.setdefault("next_step", _next_step_for(intent, approval_required))
    result.setdefault("execution_result", _execution_result_for(approval_required))
    if "reflection" not in result:
        reflection = await call_llm_reflection(
            user_query,
            _analysis_from_result(result),
            _resolution_from_result(result),
            evidence,
            config,
        )
        validated_reflection = _validate_llm_output(RiskReflection, reflection)
        if validated_reflection:
            result["reflection"] = validated_reflection.model_dump()
        else:
            result["reflection"] = _fallback_reflection(
                intent,
                priority,
                approval_required,
                evidence,
            )
            _mark_model_fallback(result)
    result = apply_policy_guardrails(result)
    if result.get("intent") == "信息不全待补充类" and not result.get("needs_human_approval"):
        result["next_step"] = "ask_for_missing_information"
        result["execution_result"] = {"status": "ready_to_send"}
    typed_result = _legacy_caseflow_result(result)
    return cast(
        CaseFlowState,
        {
            "reflection": typed_result["reflection"],
            "typed_result": typed_result,
            "caseflow_result": typed_result,
        },
    )


def _approval_route(state: CaseFlowState) -> Literal["approval", "execute"]:
    if state["caseflow_result"]["needs_human_approval"]:
        return "approval"
    return "execute"


async def request_human_approval_if_needed(state: CaseFlowState) -> CaseFlowState:
    result = dict(state["caseflow_result"])
    reflection = result.get("reflection", {})
    approval_message = (
        "需要人工审批后才能执行：\n"
        f"- intent: {result['intent']}\n"
        f"- priority: {result['priority']}\n"
        f"- evidence: {'; '.join(result['evidence'])}\n"
        f"- risk_reason: {reflection.get('reason', '未提供')}\n"
        f"- approval_reason: {reflection.get('approval_reason', '策略要求人工确认')}\n"
        f"- proposed_action: {'; '.join(result['proposed_action_plan'])}\n"
        "请回复 approve / reject，或给出修改意见。"
    )
    approval_decision = parse_caseflow_approval_decision(interrupt(approval_message))
    result["approved_action"] = approval_decision.decision
    result["approval_decision"] = approval_decision.model_dump(mode="json")
    if approval_decision.decision == "reject":
        result["execution_result"] = {
            "status": "rejected",
            "reason": approval_decision.reason or "主管拒绝执行。",
        }
        result["next_step"] = "approval_rejected"
    elif approval_decision.decision == "modify":
        result["execution_result"] = {
            "status": "modification_requested",
            "reason": approval_decision.reason or "主管要求修改。",
            "modification_notes": approval_decision.modification_notes,
        }
        result["next_step"] = "approval_modification_requested"
    else:
        result["execution_result"] = {
            "status": "approved",
            "approval_note": approval_decision.reason,
        }
        result["next_step"] = "execute_approved_action"
    typed_result = _legacy_caseflow_result(result)
    return cast(
        CaseFlowState,
        {
            "execution_result": typed_result["execution_result"],
            "typed_result": typed_result,
            "caseflow_result": typed_result,
            "approved_action": approval_decision.decision,
        },
    )


async def execute_action(state: CaseFlowState) -> CaseFlowState:
    result = dict(state.get("caseflow_result") or {})
    user_query = str(result.get("user_query") or state.get("user_query") or _latest_user_query(state))
    thread_id = str(result.get("thread_id") or state.get("thread_id") or "local-thread")
    user_id = str(result.get("user_id") or state.get("user_id") or "cust-003")
    intent = _intent_from_result_or_query(result, user_query)
    execution_result = dict(result.get("execution_result") or {"status": "ready_to_send"})
    result.setdefault("user_query", user_query)
    result.setdefault("thread_id", thread_id)
    result.setdefault("user_id", user_id)
    result.setdefault("intent", intent)
    result["execution_result"] = execution_result

    if execution_result.get("status") in {"rejected", "modification_requested"}:
        result["case_note"] = save_case_note(result["thread_id"], result)
        typed_result = _legacy_caseflow_result(result)
        return cast(
            CaseFlowState,
            {
                "execution_result": typed_result["execution_result"],
                "typed_result": typed_result,
                "caseflow_result": typed_result,
            },
        )

    if intent in {"退款 / 升级类", "投诉类", "异常类"}:
        ticket = create_ticket(thread_id, user_id, intent)
        if intent in {"退款 / 升级类", "投诉类"}:
            escalation = escalate_ticket(ticket["ticket_id"], "human_review")
            result["execution_result"] = {
                "status": "executed",
                "ticket": ticket,
                "escalation": escalation,
            }
        else:
            result["execution_result"] = {"status": "executed", "ticket": ticket}
    else:
        result["execution_result"] = {"status": execution_result.get("status", "ready_to_send")}

    result["case_note"] = save_case_note(thread_id, result)
    typed_result = _legacy_caseflow_result(result)
    return cast(
        CaseFlowState,
        {
            "execution_result": typed_result["execution_result"],
            "typed_result": typed_result,
            "caseflow_result": typed_result,
        },
    )


async def finalize_and_persist(
    state: CaseFlowState, store: BaseStore | None = None
) -> CaseFlowState:
    result = dict(state.get("caseflow_result") or {})
    user_query = str(result.get("user_query") or state.get("user_query") or _latest_user_query(state))
    user_id = str(result.get("user_id") or state.get("user_id") or "cust-003")
    profile = _safe_customer_profile(result, user_id)
    evidence = _safe_evidence(result)
    intent = _intent_from_result_or_query(result, user_query)
    priority = _priority_from_result_or_query(result, intent, user_query, profile)
    approval_required = _approval_from_result_or_policy(result, intent, priority)
    result.setdefault("user_query", user_query)
    result.setdefault("user_id", user_id)
    result.setdefault("customer_profile", profile)
    result.setdefault("evidence", evidence)
    result.setdefault("intent", intent)
    result.setdefault("priority", priority)
    result.setdefault("needs_human_approval", approval_required)
    result.setdefault("next_step", _next_step_for(intent, approval_required))
    result.setdefault("draft_response", _draft_response(intent, evidence))
    result = _legacy_caseflow_result(result)
    result["memory_result"] = await persist_case_summary(store, result)
    result["workflow_nodes"] = workflow_node_definitions()
    result["workflow_events"] = build_workflow_events(result)
    result = _legacy_caseflow_result(result)
    content = (
        "CaseFlow 处理结果\n\n"
        f"意图：{result.get('intent')}\n"
        f"优先级：{result.get('priority')}\n"
        f"下一步：{result.get('next_step')}\n\n"
        f"回复草稿：{result.get('draft_response')}"
    )
    return cast(
        CaseFlowState,
        {
            "typed_result": result,
            "caseflow_result": result,
            "messages": [AIMessage(content=content, additional_kwargs={"custom_data": result})],
        },
    )


agent = StateGraph(CaseFlowState)
agent.add_node("intake_and_classify", intake_and_classify)
agent.add_node("retrieve_evidence", retrieve_evidence)
agent.add_node("plan_actions", plan_actions)
agent.add_node("draft_resolution", draft_resolution)
agent.add_node("reflect_and_risk_check", reflect_and_risk_check)
agent.add_node("request_human_approval_if_needed", request_human_approval_if_needed)
agent.add_node("execute_action", execute_action)
agent.add_node("finalize_and_persist", finalize_and_persist)

agent.set_entry_point("intake_and_classify")
agent.add_edge("intake_and_classify", "retrieve_evidence")
agent.add_edge("retrieve_evidence", "plan_actions")
agent.add_edge("plan_actions", "draft_resolution")
agent.add_edge("draft_resolution", "reflect_and_risk_check")
agent.add_conditional_edges(
    "reflect_and_risk_check",
    _approval_route,
    {"approval": "request_human_approval_if_needed", "execute": "execute_action"},
)
agent.add_edge("request_human_approval_if_needed", "execute_action")
agent.add_edge("execute_action", "finalize_and_persist")
agent.add_edge("finalize_and_persist", END)

caseflow_agent = agent.compile()
caseflow_agent.name = "caseflow-agent"
