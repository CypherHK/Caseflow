import json
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ValidationError

from agents.caseflow_agent import (
    CaseAnalysis,
    CaseApprovalDecision,
    CaseFlowResult,
    CustomerProfile,
    DraftResolution,
    RetrievedEvidence,
    RiskReflection,
    analyze_case,
    apply_policy_guardrails,
    caseflow_agent,
    caseflow_case_summary,
    draft_resolution,
    execute_action,
    finalize_and_persist,
    parse_caseflow_approval_decision,
    persist_case_summary,
    plan_actions,
    reflect_and_risk_check,
    retrieve_evidence,
    retrieve_recent_case_memory,
)
from agents.caseflow_agent import (
    agent as caseflow_graph_builder,
)
from agents.caseflow_tools import (
    create_ticket,
    escalate_ticket,
    get_customer_profile,
    save_case_note,
    search_case_history,
    search_kb,
)
from schema.models import FakeModelName


def _retrieved_caseflow_state(
    user_query: str = "客户被重复扣费，要求马上退款",
    user_id: str = "cust-002",
    tier: str = "enterprise",
) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "thread_id": "thread-node",
        "user_id": user_id,
        "caseflow_result": {
            "user_query": user_query,
            "thread_id": "thread-node",
            "user_id": user_id,
            "customer_profile": {
                "customer_id": user_id,
                "name": "Test Customer",
                "tier": tier,
            },
            "evidence": [
                "POLICY - 退款审批政策: 退款或重复扣费必须人工审批。",
                "CASE-101: 历史案例：重复扣费后整理证据并提交审批。",
                f"Customer profile: Test Customer ({tier})",
            ],
            "approved_action": None,
            "model_used": "openai-compatible",
        },
    }


def _forbid_unexpected_node_call(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("Unexpected cross-node dependency call")


def _assert_no_pydantic_models(value: Any) -> None:
    if isinstance(value, BaseModel):
        raise AssertionError(f"Raw Pydantic model leaked into custom_data: {type(value)}")
    if isinstance(value, dict):
        for item in value.values():
            _assert_no_pydantic_models(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_pydantic_models(item)


def test_caseflow_result_model_validates_complete_business_contract():
    result = CaseFlowResult.model_validate(
        {
            "user_query": "客户问怎么重置账号密码",
            "thread_id": "thread-typed",
            "user_id": "cust-001",
            "customer_profile": {
                "customer_id": "cust-001",
                "name": "Acme Retail",
                "tier": "standard",
                "preferences": ["回复要简洁"],
            },
            "intent": "咨询类",
            "priority": "low",
            "evidence": ["FAQ - 密码重置: 使用登录页入口。"],
            "proposed_action_plan": ["引用 FAQ 生成回复。"],
            "draft_response": "您好，您可以通过登录页重置密码。",
            "needs_human_approval": False,
            "approved_action": None,
            "next_step": "send_draft_response",
            "execution_result": {"status": "ready_to_send"},
            "model_used": "deterministic-fallback",
            "llm_reasoning": {"analysis": None, "resolution": None},
            "reflection": {
                "evidence_sufficient": True,
                "risk": "normal",
                "reason": "当前可直接回复。",
                "approval_reason": None,
            },
        }
    )

    dumped = result.model_dump(mode="json")
    assert dumped["intent"] == "咨询类"
    assert dumped["customer_profile"]["customer_id"] == "cust-001"
    assert dumped["execution_result"]["status"] == "ready_to_send"


def test_caseflow_result_model_rejects_missing_required_business_fields():
    with pytest.raises(ValidationError):
        CaseFlowResult.model_validate(
            {
                "intent": "咨询类",
                "priority": "low",
                "evidence": [],
                "draft_response": "missing required fields",
            }
        )


def test_caseflow_result_dump_is_json_safe_and_preserves_custom_data_keys():
    result = CaseFlowResult.model_validate(
        analyze_case("客户问怎么重置账号密码", "thread-json", "cust-001")
    )
    dumped = result.model_dump(mode="json")

    json.dumps(dumped, ensure_ascii=False)
    _assert_no_pydantic_models(dumped)
    for key in {
        "intent",
        "priority",
        "evidence",
        "proposed_action_plan",
        "draft_response",
        "needs_human_approval",
        "next_step",
        "execution_result",
    }:
        assert key in dumped


def test_caseflow_case_summary_is_compact_and_json_safe():
    result = analyze_case("客户问怎么重置账号密码", "thread-summary", "cust-001")
    result["case_note"] = save_case_note("thread-summary", result)

    summary = caseflow_case_summary(result)

    assert summary["thread_id"] == "thread-summary"
    assert summary["user_id"] == "cust-001"
    assert summary["intent"] == "咨询类"
    assert summary["execution_status"] == "ready_to_send"
    assert "messages" not in summary
    json.dumps(summary, ensure_ascii=False)


@pytest.mark.asyncio
async def test_persist_case_summary_writes_customer_scoped_store_item():
    writes = []

    class FakeStore:
        async def aput(self, namespace, key, value):
            writes.append((namespace, key, value))

    result = analyze_case("客户问怎么重置账号密码", "thread-store", "cust-001")
    memory_result = await persist_case_summary(FakeStore(), result)

    assert memory_result == {
        "status": "saved",
        "namespace": ["caseflow", "case_summaries", "cust-001"],
        "key": "thread-store",
    }
    assert writes[0][0] == ("caseflow", "case_summaries", "cust-001")
    assert writes[0][1] == "thread-store"
    assert writes[0][2]["intent"] == "咨询类"


@pytest.mark.asyncio
async def test_persist_case_summary_failure_is_non_fatal():
    class FailingStore:
        async def aput(self, namespace, key, value):
            raise RuntimeError("store offline")

    result = analyze_case("客户问怎么重置账号密码", "thread-store-fail", "cust-001")
    memory_result = await persist_case_summary(FailingStore(), result)

    assert memory_result["status"] == "unavailable"
    assert "store offline" in memory_result["error"]


@pytest.mark.asyncio
async def test_retrieve_recent_case_memory_formats_same_customer_items():
    class FakeStore:
        async def asearch(self, namespace, query="", limit=3):
            assert namespace == ("caseflow", "case_summaries", "cust-001")
            assert query == "客户又来询问密码重置"
            assert limit == 3
            return [
                {
                    "value": {
                        "thread_id": "thread-previous",
                        "user_id": "cust-001",
                        "intent": "咨询类",
                        "execution_status": "ready_to_send",
                        "draft_response": "上次已引导客户通过登录页重置密码。",
                    }
                },
                {
                    "value": {
                        "thread_id": "thread-other",
                        "user_id": "cust-999",
                        "intent": "投诉类",
                        "execution_status": "executed",
                        "draft_response": "其他客户的记录不应出现。",
                    }
                },
            ]

    memory = await retrieve_recent_case_memory(
        FakeStore(),
        "cust-001",
        "客户又来询问密码重置",
    )

    assert memory == [
        "历史案例 / 近期记忆: thread-previous / 咨询类 / ready_to_send: 上次已引导客户通过登录页重置密码。"
    ]


@pytest.mark.asyncio
async def test_retrieve_recent_case_memory_falls_back_to_list_and_ignores_bad_items():
    class FakeItem:
        def __init__(self, value):
            self.value = value

    class FakeStore:
        async def asearch(self, namespace, query="", limit=3):
            raise NotImplementedError("semantic index disabled")

        async def alist(self, namespace, limit=3):
            assert namespace == ("caseflow", "case_summaries", "cust-002")
            return [
                None,
                FakeItem("bad-value"),
                FakeItem(
                    {
                        "thread_id": "thread-listed",
                        "user_id": "cust-002",
                        "intent": "退款 / 升级类",
                        "execution_status": "approved",
                        "draft_response": "上次重复扣费已收集凭证并提交审批。",
                    }
                ),
            ]

    memory = await retrieve_recent_case_memory(FakeStore(), "cust-002", "重复扣费")

    assert memory == [
        "历史案例 / 近期记忆: thread-listed / 退款 / 升级类 / approved: 上次重复扣费已收集凭证并提交审批。"
    ]


@pytest.mark.asyncio
async def test_retrieve_recent_case_memory_store_failure_returns_empty_list():
    class FailingStore:
        async def asearch(self, namespace, query="", limit=3):
            raise RuntimeError("store unavailable")

    assert await retrieve_recent_case_memory(FailingStore(), "cust-001", "anything") == []


def test_fallback_paths_produce_typed_valid_results():
    cases = [
        ("客户问怎么重置账号密码", "cust-001", "咨询类"),
        ("我的订单有问题", "cust-003", "信息不全待补充类"),
        ("客户被重复扣费，要求马上退款", "cust-002", "退款 / 升级类"),
    ]

    for query, user_id, expected_intent in cases:
        result = CaseFlowResult.model_validate(analyze_case(query, "thread-fallback", user_id))
        assert result.intent == expected_intent
        assert result.evidence
        assert result.draft_response


def test_faq_case_returns_structured_low_risk_result():
    result = analyze_case(
        user_query="客户问怎么重置账号密码",
        thread_id="thread-faq",
        user_id="cust-001",
    )

    assert result["intent"] == "咨询类"
    assert result["priority"] == "low"
    assert result["needs_human_approval"] is False
    assert result["next_step"] == "send_draft_response"
    assert result["draft_response"]
    assert result["evidence"]
    assert result["execution_result"]["status"] == "ready_to_send"
    assert result["model_used"] == "deterministic-fallback"
    assert result["reflection"]["risk"] == "normal"


def test_refund_case_requires_approval_before_execution():
    result = analyze_case(
        user_query="客户投诉扣费错误，要求退款并升级给主管",
        thread_id="thread-refund",
        user_id="cust-002",
    )

    assert result["intent"] == "退款 / 升级类"
    assert result["priority"] == "high"
    assert result["needs_human_approval"] is True
    assert result["next_step"] == "await_human_approval"
    assert result["execution_result"]["status"] == "pending_approval"
    assert any("退款" in item for item in result["proposed_action_plan"])
    assert result["reflection"]["approval_reason"]


def test_missing_information_case_asks_for_required_details():
    result = analyze_case(
        user_query="我的订单有问题",
        thread_id="thread-missing",
        user_id="cust-003",
    )

    assert result["intent"] == "信息不全待补充类"
    assert result["priority"] == "medium"
    assert result["needs_human_approval"] is False
    assert result["next_step"] == "ask_for_missing_information"
    assert "订单号" in result["draft_response"]


def test_caseflow_structured_llm_models_validate_expected_shapes():
    analysis = CaseAnalysis.model_validate(
        {
            "intent": "投诉类",
            "priority": "high",
            "needs_human_approval": True,
            "reasoning": "客户明确表达投诉，且需要人工介入。",
            "missing_information": [],
        }
    )
    resolution = DraftResolution.model_validate(
        {
            "proposed_action_plan": ["整理事实", "人工审批后升级"],
            "draft_response": "您好，我们会先整理证据并升级给人工处理。",
            "next_step": "await_human_approval",
            "reasoning": "投诉需要人工确认。",
        }
    )
    reflection = RiskReflection.model_validate(
        {
            "evidence_sufficient": True,
            "risk": "high",
            "reason": "涉及升级处理。",
            "approval_reason": "投诉升级前需要人工确认。",
        }
    )

    assert analysis.intent == "投诉类"
    assert resolution.proposed_action_plan[0] == "整理事实"
    assert reflection.approval_reason == "投诉升级前需要人工确认。"


def test_llm_outputs_are_merged_and_policy_guardrail_forces_approval():
    result = analyze_case(
        user_query="客户要求马上退款，但模型误判为低风险",
        thread_id="thread-llm",
        user_id="cust-001",
        llm_analysis={
            "intent": "退款 / 升级类",
            "priority": "medium",
            "needs_human_approval": False,
            "reasoning": "模型认为可直接回复。",
            "missing_information": [],
        },
        llm_resolution={
            "proposed_action_plan": ["核对退款政策", "准备退款建议"],
            "draft_response": "您好，我会先核对退款政策并整理处理建议。",
            "next_step": "send_draft_response",
            "reasoning": "先给出草稿。",
        },
        llm_reflection={
            "evidence_sufficient": True,
            "risk": "normal",
            "reason": "模型认为风险较低。",
            "approval_reason": None,
        },
        model_used="openai-compatible",
    )

    assert result["model_used"] == "openai-compatible"
    assert result["intent"] == "退款 / 升级类"
    assert result["priority"] == "high"
    assert result["needs_human_approval"] is True
    assert result["next_step"] == "await_human_approval"
    assert result["execution_result"]["status"] == "pending_approval"
    assert result["llm_reasoning"]["analysis"] == "模型认为可直接回复。"
    assert "强制人工审批" in result["reflection"]["approval_reason"]


def test_invalid_llm_outputs_fall_back_to_deterministic_oracle():
    result = analyze_case(
        user_query="客户问怎么重置账号密码",
        thread_id="thread-invalid-llm",
        user_id="cust-001",
        llm_analysis={"intent": "未知", "priority": "urgent"},
        llm_resolution={"draft_response": ""},
        llm_reflection={"risk": "critical"},
        model_used="openai-compatible",
    )

    assert result["intent"] == "咨询类"
    assert result["priority"] == "low"
    assert result["draft_response"]
    assert result["model_used"] == "openai-compatible:fallback"


def test_policy_guardrail_forces_complaint_and_escalation_approval():
    guarded = apply_policy_guardrails(
        {
            "intent": "投诉类",
            "priority": "medium",
            "needs_human_approval": False,
            "next_step": "send_draft_response",
            "execution_result": {"status": "ready_to_send"},
            "reflection": {"risk": "normal", "approval_reason": None},
            "user_query": "我要投诉并升级主管",
        }
    )

    assert guarded["priority"] == "high"
    assert guarded["needs_human_approval"] is True
    assert guarded["next_step"] == "await_human_approval"
    assert guarded["execution_result"]["status"] == "pending_approval"


def test_parse_caseflow_approval_decision_accepts_structured_and_legacy_payloads():
    structured = parse_caseflow_approval_decision(
        '{"decision":"approve","reason":"主管确认","modification_notes":""}'
    )
    assert structured == CaseApprovalDecision(
        decision="approve",
        reason="主管确认",
        modification_notes=None,
    )

    assert parse_caseflow_approval_decision("approve").decision == "approve"
    assert parse_caseflow_approval_decision("批准执行").decision == "approve"
    assert parse_caseflow_approval_decision("reject because policy mismatch").decision == "reject"
    assert parse_caseflow_approval_decision("拒绝：证据不足").decision == "reject"

    free_text = parse_caseflow_approval_decision("请先把回复改得更谨慎")
    assert free_text.decision == "modify"
    assert free_text.modification_notes == "请先把回复改得更谨慎"


def test_mock_tools_return_business_context_and_records():
    assert search_kb("退款")["source"] in {"policy", "sop", "faq"}
    assert search_case_history("退款")["case_id"].startswith("CASE-")
    assert get_customer_profile("cust-001")["customer_id"] == "cust-001"
    assert create_ticket("thread-1", "cust-001", "异常类")["ticket_id"].startswith("TCK-")
    assert escalate_ticket("TCK-1", "refund_review")["status"] == "escalated"
    assert save_case_note("thread-1", {"intent": "咨询类"})["status"] == "saved"


@pytest.mark.asyncio
async def test_caseflow_graph_returns_ai_message_with_structured_custom_data():
    result = await caseflow_agent.ainvoke(
        {"messages": [HumanMessage(content="客户问怎么重置账号密码")]},
        config=RunnableConfig(
            configurable={
                "thread_id": "thread-graph",
                "user_id": "cust-001",
                "model": FakeModelName.FAKE,
            }
        ),
    )

    message = result["messages"][-1]
    custom_data = message.additional_kwargs["custom_data"]

    assert message.content.startswith("CaseFlow 处理结果")
    assert custom_data["intent"] == "咨询类"
    assert custom_data["needs_human_approval"] is False
    assert custom_data["model_used"]
    assert custom_data["reflection"]


@pytest.mark.asyncio
async def test_retrieve_evidence_node_populates_customer_profile_and_evidence():
    state = {
        "messages": [HumanMessage(content="客户被重复扣费，要求马上退款")],
        "user_query": "客户被重复扣费，要求马上退款",
        "thread_id": "thread-retrieve",
        "user_id": "cust-002",
        "caseflow_result": {
            "user_query": "客户被重复扣费，要求马上退款",
            "thread_id": "thread-retrieve",
            "user_id": "cust-002",
        },
    }

    update = await retrieve_evidence(
        state,
        RunnableConfig(configurable={"thread_id": "thread-retrieve", "user_id": "cust-002"}),
    )
    result = update["caseflow_result"]

    assert result["customer_profile"]["customer_id"] == "cust-002"
    assert len(result["evidence"]) == 3
    assert any("退款" in item or "扣费" in item for item in result["evidence"])
    retrieved = RetrievedEvidence.model_validate(update["retrieved_context"])
    assert retrieved.customer_profile.customer_id == "cust-002"
    assert retrieved.evidence == result["evidence"]


@pytest.mark.asyncio
async def test_retrieve_evidence_appends_recent_case_memory_from_store():
    class FakeStore:
        async def asearch(self, namespace, query="", limit=3):
            assert namespace == ("caseflow", "case_summaries", "cust-001")
            return [
                {
                    "value": {
                        "thread_id": "thread-old",
                        "user_id": "cust-001",
                        "intent": "咨询类",
                        "execution_status": "ready_to_send",
                        "draft_response": "上次已发送密码重置指引。",
                    }
                }
            ]

    update = await retrieve_evidence(
        {
            "messages": [HumanMessage(content="客户问怎么重置账号密码")],
            "user_query": "客户问怎么重置账号密码",
            "thread_id": "thread-new",
            "user_id": "cust-001",
            "caseflow_result": {
                "user_query": "客户问怎么重置账号密码",
                "thread_id": "thread-new",
                "user_id": "cust-001",
            },
        },
        RunnableConfig(configurable={"thread_id": "thread-new", "user_id": "cust-001"}),
        store=FakeStore(),
    )
    result = update["caseflow_result"]

    assert len(result["evidence"]) == 4
    assert result["evidence"][-1].startswith("历史案例 / 近期记忆: thread-old")
    assert update["retrieved_context"]["evidence"] == result["evidence"]


@pytest.mark.asyncio
async def test_compiled_caseflow_graph_persists_summary_when_compiled_with_store():
    writes = []

    class FakeStore:
        async def asearch(self, namespace, query="", limit=3):
            return []

        async def aput(self, namespace, key, value):
            writes.append((namespace, key, value))

    graph = caseflow_graph_builder.compile(store=FakeStore())
    graph.name = "caseflow-agent"
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="客户问怎么重置账号密码")]},
        config={
            "configurable": {
                "thread_id": "thread-compiled-store",
                "user_id": "cust-001",
                "model": FakeModelName.FAKE,
            }
        },
    )

    custom_data = result["messages"][-1].additional_kwargs["custom_data"]
    assert custom_data["memory_result"]["status"] == "saved"
    assert writes[0][0] == ("caseflow", "case_summaries", "cust-001")
    assert writes[0][1] == "thread-compiled-store"


@pytest.mark.asyncio
async def test_plan_actions_merges_valid_llm_analysis_without_retrieval_calls(monkeypatch):
    async def valid_llm_analysis(*args, **kwargs):
        return CaseAnalysis(
            intent="投诉类",
            priority="high",
            needs_human_approval=True,
            reasoning="客户明确表达投诉并要求升级处理。",
            missing_information=[],
        )

    monkeypatch.setattr("agents.caseflow_agent.call_llm_case_analysis", valid_llm_analysis)
    monkeypatch.setattr("agents.caseflow_agent.get_customer_profile", _forbid_unexpected_node_call)
    monkeypatch.setattr("agents.caseflow_agent.search_kb", _forbid_unexpected_node_call)
    monkeypatch.setattr("agents.caseflow_agent.search_case_history", _forbid_unexpected_node_call)

    update = await plan_actions(
        _retrieved_caseflow_state("客户投诉服务体验差，要求主管升级"),
        RunnableConfig(configurable={"thread_id": "thread-node", "user_id": "cust-002"}),
    )
    result = update["caseflow_result"]

    assert result["intent"] == "投诉类"
    assert result["priority"] == "high"
    assert result["needs_human_approval"] is True
    assert result["proposed_action_plan"]
    assert result["next_step"] == "await_human_approval"
    assert result["execution_result"]["status"] == "pending_approval"
    assert result["model_used"] == "openai-compatible"
    assert result["llm_reasoning"]["analysis"] == "客户明确表达投诉并要求升级处理。"
    assert CaseAnalysis.model_validate(update["analysis"]).intent == result["intent"]
    assert update["typed_result"]["proposed_action_plan"] == result["proposed_action_plan"]


@pytest.mark.asyncio
async def test_plan_actions_falls_back_to_deterministic_plan_for_invalid_llm(monkeypatch):
    async def invalid_llm_analysis(*args, **kwargs):
        return {"intent": "unknown", "priority": "urgent"}

    monkeypatch.setattr("agents.caseflow_agent.call_llm_case_analysis", invalid_llm_analysis)
    monkeypatch.setattr("agents.caseflow_agent.get_customer_profile", _forbid_unexpected_node_call)
    monkeypatch.setattr("agents.caseflow_agent.search_kb", _forbid_unexpected_node_call)
    monkeypatch.setattr("agents.caseflow_agent.search_case_history", _forbid_unexpected_node_call)

    update = await plan_actions(
        _retrieved_caseflow_state(),
        RunnableConfig(configurable={"thread_id": "thread-node", "user_id": "cust-002"}),
    )
    result = update["caseflow_result"]

    assert result["intent"] == "退款 / 升级类"
    assert result["priority"] == "high"
    assert result["needs_human_approval"] is True
    assert any("退款" in step for step in result["proposed_action_plan"])
    assert result["next_step"] == "await_human_approval"
    assert result["execution_result"]["status"] == "pending_approval"
    assert result["model_used"] == "openai-compatible:fallback"
    assert result["llm_reasoning"]["analysis"] is None
    assert CaseAnalysis.model_validate(update["analysis"]).intent == "退款 / 升级类"


@pytest.mark.asyncio
async def test_plan_actions_handles_partial_standalone_state_without_keyerror(monkeypatch):
    async def unavailable_llm_analysis(*args, **kwargs):
        return None

    monkeypatch.setattr("agents.caseflow_agent.call_llm_case_analysis", unavailable_llm_analysis)
    monkeypatch.setattr("agents.caseflow_agent.get_customer_profile", _forbid_unexpected_node_call)
    monkeypatch.setattr("agents.caseflow_agent.search_kb", _forbid_unexpected_node_call)
    monkeypatch.setattr("agents.caseflow_agent.search_case_history", _forbid_unexpected_node_call)

    update = await plan_actions(
        {
            "messages": [HumanMessage(content="客户问怎么重置账号密码")],
            "caseflow_result": {"model_used": "openai-compatible"},
        },
        RunnableConfig(configurable={"thread_id": "thread-node", "user_id": "cust-001"}),
    )
    result = update["caseflow_result"]

    assert result["intent"] == "咨询类"
    assert result["priority"] == "low"
    assert result["proposed_action_plan"]
    assert result["execution_result"]["status"] == "ready_to_send"
    assert result["model_used"] == "openai-compatible:fallback"
    assert CaseAnalysis.model_validate(update["analysis"]).intent == "咨询类"


@pytest.mark.asyncio
async def test_draft_resolution_merges_valid_llm_draft_without_overwriting_plan(monkeypatch):
    async def valid_llm_resolution(*args, **kwargs):
        return DraftResolution(
            proposed_action_plan=["LLM tried to replace the planning node output."],
            draft_response="您好，我会先整理重复扣费证据，并提交人工审批后继续处理。",
            next_step="send_draft_response",
            reasoning="根据证据生成面向客户的回复草稿。",
        )

    state = _retrieved_caseflow_state()
    state["caseflow_result"].update(
        {
            "intent": "退款 / 升级类",
            "priority": "high",
            "needs_human_approval": True,
            "proposed_action_plan": ["Planning node owns this action plan."],
            "next_step": "await_human_approval",
            "execution_result": {"status": "pending_approval"},
            "llm_reasoning": {"analysis": "Planning node reasoning."},
        }
    )

    monkeypatch.setattr("agents.caseflow_agent.call_llm_resolution", valid_llm_resolution)
    monkeypatch.setattr("agents.caseflow_agent.get_customer_profile", _forbid_unexpected_node_call)
    monkeypatch.setattr("agents.caseflow_agent.search_kb", _forbid_unexpected_node_call)
    monkeypatch.setattr("agents.caseflow_agent.search_case_history", _forbid_unexpected_node_call)
    monkeypatch.setattr("agents.caseflow_agent.apply_policy_guardrails", _forbid_unexpected_node_call)

    update = await draft_resolution(
        state,
        RunnableConfig(configurable={"thread_id": "thread-node", "user_id": "cust-002"}),
    )
    result = update["caseflow_result"]

    assert result["proposed_action_plan"] == ["Planning node owns this action plan."]
    assert result["draft_response"] == "您好，我会先整理重复扣费证据，并提交人工审批后继续处理。"
    assert result["next_step"] == "await_human_approval"
    assert result["execution_result"]["status"] == "pending_approval"
    assert result["llm_reasoning"]["analysis"] == "Planning node reasoning."
    assert result["llm_reasoning"]["resolution"] == "根据证据生成面向客户的回复草稿。"
    assert DraftResolution.model_validate(update["draft"]).draft_response == result["draft_response"]
    assert update["typed_result"]["draft_response"] == result["draft_response"]


@pytest.mark.asyncio
async def test_draft_resolution_falls_back_without_retrieval_or_policy_guardrails(monkeypatch):
    async def unavailable_llm_resolution(*args, **kwargs):
        return None

    state = _retrieved_caseflow_state("客户问怎么重置账号密码", "cust-001", "standard")
    state["caseflow_result"].update(
        {
            "intent": "咨询类",
            "priority": "low",
            "needs_human_approval": False,
            "proposed_action_plan": ["Planning node owns the FAQ plan."],
            "next_step": "send_draft_response",
            "execution_result": {"status": "ready_to_send"},
            "llm_reasoning": {"analysis": None},
        }
    )

    monkeypatch.setattr("agents.caseflow_agent.call_llm_resolution", unavailable_llm_resolution)
    monkeypatch.setattr("agents.caseflow_agent.get_customer_profile", _forbid_unexpected_node_call)
    monkeypatch.setattr("agents.caseflow_agent.search_kb", _forbid_unexpected_node_call)
    monkeypatch.setattr("agents.caseflow_agent.search_case_history", _forbid_unexpected_node_call)
    monkeypatch.setattr("agents.caseflow_agent.apply_policy_guardrails", _forbid_unexpected_node_call)

    update = await draft_resolution(
        state,
        RunnableConfig(configurable={"thread_id": "thread-node", "user_id": "cust-001"}),
    )
    result = update["caseflow_result"]

    assert result["proposed_action_plan"] == ["Planning node owns the FAQ plan."]
    assert "重置密码" in result["draft_response"]
    assert result["next_step"] == "send_draft_response"
    assert result["execution_result"]["status"] == "ready_to_send"
    assert result["model_used"] == "openai-compatible:fallback"
    assert result["llm_reasoning"]["resolution"] is None
    assert DraftResolution.model_validate(update["draft"]).draft_response == result["draft_response"]


@pytest.mark.asyncio
async def test_downstream_nodes_handle_partial_standalone_state_without_keyerror(monkeypatch):
    async def unavailable_llm_resolution(*args, **kwargs):
        return None

    async def unavailable_llm_reflection(*args, **kwargs):
        return None

    monkeypatch.setattr("agents.caseflow_agent.call_llm_resolution", unavailable_llm_resolution)
    monkeypatch.setattr("agents.caseflow_agent.call_llm_reflection", unavailable_llm_reflection)

    state = {
        "messages": [HumanMessage(content="客户被重复扣费，要求马上退款")],
        "caseflow_result": {
            "user_query": "客户被重复扣费，要求马上退款",
            "thread_id": "thread-partial",
            "user_id": "cust-002",
            "model_used": "openai-compatible",
            "execution_result": {"status": "ready_to_send"},
        },
    }

    state.update(
        await draft_resolution(
            state,
            RunnableConfig(configurable={"thread_id": "thread-partial", "user_id": "cust-002"}),
        )
    )
    state.update(
        await reflect_and_risk_check(
            state,
            RunnableConfig(configurable={"thread_id": "thread-partial", "user_id": "cust-002"}),
        )
    )
    state.update(await execute_action(state))
    state.update(await finalize_and_persist(state))

    custom_data = state["messages"][-1].additional_kwargs["custom_data"]
    assert custom_data["intent"] == "退款 / 升级类"
    assert custom_data["priority"] == "high"
    assert custom_data["draft_response"]
    assert custom_data["case_note"]["status"] == "saved"
    CaseFlowResult.model_validate(custom_data)
    _assert_no_pydantic_models(custom_data)


@pytest.mark.asyncio
async def test_execute_action_does_not_create_ticket_for_modification_request(monkeypatch):
    calls = []

    def fake_create_ticket(*args, **kwargs):
        calls.append(("ticket", args, kwargs))
        return {"ticket_id": "TCK-SHOULD-NOT-HAPPEN", "status": "created"}

    def fake_escalate_ticket(*args, **kwargs):
        calls.append(("escalation", args, kwargs))
        return {"ticket_id": "TCK-SHOULD-NOT-HAPPEN", "status": "escalated"}

    monkeypatch.setattr("agents.caseflow_agent.create_ticket", fake_create_ticket)
    monkeypatch.setattr("agents.caseflow_agent.escalate_ticket", fake_escalate_ticket)

    update = await execute_action(
        {
            "messages": [HumanMessage(content="客户要求退款")],
            "user_query": "客户要求退款",
            "thread_id": "thread-modify",
            "user_id": "cust-002",
            "caseflow_result": {
                **analyze_case("客户要求退款", "thread-modify", "cust-002"),
                "execution_result": {
                    "status": "modification_requested",
                    "reason": "证据不足",
                    "modification_notes": "先补充退款政策依据。",
                },
                "next_step": "approval_modification_requested",
            },
        }
    )

    assert calls == []
    result = update["caseflow_result"]
    assert result["execution_result"]["status"] == "modification_requested"
    assert result["case_note"]["status"] == "saved"


@pytest.mark.asyncio
async def test_retrieval_tools_are_only_called_by_retrieve_evidence(monkeypatch):
    calls = []

    def fake_customer_profile(user_id):
        calls.append(("profile", user_id))
        return {"customer_id": user_id, "name": "Test Customer", "tier": "enterprise"}

    def fake_kb(query):
        calls.append(("kb", query))
        return {"source": "policy", "title": "退款审批政策", "summary": "退款必须审批。"}

    def fake_history(query):
        calls.append(("history", query))
        return {"case_id": "CASE-X", "summary": "历史案例：重复扣费后提交审批。"}

    async def no_llm_analysis(*args, **kwargs):
        return None

    async def no_llm_resolution(*args, **kwargs):
        return None

    async def no_llm_reflection(*args, **kwargs):
        return None

    monkeypatch.setattr("agents.caseflow_agent.get_customer_profile", fake_customer_profile)
    monkeypatch.setattr("agents.caseflow_agent.search_kb", fake_kb)
    monkeypatch.setattr("agents.caseflow_agent.search_case_history", fake_history)
    monkeypatch.setattr("agents.caseflow_agent.call_llm_case_analysis", no_llm_analysis)
    monkeypatch.setattr("agents.caseflow_agent.call_llm_resolution", no_llm_resolution)
    monkeypatch.setattr("agents.caseflow_agent.call_llm_reflection", no_llm_reflection)

    config = RunnableConfig(configurable={"thread_id": "thread-tools", "user_id": "cust-002"})
    state = {
        "messages": [HumanMessage(content="客户被重复扣费，要求马上退款")],
        "user_query": "客户被重复扣费，要求马上退款",
        "thread_id": "thread-tools",
        "user_id": "cust-002",
        "caseflow_result": {
            "user_query": "客户被重复扣费，要求马上退款",
            "thread_id": "thread-tools",
            "user_id": "cust-002",
        },
    }

    state.update(await retrieve_evidence(state, config))
    assert [name for name, _ in calls] == ["profile", "kb", "history"]

    calls.clear()
    state.update(await plan_actions(state, config))
    assert "analysis" in state
    state.update(await draft_resolution(state, config))
    assert "draft" in state
    state.update(await reflect_and_risk_check(state, config))
    assert RiskReflection.model_validate(state["reflection"]).risk == "high"
    assert CaseFlowResult.model_validate(state["typed_result"]).needs_human_approval is True

    assert calls == []


@pytest.mark.asyncio
async def test_finalize_returns_plain_dict_custom_data_matching_typed_result():
    writes = []

    class FakeStore:
        async def aput(self, namespace, key, value):
            writes.append((namespace, key, value))

    typed_state = {
        "messages": [HumanMessage(content="客户问怎么重置账号密码")],
        "user_query": "客户问怎么重置账号密码",
        "thread_id": "thread-final",
        "user_id": "cust-001",
        "caseflow_result": analyze_case("客户问怎么重置账号密码", "thread-final", "cust-001"),
    }

    update = await finalize_and_persist(typed_state, store=FakeStore())
    custom_data = update["messages"][-1].additional_kwargs["custom_data"]

    _assert_no_pydantic_models(custom_data)
    assert isinstance(custom_data, dict)
    assert custom_data == update["typed_result"]
    assert CaseFlowResult.model_validate(custom_data).intent == "咨询类"
    assert custom_data["memory_result"]["status"] == "saved"
    assert writes[0][0] == ("caseflow", "case_summaries", "cust-001")


def test_customer_profile_defaults_preferences_for_partial_profile():
    profile = CustomerProfile.model_validate(
        {"customer_id": "cust-x", "name": "Partial", "tier": "standard"}
    )

    assert profile.preferences == []
