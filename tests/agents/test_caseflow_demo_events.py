from agents.caseflow_agent import analyze_case
from agents.caseflow_demo import load_demo_cases
from agents.caseflow_events import (
    build_workflow_events,
    risk_flags_from_result,
    workflow_node_definitions,
)


def test_demo_cases_cover_required_portfolio_scenarios():
    cases = load_demo_cases()
    titles = {case.title for case in cases}

    assert len(cases) == 5
    assert "普通商品咨询 / 售后政策咨询" in titles
    assert "退款申请" in titles
    assert "高金额补偿诉求" in titles
    assert "投诉升级" in titles
    assert "信息不全" in titles
    assert all(case.user_query and case.expected_key_nodes for case in cases)


def test_workflow_node_definitions_match_portfolio_graph_contract():
    nodes = workflow_node_definitions()
    labels = [node["node_label"] for node in nodes]

    assert labels == [
        "Intent Recognition",
        "Evidence Retrieval",
        "Action Planning",
        "Draft Response",
        "Risk Review",
        "Human Approval",
        "Ticket Execution",
        "Result Persistence",
    ]
    assert any(node["is_risk_control"] for node in nodes if node["node_id"] == "risk_review")


def test_workflow_events_mark_refund_as_guarded_before_approval():
    result = analyze_case("客户被重复扣费，要求马上退款", "thread-events", "cust-002")
    events = build_workflow_events(result, run_id="run-1", case_id="demo-refund-request")
    by_node = {event["node_id"]: event for event in events}

    assert by_node["risk_review"]["status"] == "blocked"
    assert by_node["human_approval"]["status"] == "running"
    assert by_node["ticket_execution"]["status"] == "pending"
    assert by_node["risk_review"]["approval_required"] is True
    assert by_node["risk_review"]["risk_flags"]


def test_policy_consultation_demo_case_stays_low_risk():
    result = analyze_case(
        "我想知道 7 天无理由退换货政策，拆封后还能退吗？",
        "thread-policy-consult",
        "cust-001",
    )

    assert result["intent"] == "咨询类"
    assert result["priority"] == "low"
    assert result["needs_human_approval"] is False
    assert "退换货" in result["draft_response"]
    assert "重置密码" not in result["draft_response"]


def test_high_compensation_demo_case_is_classified_as_guarded_financial_action():
    result = analyze_case(
        "你们发错货导致我活动损失，要求补偿 3000 元并马上处理。",
        "thread-compensation",
        "cust-002",
    )

    assert result["intent"] == "退款 / 升级类"
    assert result["needs_human_approval"] is True
    assert any(flag.type == "compensation" for flag in risk_flags_from_result(result))


def test_risk_flags_are_business_trace_not_hidden_reasoning():
    result = analyze_case("我要投诉并要求主管升级处理", "thread-risk", "cust-002")
    flags = risk_flags_from_result(result)
    flag_types = {flag.type for flag in flags}

    assert {"complaint", "escalation"} <= flag_types
    assert all(flag.reason for flag in flags)
