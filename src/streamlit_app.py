import asyncio
import html
import json
import os
import urllib.parse
import uuid
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from agents.caseflow_demo import get_demo_case, load_demo_cases
from agents.caseflow_events import workflow_node_definitions
from client import AgentClient, AgentClientError
from schema import ChatHistory, ChatMessage
from schema.task_data import TaskData, TaskDataStatus
from voice import VoiceManager

# A Streamlit app for interacting with the langgraph agent via a simple chat interface.
# The app has three main functions which are all run async:

# - main() - sets up the streamlit app and high level structure
# - draw_messages() - draws a set of chat messages - either replaying existing messages
#   or streaming new ones.
# - handle_feedback() - Draws a feedback widget and records feedback from the user.

# The app heavily uses AgentClient to interact with the agent's FastAPI endpoints.


APP_TITLE = "CaseFlow Agent"
APP_ICON = "🧭"
USER_ID_COOKIE = "user_id"
CASEFLOW_SOURCE_URL = "https://github.com/CypherHK/Caseflow"
CASEFLOW_AUTHOR = "Yucheng"
CASEFLOW_SIDEBAR_DESCRIPTION = (
    "面向客服工单处理的 AI 工作台：分类、检索依据、生成回复、复核风险，并在高风险动作前等待人工审批。"
)
CASEFLOW_NEW_CASE_LABEL = ":material/add: 新建工单"
CASEFLOW_SETTINGS_LABEL = ":material/tune: 演示设置"
CASEFLOW_PRIVACY_LABEL = ":material/policy: 隐私与反馈"
CASEFLOW_SHARE_LABEL = ":material/upload: 分享/恢复工单"
CASEFLOW_SHARE_DIALOG_TITLE = "分享/恢复工单"
CASEFLOW_SOURCE_LINK_LABEL = "CaseFlow 项目源码"
CASEFLOW_AUTHOR_CAPTION = f"CaseFlow 原型：{CASEFLOW_AUTHOR}"
CASEFLOW_CONNECTING_MESSAGE = "正在连接 CaseFlow 服务..."
CASEFLOW_CONNECTION_ERROR_PREFIX = "连接 CaseFlow 服务失败"
CASEFLOW_SERVICE_BOOT_MESSAGE = "服务可能仍在启动，请稍后重试。"
CASEFLOW_AGENT_SELECT_LABEL = "Agent（调试）"
CASEFLOW_WELCOME_MESSAGE = (
    "你好，我是 CaseFlow 客服工作台。请描述客户问题，我会整理意图、依据、处理计划、回复草稿和审批状态。"
)
CASEFLOW_CHAT_PLACEHOLDER = "输入客户问题，例如：客户要求退款、投诉处理慢，或询问订单状态"
CASEFLOW_APPROVE_BUTTON_LABEL = "批准执行"
CASEFLOW_REJECT_BUTTON_LABEL = "拒绝执行"
CASEFLOW_MODIFY_BUTTON_LABEL = "要求修改"
CASEFLOW_APPROVAL_INTERRUPT_DISPLAY = (
    "该工单触发高风险审批。请主管选择批准执行、拒绝执行，或要求修改处理方案。"
)
CASEFLOW_APPROVAL_DISPLAY_TEXT = {
    "approve": "已选择：批准执行",
    "reject": "已选择：拒绝执行",
    "modify": "已选择：要求修改",
}
CASEFLOW_PORTFOLIO_TITLE = "可审计的电商售后工作流 Agent"
CASEFLOW_DEMO_RUN_LABEL = "运行"
CASEFLOW_DEMO_RESET_LABEL = "重置"
CASEFLOW_DEMO_REPLAY_LABEL = "重放"
CASEFLOW_DEMO_GUIDE_LABEL = "演示说明"
CASEFLOW_GUIDE_BACK_LABEL = "返回工作台"
CASEFLOW_DEMO_START_LABEL = "开始演示"
CASEFLOW_DEMO_INTENT_LABELS = {
    "demo-faq-policy": "咨询类",
    "demo-refund-request": "退款 / 升级类",
    "demo-high-compensation": "退款 / 升级类",
    "demo-complaint-escalation": "投诉类",
    "demo-missing-information": "信息不全待补充类",
}
CASEFLOW_WORKFLOW_EVENT_TYPE = "workflow_event"
CASEFLOW_LEFT_CASE_HEIGHT = 142
CASEFLOW_LEFT_INPUT_HEIGHT = 158
CASEFLOW_LEFT_CONTEXT_HEIGHT = 300
CASEFLOW_GRAPH_HEIGHT = 386
CASEFLOW_TIMELINE_HEIGHT = 224
CASEFLOW_INSPECTOR_HEIGHT = 330
CASEFLOW_APPROVAL_BAR_HEIGHT = 164
CASEFLOW_RESOLUTION_HEIGHT = 132
CASEFLOW_RUN_STATUS_LABELS = {
    "ready": "待运行",
    "running": "运行中",
    "awaiting_approval": "等待审批",
    "completed": "已完成",
    "blocked": "已阻断",
    "failed": "失败",
}
CASEFLOW_NODE_LABELS_ZH = {
    "intent_recognition": "意图识别",
    "evidence_retrieval": "证据检索",
    "action_planning": "行动规划",
    "draft_response": "回复草稿",
    "risk_review": "风险复核",
    "human_approval": "人工审批",
    "ticket_execution": "工单执行",
    "result_persistence": "结果持久化",
}
CASEFLOW_NODE_GOALS_ZH = {
    "intent_recognition": "捕获客户问题和运行身份，确定进入工作流的业务入口。",
    "evidence_retrieval": "检索本地政策、SOP、订单上下文和客户历史证据。",
    "action_planning": "判断意图和优先级，决定是否需要审批，并生成行动计划。",
    "draft_response": "基于证据生成面向客户的回复草稿。",
    "risk_review": "复核证据是否充分，并执行退款、补偿、投诉、升级等确定性风险规则。",
    "human_approval": "高风险动作暂停执行，等待人工批准、拒绝或要求修改。",
    "ticket_execution": "在策略允许且审批通过后，创建或升级模拟工单。",
    "result_persistence": "保存可审计结果数据和紧凑案例摘要。",
}
CASEFLOW_EVIDENCE_SOURCE_LABELS = {
    "policy": "政策",
    "sop": "SOP",
    "faq": "FAQ",
    "kb": "知识库",
    "customer": "客户资料",
    "customer_history": "历史案例",
    "case_history": "历史案例",
    "case_memory": "近期记忆",
    "order": "订单",
    "evidence": "证据",
}
CASEFLOW_RISK_TYPE_LABELS = {
    "refund": "退款",
    "compensation": "补偿",
    "complaint": "投诉",
    "escalation": "升级",
    "policy_guardrail": "策略规则",
    "human_in_the_loop": "人工审批",
}
CASEFLOW_RISK_SEVERITY_LABELS = {
    "low": "低",
    "medium": "中",
    "high": "高",
}
CASEFLOW_COPY_SURFACE = (
    APP_TITLE,
    CASEFLOW_SOURCE_URL,
    CASEFLOW_AUTHOR,
    CASEFLOW_SIDEBAR_DESCRIPTION,
    CASEFLOW_NEW_CASE_LABEL,
    CASEFLOW_SETTINGS_LABEL,
    CASEFLOW_PRIVACY_LABEL,
    CASEFLOW_SHARE_LABEL,
    CASEFLOW_SHARE_DIALOG_TITLE,
    CASEFLOW_SOURCE_LINK_LABEL,
    CASEFLOW_AUTHOR_CAPTION,
    CASEFLOW_CONNECTING_MESSAGE,
    CASEFLOW_CONNECTION_ERROR_PREFIX,
    CASEFLOW_SERVICE_BOOT_MESSAGE,
    CASEFLOW_AGENT_SELECT_LABEL,
    CASEFLOW_WELCOME_MESSAGE,
    CASEFLOW_CHAT_PLACEHOLDER,
    CASEFLOW_APPROVE_BUTTON_LABEL,
    CASEFLOW_REJECT_BUTTON_LABEL,
    CASEFLOW_MODIFY_BUTTON_LABEL,
    CASEFLOW_APPROVAL_INTERRUPT_DISPLAY,
    CASEFLOW_PORTFOLIO_TITLE,
    CASEFLOW_DEMO_RUN_LABEL,
    CASEFLOW_DEMO_RESET_LABEL,
    CASEFLOW_DEMO_REPLAY_LABEL,
    CASEFLOW_DEMO_GUIDE_LABEL,
    CASEFLOW_GUIDE_BACK_LABEL,
    CASEFLOW_DEMO_START_LABEL,
    *CASEFLOW_APPROVAL_DISPLAY_TEXT.values(),
)
CASEFLOW_APPROVAL_PENDING_STATUSES = {
    "pending_approval",
    "await_human_approval",
    "awaiting_human_approval",
}
CASEFLOW_REQUIRED_KEYS = {
    "intent",
    "priority",
    "evidence",
    "proposed_action_plan",
    "draft_response",
    "needs_human_approval",
    "next_step",
    "execution_result",
}
CASEFLOW_STATUS_LABELS = {
    "executed": "已执行",
    "ready_to_send": "待发送",
    "await_human_approval": "等待人工审批",
    "awaiting_human_approval": "等待人工审批",
    "pending_approval": "等待人工审批",
    "approved": "已批准",
    "modification_requested": "要求修改",
    "rejected": "已拒绝",
    "cancelled": "已取消",
    "open": "已创建",
    "created": "已创建",
    "escalated": "已升级",
    "human_review": "人工复核",
    "unknown": "未知",
}
CASEFLOW_QUEUE_LABELS = {
    "human_review": "人工复核队列",
    "supervisor_review": "主管复核队列",
    "billing_review": "计费复核队列",
}
CASEFLOW_EVIDENCE_GROUPS = {
    "kb": "知识库 / FAQ / SOP / Policy",
    "customer": "客户资料",
    "history": "历史案例",
    "other": "其他依据",
}
CASEFLOW_KB_SOURCE_KEYS = {"faq", "sop", "policy", "kb", "knowledge_base"}
CASEFLOW_HISTORY_SOURCE_KEYS = {"history", "case", "case_history"}
CASEFLOW_CUSTOMER_SOURCE_KEYS = {"customer", "customer_profile", "profile"}
CASEFLOW_EVIDENCE_EMPTY_MESSAGE = "暂无可用依据。"


def is_caseflow_result(custom_data: dict) -> bool:
    return CASEFLOW_REQUIRED_KEYS.issubset(custom_data.keys())


def is_caseflow_workflow_event(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("type") == CASEFLOW_WORKFLOW_EVENT_TYPE
        and isinstance(value.get("content"), Mapping)
    )


def caseflow_latest_custom_data(messages: list[ChatMessage]) -> dict[str, Any]:
    for message in reversed(messages):
        if message.type == "ai" and is_caseflow_result(message.custom_data):
            return dict(message.custom_data)
    return dict(st.session_state.get("caseflow_last_custom_data") or {})


def caseflow_record_workflow_event(event: Mapping[str, Any]) -> None:
    events = list(st.session_state.get("caseflow_workflow_events") or [])
    events.append(dict(event))
    st.session_state.caseflow_workflow_events = events


def caseflow_store_result_payload(custom_data: dict) -> None:
    st.session_state.caseflow_last_custom_data = dict(custom_data)
    events = custom_data.get("workflow_events")
    if isinstance(events, list) and events:
        st.session_state.caseflow_workflow_events = [
            dict(event) for event in events if isinstance(event, Mapping)
        ]


def caseflow_workflow_events(custom_data: dict | None = None) -> list[dict[str, Any]]:
    if custom_data:
        events = custom_data.get("workflow_events")
        if isinstance(events, list) and events:
            return [dict(event) for event in events if isinstance(event, Mapping)]
    return [dict(event) for event in st.session_state.get("caseflow_workflow_events", [])]


def caseflow_workflow_nodes(custom_data: dict | None = None) -> list[dict[str, Any]]:
    if custom_data:
        nodes = custom_data.get("workflow_nodes")
        if isinstance(nodes, list) and nodes:
            return [dict(node) for node in nodes if isinstance(node, Mapping)]
    return workflow_node_definitions()


def caseflow_status_by_node(events: list[dict[str, Any]]) -> dict[str, str]:
    statuses = {node["node_id"]: "pending" for node in workflow_node_definitions()}
    for event in events:
        node_id = str(event.get("node_id") or "")
        status = str(event.get("status") or "")
        if node_id and status:
            statuses[node_id] = status
    return statuses


def caseflow_latest_event_for_node(
    events: list[dict[str, Any]],
    node_id: str,
) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("node_id") == node_id:
            return event
    return None


def caseflow_latest_active_node_id(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("status") in {"running", "blocked", "failed"}:
            return str(event.get("node_id") or "intent_recognition")
    for event in reversed(events):
        if event.get("status") == "succeeded":
            return str(event.get("node_id") or "intent_recognition")
    return "intent_recognition"


def caseflow_html_escape(value: object) -> str:
    return html.escape(str(value or ""))


def caseflow_workflow_status_label(status: str) -> str:
    return {
        "pending": "未执行",
        "running": "运行中",
        "succeeded": "已完成",
        "blocked": "已阻断",
        "failed": "失败",
        "skipped": "已跳过",
    }.get(status, status or "未执行")


def caseflow_node_label_zh(node_id: object, fallback: object = "") -> str:
    node_key = str(node_id or "")
    return CASEFLOW_NODE_LABELS_ZH.get(node_key, str(fallback or node_key))


def caseflow_node_goal_zh(node_id: object, fallback: object = "") -> str:
    node_key = str(node_id or "")
    return CASEFLOW_NODE_GOALS_ZH.get(node_key, str(fallback or "等待运行。"))


def caseflow_evidence_source_label(value: object) -> str:
    source_key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return CASEFLOW_EVIDENCE_SOURCE_LABELS.get(source_key, str(value or "证据"))


def caseflow_risk_type_label(value: object) -> str:
    risk_key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return CASEFLOW_RISK_TYPE_LABELS.get(risk_key, str(value or "风险"))


def caseflow_risk_severity_label(value: object) -> str:
    severity_key = str(value or "").strip().lower()
    return CASEFLOW_RISK_SEVERITY_LABELS.get(severity_key, str(value or "未知"))


def caseflow_localize_trace_text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    approval_decision_text = {
        "approve": "已批准执行。",
        "reject": "已拒绝执行。",
        "modify": "已要求修改。",
    }.get(text.lower())
    if approval_decision_text:
        return approval_decision_text

    replacements = (
        ("Node execution started.", "节点开始执行。"),
        ("Waiting for node output.", "等待节点输出。"),
        ("Captured issue for workflow. Current intent:", "已进入工作流。当前意图："),
        (", priority:", "，优先级："),
        ("Retrieved ", "已检索到 "),
        (" evidence item(s).", " 条证据。"),
        ("Planned ", "已生成 "),
        (" action step(s).", " 步处理计划。"),
        (" with evidence context", "，已带入证据上下文"),
        ("approval_required=True", "需要审批"),
        ("approval_required=False", "无需审批"),
        ("Guarded execution status:", "受控执行状态："),
        ("Execution status:", "执行状态："),
        ("Approved action:", "审批动作："),
        ("Waiting for approve / reject / modify", "等待人工批准 / 拒绝 / 修改"),
        ("Waiting for human approve / reject / modify.", "等待人工审批：批准 / 拒绝 / 修改。"),
        ("High-risk action is paused by LangGraph interrupt().", "高风险动作已由 LangGraph interrupt() 暂停。"),
        (
            "Graph execution is paused before ticket/refund/escalation execution.",
            "工作流已在工单、退款或升级执行前暂停。",
        ),
        ("Final typed payload", "最终结构化结果"),
        ("Result persisted with execution status:", "结果已持久化，执行状态："),
        ("not_started", "未开始"),
        ("pending_approval", "等待审批"),
        ("ready_to_send", "待发送"),
        ("executed", "已执行"),
        ("unknown", "未知"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    text = text.replace("： ", "：").replace("，优先级： ", "，优先级：")
    for source, target in (("low", "低"), ("medium", "中"), ("high", "高")):
        text = text.replace(f"优先级：{source}", f"优先级：{target}")
    if "当前意图：未知" in text:
        return "客户问题已进入工作流，结构化意图和优先级将在后续节点更新。"
    if "受控执行状态：未知" in text:
        return "受控执行状态等待审批或执行结果更新。"
    if "审批动作：None" in text:
        return "尚未产生审批动作。"
    if text.endswith("."):
        text = f"{text[:-1]}。"
    return text


def caseflow_portfolio_css() -> str:
    return """
    <style>
    :root {
        color-scheme: light;
        --cf-surface: oklch(98% 0.006 230);
        --cf-surface-2: oklch(95.5% 0.008 230);
        --cf-panel: oklch(99% 0.004 230);
        --cf-panel-strong: oklch(93% 0.012 230);
        --cf-text: oklch(22% 0.018 240);
        --cf-muted: oklch(52% 0.02 240);
        --cf-line: oklch(86% 0.012 235);
        --cf-accent: oklch(58% 0.13 225);
        --cf-accent-soft: oklch(93% 0.035 225);
        --cf-success: oklch(55% 0.12 155);
        --cf-success-soft: oklch(93% 0.04 155);
        --cf-warning: oklch(69% 0.14 80);
        --cf-warning-soft: oklch(94% 0.052 80);
        --cf-danger: oklch(56% 0.14 25);
        --cf-danger-soft: oklch(94% 0.042 25);
        --cf-skipped: oklch(68% 0.014 250);
        --cf-shadow: 0 14px 36px oklch(32% 0.025 240 / 9%);
        --cf-radius: 8px;
        --cf-mono: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
        --cf-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    }

    section[data-testid="stSidebar"],
    div[data-testid="collapsedControl"],
    button[kind="header"],
    [data-testid="stToolbar"] {
        display: none !important;
    }

    .stApp {
        background: var(--cf-surface);
        color: var(--cf-text);
        font-family: var(--cf-sans);
        height: 100vh;
        overflow: hidden;
    }

    html,
    body,
    div[data-testid="stAppViewContainer"],
    section.main {
        height: 100vh;
        overflow: hidden;
    }

    .block-container {
        max-width: 1500px;
        height: 100vh;
        max-height: 100vh;
        overflow: hidden;
        padding: 10px 16px 12px !important;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.55rem;
    }

    div[data-testid="stHorizontalBlock"] {
        gap: 0.75rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--cf-line) !important;
        border-radius: var(--cf-radius) !important;
        background: var(--cf-panel) !important;
        box-shadow: var(--cf-shadow);
        overflow: hidden;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        gap: 0.75rem;
    }

    .cf-topbar {
        display: grid;
        grid-template-columns: minmax(260px, 1fr) auto;
        gap: 14px;
        align-items: center;
        padding: 10px 14px;
        border: 1px solid var(--cf-line);
        border-radius: var(--cf-radius);
        background: oklch(97.5% 0.006 230);
        box-shadow: var(--cf-shadow);
        margin-bottom: 6px;
    }

    .cf-brand {
        display: flex;
        gap: 14px;
        align-items: center;
        min-width: 0;
    }

    .cf-mark {
        width: 32px;
        height: 32px;
        border: 1px solid oklch(74% 0.035 225);
        border-radius: 8px;
        display: grid;
        place-items: center;
        color: var(--cf-accent);
        font-weight: 780;
        background: var(--cf-panel);
        box-shadow: inset 0 -10px 18px oklch(90% 0.02 225 / 55%);
    }

    .stApp .cf-title {
        margin: 0;
        font-size: 18px !important;
        line-height: 1.2;
        font-weight: 760;
        letter-spacing: 0;
    }

    .cf-subtitle {
        margin: 3px 0 0;
        color: var(--cf-muted);
        font-size: 13px;
    }

    .cf-top-meta {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }

    .cf-pill {
        border: 1px solid var(--cf-line);
        background: var(--cf-panel);
        border-radius: 999px;
        padding: 5px 9px;
        white-space: nowrap;
        color: var(--cf-muted);
        font-size: 13px;
    }

    .cf-pill strong {
        color: var(--cf-accent);
        font-weight: 760;
    }

    .cf-panel {
        border: 1px solid var(--cf-line);
        border-radius: var(--cf-radius);
        background: var(--cf-panel);
        box-shadow: var(--cf-shadow);
        overflow: hidden;
    }

    .cf-panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 10px 12px;
        border-bottom: 1px solid var(--cf-line);
        background: oklch(97% 0.006 230);
    }

    .stApp .cf-panel-title {
        margin: 0;
        font-size: 13px !important;
        line-height: 1.25 !important;
        font-weight: 740;
        color: oklch(31% 0.02 240);
        letter-spacing: 0;
    }

    .cf-panel-kicker {
        color: var(--cf-muted);
        font-size: 12px;
        white-space: nowrap;
    }

    .cf-panel-body {
        padding: 10px 12px;
    }

    .cf-muted {
        color: var(--cf-muted);
    }

    .cf-business-context {
        display: grid;
        gap: 7px;
        border: 1px solid var(--cf-line);
        border-radius: 8px;
        background: oklch(98.5% 0.005 230);
        padding: 9px;
        line-height: 1.42;
    }

    .cf-context-row {
        display: grid;
        grid-template-columns: 64px 1fr;
        gap: 10px;
    }

    .cf-context-row span:first-child {
        color: var(--cf-muted);
    }

    div[data-testid="stRadio"] label,
    div[data-testid="stTextArea"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stTextInput"] label {
        color: var(--cf-muted) !important;
        font-size: 12px !important;
        font-weight: 650 !important;
    }

    div[data-testid="stRadio"] [role="radiogroup"] {
        gap: 0.15rem;
    }

    div[data-testid="stRadio"] [role="radio"] {
        min-height: 34px;
        align-items: flex-start;
    }

    div[data-testid="stTextArea"] textarea,
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        border-color: var(--cf-line) !important;
        border-radius: 8px !important;
        background: oklch(98.5% 0.005 230) !important;
        color: var(--cf-text) !important;
        box-shadow: none !important;
    }

    .stButton > button {
        width: 100%;
        min-height: 34px !important;
        border-radius: 7px;
        border: 1px solid var(--cf-line);
        background: var(--cf-panel);
        color: var(--cf-text);
        padding: 6px 10px !important;
        white-space: nowrap;
        font-weight: 650;
        transition: background 160ms ease-out, border-color 160ms ease-out, transform 160ms ease-out;
    }

    .stButton > button p {
        font-size: 14px !important;
        line-height: 1.1 !important;
    }

    .stButton > button:hover {
        border-color: oklch(70% 0.035 230);
        background: oklch(97% 0.008 230);
    }

    .stButton > button:active {
        transform: translateY(1px);
    }

    .cf-graph-wrap {
        min-height: 292px;
        display: grid;
        grid-template-rows: 1fr auto;
        gap: 10px;
    }

    .cf-graph {
        position: relative;
        display: grid;
        grid-template-columns: repeat(4, minmax(118px, 1fr));
        gap: 18px 20px;
        align-items: stretch;
    }

    .cf-graph::before,
    .cf-graph::after {
        content: "";
        position: absolute;
        left: 9%;
        right: 9%;
        height: 1px;
        background: var(--cf-line);
        z-index: 0;
    }

    .cf-graph::before {
        top: 47px;
    }

    .cf-graph::after {
        bottom: 102px;
    }

    .cf-node {
        position: relative;
        z-index: 1;
        min-height: 112px;
        border: 1px solid var(--cf-line);
        border-radius: 8px;
        background: var(--cf-panel);
        padding: 10px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        gap: 7px;
        transition: transform 160ms ease-out, border-color 160ms ease-out, background 160ms ease-out;
    }

    .cf-node-active {
        transform: translateY(-1px);
        border-color: var(--cf-accent);
    }

    .cf-node-title {
        font-size: 12.5px;
        font-weight: 760;
        line-height: 1.25;
        color: var(--cf-text);
    }

    .cf-node-goal {
        color: var(--cf-muted);
        font-size: 11.5px;
        line-height: 1.32;
    }

    .cf-node-state {
        display: inline-flex;
        width: fit-content;
        align-items: center;
        border-radius: 999px;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 720;
        background: var(--cf-surface-2);
        color: var(--cf-muted);
    }

    .cf-node-running {
        border-color: var(--cf-accent);
        background: var(--cf-accent-soft);
    }

    .cf-node-running .cf-node-state {
        background: var(--cf-accent);
        color: oklch(98% 0.004 230);
    }

    .cf-node-succeeded {
        border-color: oklch(68% 0.075 155);
        background: var(--cf-success-soft);
    }

    .cf-node-succeeded .cf-node-state {
        background: var(--cf-success);
        color: oklch(98% 0.004 230);
    }

    .cf-node-blocked {
        border-color: oklch(73% 0.115 80);
        background: var(--cf-warning-soft);
    }

    .cf-node-blocked .cf-node-state {
        background: var(--cf-warning);
        color: oklch(24% 0.03 80);
    }

    .cf-node-failed {
        border-color: oklch(68% 0.11 25);
        background: var(--cf-danger-soft);
    }

    .cf-node-failed .cf-node-state {
        background: var(--cf-danger);
        color: oklch(98% 0.004 230);
    }

    .cf-node-skipped {
        border-color: oklch(84% 0.01 250);
        background: oklch(97% 0.004 250);
        color: var(--cf-skipped);
    }

    .cf-node-skipped .cf-node-state {
        background: oklch(91% 0.008 250);
        color: var(--cf-skipped);
    }

    .cf-risk-node {
        outline: 2px solid oklch(88% 0.075 80);
        outline-offset: 2px;
    }

    .cf-approval-node {
        outline: 2px solid oklch(86% 0.065 25);
        outline-offset: 2px;
    }

    .cf-path-summary {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        align-items: center;
        padding-top: 2px;
    }

    .cf-node-index {
        position: absolute;
        top: 8px;
        right: 9px;
        color: oklch(68% 0.014 240);
        font-family: var(--cf-mono);
        font-size: 10px;
        font-weight: 760;
    }

    .cf-node-arrow {
        position: absolute;
        z-index: 2;
        display: grid;
        place-items: center;
        width: 20px;
        height: 20px;
        border: 1px solid var(--cf-line);
        border-radius: 999px;
        background: var(--cf-panel);
        color: var(--cf-muted);
        font-size: 13px;
        font-weight: 760;
    }

    .cf-node-arrow-right {
        right: -20px;
        top: 42px;
    }

    .cf-node-arrow-wrap {
        right: -20px;
        bottom: -21px;
        color: var(--cf-accent);
        border-color: oklch(74% 0.04 225);
    }

    .cf-node-succeeded .cf-node-arrow,
    .cf-node-running .cf-node-arrow,
    .cf-node-blocked .cf-node-arrow {
        color: var(--cf-accent);
        border-color: oklch(74% 0.04 225);
    }

    .cf-bypass-note {
        border: 1px dashed oklch(80% 0.02 240);
        border-radius: 999px;
        color: var(--cf-muted);
        background: oklch(98% 0.005 230);
        padding: 5px 9px;
        font-size: 12px;
    }

    .cf-path-chip {
        border: 1px solid var(--cf-line);
        background: oklch(98% 0.006 230);
        color: var(--cf-muted);
        border-radius: 999px;
        padding: 5px 9px;
        font-size: 12px;
    }

    .cf-path-chip-active {
        color: var(--cf-accent);
        border-color: oklch(73% 0.05 225);
        background: var(--cf-accent-soft);
    }

    .cf-inspector-body,
    .cf-timeline,
    .cf-resolution {
        display: grid;
        gap: 10px;
    }

    .cf-node-heading {
        display: flex;
        align-items: start;
        justify-content: space-between;
        gap: 12px;
    }

    .cf-node-heading .cf-inspector-title {
        margin: 0;
        font-size: 17px !important;
        line-height: 1.25;
        font-weight: 760;
    }

    .cf-badge {
        border-radius: 999px;
        padding: 4px 8px;
        background: var(--cf-surface-2);
        color: var(--cf-muted);
        font-size: 11px;
        font-weight: 740;
        white-space: nowrap;
    }

    .cf-badge-high {
        background: var(--cf-warning-soft);
        color: oklch(43% 0.08 70);
    }

    .cf-section {
        display: grid;
        gap: 5px;
    }

    .cf-section h3 {
        margin: 0;
        font-size: 12px;
        color: var(--cf-muted);
        font-weight: 760;
        text-transform: uppercase;
        letter-spacing: 0.02em;
    }

    .cf-section p,
    .cf-section li {
        margin: 0;
        line-height: 1.45;
        color: oklch(32% 0.018 240);
    }

    .cf-evidence-list {
        display: grid;
        gap: 8px;
        padding: 0;
        margin: 0;
        list-style: none;
    }

    .cf-evidence-item,
    .cf-risk-item,
    .cf-metric {
        border: 1px solid var(--cf-line);
        border-radius: 8px;
        padding: 9px;
        background: oklch(98% 0.005 230);
    }

    .cf-evidence-card {
        display: grid;
        gap: 4px;
        border: 1px solid var(--cf-line);
        border-radius: 8px;
        padding: 9px;
        background: oklch(98.5% 0.005 230);
    }

    .cf-evidence-card strong {
        color: oklch(31% 0.02 240);
        font-size: 12.5px;
    }

    .cf-evidence-card span {
        color: var(--cf-muted);
        font-size: 11.5px;
        font-weight: 720;
    }

    .cf-risk-item {
        border-color: oklch(80% 0.08 80);
        background: var(--cf-warning-soft);
    }

    .cf-evidence-item strong,
    .cf-risk-item strong,
    .cf-metric span {
        display: block;
        margin-bottom: 3px;
        font-size: 12px;
        color: var(--cf-muted);
    }

    .cf-timeline-row {
        display: grid;
        grid-template-columns: 72px 108px 1fr;
        gap: 10px;
        align-items: start;
        padding: 6px 0;
        border-bottom: 1px solid var(--cf-line);
    }

    .cf-timeline-row:last-child {
        border-bottom: 0;
    }

    .cf-timestamp,
    .cf-event-node {
        color: var(--cf-muted);
        font-size: 12px;
        font-family: var(--cf-mono);
    }

    .cf-resolution-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
    }

    .cf-draft {
        border: 1px solid var(--cf-line);
        border-radius: 8px;
        background: oklch(98% 0.005 230);
        padding: 9px;
        line-height: 1.45;
        max-height: 62px;
        overflow: auto;
    }

    .cf-approval-bar {
        display: grid;
        gap: 8px;
    }

    .cf-approval-lede {
        margin: 0;
        color: oklch(34% 0.018 240);
        line-height: 1.42;
        font-size: 13px;
    }

    .cf-approval-idle {
        color: var(--cf-muted);
        line-height: 1.45;
        font-size: 13px;
    }

    .cf-guide-hero {
        display: grid;
        gap: 12px;
        padding: 22px;
        border: 1px solid var(--cf-line);
        border-radius: var(--cf-radius);
        background: linear-gradient(
            135deg,
            oklch(97.5% 0.008 225),
            oklch(99% 0.004 230)
        );
        box-shadow: var(--cf-shadow);
    }

    .cf-guide-hero h1 {
        margin: 0;
        font-size: 28px !important;
        line-height: 1.2;
        letter-spacing: 0;
        color: var(--cf-text);
    }

    .cf-guide-hero p {
        max-width: 880px;
        margin: 0;
        color: var(--cf-muted);
        line-height: 1.6;
    }

    .cf-guide-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
    }

    .cf-guide-card {
        border: 1px solid var(--cf-line);
        border-radius: 8px;
        background: var(--cf-panel);
        box-shadow: var(--cf-shadow);
        padding: 14px;
        min-height: 166px;
        display: grid;
        gap: 9px;
        align-content: start;
    }

    .cf-guide-card h2 {
        margin: 0;
        font-size: 15px !important;
        line-height: 1.28;
        color: var(--cf-text);
        letter-spacing: 0;
    }

    .cf-guide-card p,
    .cf-guide-card li {
        margin: 0;
        color: oklch(34% 0.018 240);
        line-height: 1.58;
    }

    .cf-guide-card ul {
        margin: 0;
        padding-left: 1.1rem;
        display: grid;
        gap: 5px;
    }

    .cf-guide-wide {
        grid-column: span 2;
    }

    @media (max-width: 1180px) {
        .cf-graph {
            grid-template-columns: repeat(2, minmax(150px, 1fr));
        }

        .cf-graph::before,
        .cf-graph::after {
            display: none;
        }
    }

    @media (max-width: 760px) {
        .block-container {
            padding: 10px !important;
        }

        .cf-topbar {
            grid-template-columns: 1fr;
        }

        .cf-top-meta {
            justify-content: flex-start;
        }

        .cf-graph {
            grid-template-columns: 1fr;
        }

        .cf-timeline-row,
        .cf-resolution-grid,
        .cf-guide-grid {
            grid-template-columns: 1fr;
        }

        .cf-guide-wide {
            grid-column: span 1;
        }
    }
    </style>
    """


def caseflow_run_status_label(custom_data: dict, events: list[dict[str, Any]]) -> str:
    session_status = st.session_state.get("caseflow_run_status")
    if caseflow_has_pending_approval_state(custom_data, events):
        return CASEFLOW_RUN_STATUS_LABELS["awaiting_approval"]
    if session_status in CASEFLOW_RUN_STATUS_LABELS:
        return CASEFLOW_RUN_STATUS_LABELS[str(session_status)]

    latest_status = str(events[-1].get("status") or "") if events else ""
    latest_node = str(events[-1].get("node_id") or "") if events else ""
    if latest_status == "failed":
        return CASEFLOW_RUN_STATUS_LABELS["failed"]
    if latest_status == "blocked":
        return CASEFLOW_RUN_STATUS_LABELS["blocked"]
    if latest_node == "human_approval" and latest_status == "running":
        return CASEFLOW_RUN_STATUS_LABELS["awaiting_approval"]
    if latest_status == "running":
        return CASEFLOW_RUN_STATUS_LABELS["running"]
    if latest_node == "result_persistence" and latest_status == "succeeded":
        return CASEFLOW_RUN_STATUS_LABELS["completed"]
    return CASEFLOW_RUN_STATUS_LABELS["ready"]


def caseflow_event_path(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path = []
    seen: set[str] = set()
    for event in events:
        node_id = str(event.get("node_id") or "")
        status = str(event.get("status") or "")
        if not node_id or node_id in seen or status not in {"succeeded", "running", "blocked"}:
            continue
        path.append(event)
        seen.add(node_id)
    return path


def caseflow_workflow_graph_html(
    nodes: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> str:
    statuses = caseflow_status_by_node(events)
    active_node_id = caseflow_latest_active_node_id(events)
    cards = []
    for index, node in enumerate(nodes):
        node_id = str(node.get("node_id") or "")
        status = statuses.get(node_id, "pending")
        risk_class = " cf-risk-node" if node.get("is_risk_control") else ""
        approval_class = " cf-approval-node" if node_id == "human_approval" else ""
        active_class = " cf-node-active" if node_id == active_node_id else ""
        node_label = caseflow_node_label_zh(node_id, node.get("node_label"))
        node_goal = caseflow_node_goal_zh(node_id, node.get("goal"))
        arrow = ""
        if index < len(nodes) - 1:
            arrow_class = "cf-node-arrow-wrap" if index == 3 else "cf-node-arrow-right"
            arrow_symbol = "↴" if index == 3 else "→"
            arrow = f"<div class='cf-node-arrow {arrow_class}'>{arrow_symbol}</div>"
        cards.append(
            "<div class='cf-node "
            f"cf-node-{caseflow_html_escape(status)}{risk_class}{approval_class}{active_class}'>"
            f"<div class='cf-node-index'>{index + 1:02d}</div>"
            f"<div class='cf-node-title'>{caseflow_html_escape(node_label)}</div>"
            f"<div class='cf-node-goal'>{caseflow_html_escape(node_goal)}</div>"
            f"<div class='cf-node-state'>{caseflow_workflow_status_label(status)}</div>"
            f"{arrow}"
            "</div>"
        )
    chips = []
    if statuses.get("human_approval") == "skipped":
        chips.append("<span class='cf-bypass-note'>低风险路径：跳过人工审批</span>")
    for event in caseflow_event_path(events):
        chip_class = "cf-path-chip"
        if str(event.get("node_id") or "") == active_node_id:
            chip_class += " cf-path-chip-active"
        chips.append(
            f"<span class='{chip_class}'>"
            f"{caseflow_html_escape(caseflow_node_label_zh(event.get('node_id'), event.get('node_label')))}"
            "</span>"
        )
    if not chips:
        chips.append("<span class='cf-path-chip'>等待运行后生成事件流</span>")
    return (
        "<div class='cf-graph-wrap'>"
        f"<div class='cf-graph'>{''.join(cards)}</div>"
        f"<div class='cf-path-summary'>{''.join(chips)}</div>"
        "</div>"
    )


def draw_caseflow_workflow_graph(
    nodes: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    st.markdown(caseflow_workflow_graph_html(nodes, events), unsafe_allow_html=True)


def caseflow_event_evidence(event: Mapping[str, Any], custom_data: dict) -> list[Mapping[str, Any]]:
    evidence = event.get("evidence")
    if isinstance(evidence, list) and evidence:
        return [item for item in evidence if isinstance(item, Mapping)]

    if event.get("node_id") != "evidence_retrieval":
        return []

    fallback = []
    for group_label, items in caseflow_group_evidence(custom_data):
        for item in items:
            fallback.append(
                {
                    "source": group_label,
                    "title": "CaseFlow evidence",
                    "snippet": item,
                }
            )
    return fallback


def caseflow_active_demo_case(event: Mapping[str, Any] | None = None) -> Any:
    case_id = ""
    if event:
        case_id = str(event.get("case_id") or "")
    case_id = case_id or str(st.session_state.get("caseflow_active_case_id") or "")
    return get_demo_case(case_id)


def caseflow_audit_evidence_cards(
    event: Mapping[str, Any],
    custom_data: dict,
) -> list[dict[str, str]]:
    evidence = caseflow_event_evidence(event, custom_data)
    if event.get("node_id") != "evidence_retrieval":
        return [
            {
                "source": caseflow_evidence_source_label(item.get("source")),
                "title": str(item.get("title") or "证据"),
                "snippet": str(item.get("snippet") or ""),
            }
            for item in evidence
        ]

    demo_case = caseflow_active_demo_case(event)
    policy_title = "售后处理规则"
    policy_snippet = demo_case.product_summary
    for item in evidence:
        source_key = str(item.get("source") or "").lower()
        if source_key in {"policy", "faq", "sop", "kb"}:
            policy_title = str(item.get("title") or policy_title)
            policy_snippet = str(item.get("snippet") or policy_snippet)
            break

    return [
        {
            "source": "订单记录",
            "title": "订单 / 商品上下文",
            "snippet": demo_case.order_summary,
        },
        {
            "source": "售后政策",
            "title": policy_title,
            "snippet": policy_snippet,
        },
        {
            "source": "客户历史",
            "title": "客户历史 / 风险信号",
            "snippet": demo_case.customer_history_summary,
        },
    ]


def caseflow_inspector_html(
    selected_node: Mapping[str, Any],
    event: Mapping[str, Any],
    custom_data: dict,
) -> str:
    status = str(event.get("status") or "pending")
    evidence = caseflow_audit_evidence_cards(event, custom_data)
    risk_flags = event.get("risk_flags")
    if not isinstance(risk_flags, list):
        risk_flags = []

    evidence_html = ""
    if evidence:
        evidence_items = []
        for item in evidence:
            evidence_items.append(
                "<li class='cf-evidence-card'>"
                f"<span>{caseflow_html_escape(item.get('source', '证据'))}</span>"
                f"<strong>{caseflow_html_escape(item.get('title', ''))}</strong>"
                f"{caseflow_html_escape(item.get('snippet', ''))}"
                "</li>"
            )
        evidence_html = (
            "<div class='cf-section'><h3>证据</h3>"
            f"<ul class='cf-evidence-list'>{''.join(evidence_items)}</ul></div>"
        )

    risk_html = ""
    if risk_flags:
        risk_items = []
        for flag in risk_flags:
            if isinstance(flag, Mapping):
                risk_items.append(
                    "<li class='cf-risk-item'>"
                    f"<strong>{caseflow_html_escape(caseflow_risk_type_label(flag.get('type')))} · "
                    f"{caseflow_html_escape(caseflow_risk_severity_label(flag.get('severity')))}</strong>"
                    f"{caseflow_html_escape(caseflow_localize_trace_text(flag.get('reason')))}"
                    "</li>"
                )
        risk_html = (
            "<div class='cf-section'><h3>风险规则</h3>"
            f"<ul class='cf-evidence-list'>{''.join(risk_items)}</ul></div>"
        )

    error_html = ""
    if event.get("error"):
        error_html = (
            "<div class='cf-section'><h3>错误 / 兜底</h3>"
            f"<p>{caseflow_html_escape(caseflow_localize_trace_text(event.get('error')))}</p></div>"
        )

    selected_node_id = selected_node.get("node_id")
    event_node_id = event.get("node_id") or selected_node_id
    node_label = caseflow_node_label_zh(selected_node_id, selected_node.get("node_label"))
    node_goal = caseflow_node_goal_zh(event_node_id, event.get("goal") or selected_node.get("goal"))
    input_summary = caseflow_localize_trace_text(event.get("input_summary")) or "尚无输入摘要。"
    output_summary = caseflow_localize_trace_text(event.get("output_summary")) or "尚无输出摘要。"

    return (
        "<div class='cf-inspector-body'>"
        "<div class='cf-node-heading'>"
        f"<div class='cf-inspector-title'>{caseflow_html_escape(node_label)}</div>"
        f"<span class='cf-badge cf-badge-high'>{caseflow_workflow_status_label(status)}</span>"
        "</div>"
        "<div class='cf-section'><h3>节点目标</h3>"
        f"<p>{caseflow_html_escape(node_goal)}</p>"
        "</div>"
        "<div class='cf-section'><h3>输入摘要</h3>"
        f"<p>{caseflow_html_escape(input_summary)}</p>"
        "</div>"
        "<div class='cf-section'><h3>输出摘要</h3>"
        f"<p>{caseflow_html_escape(output_summary)}</p>"
        "</div>"
        f"{evidence_html}{risk_html}{error_html}"
        "</div>"
    )


def draw_caseflow_node_inspector(
    nodes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    custom_data: dict,
    *,
    show_approval_controls: bool = True,
) -> None:
    node_options = [str(node["node_id"]) for node in nodes]
    node_labels = {str(node["node_id"]): str(node["node_label"]) for node in nodes}
    default_node_id = caseflow_latest_active_node_id(events)
    default_index = node_options.index(default_node_id) if default_node_id in node_options else 0
    selected_node_id = st.selectbox(
        "当前节点",
        options=node_options,
        index=default_index,
        format_func=lambda value: caseflow_node_label_zh(value, node_labels.get(value, value)),
        key=f"caseflow-node-inspector-select-{default_node_id}-{len(events)}",
    )
    selected_node = next(node for node in nodes if node["node_id"] == selected_node_id)
    event = caseflow_latest_event_for_node(events, selected_node_id) or {}
    status = str(event.get("status") or "pending")

    st.markdown(caseflow_inspector_html(selected_node, event, custom_data), unsafe_allow_html=True)

    if show_approval_controls and selected_node_id == "human_approval":
        if caseflow_is_approval_pending(custom_data) or status in {"running", "blocked"}:
            st.markdown(
                "<div class='cf-section'><h3>审批操作</h3></div>",
                unsafe_allow_html=True,
            )
            draw_caseflow_structured_approval_controls(
                str(custom_data.get("thread_id") or st.session_state.get("thread_id", "caseflow")),
                "portfolio",
            )


def draw_caseflow_portfolio_approval_bar(
    events: list[dict[str, Any]],
    custom_data: dict,
) -> None:
    pending_approval = caseflow_has_pending_approval_state(custom_data, events)
    if not pending_approval:
        st.markdown(
            "<div class='cf-approval-bar'>"
            "<p class='cf-approval-idle'>当前没有待审批动作。低风险咨询会跳过人工审批，"
            "退款、补偿、投诉或主管升级会在这里暂停等待处理。</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    human_event = caseflow_latest_event_for_node(events, "human_approval") or {}
    risk_flags = human_event.get("risk_flags")
    risk_reason = ""
    if isinstance(risk_flags, list) and risk_flags:
        first_flag = risk_flags[0]
        if isinstance(first_flag, Mapping):
            risk_reason = caseflow_localize_trace_text(first_flag.get("reason"))
    risk_reason = risk_reason or "高风险动作需要人工确认后才能进入工单执行。"
    st.markdown(
        "<div class='cf-approval-bar'>"
        "<p class='cf-approval-lede'>"
        f"{caseflow_html_escape(risk_reason)}"
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    draw_caseflow_structured_approval_controls(
        str(custom_data.get("thread_id") or st.session_state.get("thread_id", "caseflow")),
        "portfolio-sticky",
        compact=True,
    )


def draw_caseflow_execution_timeline(events: list[dict[str, Any]]) -> None:
    if not events:
        st.markdown(
            "<div class='cf-timeline'><div class='cf-muted'>还没有运行事件。</div></div>",
            unsafe_allow_html=True,
        )
        return
    rows = []
    display_events = []
    active_node_id = caseflow_latest_active_node_id(events)
    for event in events:
        status = str(event.get("status") or "")
        node_id = str(event.get("node_id") or "")
        if status == "running" and node_id != active_node_id:
            continue
        display_events.append(event)

    for event in display_events[-8:]:
        timestamp = str(event.get("timestamp") or "")
        time_label = timestamp.split("T")[-1][:8] if "T" in timestamp else timestamp[:8]
        detail = caseflow_localize_trace_text(
            event.get("output_summary") or event.get("input_summary") or ""
        )
        if detail == "等待节点输出。":
            detail = "节点正在处理。"
        node_label = caseflow_node_label_zh(event.get("node_id"), event.get("node_label"))
        rows.append(
            "<div class='cf-timeline-row'>"
            f"<span class='cf-timestamp'>{caseflow_html_escape(time_label)}</span>"
            f"<span class='cf-event-node'>{caseflow_html_escape(node_label)}</span>"
            "<span>"
            f"<strong>{caseflow_workflow_status_label(str(event.get('status') or ''))}</strong>"
            f"<br><span class='cf-muted'>{caseflow_html_escape(detail)}</span>"
            "</span></div>"
        )
    st.markdown(f"<div class='cf-timeline'>{''.join(rows)}</div>", unsafe_allow_html=True)


def caseflow_final_resolution_fallback_html(events: list[dict[str, Any]]) -> str:
    statuses = caseflow_status_by_node(events)
    if statuses.get("result_persistence") != "succeeded":
        return ""

    case_event = next((event for event in reversed(events) if event.get("case_id")), None)
    demo_case = caseflow_active_demo_case(case_event)
    intent = CASEFLOW_DEMO_INTENT_LABELS.get(demo_case.case_id, demo_case.category)
    approval_status = statuses.get("human_approval")
    approval_label = "无需审批" if approval_status == "skipped" else "需要审批"
    ticket_status = statuses.get("ticket_execution")
    execution_label = "已执行" if ticket_status == "succeeded" else "待发送"
    ticket_label = "已创建（模拟）" if ticket_status == "succeeded" else "未创建"
    draft_event = caseflow_latest_event_for_node(events, "draft_response") or {}
    draft = (
        caseflow_localize_trace_text(draft_event.get("output_summary"))
        or "已生成面向客户的处理口径，完整结构化结果待消息同步。"
    )

    return (
        "<div class='cf-resolution'>"
        "<div class='cf-resolution-grid'>"
        f"<div class='cf-metric'><span>意图</span><strong>{caseflow_html_escape(intent)}</strong></div>"
        f"<div class='cf-metric'><span>审批</span><strong>{caseflow_html_escape(approval_label)}</strong></div>"
        f"<div class='cf-metric'><span>执行</span><strong>{caseflow_html_escape(execution_label)}</strong></div>"
        f"<div class='cf-metric'><span>工单</span><strong>{caseflow_html_escape(ticket_label)}</strong></div>"
        "</div>"
        "<div class='cf-section'><h3>回复草稿</h3>"
        f"<div class='cf-draft'>{caseflow_html_escape(draft)}</div></div>"
        "</div>"
    )


def draw_caseflow_final_resolution(custom_data: dict, events: list[dict[str, Any]] | None = None) -> None:
    if not is_caseflow_result(custom_data):
        fallback_html = caseflow_final_resolution_fallback_html(events or [])
        if fallback_html:
            st.markdown(fallback_html, unsafe_allow_html=True)
            return
        st.markdown(
            "<div class='cf-resolution'><div class='cf-muted'>"
            "运行完成后展示最终意图、审批状态、工单结果和持久化记录。"
            "</div></div>",
            unsafe_allow_html=True,
        )
        return
    execution = normalize_caseflow_execution(custom_data)
    ticket = execution.get("ticket") if isinstance(execution, Mapping) else None
    ticket_label = str(ticket.get("ticket_id")) if isinstance(ticket, Mapping) else "未创建"
    approval_label = "需要审批" if custom_data.get("needs_human_approval") else "无需审批"
    draft = str(custom_data.get("draft_response") or "")
    st.markdown(
        "<div class='cf-resolution'>"
        "<div class='cf-resolution-grid'>"
        f"<div class='cf-metric'><span>意图</span><strong>{caseflow_html_escape(custom_data.get('intent', 'unknown'))}</strong></div>"
        f"<div class='cf-metric'><span>审批</span><strong>{caseflow_html_escape(approval_label)}</strong></div>"
        f"<div class='cf-metric'><span>执行</span><strong>{caseflow_html_escape(caseflow_display_status(execution.get('status')))}</strong></div>"
        f"<div class='cf-metric'><span>工单</span><strong>{caseflow_html_escape(ticket_label)}</strong></div>"
        "</div>"
        "<div class='cf-section'><h3>回复草稿</h3>"
        f"<div class='cf-draft'>{caseflow_html_escape(draft)}</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )


def caseflow_is_guide_view() -> bool:
    return str(st.query_params.get("view") or "") == "guide"


def caseflow_open_guide_page() -> None:
    st.query_params["view"] = "guide"
    st.rerun()


def caseflow_open_workbench_page() -> None:
    if "view" in st.query_params:
        del st.query_params["view"]
    st.rerun()


def draw_caseflow_guide_page() -> None:
    st.markdown(caseflow_portfolio_css(), unsafe_allow_html=True)
    st.markdown(
        "<style>.block-container{overflow:auto !important;}</style>",
        unsafe_allow_html=True,
    )

    header_left, header_back = st.columns([1, 0.16])
    with header_left:
        st.markdown(
            "<div class='cf-topbar'>"
            "<div class='cf-brand'>"
            "<div class='cf-mark'>CF</div>"
            "<div>"
            f"<div class='cf-title'>{caseflow_html_escape(APP_TITLE)}</div>"
            f"<p class='cf-subtitle'>{caseflow_html_escape(CASEFLOW_PORTFOLIO_TITLE)}</p>"
            "</div>"
            "</div>"
            "<div class='cf-top-meta'>"
            "<span class='cf-pill'><strong>系统说明</strong></span>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with header_back:
        if st.button(CASEFLOW_GUIDE_BACK_LABEL, use_container_width=True):
            caseflow_open_workbench_page()

    st.markdown(
        "<div class='cf-guide-hero'>"
        "<h1>CaseFlow Agent 使用指南</h1>"
        "<p>"
        "CaseFlow Agent 面向电商售后、客服工单和投诉升级场景，"
        "把客户问题处理为一条可追踪、可复核、可审批的业务工作流。"
        "系统会展示每个节点的输入摘要、输出摘要、证据引用、风险规则和最终工单结果。"
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    guide_cta_left, guide_cta_right = st.columns([0.82, 0.18])
    with guide_cta_left:
        st.empty()
    with guide_cta_right:
        if st.button(CASEFLOW_DEMO_START_LABEL, use_container_width=True, type="primary"):
            caseflow_open_workbench_page()

    st.markdown(
        "<div class='cf-guide-grid'>"
        "<section class='cf-guide-card'>"
        "<h2>这个系统是什么</h2>"
        "<p>"
        "CaseFlow 把售后请求拆成意图识别、证据检索、行动规划、回复草稿、"
        "风险复核、人工审批、工单执行和结果持久化 8 个节点。"
        "</p>"
        "</section>"
        "<section class='cf-guide-card'>"
        "<h2>为什么不是普通聊天机器人</h2>"
        "<p>"
        "普通聊天机器人主要展示回答质量；CaseFlow 展示的是业务动作是否受控、"
        "证据是否可追踪、高风险动作是否会被阻断。"
        "</p>"
        "</section>"
        "<section class='cf-guide-card'>"
        "<h2>核心能力</h2>"
        "<p>"
        "工作流图用于展示处理进度；节点详情用于展示可审计依据；"
        "人工审批用于拦截退款、补偿、投诉升级等高风险动作。"
        "</p>"
        "</section>"
        "<section class='cf-guide-card cf-guide-wide'>"
        "<h2>如何操作</h2>"
        "<ul>"
        "<li>选择左侧预设案例，例如普通售后咨询或高金额补偿诉求。</li>"
        "<li>点击“运行”，观察 LangGraph 工作流节点如何从未执行变为运行中和已完成。</li>"
        "<li>点击任意节点，在右侧查看输入摘要、输出摘要、命中证据和风险规则。</li>"
        "<li>遇到高风险案例时，在人工审批节点选择批准、拒绝或要求修改。</li>"
        "<li>最后查看底部执行时间线和工单摘要，确认处理结果可复盘。</li>"
        "</ul>"
        "</section>"
        "<section class='cf-guide-card'>"
        "<h2>推荐演示路径</h2>"
        "<p>"
        "可以先运行“普通商品咨询 / 售后政策咨询”，观察低风险咨询如何检索政策并跳过审批；"
        "再运行“高金额补偿诉求”，观察风险复核如何阻断直接执行并进入人工审批。"
        "</p>"
        "</section>"
        "<section class='cf-guide-card'>"
        "<h2>风险控制 / 人工介入</h2>"
        "<p>"
        "退款、补偿、投诉、主管升级等动作会触发确定性规则。高风险动作不会直接执行，"
        "需要人工审批后才进入模拟工单执行。"
        "</p>"
        "</section>"
        "<section class='cf-guide-card'>"
        "<h2>原型边界</h2>"
        "<p>"
        "当前演示使用本地模拟数据和模拟工单执行，不连接真实 CRM、退款或赔付系统；"
        "界面只展示业务证据和结构化审计轨迹，不展示模型隐藏推理链路。"
        "</p>"
        "</section>"
        "<section class='cf-guide-card'>"
        "<h2>数据与安全</h2>"
        "<p>"
        "请不要输入真实客户隐私、支付凭证或账号密钥。"
        "当前版本用于展示工作流形态和交互边界，真实上线前需要接入权限、审计日志和业务系统。"
        "</p>"
        "</section>"
        "</div>",
        unsafe_allow_html=True,
    )


def caseflow_has_pending_approval_state(
    custom_data: dict,
    events: list[dict[str, Any]],
) -> bool:
    if caseflow_is_approval_pending(custom_data):
        return True
    for event in reversed(events):
        node_id = str(event.get("node_id") or "")
        status = str(event.get("status") or "")
        if node_id == "human_approval" and status == "running":
            return True
        if node_id in {"ticket_execution", "result_persistence"} and status == "succeeded":
            return False
    return False


def draw_caseflow_portfolio_shell(messages: list[ChatMessage]) -> None:
    st.markdown(caseflow_portfolio_css(), unsafe_allow_html=True)

    if "caseflow_workflow_events" not in st.session_state:
        st.session_state.caseflow_workflow_events = []

    demo_cases = load_demo_cases()
    case_ids = [case.case_id for case in demo_cases]
    active_case_id = (
        st.session_state.get("caseflow-demo-case-radio")
        or st.session_state.get("caseflow_active_case_id")
        or case_ids[0]
    )
    active_index = case_ids.index(active_case_id) if active_case_id in case_ids else 0
    previous_case_id = st.session_state.get("caseflow_active_case_id")
    if previous_case_id and active_case_id != previous_case_id:
        st.session_state.messages = []
        messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.caseflow_workflow_events = []
        st.session_state.caseflow_last_custom_data = {}
        st.session_state.caseflow_run_status = "ready"
    st.session_state.caseflow_active_case_id = active_case_id

    custom_data = caseflow_latest_custom_data(messages)
    events = caseflow_workflow_events(custom_data)
    nodes = caseflow_workflow_nodes(custom_data)
    selected_case = get_demo_case(active_case_id)
    run_status = caseflow_run_status_label(custom_data, events)

    header_action = "可运行 / 可重放" if not events else "可重放"
    st.markdown(
        "<div class='cf-topbar'>"
        "<div class='cf-brand'>"
        "<div class='cf-mark'>CF</div>"
        "<div>"
        f"<div class='cf-title'>{caseflow_html_escape(APP_TITLE)}</div>"
        f"<p class='cf-subtitle'>{caseflow_html_escape(CASEFLOW_PORTFOLIO_TITLE)}</p>"
        "</div>"
        "</div>"
        "<div class='cf-top-meta'>"
        f"<span class='cf-pill'>当前案例：{caseflow_html_escape(selected_case.title)}</span>"
        f"<span class='cf-pill'>运行状态: <strong>{caseflow_html_escape(run_status)}</strong></span>"
        f"<span class='cf-pill'>演示模式：{caseflow_html_escape(header_action)}</span>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    top_left, top_guide, top_run, top_reset, top_replay = st.columns(
        [1, 0.11, 0.08, 0.08, 0.09]
    )
    with top_left:
        st.empty()
    if top_guide.button(CASEFLOW_DEMO_GUIDE_LABEL, use_container_width=True):
        caseflow_open_guide_page()

    left, center, right = st.columns([0.30, 0.42, 0.28], gap="medium")

    with left:
        with st.container(border=True, height=CASEFLOW_LEFT_CASE_HEIGHT):
            st.markdown(
                "<div class='cf-panel-header'>"
                "<div class='cf-panel-title'>预设案例</div>"
                "<span class='cf-panel-kicker'>作品集演示</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            selected_case_id = st.radio(
                "选择演示案例",
                options=case_ids,
                index=active_index,
                format_func=lambda value: get_demo_case(value).title,
                label_visibility="collapsed",
                key="caseflow-demo-case-radio",
            )
            selected_case = get_demo_case(selected_case_id)
            st.session_state.caseflow_active_case_id = selected_case.case_id

        with st.container(border=True, height=CASEFLOW_LEFT_INPUT_HEIGHT):
            st.markdown(
                "<div class='cf-panel-header'>"
                "<div class='cf-panel-title'>客户输入 / 工单入口</div>"
                "<span class='cf-panel-kicker'>工单入口</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            demo_input = st.text_area(
                "客户问题",
                value=selected_case.user_query,
                height=116,
                key=f"caseflow-demo-input-{selected_case.case_id}",
            )

        with st.container(border=True, height=CASEFLOW_LEFT_CONTEXT_HEIGHT):
            st.markdown(
                "<div class='cf-panel-header'>"
                "<div class='cf-panel-title'>业务上下文</div>"
                "<span class='cf-panel-kicker'>订单 / 用户 / 历史</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='cf-business-context'>"
                f"<div class='cf-context-row'><span>订单</span><strong>{caseflow_html_escape(selected_case.order_summary)}</strong></div>"
                f"<div class='cf-context-row'><span>商品</span><strong>{caseflow_html_escape(selected_case.product_summary)}</strong></div>"
                f"<div class='cf-context-row'><span>历史</span><strong>{caseflow_html_escape(selected_case.customer_history_summary)}</strong></div>"
                f"<div class='cf-context-row'><span>信号</span><strong>{caseflow_html_escape(selected_case.expected_signal)}</strong></div>"
                "</div>",
                unsafe_allow_html=True,
            )

    if top_run.button(CASEFLOW_DEMO_RUN_LABEL, use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.caseflow_demo_pending_input = demo_input
        st.session_state.caseflow_demo_pending_user_id = selected_case.customer_id
        st.session_state.caseflow_demo_pending_case_id = selected_case.case_id
        st.session_state.caseflow_active_user_id = selected_case.customer_id
        st.session_state.caseflow_workflow_events = []
        st.session_state.caseflow_last_custom_data = {}
        st.session_state.caseflow_run_status = "running"
        st.rerun()
    if top_reset.button(CASEFLOW_DEMO_RESET_LABEL, use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.caseflow_workflow_events = []
        st.session_state.caseflow_last_custom_data = {}
        st.session_state.caseflow_run_status = "ready"
        st.rerun()
    if top_replay.button(CASEFLOW_DEMO_REPLAY_LABEL, use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.caseflow_demo_pending_input = demo_input
        st.session_state.caseflow_demo_pending_user_id = selected_case.customer_id
        st.session_state.caseflow_demo_pending_case_id = selected_case.case_id
        st.session_state.caseflow_active_user_id = selected_case.customer_id
        st.session_state.caseflow_workflow_events = []
        st.session_state.caseflow_last_custom_data = {}
        st.session_state.caseflow_run_status = "running"
        st.rerun()

    with center:
        with st.container(border=True, height=CASEFLOW_GRAPH_HEIGHT):
            st.markdown(
                "<div class='cf-panel-header'>"
                "<div class='cf-panel-title'>业务工作流 / LangGraph Trace</div>"
                "<span class='cf-panel-kicker'>执行路径与节点状态</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            draw_caseflow_workflow_graph(nodes, events)
        with st.container(border=True, height=CASEFLOW_TIMELINE_HEIGHT):
            st.markdown(
                "<div class='cf-panel-header'>"
                "<div class='cf-panel-title'>执行时间线</div>"
                "<span class='cf-panel-kicker'>事件流</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            draw_caseflow_execution_timeline(events)

    with right:
        with st.container(border=True, height=CASEFLOW_INSPECTOR_HEIGHT):
            st.markdown(
                "<div class='cf-panel-header'>"
                "<div class='cf-panel-title'>节点详情</div>"
                "<span class='cf-panel-kicker'>审计轨迹</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            draw_caseflow_node_inspector(
                nodes,
                events,
                custom_data,
                show_approval_controls=False,
            )
        with st.container(border=True, height=CASEFLOW_APPROVAL_BAR_HEIGHT):
            st.markdown(
                "<div class='cf-panel-header'>"
                "<div class='cf-panel-title'>人工审批</div>"
                "<span class='cf-panel-kicker'>Approve / Reject / Modify</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            draw_caseflow_portfolio_approval_bar(events, custom_data)
        with st.container(border=True, height=CASEFLOW_RESOLUTION_HEIGHT):
            st.markdown(
                "<div class='cf-panel-header'>"
                "<div class='cf-panel-title'>最终处理结果 / 工单摘要</div>"
                "<span class='cf-panel-kicker'>模拟执行</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            draw_caseflow_final_resolution(custom_data, events)


def is_caseflow_approval_interrupt(content: str) -> bool:
    return content.startswith("需要人工审批后才能执行：")


def normalize_caseflow_execution(custom_data: dict) -> dict:
    execution = custom_data.get("execution_result")
    return execution if isinstance(execution, dict) else {}


def caseflow_status_key(value: object) -> str:
    if value in (None, ""):
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def caseflow_display_status(value: object, default: str = "未执行") -> str:
    status_key = caseflow_status_key(value)
    if not status_key:
        return default
    return CASEFLOW_STATUS_LABELS.get(status_key, str(value).strip())


def caseflow_display_queue(value: object, default: str = "未知") -> str:
    queue_key = caseflow_status_key(value)
    if not queue_key:
        return default
    return CASEFLOW_QUEUE_LABELS.get(queue_key, str(value).strip())


def caseflow_is_approval_pending(custom_data: dict) -> bool:
    if not custom_data.get("needs_human_approval"):
        return False
    execution = normalize_caseflow_execution(custom_data)
    return caseflow_status_key(execution.get("status")) in CASEFLOW_APPROVAL_PENDING_STATUSES


def caseflow_display_ai_content(content: str) -> str:
    if is_caseflow_approval_interrupt(content):
        return CASEFLOW_APPROVAL_INTERRUPT_DISPLAY
    return content


def caseflow_approval_payload(
    decision: str,
    reason: str = "",
    modification_notes: str = "",
) -> str:
    return json.dumps(
        {
            "decision": decision,
            "reason": reason.strip(),
            "modification_notes": modification_notes.strip(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _caseflow_approval_payload_parts(value: str) -> tuple[str, str, str]:
    if value in CASEFLOW_APPROVAL_DISPLAY_TEXT:
        return value, "", ""
    try:
        payload = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value, "", ""
    if not isinstance(payload, dict):
        return value, "", ""
    return (
        str(payload.get("decision", "")).strip(),
        str(payload.get("reason", "")).strip(),
        str(payload.get("modification_notes", "")).strip(),
    )


def caseflow_approval_display_text(value: str) -> str:
    decision, reason, modification_notes = _caseflow_approval_payload_parts(value)
    display = CASEFLOW_APPROVAL_DISPLAY_TEXT.get(decision, value)
    details = []
    if reason:
        details.append(f"原因：{reason}")
    if modification_notes:
        details.append(f"修改要求：{modification_notes}")
    if details:
        return f"{display}（{'；'.join(details)}）"
    return display


def caseflow_status_badge(label: str, value: object, default: str = "未执行") -> str:
    label_text = html.escape(label)
    value_text = html.escape(caseflow_display_status(value, default=default))
    return (
        '<span style="display:inline-block; border:1px solid #D0D5DD; '
        'border-radius:6px; padding:4px 10px; margin:2px 6px 2px 0; '
        'background:#F9FAFB; color:#344054; font-size:0.9rem;">'
        f"{label_text}：<strong>{value_text}</strong></span>"
    )


def caseflow_overview_items(custom_data: dict) -> tuple[tuple[str, str], ...]:
    approval = "需要审批" if custom_data["needs_human_approval"] else "无需审批"
    return (
        ("意图", str(custom_data["intent"])),
        ("优先级", str(custom_data["priority"])),
        ("审批", approval),
        ("模型", str(custom_data.get("model_used", "unknown"))),
    )


def _caseflow_evidence_source_key(value: object) -> str:
    if value in (None, ""):
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _caseflow_group_key_from_source(value: object) -> str:
    source_key = _caseflow_evidence_source_key(value)
    if source_key in CASEFLOW_KB_SOURCE_KEYS:
        return "kb"
    if source_key in CASEFLOW_HISTORY_SOURCE_KEYS:
        return "history"
    if source_key in CASEFLOW_CUSTOMER_SOURCE_KEYS:
        return "customer"
    if source_key == "other":
        return "other"
    return ""


def _caseflow_join_title_summary(title: object, summary: object) -> str:
    title_text = str(title).strip() if title not in (None, "") else ""
    summary_text = str(summary).strip() if summary not in (None, "") else ""
    if title_text and summary_text:
        return f"{title_text}: {summary_text}"
    return title_text or summary_text


def _caseflow_group_text_evidence(item: str, group_hint: str = "") -> tuple[str, str]:
    text = item.strip()
    if not text:
        return "", ""

    upper_text = text.upper()
    if upper_text.startswith(("FAQ -", "SOP -", "POLICY -", "KB -")):
        return "kb", text
    if upper_text.startswith("CASE-") or "历史案例" in text:
        return "history", text
    if upper_text.startswith("CUSTOMER PROFILE:"):
        return "customer", text
    return group_hint or "other", text


def _caseflow_group_dict_evidence(
    item: Mapping[str, Any],
    group_hint: str = "",
) -> tuple[str, str]:
    case_id = item.get("case_id")
    if case_id not in (None, ""):
        evidence_text = _caseflow_join_title_summary(case_id, item.get("summary") or item.get("title"))
        return "history", evidence_text

    source = item.get("source") or item.get("type") or group_hint
    group_key = _caseflow_group_key_from_source(source) or group_hint or "other"
    title = item.get("title") or item.get("name")
    summary = item.get("summary") or item.get("content") or item.get("text")

    if group_key == "kb" and source not in (None, ""):
        source_label = str(source).strip().upper()
        evidence_text = _caseflow_join_title_summary(title, summary)
        return group_key, f"{source_label} - {evidence_text}" if evidence_text else source_label

    evidence_text = _caseflow_join_title_summary(title, summary)
    if evidence_text:
        return group_key, evidence_text

    return group_key, str(dict(item))


def _caseflow_group_evidence_item(item: object, group_hint: str = "") -> tuple[str, str]:
    if item in (None, ""):
        return "", ""
    if isinstance(item, str):
        return _caseflow_group_text_evidence(item, group_hint=group_hint)
    if isinstance(item, Mapping):
        return _caseflow_group_dict_evidence(item, group_hint=group_hint)
    return group_hint or "other", str(item).strip()


def _caseflow_iter_evidence_items(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, str | bytes):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _caseflow_format_customer_profile(profile: object) -> str:
    if not isinstance(profile, Mapping):
        return ""

    name = profile.get("name") or profile.get("customer_id")
    tier = profile.get("tier")
    if name in (None, ""):
        return ""
    if tier not in (None, ""):
        return f"{name}（{tier}）"
    return str(name)


def caseflow_group_evidence(custom_data: dict) -> tuple[tuple[str, tuple[str, ...]], ...]:
    groups: dict[str, list[str]] = {group_key: [] for group_key in CASEFLOW_EVIDENCE_GROUPS}
    seen_items: set[tuple[str, str]] = set()

    def add_item(group_key: str, evidence_text: str) -> None:
        if group_key not in groups:
            group_key = "other"
        text = evidence_text.strip()
        if not text:
            return
        seen_key = (group_key, text)
        if seen_key in seen_items:
            return
        groups[group_key].append(text)
        seen_items.add(seen_key)

    grouped_evidence = custom_data.get("grouped_evidence")
    if isinstance(grouped_evidence, Mapping):
        for source, value in grouped_evidence.items():
            group_hint = _caseflow_group_key_from_source(source) or "other"
            for item in _caseflow_iter_evidence_items(value):
                group_key, evidence_text = _caseflow_group_evidence_item(item, group_hint=group_hint)
                add_item(group_key, evidence_text)

    for item in _caseflow_iter_evidence_items(custom_data.get("evidence")):
        group_key, evidence_text = _caseflow_group_evidence_item(item)
        add_item(group_key, evidence_text)

    if not groups["customer"]:
        customer_profile_text = _caseflow_format_customer_profile(custom_data.get("customer_profile"))
        add_item("customer", customer_profile_text)

    return tuple(
        (label, tuple(groups[group_key]))
        for group_key, label in CASEFLOW_EVIDENCE_GROUPS.items()
        if groups[group_key]
    )


def caseflow_status_summary(custom_data: dict) -> str:
    approval = "需要人工审批" if custom_data["needs_human_approval"] else "无需人工审批"
    execution = normalize_caseflow_execution(custom_data)
    execution_status = caseflow_display_status(execution.get("status"))
    model_used = custom_data.get("model_used", "unknown-model")
    ticket = execution.get("ticket")
    ticket_id = ticket.get("ticket_id") if isinstance(ticket, Mapping) else None
    ticket_suffix = f" | 工单={ticket_id}" if ticket_id else ""
    return (
        f"{custom_data['intent']} | 优先级={custom_data['priority']} | "
        f"{approval} | 模型={model_used} | 执行状态={execution_status}{ticket_suffix}"
    )


def draw_caseflow_workbench(custom_data: dict) -> None:
    st.divider()
    st.caption("CaseFlow 工单工作台")
    overview_cols = st.columns(4)
    for column, (label, value) in zip(
        overview_cols,
        caseflow_overview_items(custom_data),
        strict=True,
    ):
        with column:
            st.caption(label)
            st.markdown(f"**{value}**")

    st.info(caseflow_status_summary(custom_data))

    evidence_col, plan_col = st.columns(2)
    with evidence_col:
        st.subheader("依据")
        evidence_groups = caseflow_group_evidence(custom_data)
        if not evidence_groups:
            st.caption(CASEFLOW_EVIDENCE_EMPTY_MESSAGE)
        for group_label, items in evidence_groups:
            st.markdown(f"**{group_label}**")
            for item in items:
                st.markdown(f"- {item}")

    with plan_col:
        st.subheader("处理计划")
        for index, item in enumerate(custom_data["proposed_action_plan"], start=1):
            st.markdown(f"{index}. {item}")

    st.subheader("客户回复草稿")
    st.write(custom_data["draft_response"])

    reflection = custom_data.get("reflection", {})
    reasoning = custom_data.get("llm_reasoning", {})
    st.subheader("风险复核")
    reflection_col, reason_col = st.columns(2)
    with reflection_col:
        st.markdown(f"**风险：** {reflection.get('risk', 'unknown')}")
        st.markdown(
            f"**依据是否充分：** {reflection.get('evidence_sufficient', 'unknown')}"
        )
        if reflection.get("approval_reason"):
            st.warning(f"审批原因：{reflection['approval_reason']}")
    with reason_col:
        if reflection.get("reason"):
            st.write(reflection["reason"])
        if reasoning.get("analysis"):
            st.caption(f"分析：{reasoning['analysis']}")
        if reasoning.get("resolution"):
            st.caption(f"回复依据：{reasoning['resolution']}")

    execution = normalize_caseflow_execution(custom_data)
    ticket = execution.get("ticket")
    escalation = execution.get("escalation")
    if ticket or escalation:
        st.subheader("工单 / 升级")
        if ticket:
            ticket_id = html.escape(str(ticket.get("ticket_id", "未知")))
            st.markdown(
                f"工单：{ticket_id} "
                f"{caseflow_status_badge('状态', ticket.get('status'), default='未知')}",
                unsafe_allow_html=True,
            )
        if escalation:
            escalation_queue = html.escape(caseflow_display_queue(escalation.get("queue")))
            st.markdown(
                f"升级队列：{escalation_queue} "
                f"{caseflow_status_badge('状态', escalation.get('status'), default='未知')}",
                unsafe_allow_html=True,
    )
    if caseflow_is_approval_pending(custom_data):
        st.warning("高风险动作执行前需要人工审批。请选择批准、拒绝或要求修改。")
        approval_key = custom_data.get("thread_id", st.session_state.get("thread_id", "caseflow"))
        draw_caseflow_structured_approval_controls(approval_key, "workbench")
    st.markdown(
        caseflow_status_badge("执行状态", execution.get("status")),
        unsafe_allow_html=True,
    )


def draw_caseflow_structured_approval_controls(
    thread_key: str,
    key_prefix: str,
    *,
    compact: bool = False,
) -> None:
    if compact:
        reason = st.text_input(
            "审批备注（可选）",
            key=f"{key_prefix}-approval-reason-{thread_key}",
            placeholder="例如：证据充分，可执行；或需要补充政策依据",
        )
        approve_col, reject_col, modify_col = st.columns(3)
        if approve_col.button(CASEFLOW_APPROVE_BUTTON_LABEL, key=f"{key_prefix}-approve-{thread_key}"):
            st.session_state.approval_decision = caseflow_approval_payload("approve", reason)
            st.rerun()
        if reject_col.button(CASEFLOW_REJECT_BUTTON_LABEL, key=f"{key_prefix}-reject-{thread_key}"):
            st.session_state.approval_decision = caseflow_approval_payload("reject", reason)
            st.rerun()
        if modify_col.button(CASEFLOW_MODIFY_BUTTON_LABEL, key=f"{key_prefix}-modify-{thread_key}"):
            st.session_state.approval_decision = caseflow_approval_payload("modify", reason, reason)
            st.rerun()
        return

    reason = st.text_input(
        "审批原因（可选）",
        key=f"{key_prefix}-approval-reason-{thread_key}",
        placeholder="例如：证据充分，可执行；或证据不足，需补充",
    )
    modification_notes = st.text_area(
        "修改要求（选择“要求修改”时填写）",
        key=f"{key_prefix}-approval-modification-{thread_key}",
        placeholder="例如：先补充退款政策依据，并把客户回复改得更谨慎",
        height=90,
    )
    approve_col, reject_col, modify_col = st.columns(3)
    if approve_col.button(CASEFLOW_APPROVE_BUTTON_LABEL, key=f"{key_prefix}-approve-{thread_key}"):
        st.session_state.approval_decision = caseflow_approval_payload("approve", reason)
        st.rerun()
    if reject_col.button(CASEFLOW_REJECT_BUTTON_LABEL, key=f"{key_prefix}-reject-{thread_key}"):
        st.session_state.approval_decision = caseflow_approval_payload("reject", reason)
        st.rerun()
    if modify_col.button(CASEFLOW_MODIFY_BUTTON_LABEL, key=f"{key_prefix}-modify-{thread_key}"):
        st.session_state.approval_decision = caseflow_approval_payload(
            "modify",
            reason,
            modification_notes,
        )
        st.rerun()


def draw_caseflow_approval_controls(thread_key: str) -> None:
    st.warning("高风险动作执行前需要人工审批。请选择批准、拒绝或要求修改。")
    draw_caseflow_structured_approval_controls(thread_key, "interrupt")


def get_or_create_user_id() -> str:
    """Get the user ID from session state or URL parameters, or create a new one if it doesn't exist."""
    # Check if user_id exists in session state
    if USER_ID_COOKIE in st.session_state:
        return st.session_state[USER_ID_COOKIE]

    # Try to get from URL parameters using the new st.query_params
    if USER_ID_COOKIE in st.query_params:
        user_id = st.query_params[USER_ID_COOKIE]
        st.session_state[USER_ID_COOKIE] = user_id
        return user_id

    # Generate a new user_id if not found
    user_id = str(uuid.uuid4())

    # Store in session state for this session
    st.session_state[USER_ID_COOKIE] = user_id

    # Also add to URL parameters so it can be bookmarked/shared
    st.query_params[USER_ID_COOKIE] = user_id

    return user_id


async def run_caseflow_portfolio_interaction(
    agent_client: AgentClient,
    *,
    user_input: str,
    display_user_input: str,
    model: str,
    thread_id: str,
    user_id: str,
    agent_config: dict[str, Any] | None,
) -> None:
    st.session_state.messages.append(ChatMessage(type="human", content=display_user_input))
    stream_kwargs: dict[str, Any] = {
        "message": user_input,
        "model": model,
        "thread_id": thread_id,
        "user_id": user_id,
    }
    if agent_config:
        stream_kwargs["agent_config"] = agent_config

    try:
        async for msg in agent_client.astream(**stream_kwargs):
            if is_caseflow_workflow_event(msg):
                caseflow_record_workflow_event(msg["content"])
                continue
            if isinstance(msg, str):
                continue
            if not isinstance(msg, ChatMessage):
                continue
            st.session_state.messages.append(msg)
            if msg.type == "ai" and is_caseflow_result(msg.custom_data):
                caseflow_store_result_payload(msg.custom_data)
            elif msg.type == "ai" and msg.content and is_caseflow_approval_interrupt(msg.content):
                st.session_state.caseflow_run_status = "awaiting_approval"
    except AgentClientError as exc:
        st.session_state.caseflow_run_status = "failed"
        st.session_state.caseflow_run_error = f"Error generating response: {exc}"
        return

    if st.session_state.get("caseflow_run_status") == "running":
        custom_data = caseflow_latest_custom_data(st.session_state.messages)
        events = caseflow_workflow_events(custom_data)
        st.session_state.caseflow_run_status = (
            "awaiting_approval" if caseflow_has_pending_approval_state(custom_data, events) else "completed"
        )


async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={},
    )

    # Hide the streamlit upper-right chrome
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    # Get or create user ID
    user_id = get_or_create_user_id()

    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0")
            port = os.getenv("PORT", 8080)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner(CASEFLOW_CONNECTING_MESSAGE):
                st.session_state.agent_client = AgentClient(base_url=agent_url)
        except AgentClientError as e:
            st.error(f"{CASEFLOW_CONNECTION_ERROR_PREFIX}（{agent_url}）：{e}")
            st.markdown(CASEFLOW_SERVICE_BOOT_MESSAGE)
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    # Initialize voice manager (once per session)
    if "voice_manager" not in st.session_state:
        st.session_state.voice_manager = VoiceManager.from_env()
    voice = st.session_state.voice_manager

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            messages = []
        else:
            try:
                messages: ChatHistory = agent_client.get_history(thread_id=thread_id).messages
            except AgentClientError:
                st.error("No message history found for this Thread ID.")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    model = agent_client.info.default_model
    use_streaming = True
    enable_audio = False

    # Config options. CaseFlow owns a portfolio workbench, so the generic sidebar is
    # kept only for non-portfolio agents and tests that exercise the upstream chat UI.
    if agent_client.agent != "caseflow-agent":
        with st.sidebar:
            st.header(f"{APP_ICON} {APP_TITLE}")

            ""
            CASEFLOW_SIDEBAR_DESCRIPTION
            ""

            if st.button(CASEFLOW_NEW_CASE_LABEL, use_container_width=True):
                st.session_state.messages = []
                st.session_state.thread_id = str(uuid.uuid4())
                # Clear saved audio when starting new chat
                if "last_audio" in st.session_state:
                    del st.session_state.last_audio
                st.rerun()

            with st.popover(CASEFLOW_SETTINGS_LABEL, use_container_width=True):
                model_idx = agent_client.info.models.index(agent_client.info.default_model)
                model = st.selectbox("模型", options=agent_client.info.models, index=model_idx)
                show_example_agents = st.checkbox("显示示例 Agent", value=False)
                all_agents = [a.key for a in agent_client.info.agents]
                if show_example_agents:
                    agent_list = all_agents
                else:
                    agent_list = ["caseflow-agent"] if "caseflow-agent" in all_agents else all_agents
                agent_idx = agent_list.index(agent_client.info.default_agent)
                agent_client.agent = st.selectbox(
                    CASEFLOW_AGENT_SELECT_LABEL,
                    options=agent_list,
                    index=agent_idx,
                )
                use_streaming = st.toggle("流式输出", value=True)
                # Audio toggle with callback: clears cached audio when toggled off
                enable_audio = st.toggle(
                    "启用语音生成",
                    value=True,
                    disabled=not voice or not voice.tts,
                    help="在 .env 中配置 VOICE_TTS_PROVIDER 后启用"
                    if not voice or not voice.tts
                    else None,
                    on_change=lambda: st.session_state.pop("last_audio", None)
                    if not st.session_state.get("enable_audio", True)
                    else None,
                    key="enable_audio",
                )

                # Display user ID (for debugging or user information)
                st.text_input("用户 ID（只读）", value=user_id, disabled=True)

            with st.popover(CASEFLOW_PRIVACY_LABEL, use_container_width=True):
                st.write(
                    "本地演示会保留会话、反馈和运行信息，用于观察 CaseFlow 的处理效果。请不要输入真实敏感客户信息。"
                )

            @st.dialog(CASEFLOW_SHARE_DIALOG_TITLE)
            def share_chat_dialog() -> None:
                session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]
                st_base_url = urllib.parse.urlunparse(
                    [session.client.request.protocol, session.client.request.host, "", "", "", ""]
                )
                # if it's not localhost, switch to https by default
                if not st_base_url.startswith("https") and "localhost" not in st_base_url:
                    st_base_url = st_base_url.replace("http", "https")
                # Include both thread_id and user_id in the URL for sharing to maintain user identity
                chat_url = (
                    f"{st_base_url}?thread_id={st.session_state.thread_id}&{USER_ID_COOKIE}={user_id}"
                )
                st.markdown(f"**工单链接：**\n```text\n{chat_url}\n```")
                st.info("复制上方链接，可分享或恢复当前工单线程。")

            if st.button(CASEFLOW_SHARE_LABEL, use_container_width=True):
                share_chat_dialog()

            st.markdown(f"[{CASEFLOW_SOURCE_LINK_LABEL}]({CASEFLOW_SOURCE_URL})")
            st.caption(CASEFLOW_AUTHOR_CAPTION)

    # Draw existing messages
    messages: list[ChatMessage] = st.session_state.messages

    if agent_client.agent == "caseflow-agent":
        if caseflow_is_guide_view():
            draw_caseflow_guide_page()
            return

        approval_decision = st.session_state.pop("approval_decision", None)
        demo_input = st.session_state.pop("caseflow_demo_pending_input", None)
        if approval_decision or demo_input:
            st.session_state.caseflow_run_status = "running"
            draw_caseflow_portfolio_shell(messages)

            active_case_id = st.session_state.get("caseflow_active_case_id")
            run_agent_config: dict[str, Any] | None = None
            run_user_id = str(st.session_state.get("caseflow_active_user_id") or user_id)
            if approval_decision:
                user_input = str(approval_decision)
                display_user_input = caseflow_approval_display_text(str(approval_decision))
                if active_case_id:
                    run_agent_config = {"demo_mode": True, "case_id": active_case_id}
            else:
                user_input = str(demo_input)
                display_user_input = user_input
                run_user_id = str(st.session_state.get("caseflow_demo_pending_user_id") or user_id)
                active_case_id = st.session_state.get("caseflow_demo_pending_case_id")
                if active_case_id:
                    st.session_state.caseflow_active_case_id = active_case_id
                    run_agent_config = {"demo_mode": True, "case_id": active_case_id}

            with st.spinner("正在运行 CaseFlow 工作流事件流..."):
                await run_caseflow_portfolio_interaction(
                    agent_client,
                    user_input=user_input,
                    display_user_input=display_user_input,
                    model=str(model),
                    thread_id=st.session_state.thread_id,
                    user_id=run_user_id,
                    agent_config=run_agent_config,
                )
            st.rerun()

        draw_caseflow_portfolio_shell(messages)
        if st.session_state.get("caseflow_run_error"):
            st.error(st.session_state.caseflow_run_error)
        return

    if len(messages) == 0:
        match agent_client.agent:
            case "caseflow-agent":
                WELCOME = CASEFLOW_WELCOME_MESSAGE
            case "chatbot":
                WELCOME = "Hello! I'm a simple chatbot. Ask me anything!"
            case "interrupt-agent":
                WELCOME = "Hello! I'm an interrupt agent. Tell me your birthday and I will predict your personality!"
            case "research-assistant":
                WELCOME = "Hello! I'm an AI-powered research assistant with web search and a calculator. Ask me anything!"
            case "rag-assistant":
                WELCOME = """Hello! I'm an AI-powered Company Policy & HR assistant with access to AcmeTech's Employee Handbook.
                I can help you find information about benefits, remote work, time-off policies, company values, and more. Ask me anything!"""
            case _:
                WELCOME = "Hello! I'm an AI agent. Ask me anything!"

        with st.chat_message("ai"):
            st.write(WELCOME)

    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # Render saved audio for the last AI message (if it exists)
    # This ensures audio persists across st.rerun() calls
    if (
        voice
        and enable_audio
        and "last_audio" in st.session_state
        and st.session_state.last_message
        and len(messages) > 0
        and messages[-1].type == "ai"
    ):
        with st.session_state.last_message:
            audio_data = st.session_state.last_audio
            st.audio(audio_data["data"], format=audio_data["format"])

    # Generate new message if the user provided new input
    # Use voice manager if available, otherwise fall back to regular input
    # REQUIRED: Set VOICE_STT_PROVIDER, VOICE_TTS_PROVIDER, OPENAI_API_KEY
    # in app .env (NOT service .env) to enable voice features.
    approval_decision = st.session_state.pop("approval_decision", None)
    demo_input = st.session_state.pop("caseflow_demo_pending_input", None)
    run_user_id = user_id
    run_agent_config: dict[str, Any] | None = None
    if approval_decision:
        user_input = approval_decision
        display_user_input = caseflow_approval_display_text(approval_decision)
        run_user_id = str(st.session_state.get("caseflow_active_user_id") or user_id)
        active_case_id = st.session_state.get("caseflow_active_case_id")
        if active_case_id:
            run_agent_config = {"demo_mode": True, "case_id": active_case_id}
    elif demo_input:
        user_input = str(demo_input)
        display_user_input = user_input
        run_user_id = str(st.session_state.get("caseflow_demo_pending_user_id") or user_id)
        active_case_id = st.session_state.get("caseflow_demo_pending_case_id")
        if active_case_id:
            st.session_state.caseflow_active_case_id = active_case_id
            run_agent_config = {"demo_mode": True, "case_id": active_case_id}
    elif voice:
        user_input = voice.get_chat_input()
        display_user_input = user_input
    else:
        user_input = st.chat_input(CASEFLOW_CHAT_PLACEHOLDER)
        display_user_input = user_input

    if user_input:
        messages.append(ChatMessage(type="human", content=display_user_input))
        st.chat_message("human").write(display_user_input)
        try:
            if use_streaming:
                stream_kwargs: dict[str, Any] = {
                    "message": user_input,
                    "model": model,
                    "thread_id": st.session_state.thread_id,
                    "user_id": run_user_id,
                }
                if run_agent_config:
                    stream_kwargs["agent_config"] = run_agent_config
                stream = agent_client.astream(**stream_kwargs)
                await draw_messages(stream, is_new=True)
                # Generate TTS audio for streaming response
                # Note: draw_messages() stores the final message in st.session_state.messages
                # and the container reference in st.session_state.last_message
                if voice and enable_audio and st.session_state.messages:
                    last_msg = st.session_state.messages[-1]
                    # Only generate audio for AI responses with content
                    if last_msg.type == "ai" and last_msg.content:
                        # Use audio_only=True since text was already streamed by draw_messages()
                        voice.render_message(
                            last_msg.content,
                            container=st.session_state.last_message,
                            audio_only=True,
                        )
            else:
                invoke_kwargs: dict[str, Any] = {
                    "message": user_input,
                    "model": model,
                    "thread_id": st.session_state.thread_id,
                    "user_id": run_user_id,
                }
                if run_agent_config:
                    invoke_kwargs["agent_config"] = run_agent_config
                response = await agent_client.ainvoke(**invoke_kwargs)
                messages.append(response)
                # Render AI response with optional voice
                with st.chat_message("ai"):
                    display_response_content = caseflow_display_ai_content(response.content)
                    if voice and enable_audio:
                        voice.render_message(display_response_content)
                    else:
                        st.write(display_response_content)
                    if is_caseflow_result(response.custom_data):
                        caseflow_store_result_payload(response.custom_data)
                        draw_caseflow_workbench(response.custom_data)
                    elif is_caseflow_approval_interrupt(response.content):
                        draw_caseflow_approval_controls(st.session_state.thread_id)
            st.rerun()  # Clear stale containers
        except AgentClientError as e:
            st.error(f"Error generating response: {e}")
            st.stop()

    # If messages have been generated, show feedback widget
    if len(messages) > 0 and st.session_state.last_message:
        with st.session_state.last_message:
            await handle_feedback()


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str | dict[str, Any], None],
    is_new: bool = False,
) -> None:
    """
    Draws a set of chat messages - either replaying existing messages
    or streaming new ones.

    This function has additional logic to handle streaming tokens and tool calls.
    - Use a placeholder container to render streaming tokens as they arrive.
    - Use a status container to render tool calls. Track the tool inputs and outputs
      and update the status container accordingly.

    The function also needs to track the last message container in session state
    since later messages can draw to the same container. This is also used for
    drawing the feedback widget in the latest chat message.

    Args:
        messages_aiter: An async iterator over messages to draw.
        is_new: Whether the messages are new or not.
    """

    # Keep track of the last message container
    last_message_type = None
    st.session_state.last_message = None
    active_replay_interrupt: ChatMessage | None = None

    if not is_new:
        replay_messages = []
        while msg := await anext(messages_agen, None):
            replay_messages.append(msg)
        latest_message = replay_messages[-1] if replay_messages else None
        if (
            isinstance(latest_message, ChatMessage)
            and latest_message.type == "ai"
            and latest_message.content
            and is_caseflow_approval_interrupt(latest_message.content)
        ):
            active_replay_interrupt = latest_message

        async def replay_iter() -> AsyncGenerator[ChatMessage | str, None]:
            for replay_message in replay_messages:
                yield replay_message

        messages_agen = replay_iter()

    # Placeholder for intermediate streaming tokens
    streaming_content = ""
    streaming_placeholder = None

    # Iterate over the messages and draw them
    while msg := await anext(messages_agen, None):
        if is_caseflow_workflow_event(msg):
            caseflow_record_workflow_event(msg["content"])
            continue
        # str message represents an intermediate token being streamed
        if isinstance(msg, str):
            # If placeholder is empty, this is the first token of a new message
            # being streamed. We need to do setup.
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()

        match msg.type:
            # A message from the user, the easiest case
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # A message from the agent is the most complex case, since we need to
            # handle streaming tokens and tool calls.
            case "ai":
                # If we're rendering new messages, store the message in session state
                if is_new:
                    st.session_state.messages.append(msg)

                # If the last message type was not AI, create a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    # If the message has content, write it out.
                    # Reset the streaming variables to prepare for the next message.
                    if msg.content:
                        display_content = caseflow_display_ai_content(msg.content)
                        if streaming_placeholder:
                            streaming_placeholder.write(display_content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(display_content)

                    if is_caseflow_result(msg.custom_data):
                        caseflow_store_result_payload(msg.custom_data)
                        draw_caseflow_workbench(msg.custom_data)
                    elif msg.content and is_caseflow_approval_interrupt(msg.content):
                        if is_new or msg is active_replay_interrupt:
                            draw_caseflow_approval_controls(st.session_state.thread_id)

                    if msg.tool_calls:
                        # Create a status container for each tool call and store the
                        # status container by ID to ensure results are mapped to the
                        # correct status container.
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            # Use different labels for transfer vs regular tool calls
                            if "transfer_to" in tool_call["name"]:
                                label = f"""💼 Sub Agent: {tool_call["name"]}"""
                            else:
                                label = f"""🛠️ Tool Call: {tool_call["name"]}"""

                            status = st.status(
                                label,
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status

                        # Expect one ToolMessage for each tool call.
                        for tool_call in msg.tool_calls:
                            if "transfer_to" in tool_call["name"]:
                                status = call_results[tool_call["id"]]
                                status.update(expanded=True)
                                await handle_sub_agent_msgs(messages_agen, status, is_new)
                                break

                            # Only non-transfer tool calls reach this point
                            status = call_results[tool_call["id"]]
                            status.write("Input:")
                            status.write(tool_call["args"])
                            tool_result: ChatMessage = await anext(messages_agen)

                            if tool_result.type != "tool":
                                st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()

                            # Record the message if it's new, and update the correct
                            # status container with the result
                            if is_new:
                                st.session_state.messages.append(tool_result)
                            if tool_result.tool_call_id:
                                status = call_results[tool_result.tool_call_id]
                            status.write("Output:")
                            status.write(tool_result.content)
                            status.update(state="complete")

            case "custom":
                # CustomData example used by the bg-task-agent
                # See:
                # - src/agents/utils.py CustomData
                # - src/agents/bg_task_agent/task.py
                try:
                    task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    st.error("Unexpected CustomData message received from agent")
                    st.write(msg.custom_data)
                    st.stop()

                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message(
                        name="task", avatar=":material/manufacturing:"
                    )
                    with st.session_state.last_message:
                        status = TaskDataStatus()

                status.add_and_draw_task_data(task_data)

            # In case of an unexpected message type, log an error and stop
            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback() -> None:
    """Draws a feedback widget and records feedback from the user."""

    # Keep track of last feedback sent to avoid sending duplicates
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)

    # If the feedback value or run ID has changed, send a new feedback record
    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
        # Normalize the feedback value (an index) to a score between 0 and 1
        normalized_score = (feedback + 1) / 5.0

        agent_client: AgentClient = st.session_state.agent_client
        try:
            await agent_client.acreate_feedback(
                run_id=latest_run_id,
                key="human-feedback-stars",
                score=normalized_score,
                kwargs={"comment": "In-line human feedback"},
            )
        except AgentClientError as e:
            st.error(f"Error recording feedback: {e}")
            st.stop()
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon=":material/reviews:")


async def handle_sub_agent_msgs(messages_agen, status, is_new):
    """
    This function segregates agent output into a status container.
    It handles all messages after the initial tool call message
    until it reaches the final AI message.

    Enhanced to support nested multi-agent hierarchies with handoff back messages.

    Args:
        messages_agen: Async generator of messages
        status: the status container for the current agent
        is_new: Whether messages are new or replayed
    """
    nested_popovers = {}

    # looking for the transfer Success tool call message
    first_msg = await anext(messages_agen)
    if is_new:
        st.session_state.messages.append(first_msg)

    # Continue reading until we get an explicit handoff back
    while True:
        # Read next message
        sub_msg = await anext(messages_agen)

        # this should only happen is skip_stream flag is removed
        # if isinstance(sub_msg, str):
        #     continue

        if is_new:
            st.session_state.messages.append(sub_msg)

        # Handle tool results with nested popovers
        if sub_msg.type == "tool" and sub_msg.tool_call_id in nested_popovers:
            popover = nested_popovers[sub_msg.tool_call_id]
            popover.write("**Output:**")
            popover.write(sub_msg.content)
            continue

        # Handle transfer_back_to tool calls - these indicate a sub-agent is returning control
        if (
            hasattr(sub_msg, "tool_calls")
            and sub_msg.tool_calls
            and any("transfer_back_to" in tc.get("name", "") for tc in sub_msg.tool_calls)
        ):
            # Process transfer_back_to tool calls
            for tc in sub_msg.tool_calls:
                if "transfer_back_to" in tc.get("name", ""):
                    # Read the corresponding tool result
                    transfer_result = await anext(messages_agen)
                    if is_new:
                        st.session_state.messages.append(transfer_result)

            # After processing transfer back, we're done with this agent
            if status:
                status.update(state="complete")
            break

        # Display content and tool calls in the same nested status
        if status:
            if sub_msg.content:
                status.write(sub_msg.content)

            if hasattr(sub_msg, "tool_calls") and sub_msg.tool_calls:
                for tc in sub_msg.tool_calls:
                    # Check if this is a nested transfer/delegate
                    if "transfer_to" in tc["name"]:
                        # Create a nested status container for the sub-agent
                        nested_status = status.status(
                            f"""💼 Sub Agent: {tc["name"]}""",
                            state="running" if is_new else "complete",
                            expanded=True,
                        )

                        # Recursively handle sub-agents of this sub-agent
                        await handle_sub_agent_msgs(messages_agen, nested_status, is_new)
                    else:
                        # Regular tool call - create popover
                        popover = status.popover(f"{tc['name']}", icon="🛠️")
                        popover.write(f"**Tool:** {tc['name']}")
                        popover.write("**Input:**")
                        popover.write(tc["args"])
                        # Store the popover reference using the tool call ID
                        nested_popovers[tc["id"]] = popover


if __name__ == "__main__":
    asyncio.run(main())
