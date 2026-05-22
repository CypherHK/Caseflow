import inspect

from streamlit_app import (
    CASEFLOW_AGENT_SELECT_LABEL,
    CASEFLOW_APPROVAL_DISPLAY_TEXT,
    CASEFLOW_APPROVAL_INTERRUPT_DISPLAY,
    CASEFLOW_APPROVE_BUTTON_LABEL,
    CASEFLOW_AUTHOR_CAPTION,
    CASEFLOW_CHAT_PLACEHOLDER,
    CASEFLOW_CONNECTING_MESSAGE,
    CASEFLOW_COPY_SURFACE,
    CASEFLOW_DEMO_GUIDE_LABEL,
    CASEFLOW_MODIFY_BUTTON_LABEL,
    CASEFLOW_PORTFOLIO_TITLE,
    CASEFLOW_REJECT_BUTTON_LABEL,
    CASEFLOW_RUN_STATUS_LABELS,
    CASEFLOW_SHARE_DIALOG_TITLE,
    CASEFLOW_SOURCE_LINK_LABEL,
    CASEFLOW_WELCOME_MESSAGE,
    caseflow_approval_display_text,
    caseflow_approval_payload,
    caseflow_display_ai_content,
    caseflow_display_queue,
    caseflow_display_status,
    caseflow_final_resolution_fallback_html,
    caseflow_group_evidence,
    caseflow_has_pending_approval_state,
    caseflow_is_approval_pending,
    caseflow_localize_trace_text,
    caseflow_node_label_zh,
    caseflow_overview_items,
    caseflow_status_badge,
    caseflow_status_summary,
    draw_caseflow_guide_page,
    is_caseflow_approval_interrupt,
    is_caseflow_result,
    normalize_caseflow_execution,
)


def test_is_caseflow_result_requires_business_keys():
    assert is_caseflow_result(
        {
            "intent": "咨询类",
            "priority": "low",
            "evidence": [],
            "proposed_action_plan": [],
            "draft_response": "hello",
            "needs_human_approval": False,
            "next_step": "send_draft_response",
            "execution_result": {"status": "ready_to_send"},
            "model_used": "openai-compatible",
            "reflection": {"risk": "normal", "reason": "可直接回复"},
        }
    )
    assert not is_caseflow_result({"intent": "咨询类"})


def test_caseflow_status_summary_names_approval_and_ticket_state():
    summary = caseflow_status_summary(
        {
            "intent": "退款 / 升级类",
            "priority": "high",
            "needs_human_approval": True,
            "next_step": "await_human_approval",
            "execution_result": {
                "status": "executed",
                "ticket": {"ticket_id": "TCK-123"},
                "escalation": {"queue": "human_review"},
            },
            "model_used": "openai-compatible",
        }
    )

    assert "退款 / 升级类" in summary
    assert "high" in summary
    assert "需要人工审批" in summary
    assert "已执行" in summary
    assert "executed" not in summary
    assert "openai-compatible" in summary
    assert "TCK-123" in summary
    assert "优先级=" in summary
    assert "模型=" in summary
    assert "执行状态=" in summary
    assert "priority=" not in summary
    assert "model=" not in summary
    assert "execution=" not in summary


def test_caseflow_status_summary_handles_missing_execution_result():
    summary = caseflow_status_summary(
        {
            "intent": "信息不全待补充类",
            "priority": "low",
            "needs_human_approval": False,
            "next_step": "ask_for_missing_information",
            "execution_result": None,
            "model_used": "openai-compatible:fallback",
        }
    )

    assert "信息不全待补充类" in summary
    assert "无需人工审批" in summary
    assert "执行状态=未执行" in summary
    assert normalize_caseflow_execution({"execution_result": None}) == {}


def test_caseflow_status_summary_handles_null_ticket_value():
    summary = caseflow_status_summary(
        {
            "intent": "咨询类",
            "priority": "medium",
            "needs_human_approval": False,
            "next_step": "send_draft_response",
            "execution_result": {"status": "ready_to_send", "ticket": None},
            "model_used": "openai-compatible",
        }
    )

    assert "咨询类" in summary
    assert "无需人工审批" in summary
    assert "执行状态=待发送" in summary
    assert "工单=" not in summary


def test_caseflow_status_values_use_chinese_display_labels_and_badges():
    assert caseflow_display_status("executed") == "已执行"
    assert caseflow_display_status(" Executed ") == "已执行"
    assert caseflow_display_status("await-human-approval") == "等待人工审批"
    assert caseflow_display_status("approved") == "已批准"
    assert caseflow_display_status("modification_requested") == "要求修改"
    assert caseflow_display_status("ready_to_send") == "待发送"
    assert caseflow_display_status("human_review") == "人工复核"
    assert caseflow_display_status(None) == "未执行"

    badge = caseflow_status_badge("执行状态", "<executed>")
    assert "border:1px" in badge
    assert "执行状态" in badge
    assert "&lt;executed&gt;" in badge


def test_caseflow_queue_display_is_separate_from_status_display():
    assert caseflow_display_queue("human_review") == "人工复核队列"
    assert caseflow_display_status("human_review") == "人工复核"


def test_caseflow_approval_controls_only_show_for_pending_execution():
    pending = {
        "needs_human_approval": True,
        "execution_result": {"status": "pending_approval"},
    }
    executed = {
        "needs_human_approval": True,
        "execution_result": {"status": "executed"},
    }

    assert caseflow_is_approval_pending(pending)
    assert not caseflow_is_approval_pending(executed)


def test_caseflow_pending_approval_state_can_come_from_running_event():
    assert caseflow_has_pending_approval_state(
        {},
        [{"node_id": "human_approval", "status": "running"}],
    )
    assert not caseflow_has_pending_approval_state(
        {},
        [{"node_id": "result_persistence", "status": "succeeded"}],
    )


def test_caseflow_overview_items_use_compact_text_values():
    items = caseflow_overview_items(
        {
            "intent": "退款 / 升级类",
            "priority": "high",
            "needs_human_approval": True,
            "model_used": "openai-compatible:fallback",
        }
    )

    assert items == (
        ("意图", "退款 / 升级类"),
        ("优先级", "high"),
        ("审批", "需要审批"),
        ("模型", "openai-compatible:fallback"),
    )


def test_caseflow_group_evidence_groups_legacy_flat_items_without_mutation():
    custom_data = {
        "evidence": [
            "POLICY - 退款审批政策: 涉及退款时必须人工审批。",
            "CASE-1002: 客户重复扣费，客服提交退款审批工单。",
            "Customer profile: Default Customer (standard)",
            "Unmapped source evidence",
        ],
        "customer_profile": {
            "name": "Default Customer",
            "tier": "standard",
            "preferences": {"language": "zh"},
        },
    }
    original_evidence = list(custom_data["evidence"])
    original_profile = dict(custom_data["customer_profile"])

    assert caseflow_group_evidence(custom_data) == (
        ("知识库 / FAQ / SOP / Policy", ("POLICY - 退款审批政策: 涉及退款时必须人工审批。",)),
        ("客户资料", ("Customer profile: Default Customer (standard)",)),
        ("历史案例", ("CASE-1002: 客户重复扣费，客服提交退款审批工单。",)),
        ("其他依据", ("Unmapped source evidence",)),
    )
    assert custom_data["evidence"] == original_evidence
    assert custom_data["customer_profile"] == original_profile


def test_caseflow_group_evidence_places_recent_memory_under_history():
    grouped = caseflow_group_evidence(
        {
            "evidence": [
                "FAQ - 密码重置: 使用登录页入口。",
                "历史案例 / 近期记忆: thread-old / 咨询类 / ready_to_send: 上次已发送密码重置指引。",
            ],
            "customer_profile": {"name": "Acme Retail", "tier": "standard"},
        }
    )

    assert grouped == (
        ("知识库 / FAQ / SOP / Policy", ("FAQ - 密码重置: 使用登录页入口。",)),
        ("客户资料", ("Acme Retail（standard）",)),
        (
            "历史案例",
            ("历史案例 / 近期记忆: thread-old / 咨询类 / ready_to_send: 上次已发送密码重置指引。",),
        ),
    )


def test_caseflow_group_evidence_uses_customer_profile_when_flat_item_is_absent():
    grouped = caseflow_group_evidence(
        {
            "evidence": ["FAQ - 订单查询: 可直接查询物流状态。"],
            "customer_profile": {"name": "Default Customer", "tier": "standard"},
        }
    )

    assert grouped == (
        ("知识库 / FAQ / SOP / Policy", ("FAQ - 订单查询: 可直接查询物流状态。",)),
        ("客户资料", ("Default Customer（standard）",)),
    )


def test_caseflow_group_evidence_supports_structured_and_grouped_future_shapes():
    grouped = caseflow_group_evidence(
        {
            "grouped_evidence": {
                "history": [{"case_id": "CASE-9001", "summary": "历史退款审批已通过。"}],
            },
            "evidence": [
                {"source": "sop", "title": "升级处理", "summary": "高风险升级需审批。"},
                {"case_id": "CASE-1003", "summary": "投诉类工单转主管复核。"},
                {"source": "unknown", "title": "External note", "summary": "Not classified."},
            ],
        }
    )

    assert grouped == (
        ("知识库 / FAQ / SOP / Policy", ("SOP - 升级处理: 高风险升级需审批。",)),
        (
            "历史案例",
            (
                "CASE-9001: 历史退款审批已通过。",
                "CASE-1003: 投诉类工单转主管复核。",
            ),
        ),
        ("其他依据", ("External note: Not classified.",)),
    )


def test_caseflow_group_evidence_handles_missing_or_malformed_evidence_safely():
    assert caseflow_group_evidence({}) == ()
    assert caseflow_group_evidence({"evidence": None}) == ()
    assert caseflow_group_evidence({"evidence": [None, "", 42]}) == (
        ("其他依据", ("42",)),
    )


def test_caseflow_approval_interrupt_is_detected_from_service_message():
    assert is_caseflow_approval_interrupt("需要人工审批后才能执行：\n- intent: 退款 / 升级类")
    assert not is_caseflow_approval_interrupt("CaseFlow 处理结果")


def test_caseflow_approval_interrupt_uses_safe_display_copy():
    raw_interrupt = "需要人工审批后才能执行：\n请回复 approve / reject，或给出修改意见。"

    assert caseflow_display_ai_content(raw_interrupt) == CASEFLOW_APPROVAL_INTERRUPT_DISPLAY
    assert "approve" not in CASEFLOW_APPROVAL_INTERRUPT_DISPLAY
    assert "或给出修改意见" not in CASEFLOW_APPROVAL_INTERRUPT_DISPLAY


def test_caseflow_visible_copy_is_product_owned_and_localized():
    copy = "\n".join(CASEFLOW_COPY_SURFACE)

    assert "Yucheng" in copy
    assert "https://github.com/CypherHK/Caseflow" in copy
    assert "客服" in CASEFLOW_WELCOME_MESSAGE
    assert "客户问题" in CASEFLOW_CHAT_PLACEHOLDER
    assert CASEFLOW_APPROVE_BUTTON_LABEL == "批准执行"
    assert CASEFLOW_REJECT_BUTTON_LABEL == "拒绝执行"
    assert CASEFLOW_APPROVAL_DISPLAY_TEXT == {
        "approve": "已选择：批准执行",
        "reject": "已选择：拒绝执行",
        "modify": "已选择：要求修改",
    }
    assert CASEFLOW_MODIFY_BUTTON_LABEL == "要求修改"
    assert CASEFLOW_PORTFOLIO_TITLE == "可审计的电商售后工作流 Agent"
    assert CASEFLOW_DEMO_GUIDE_LABEL == "演示说明"
    assert CASEFLOW_RUN_STATUS_LABELS["awaiting_approval"] == "等待审批"


def test_caseflow_portfolio_node_labels_and_trace_copy_are_chinese():
    assert caseflow_node_label_zh("risk_review") == "风险复核"
    assert caseflow_node_label_zh("human_approval") == "人工审批"
    assert caseflow_localize_trace_text(
        "Captured issue for workflow. Current intent: 咨询类, priority: low."
    ) == "已进入工作流。当前意图：咨询类，优先级：低。"
    assert "未知" not in caseflow_localize_trace_text(
        "Captured issue for workflow. Current intent: unknown, priority: unknown."
    )
    assert "Waiting for" not in caseflow_localize_trace_text(
        "Waiting for human approve / reject / modify."
    )
    assert caseflow_localize_trace_text("approve") == "已批准执行。"
    assert caseflow_localize_trace_text("reject") == "已拒绝执行。"
    assert caseflow_localize_trace_text("modify") == "已要求修改。"


def test_caseflow_final_resolution_fallback_uses_events_without_unknown_copy():
    html = caseflow_final_resolution_fallback_html(
        [
            {
                "case_id": "demo-high-compensation",
                "node_id": "draft_response",
                "status": "succeeded",
                "output_summary": "您好，退款或升级类事项需要人工审批。",
            },
            {
                "case_id": "demo-high-compensation",
                "node_id": "human_approval",
                "status": "succeeded",
            },
            {
                "case_id": "demo-high-compensation",
                "node_id": "ticket_execution",
                "status": "succeeded",
            },
            {
                "case_id": "demo-high-compensation",
                "node_id": "result_persistence",
                "status": "succeeded",
            },
        ]
    )

    assert "退款 / 升级类" in html
    assert "需要审批" in html
    assert "已执行" in html
    assert "已创建（模拟）" in html
    assert "unknown" not in html
    assert "未知" not in html


def test_caseflow_guide_page_uses_public_product_copy():
    guide_source = inspect.getsource(draw_caseflow_guide_page)

    assert "CaseFlow Agent 使用指南" in guide_source
    assert "系统说明" in guide_source
    assert "view=guide" not in guide_source


def test_caseflow_structured_approval_payload_keeps_json_internal_and_copy_chinese():
    payload = caseflow_approval_payload(
        "modify",
        reason="证据不足",
        modification_notes="请先补充退款政策依据。",
    )

    assert '"decision":"modify"' in payload
    display_text = caseflow_approval_display_text(payload)
    assert display_text == "已选择：要求修改（原因：证据不足；修改要求：请先补充退款政策依据。）"
    assert "decision" not in display_text
    assert "modify" not in display_text

    assert caseflow_approval_display_text("approve") == "已选择：批准执行"


def test_caseflow_copy_surface_includes_live_sidebar_and_dialog_copy():
    copy_surface = set(CASEFLOW_COPY_SURFACE)

    assert CASEFLOW_SHARE_DIALOG_TITLE in copy_surface
    assert CASEFLOW_SOURCE_LINK_LABEL in copy_surface
    assert CASEFLOW_AUTHOR_CAPTION in copy_surface
    assert CASEFLOW_CONNECTING_MESSAGE in copy_surface
    assert CASEFLOW_AGENT_SELECT_LABEL in copy_surface


def test_caseflow_visible_copy_excludes_upstream_template_identity():
    copy = "\n".join(CASEFLOW_COPY_SURFACE)
    forbidden = [
        "agent-service-toolkit",
        "JoshuaC215",
        "Joshua",
        "Full toolkit for running an AI agent service",
        "View the source code",
        "github.com/JoshuaC215/agent-service-toolkit",
        "agent_architecture.png",
        "App hosted on Streamlit Cloud",
        "Made with",
        "in Oakland",
    ]

    for item in forbidden:
        assert item not in copy
