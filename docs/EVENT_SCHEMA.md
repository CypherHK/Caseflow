# CaseFlow Workflow Event Schema

## 目的

Workflow event 是后端向前端同步 CaseFlow 节点状态的稳定 typed contract。它服务于 portfolio demo UI：workflow graph、Node Inspector、Execution Timeline、Human Approval 面板和 Final Resolution。

事件只展示可审计业务依据，不展示模型隐藏推理链。

## 产生方式

当前实现有两层：

1. **Streaming projection**
   - FastAPI `/caseflow-agent/stream` 在 `demo_mode=true` 或存在 `case_id` 时，从 LangGraph `updates` stream 投影 `workflow_event` SSE。
   - 这样前端可以在节点开始、完成、阻断或等待审批时更新状态。

2. **Final payload projection**
   - `finalize_and_persist` 会把完整 `workflow_nodes` 和 `workflow_events` 写入 `AIMessage.additional_kwargs["custom_data"]`。
   - 非 streaming 或页面刷新后，前端仍能从最终 payload 重建完整执行路径。

## SSE 形态

```json
{
  "type": "workflow_event",
  "content": {
    "run_id": "019e...",
    "case_id": "demo-high-compensation",
    "node_id": "risk_review",
    "node_label": "Risk Review",
    "status": "blocked",
    "timestamp": "2026-05-22T10:40:06.000000+00:00",
    "latency_ms": null,
    "input_summary": "退款 / 升级类 / approval_required=True",
    "output_summary": "强制人工审批：涉及退款、补偿、账单升级或主管介入，必须强制人工审批。",
    "evidence": [],
    "risk_flags": [
      {
        "type": "compensation",
        "severity": "high",
        "reason": "客户要求赔偿或补偿，不能由模型直接承诺。"
      }
    ],
    "approval_required": true,
    "error": null
  }
}
```

## 字段说明

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `run_id` | string | yes | FastAPI/LangGraph run id。 |
| `case_id` | string | no | Portfolio preset case id；普通聊天可为空。 |
| `node_id` | string | yes | 前端稳定节点 id。 |
| `node_label` | string | yes | 前端显示名称。 |
| `status` | enum | yes | 节点状态。 |
| `timestamp` | ISO string | yes | 事件产生时间。 |
| `latency_ms` | integer/null | no | 节点耗时；当前 projection 可为空。 |
| `input_summary` | string | no | 产品可解释的输入摘要。 |
| `output_summary` | string | no | 产品可解释的输出摘要。 |
| `evidence` | array | yes | Evidence Retrieval 节点常用。 |
| `risk_flags` | array | yes | Risk Review / Human Approval 节点常用。 |
| `approval_required` | boolean | yes | 当前 result 是否需要人工审批。 |
| `error` | string/null | no | 失败摘要或 fallback 说明。 |

## Node Status

| Status | Meaning |
|---|---|
| `pending` | 尚未执行。 |
| `running` | 正在执行，或 LangGraph interrupt 正在等待人工审批。 |
| `succeeded` | 节点已完成。 |
| `blocked` | 节点被 guardrail、审批或 reject/modify 阻断。 |
| `failed` | 节点失败。 |
| `skipped` | 当前分支跳过该节点。 |

## Canonical Nodes

| `node_id` | Label | Internal graph source |
|---|---|---|
| `intent_recognition` | Intent Recognition | `intake_and_classify` / `plan_actions` output |
| `evidence_retrieval` | Evidence Retrieval | `retrieve_evidence` |
| `action_planning` | Action Planning | `plan_actions` |
| `draft_response` | Draft Response | `draft_resolution` |
| `risk_review` | Risk Review | `reflect_and_risk_check` |
| `human_approval` | Human Approval | `request_human_approval_if_needed` |
| `ticket_execution` | Ticket Execution | `execute_action` |
| `result_persistence` | Result Persistence | `finalize_and_persist` |

## Evidence Item

```json
{
  "source": "policy",
  "title": "退款审批政策",
  "snippet": "涉及退款、账单减免或补偿承诺时，必须经过人工审批后才能执行。"
}
```

`source` 常见值：

- `faq`
- `sop`
- `policy`
- `customer_history`
- `customer`
- `case_memory`

## Risk Flag

```json
{
  "type": "refund",
  "severity": "high",
  "reason": "涉及退款、退费、扣费或资金类动作。"
}
```

常见 `type`：

- `refund`
- `compensation`
- `complaint`
- `escalation`
- `policy_guardrail`
- `human_in_the_loop`

## 前后端同步

1. Streamlit 点击 preset case 的 **Run**。
2. 前端调用 `AgentClient.astream(..., agent_config={"demo_mode": true, "case_id": "..."})`。
3. FastAPI 读取 LangGraph `updates`，投影为 `workflow_event` SSE。
4. `AgentClient._parse_stream_line()` 将 `workflow_event` 作为 dict 传给 Streamlit。
5. Streamlit 写入 `st.session_state.caseflow_workflow_events`。
6. Workflow graph、Node Inspector 和 Execution Timeline 读取最新事件刷新。
7. 最终 `custom_data.workflow_events` 覆盖 session 中的临时事件，作为完整 replay source。

## 当前限制

- Streaming event 是 service projection，不是每个 LangGraph node 内部手写埋点。
- `latency_ms` 目前保留为空，后续可以在 service 层按 node 计时补齐。
- `Intent Recognition` 展示节点是产品化抽象；当前内部真实意图判断主要在 `plan_actions` 完成。
- `Modify` 当前记录修改要求并阻断执行，不会自动二次生成新草稿。
