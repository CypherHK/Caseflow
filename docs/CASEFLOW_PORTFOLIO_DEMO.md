# CaseFlow Portfolio Demo

## 一句话定位

CaseFlow Agent 是一个面向电商售后、客服工单和投诉升级场景的可审计 workflow agent demo，用来展示 AI 如何在证据、规则、人工审批和 mock 工单执行之间完成受控流转。

## 解决的问题

普通客服 chatbot 往往只展示“回答得像不像”。CaseFlow 展示的是“业务处理是否可控”：

- 用户问题如何进入系统；
- 系统用了哪些政策、订单、历史案例和客户上下文；
- Agent 如何形成行动计划和客户回复草稿；
- 哪些退款、补偿、投诉、升级动作会触发 guardrail；
- 人工审批如何接管高风险动作；
- 最终工单结果和处理记录如何被保留。

## 为什么不是普通 Chatbot

CaseFlow 的核心卖点不是自由聊天，而是显式 workflow boundary。现有 LangGraph 仍保留 8 个节点：

1. `intake_and_classify`
2. `retrieve_evidence`
3. `plan_actions`
4. `draft_resolution`
5. `reflect_and_risk_check`
6. `request_human_approval_if_needed`
7. `execute_action`
8. `finalize_and_persist`

Portfolio UI 把这些内部节点映射成更容易理解的产品节点：

| 展示节点 | 内部来源 | 产品说明 |
|---|---|---|
| Intent Recognition | `intake_and_classify` / `plan_actions` | 捕获用户问题，并在 planning 阶段产出意图、优先级和审批判断。 |
| Evidence Retrieval | `retrieve_evidence` | 检索本地政策、SOP、历史案例、客户资料和近期 case memory。 |
| Action Planning | `plan_actions` | 形成业务行动计划，不直接执行高风险动作。 |
| Draft Response | `draft_resolution` | 生成面向客户的回复草稿。 |
| Risk Review | `reflect_and_risk_check` | 用 deterministic guardrail 复核退款、补偿、投诉和升级风险。 |
| Human Approval | `request_human_approval_if_needed` | 通过 LangGraph `interrupt()` 暂停，等待 approve / reject / modify。 |
| Ticket Execution | `execute_action` | 审批后创建或升级 mock 工单；拒绝和修改不会执行。 |
| Result Persistence | `finalize_and_persist` | 输出稳定 `custom_data`，写入 compact case summary。 |

## Guardrail / HITL 设计

当前 guardrail 是规则优先：

- `退款 / 升级类` 和 `投诉类` 必须人工审批；
- `high` priority 必须人工审批；
- 命中 `退款`、`退费`、`扣费`、`赔偿`、`补偿`、`升级`、`主管`、`投诉` 等关键词会强制审批；
- LLM 输出即使误判为低风险，也会被 deterministic policy 覆盖。

Human Approval 支持三种动作：

- **Approve**：继续进入 mock ticket execution；
- **Reject**：记录拒绝原因，不创建批准后的工单动作；
- **Modify**：记录修改要求，不创建批准后的工单动作。

这不是 UI-only 状态。真实高风险路径通过 LangGraph `interrupt()` 暂停，并用同一 `thread_id` resume。

## Demo Path

建议用 2 到 3 个预设案例展示，不要全讲完。右上角 **演示说明** 是公开使用指南，只解释系统定位、操作步骤、风险控制/人工介入能力和原型边界，不写内部展示策略。

1. **普通商品咨询 / 售后政策咨询**
   - 目标：展示低风险咨询如何检索政策、生成草稿、跳过人工审批。
   - 看点：Evidence Retrieval、Draft Response、Human Approval skipped。

2. **高金额补偿诉求**
   - 目标：展示高风险资金承诺如何被 Risk Review 阻断。
   - 看点：Risk flags、approval_required、Ticket Execution pending。

3. **投诉升级**
   - 目标：展示主管介入或投诉升级如何进入 HITL。
   - 看点：Human Approval 的 Approve / Reject / Modify。

备用 case：

- **退款申请**：适合讲订单和退款政策证据；
- **信息不全**：适合讲不充分信息下不提前建单、不承诺退款。

## 电商售后迁移说明

当前数据是本地 mock JSON，不连接真实电商系统。但产品结构可以迁移到常见售后系统：

- `search_kb()` 可替换为政策库、FAQ、SOP 或向量检索；
- `search_case_history()` 可替换为 CRM / 工单系统历史记录；
- `get_customer_profile()` 可替换为会员等级、SLA、黑白名单和历史风险；
- `create_ticket()` / `escalate_ticket()` 可替换为真实工单、退款、补偿或主管队列 API；
- `Risk Review` 可接入金额阈值、品类规则、欺诈风险、地区政策和人工审批矩阵。

对外说明时不要说它已经接入真实 CRM 或退款系统。更准确的说法是：CaseFlow 展示了售后 agent 的 workflow product shape，真实落地时需要替换工具适配器、权限、审计日志和线上评估。
