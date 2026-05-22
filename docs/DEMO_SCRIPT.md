# CaseFlow 2 分钟产品演示稿

## 15 秒：项目定位

CaseFlow Agent 是一个售后客服 workflow agent demo。它不是普通 chatbot，而是把用户问题放进一条可审计流程：识别意图、检索证据、规划动作、生成回复、复核风险、人工审批、执行模拟工单、保存结果。

## 30 秒：用户问题与业务痛点

电商售后里，退款、补偿、投诉升级都不能只靠模型直接回复。比如客户说“发错货导致活动损失，要求补偿 3000 元”，系统需要先看订单、政策、历史记录，再判断是否触发审批。

## 45 秒：Workflow 如何执行

在左侧选择 preset case 后，系统把用户输入送进 LangGraph workflow。中间可以看到完整节点：Intent Recognition、Evidence Retrieval、Action Planning、Draft Response、Risk Review、Human Approval、Ticket Execution、Result Persistence。右侧 Node Inspector 展示当前节点的输入摘要、输出摘要、证据引用和业务判断。这里展示的是可审计 trace，不是模型隐藏推理链。

## 20 秒：Risk Review 与 Human Approval

高金额补偿 case 会命中补偿和资金风险。Risk Review 会把节点标为 blocked，并要求 Human Approval。操作员可以 Approve、Reject 或 Modify。只有 Approve 后才进入 Ticket Execution；Reject 和 Modify 只记录决定，不会创建批准后的工单动作。

## 10 秒：Eval / 下一步迭代

当前有 deterministic eval 覆盖 14 个 case，检查 intent、approval、evidence、draft grounding 和 priority。继续产品化时，可以替换真实订单/工单适配器，增加金额阈值、权限、审计日志和线上行为评估。
