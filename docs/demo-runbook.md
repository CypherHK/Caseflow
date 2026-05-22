# CaseFlow Demo Runbook

This runbook is for a portfolio-grade local demo of CaseFlow Agent. It shows how the project handles FAQ, missing information, abnormal orders, high-risk approval, mock ticket execution, and evaluation reporting. It is not a production CRM, refund, billing, or ticketing integration.

Run all commands from the repository root.

## Prerequisites

- Python 3.11 or newer.
- `uv`.
- A local `.env` file. Do not commit real API keys.
- Gemini-compatible credentials only if you want the optional live LLM demo.
- Local network access for the optional live validation path.

Install dependencies:

```sh
uv sync --frozen
```

For reliable local service and UI connectivity, set proxy bypass values before starting the app:

```sh
export NO_PROXY=127.0.0.1,localhost,0.0.0.0
export no_proxy=127.0.0.1,localhost,0.0.0.0
```

For a deterministic no-key demo, configure the fake model explicitly:

```env
USE_FAKE_MODEL=true
DEFAULT_MODEL=fake
```

Leaving live credentials unset is not enough by itself. A fresh checkout with no active
provider or fake model cannot start the service.

For a live Gemini demo through the OpenAI-compatible endpoint, configure:

```env
USE_FAKE_MODEL=false
DEFAULT_MODEL=openai-compatible
COMPATIBLE_MODEL=gemini-3-flash-preview
COMPATIBLE_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
COMPATIBLE_API_KEY=your_gemini_api_key
```

## Start The Demo

Start the FastAPI service:

```sh
uv run python src/run_service.py
```

Start the Streamlit workbench in another terminal:

```sh
AGENT_URL=http://127.0.0.1:8080 uv run python -m streamlit run src/streamlit_app.py --server.port 8501 --server.address 127.0.0.1
```

Use the `python -m streamlit` form in this checkout because it avoids stale or missing console-script launcher issues.

Open the workbench:

- `http://127.0.0.1:8501`

Check service health and metadata:

- `http://127.0.0.1:8080/health`
- `http://127.0.0.1:8080/info`
- `http://127.0.0.1:8080/docs`
- `http://127.0.0.1:8080/redoc`

`/info` should show `caseflow-agent` as the default business agent. In a live Gemini demo, model metadata should show `openai-compatible` when configured correctly.

## Portfolio Demo Mode

The Streamlit workbench now includes a portfolio demo surface when `caseflow-agent` is selected.

Use the preset selector on the left:

1. 普通商品咨询 / 售后政策咨询
2. 退款申请
3. 高金额补偿诉求
4. 投诉升级
5. 信息不全

Click **Run** to send the selected case through the backend. The workbench passes `demo_mode=true` and the preset `case_id`, so the FastAPI stream emits workflow events for the graph, node inspector, timeline, and approval controls.

For the clearest guided demo path, run:

1. **普通商品咨询 / 售后政策咨询** to show low-risk evidence-grounded drafting.
2. **高金额补偿诉求** to show Risk Review blocking direct execution.
3. **投诉升级** or **退款申请** to show Human Approval and same-thread resume.

The older chat area still works below the portfolio surface. It is useful for showing that the UI is not a static mock, but the primary demo should be the workflow graph and Node Inspector.

## Demo Script

Use one thread for a short end-to-end walkthrough, or use a fresh thread per scenario if you want cleaner before-and-after states.

### 1. Low-Risk FAQ

Prompt:

```text
我忘记密码了，怎么重置？
```

Expected signal:

- Intent: `咨询类`.
- Priority: `low`.
- Approval: not required.
- Workbench: FAQ/password evidence is visible, the action plan is reply-oriented, and the draft is ready to send.

Presenter note: point out that the agent grounds a low-risk answer in local FAQ evidence and does not create a ticket or request human approval.

### 2. Missing Information

Prompt:

```text
我的订单有问题
```

Expected signal:

- Intent: `信息不全待补充类`.
- Priority: `medium`.
- Approval: not required.
- Workbench: the draft asks for missing details such as order ID, screenshot, occurrence time, or error code.

Presenter note: point out that the agent does not prematurely create a ticket or promise a refund when the request is underspecified.

### 3. Abnormal Order And Mock Ticket Path

Prompt:

```text
订单号 #A100 支付成功但服务无法使用，页面报错 E401，有截图
```

Expected signal:

- Intent: `异常类`.
- Priority: `medium`.
- Approval: not required.
- Workbench: SOP/order evidence is visible, the action plan is ticket-oriented, and execution can create a mock ticket.

Presenter note: point out that this is still a local mock ticket path. It demonstrates workflow shape, not a real external ticketing integration.

### 4. Complaint HITL Approval

Prompt:

```text
你们处理太慢了，我要投诉
```

Expected signal:

- Intent: `投诉类`.
- Approval: required.
- Workbench: approval controls are visible and execution is paused before ticket escalation.

Presenter approve path:

1. Click **批准执行** or send legacy `approve`.
2. Keep the same thread so LangGraph can resume from the interrupt.
3. Confirm the workbench shows approved execution and mock ticket/escalation output.

Presenter reject path:

1. Click **拒绝执行** with a short reason, or send legacy `reject`.
2. Keep the same thread.
3. Confirm execution is marked rejected and no approved ticket action is claimed.

Presenter modify path:

1. Click **要求修改** and enter concise modification notes.
2. Confirm execution is marked as modification requested and no mock ticket/escalation is created.

Presenter note: this is the core human-in-the-loop demo. The model can draft and recommend, but high-risk complaint execution waits for human approval.

### 5. Refund Or Escalation Approval

Prompt:

```text
客户被重复扣费，要求马上退款
```

Expected signal:

- Intent: `退款 / 升级类`.
- Priority: `high`.
- Approval: required.
- Workbench: policy evidence, approval reason, and pending execution are visible.

Presenter approve path:

1. Click **批准执行** or send legacy `approve` on the same thread.
2. Confirm the resumed result includes mock ticket creation or escalation status.
3. State clearly that refund execution is not real; this project only demonstrates the guarded workflow.

Presenter reject path:

1. Click **拒绝执行** with a short reason, or send legacy `reject` on the same thread.
2. Confirm the result records rejection and does not claim an approved refund or escalation.

Presenter modify path:

1. Click **要求修改** and enter what must change before approval.
2. Confirm the result records `modification_requested` and does not create or escalate a mock ticket.

## Evaluation Checks

Run deterministic evaluation first:

```sh
uv run python scripts/evaluate_caseflow.py
```

Expected deterministic report shape:

- `total_cases`
- `intent_accuracy`
- `approval_accuracy`
- `evidence_hit_rate`
- `draft_groundedness_rate`
- `priority_accuracy`
- `mode: deterministic`

Run optional live evaluation when credentials and network access are available:

```sh
uv run python scripts/evaluate_caseflow.py --live-llm --examples-limit 5 --output reports/live-llm-report.json
```

Read the saved report before claiming the live path worked:

- `fallback_rate`: fraction of cases that used deterministic fallback.
- `models_used`: model labels observed across the run.
- `live_model_verification.verified`: true only when at least one case used non-fallback model output.
- `qualitative_examples`: bounded case examples for inspection.
- `mismatches`: fields where live output differed from deterministic expectations.

If the report says `fallback-only` or `live_model_verification.verified` is false, the run was transparent but not proof of a working live Gemini path. Check credentials, model name, endpoint, JSON schema compliance, and local network access.

Latest known Story 5.2 live validation after network access was enabled:

```text
fallback_rate: 0.0
models_used: ["openai-compatible"]
non_fallback_cases: 14
live_model_verification.verified: true
```

Live metrics may be lower than deterministic metrics. That is acceptable for this demo because live evaluation inspects model behavior; it is not forced to match deterministic expectations.

## Troubleshooting

- If Streamlit cannot reach the backend, verify `AGENT_URL=http://127.0.0.1:8080` and the proxy bypass exports.
- If `/health` fails, restart the FastAPI service from the repository root.
- If `/info` does not show `caseflow-agent`, check agent registration before demoing.
- If live eval falls back, do not present it as model-driven validation. Present the deterministic demo and explain the live report diagnostics honestly.
- If approval resume does not work, use the same conversation thread and choose the UI approval buttons. Legacy text `approve`, `reject`, `批准`, and `拒绝` still works.
- Free-form approval text that is not an explicit approve/reject is treated as a modification request, not as approval.

## Scope Boundaries

- No real refunds, compensation, billing edits, or CRM actions are executed.
- Tickets and escalations are deterministic local mock outputs.
- Approval modification records notes and blocks execution, but it does not yet regenerate a revised draft automatically.
- The runbook should remain text-first and reproducible without screenshots or secret-bearing artifacts.
