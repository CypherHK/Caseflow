# CaseFlow Agent

CaseFlow Agent is a business-oriented customer support and ticket collaboration agent prototype built with LangGraph, FastAPI, AgentClient, Streamlit, and LangGraph-compatible persistence.

The goal is not to create a generic chatbot. The goal is to demonstrate how an AI agent can move a customer support case through classification, evidence retrieval, planning, risk reflection, human approval, mock execution, and case-note persistence.

## Product Scenario

CaseFlow Agent targets internal support, operations, and after-sales teams. A support operator enters a customer issue, and the agent returns a structured handling package instead of a free-form answer.

The MVP supports:

- FAQ and policy questions
- Complaints
- Order or service exceptions
- Refund and escalation requests
- Missing-information cases

## Agent Graph

The current graph is implemented in `src/agents/caseflow_agent.py`.

Flow:

1. `intake_and_classify`
2. `retrieve_evidence`
3. `plan_actions`
4. `draft_resolution`
5. `reflect_and_risk_check`
6. `request_human_approval_if_needed`
7. `execute_action`
8. `finalize_and_persist`

High-risk cases, such as refund, complaint, or escalation requests, trigger LangGraph `interrupt()` before execution. The operator can resume the same thread by approving or rejecting the action.

## Structured Output

CaseFlow messages include business data in `ChatMessage.custom_data`:

- `intent`
- `priority`
- `evidence`
- `proposed_action_plan`
- `draft_response`
- `needs_human_approval`
- `next_step`
- `execution_result`
- `model_used`
- `reflection`

This keeps the existing chat API compatible while allowing the Streamlit UI to render a workbench with intent, priority, evidence, action plan, model used, reflection, approval, and ticket status panels.

## LLM-driven Nodes

The current CaseFlow graph uses the configured project model through `core.get_model()`. For the local demo, the target model is Gemini through the OpenAI-compatible API:

```env
USE_FAKE_MODEL=false
DEFAULT_MODEL=openai-compatible
COMPATIBLE_MODEL=gemini-3-flash-preview
COMPATIBLE_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
COMPATIBLE_API_KEY=your_gemini_api_key
```

The LLM is used for:

- intent and priority analysis
- action plan and reply drafting
- reflection and risk assessment

Each LLM response is parsed as JSON and validated by Pydantic models in `src/agents/caseflow_agent.py`:

- `CaseAnalysis`
- `DraftResolution`
- `RiskReflection`

If validation fails or the model call errors, the agent falls back to deterministic rule functions so the demo and tests remain stable.

## Tools and Data

Mock tools live in `src/agents/caseflow_tools.py`.

Tools:

- `search_kb`
- `search_case_history`
- `get_customer_profile`
- `create_ticket`
- `escalate_ticket`
- `save_case_note`

Local data lives in `data/caseflow/`:

- `kb.json`
- `case_history.json`
- `customers.json`
- `eval_cases.json`

The MVP intentionally avoids real CRM, refund, or ticket-system integrations. Those actions are represented by deterministic local mock records.

## Memory and Human-in-the-loop

The project reuses the existing service memory initialization in `src/service/service.py` and `src/memory/`.

- Short-term thread memory is handled through LangGraph checkpointers.
- Case handling state is carried in the graph state and surfaced in `custom_data`.
- High-risk execution uses `interrupt()` so the workflow can pause and resume.

## Model Configuration

The project still supports the fake model for automated tests or no-key environments:

```env
USE_FAKE_MODEL=true
```

Daily local demos should use Gemini through the existing OpenAI-compatible path:

```env
USE_FAKE_MODEL=false
DEFAULT_MODEL=openai-compatible
COMPATIBLE_MODEL=gemini-3-flash-preview
COMPATIBLE_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
COMPATIBLE_API_KEY=your_gemini_api_key
```

Do not commit real API keys.

## Evaluation

The lightweight evaluation set is `data/caseflow/eval_cases.json`.

It covers:

- FAQ question
- Complaint
- Missing information
- Exception with enough details
- Refund request
- Escalation request
- Approval-required compensation case

Evaluation dimensions:

- Intent correctness
- Priority correctness
- Evidence relevance
- Whether approval is triggered when required
- Whether final response is grounded in evidence
- Whether mock ticket action is correct

Run deterministic evaluation:

```sh
./.venv/bin/python scripts/evaluate_caseflow.py
```

Run optional live Gemini evaluation:

```sh
./.venv/bin/python scripts/evaluate_caseflow.py --live-llm
```

## Demo Script

Suggested local demo:

1. Start the service from the repository root:

   ```sh
   ./.venv/bin/python src/run_service.py
   ```

2. Start Streamlit:

   ```sh
   ./.venv/bin/python -m streamlit run src/streamlit_app.py --server.port 8501 --server.address 127.0.0.1
   ```

3. Open `http://127.0.0.1:8501`.
4. Try a low-risk FAQ:

   ```text
   我忘记密码了，怎么重置？
   ```

5. Try a high-risk refund case:

   ```text
   客户投诉扣费错误，要求退款并升级给主管
   ```

6. Approve or reject the pending action in the workbench.

## Known Limits

- Retrieval uses local JSON mock data, not a production vector database.
- Ticket creation and escalation are deterministic mock actions.
- The Streamlit workbench is functional but not yet a polished enterprise UI.
- Real Gemini execution depends on the user providing a valid API key.
- LLM structured output has deterministic fallback, so a degraded model response will still produce a valid CaseFlow result.
