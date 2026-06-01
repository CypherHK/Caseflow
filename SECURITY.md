# Security Policy

CaseFlow is an early-stage reference implementation and demo workbench. It uses local mock data and deterministic mock ticket actions by default.

## Supported scope

Security reports are most useful when they affect:

- secret handling or accidental credential exposure;
- unsafe demo data or non-synthetic customer/order information;
- guardrail bypasses for refund, compensation, escalation, or complaint handling;
- HITL approval bypasses or misleading workflow traces;
- dependency, CI, or local setup issues that could expose user environments.

## Reporting a vulnerability

Do not include secrets, customer data, private business processes, or exploit details in a public issue.

Preferred reporting path:

1. Use GitHub private vulnerability reporting if it is enabled for this repository.
2. If private reporting is not available, open a minimal public issue that describes the affected area without sensitive details and ask for a private coordination path.

Maintainers should acknowledge credible reports, reproduce the issue in mock mode where possible, and publish a fix or mitigation note before disclosing sensitive details.

## Demo safety boundaries

- `.env` files and real API keys must not be committed.
- Demo screenshots and JSON data must remain synthetic.
- `create_ticket()`, `escalate_ticket()`, and `save_case_note()` are deterministic mock actions.
- The project must not call real CRM, refund, compensation, payment, or ticketing systems without an explicit adapter and a separate security review.
- The UI should expose auditable workflow traces, not hidden model reasoning chains.

## Local checks

Before publishing changes, run:

```sh
git status --short
rg -n "API_KEY|SECRET|TOKEN|PASSWORD" README.md docs src data tests .gitignore
uv run python -m ruff check src tests scripts
uv run python -m pytest -q
USE_FAKE_MODEL=true DEFAULT_MODEL=fake OPENAI_API_KEY=test-openai-key uv run python scripts/evaluate_caseflow.py
```
