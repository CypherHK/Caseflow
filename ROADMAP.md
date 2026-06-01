# CaseFlow Roadmap

CaseFlow is maintained as an early-stage open-source reference implementation for safer business-facing AI agent workflows. The roadmap is intentionally focused on reproducibility, auditability, and maintainer workflows.

## Near term

- Add and maintain CI for lint, tests, and deterministic eval.
- Expand eval cases for multilingual support, refund, compensation, and complaint escalation scenarios.
- Add a threat model for HITL approval, guardrail bypass, and mock ticket actions.
- Add a Docker quickstart for one-command local demo setup.
- Add a release checklist for demo snapshots and public application reviews.
- Improve event latency measurement in the workflow event schema.
- Add stronger workflow event contract tests across FastAPI, Streamlit, and LangGraph.

## Maintainer automation

- Use Codex-assisted PR review for workflow contract, guardrail, and docs changes.
- Generate regression tests from issue reports and eval failures.
- Keep issue triage focused on reproducible mock-mode cases.
- Track release readiness through lint, tests, eval, docs, and screenshot freshness.

## Later

- Add optional production-style adapters behind explicit interfaces.
- Add role-based approval policies and audit-log persistence.
- Add observability examples for workflow events and eval drift.
- Improve local demo packaging for workshops and OSS onboarding.

## Out of scope for this reference implementation

- Real customer data.
- Real order, refund, compensation, CRM, payment, or ticket-system execution.
- Private company workflows or confidential operational policies.
