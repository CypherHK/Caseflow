# Contributing to CaseFlow

CaseFlow is an open-source reference implementation for auditable, human-in-the-loop AI agent workflows in customer support and after-sales operations.

The most useful contributions improve the maintainer workflow and the reproducibility of the agent workflow:

- LangGraph workflow contracts and typed workflow events
- evidence retrieval, guardrail, and HITL approval behavior
- deterministic eval cases and regression checks
- local mock data and safe demo scenarios
- documentation, runbooks, CI, and release readiness

## Before you start

Open or comment on an issue before starting larger changes. This keeps the scope reviewable and avoids private product assumptions.

Use local mock mode by default:

```sh
USE_FAKE_MODEL=true
DEFAULT_MODEL=fake
```

Do not add real customer data, real orders, production credentials, confidential business processes, or live CRM/refund/ticket integrations.

## Development setup

This project uses `uv` for Python package management.

```sh
uv sync --frozen
```

Run the standard validation commands before opening a pull request:

```sh
uv run python -m ruff check src tests scripts
uv run python -m pytest -q
USE_FAKE_MODEL=true DEFAULT_MODEL=fake OPENAI_API_KEY=test-openai-key uv run python scripts/evaluate_caseflow.py
```

## Pull request checklist

- The change is covered by tests or deterministic eval cases where appropriate.
- Workflow event schema changes are reflected in `docs/EVENT_SCHEMA.md`.
- New demo data is synthetic and safe to publish.
- High-risk actions remain behind guardrails or HITL approval.
- Documentation and runbooks are updated when behavior changes.
- The standard validation commands pass locally.

## Maintainer workflow

Maintainers should prioritize:

- issue triage for bugs, eval gaps, and safety/guardrail improvements;
- PR review for evidence grounding, workflow traceability, and mock-data boundaries;
- regression test generation for new support scenarios;
- release-readiness checks covering lint, tests, eval, docs, and demo screenshots.

The repository is early-stage, so clear issue threads, small PRs, and reproducible validation output matter more than large feature drops.
