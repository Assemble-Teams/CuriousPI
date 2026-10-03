# Onion — Private Consulting OS Pilot

Onion is the private operating-system prototype for Uday Teki's consulting practice. By explicit founder decision on 2026-10-03, the bootstrap source is being staged inside `Assemble-Teams/CuriousPI` under the `/onion` namespace on a dedicated branch. This repository-hosting exception does not authorize reuse of Assemble Teams/GameChangers branding, product data, customer data, analytics, secrets, or runtime infrastructure.

> **Repository state (Cursor audit, 2026-10-03):** this branch contains Onion **documentation only**. The `index.html` / `site.html` prototype described below is not in the repository; the "Run locally" instructions therefore do not work here. Start with `CURSOR_AUDIT.md`, then `PRD_TRACEABILITY.md`, `ARCHITECTURE_CURRENT.md`, `SPRINT_1_PROPOSAL.md`, `UI_UX_SYSTEM.md`, `AI_CAPABILITY_ARCHITECTURE.md` and the open Founder decisions in `DECISION_REQUESTS.md`.

## Surfaces (reported prototype — not in this repository; see DR-06)
- `index.html` — private owner-operated consulting OS prototype.
- `site.html` — public consulting website/intake preview. The form is deliberately non-transmitting until the production intake backend/privacy controls are wired.

## Current capabilities
Dashboard, client/provider leads, engagement playbooks, delivery records, risks/decisions/approvals, lightweight finance with per-record currency, growth signals, reusable templates, JSON backup/import, CSV finance export, Google source links, light/dark appearance, mobile layout.

## Important pilot boundaries
- Browser-local storage is not a secure multi-user database.
- No public signup.
- No automatic Google synchronization yet.
- No automatic ChatGPT/API transmission yet.
- External collaborator/client/provider access remains deferred until authenticated engagement-scoped permissions are implemented.
- Consequential communications, commitments, and financial changes remain human-approved.

## Run locally
Not applicable on this branch — there is no runnable Onion artifact in the repository. (Historical instruction for the reported prototype: `python3 -m http.server 4173`, then open `index.html` / `site.html`.)

## Document map
| File | Purpose |
|---|---|
| `PRD.md` | canonical product requirements |
| `CURSOR_RULES.md` | engineering and audit contract |
| `CURSOR_KICKOFF_PROMPT.md`, `CURSOR_CONTINUATION_BRIEF.md` | Cursor mandate and governance (Founder Council, AXE, RAS, R-process, RAS-100) |
| `ONION_STATE.md` | master state; `ONION_SESSION_HANDOFF_TEMPLATE.md` + `handoffs/` | session continuity |
| `REPOSITORY_DECISION.md` | hosting exception record |
| `DELIVERY-AUDIT.md` | prior-session delivery claims (annotated; superseded by traceability) |
| `CURSOR_AUDIT.md` | repository engineering audit, findings by severity, core-test answer |
| `PRD_TRACEABILITY.md` | requirement → code → Built/Wired/Proven matrix |
| `ARCHITECTURE_CURRENT.md` | observed architecture and target direction for AXE |
| `SPRINT_1_PROPOSAL.md` | Sprint 0 prerequisites and Sprint 1 vertical slice |
| `UI_UX_SYSTEM.md` | personas, navigation, journeys, tokens, patterns, accessibility |
| `AI_CAPABILITY_ARCHITECTURE.md` | agent control plane and first five capability specs |
| `DECISION_REQUESTS.md` | open structured Founder decision requests (DR-01…DR-07) |
