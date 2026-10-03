# ARCHITECTURE_CURRENT.md
# Onion — Architecture as Observed, and Target Direction for AXE Review

**Observed at:** `Assemble-Teams/CuriousPI` @ `onion-bootstrap` / `669c356debe998845e9d8785360692cc2b2b30b2`, 2026-10-03  
**Sections 1–4 are observation. Section 5 onward is a PROPOSAL and is labelled as such. Nothing in §5+ is Built.**

---

## 1. Observed repository architecture

```
Assemble-Teams/CuriousPI  (PUBLIC, GitHub org: Assemble-Teams)
│
├── main ──────────────────────── CuriousPI: Python/PyTorch STEM LLM project
│     README.md, ARCHITECTURE.md, LICENSE (Apache-2.0), requirements.txt,
│     curiouspi.ipynb, src/, training/, data/, configs/, test_*.py,
│     Kimi-K2.6 (gitlink, no .gitmodules), .gitignore (Python template)
│
└── onion-bootstrap ───────────── main + /onion  (only delta; 9 Markdown files)
      onion/PRD.md                          canonical product requirements
      onion/CURSOR_RULES.md                 engineering & audit contract
      onion/CURSOR_KICKOFF_PROMPT.md        first assignment
      onion/CURSOR_CONTINUATION_BRIEF.md    governance, AXE/RAS, UI & AI leadership brief
      onion/ONION_STATE.md                  master state
      onion/ONION_SESSION_HANDOFF_TEMPLATE.md
      onion/DELIVERY-AUDIT.md               prior-session delivery claims (see §2)
      onion/REPOSITORY_DECISION.md          hosting exception record
      onion/README.md                       bootstrap readme (references absent files)
```

- No `.github/` workflows, no deployment manifests, no environment files, no application manifests, no tests, on either branch.
- No Onion code exists in any ref or unreachable object (`CURSOR_AUDIT.md` §1.1 #3–4).
- Draft PR #1 proposes `onion-bootstrap` → `main`. It is explicitly not to be merged until visibility/hosting and Cursor/RAS findings are reviewed.

## 2. Reported (not observed) prototype

`onion/README.md` and `onion/DELIVERY-AUDIT.md` describe a prototype that is **not in this repository**. For design continuity it is recorded here as *reported*:

| Aspect | Reported description | Observed |
|---|---|---|
| Private app | single static `index.html`, owner-operated | absent |
| Public site | single static `site.html`, non-transmitting form, non-affiliation notice | absent |
| Persistence | browser `localStorage`; JSON backup/import; CSV finance export | absent |
| Surfaces | dashboard, leads (client vs provider), engagements, delivery, finance (per-record currency), growth, templates | absent |
| Google | manual source links only | absent |
| AI | no API calls | absent |
| Auth / authz / shared DB / retention / abuse controls | reported as not built | consistent with absence |
| Verification | "static + HTTP checks", no browser automation | not reproducible |

Architectural implication: even if delivered, a localStorage single-file prototype has no persistence, identity, authority or audit layer to build on. It is a **design reference**, not a foundation. The Sprint 1 proposal therefore starts from a server-side skeleton rather than migrating the prototype. Disposition of the prototype files is DR-06.

## 3. Observed boundaries

| Boundary | Observation |
|---|---|
| Onion ↔ CuriousPI code | No shared code. `/onion` is a sibling directory of unrelated Python sources. Risk: generic root tooling (pytest discovery, root `.gitignore`, root `requirements.txt`) could accidentally include or exclude Onion files. Mitigation: Onion keeps its own manifest, lockfile, `.gitignore` and CI scoped to `onion/app/**`. |
| Onion ↔ Assemble Teams org | Organizational: the org owns the repo. No runtime coupling exists. See DR-01. |
| Onion ↔ public internet | Everything on every branch is public. |
| Onion private ↔ public site | Not yet instantiated. Rule for target: separate deployable, separate origin, no shared session, no direct DB access from the public site (intake writes go through a quarantine API). |
| Onion ↔ Google Workspace | Not instantiated. Target: identity first; references second; API read/write only when Wired + Proven. |
| Onion ↔ LLM provider | Not instantiated. Target: provider interface behind a tool gateway; never called from the browser. |

## 4. Observed state per layer

| Layer | Observed | Notes |
|---|---|---|
| Identity / authentication | none | — |
| Authorization | none | — |
| Domain model / persistence | none | — |
| Workflow / state machines | none | — |
| Application UI | none | — |
| Public site | none | — |
| Integrations (Google) | none | — |
| Agent control plane | none | — |
| Audit / evidence | none | — |
| Observability / backup | none | — |
| CI / tests | none | — |
| Deployment | none | — |

---

# PROPOSAL — Target architecture direction (for AXE review; not Built)

## 5. Architecture rules (to be encoded as ADRs after Founder/AXE review)

| Rule | Statement | Source |
|---|---|---|
| R-1 Modular monolith | One deployable application with explicit internal modules and no cross-module table access. No microservices in the pilot. | CURSOR_RULES §Architecture |
| R-2 Server-side authority | Every authorization decision, approval check, agent tool call and side effect is executed and logged server-side. The browser holds no credentials and no authority. | CURSOR_RULES §Security, PRD §Security |
| R-3 Commands with provenance | All writes go through named commands (`SubmitLead`, `QualifyLead`, `RecordDecision`…) carrying `actor` (human user or agent run), `operation_key`, `reason`, and producing an `audit_event`. No ad-hoc writes. | PRD §Evidence/Audit |
| R-4 Evidence is a first-class entity | Facts link to `evidence`; agent claims link to evidence or are labelled analysis/assumption. | PRD §Evidence/Audit |
| R-5 Honest integrations | Google objects are **Linked**, **Synced** or **Writes**. The label is derived from the integration's proven capability, not from UI copy. | PRD §Google Workspace |
| R-6 Agents are governed capabilities | No LLM call outside the agent runtime; every call belongs to a registered agent, a run, a scope and a tool allowlist enforced in code. | PRD §Agent governance |
| R-7 No commercial constants | Prices, percentages, commissions, terms live in approved data records, never in code or prompts. | PRD §Power Sprint |
| R-8 Portable persistence | SQL schema expressed through a migration/ORM layer that targets both the pilot store and PostgreSQL; no store-specific features without an ADR. | PRD §Productization gate |
| R-9 Synthetic data only outside pilot | Dev/preview/test use obviously fictional fixtures; pilot data never copied downward. | PRD §Security |
| R-10 Separate public deployable | Public front door is a separate app/origin with a single quarantine write path. | PRD §Public website |

## 6. Module map (modular monolith)

```
onion/app
├── identity          sign-in (Google Workspace OIDC), users, sessions, roles
├── authorization     policy engine: authority verbs × roles × record scope; deterministic, unit-tested
├── crm               organizations, people, organization_role, dedupe
├── leads             lead intake record, lead state machine, qualification record
├── opportunities     opportunity, stage, owner, next action
├── work              action, decision, approval (shared primitives used later by delivery/commercial)
├── evidence          evidence records, provenance, classification, links
├── knowledge         versioned templates (Sprint 1: Qualification Guide)
├── agents            registry, runtime, tool gateway, runs, outputs, evals, provider adapters
│   ├── capabilities/lead-qualification
│   ├── capabilities/atlas-executive-briefing
│   └── capabilities/ras-evidence-quality
├── audit             append-only audit_event, query API, export
├── integrations      google (identity now; drive/gmail/calendar adapters later), llm providers
├── ui                app shell, Today, Leads & Organizations, Evidence/Audit, Agent Activity/Approvals, Settings
└── platform          config, logging, health, backup, migrations, fixtures
```

Dependency direction: `ui → application commands/queries → domain modules → persistence`. `agents` may read via module query APIs and may **propose** via commands that require human approval; agents never write to tables directly.

## 7. Sprint 1 data model (v0, for review)

Conventions: `id` ULID/UUIDv7; `created_at`/`updated_at` UTC; `created_by` = user id or `agent_run` id (typed actor); soft delete only; `classification ∈ {internal, confidential, restricted}`; every state column is an enum with a transition table.

| Entity | Key fields | Notes |
|---|---|---|
| `practice` | id, name | single row in pilot; present so multi-practice is not a rewrite |
| `user` | id, email (Workspace), display_name, role ∈ {principal, consultant}, status | roles map to authority verbs in `authorization` |
| `organization` | id, legal_name, display_name, normalized_name, primary_domain, country, status | dedupe on normalized_name + primary_domain |
| `person` | id, organization_id?, full_name, email?, title?, consent_state | — |
| `lead` | id, source ∈ {internal_entry, referral, inbound_web, event, outbound, other}, channel_ref?, submitted_by (user) / submitter_contact, organization_name_raw, organization_id?, person_id?, engagement_model ∈ {client_transformation, provider_advisory, unknown}, organization_role ∈ {client, provider, partner, unknown}, objective, timeline, scale, contact_method, consent_recorded_at?, status ∈ {new, in_review, needs_info, qualified, nurture, declined, duplicate}, owner_id, next_action_id? | `new` = quarantine |
| `qualification` | id, lead_id, origin ∈ {human, agent}, agent_run_id?, fit_assessment (enum + rationale), engagement_model_proposed, missing_information[], risks[], suggested_next_action, questions[], evidence_refs[], status ∈ {proposed, accepted, rejected, superseded} | agent proposals are never auto-accepted |
| `decision` | id, subject_type/subject_id, decided_by (user), decision ∈ {qualify, nurture, decline, needs_info, …}, rationale, evidence_refs[], decided_at | human-only writer |
| `opportunity` | id, organization_id, organization_role, engagement_model, title, stage ∈ {qualified, discovery, diagnostic, recommendation, proposal, …}, owner_id, source_lead_id, created_from_decision_id | created only by `QualifyLead` command after a decision |
| `action` | id, subject_type/subject_id, title, owner_id, due_at?, status ∈ {open, done, cancelled}, origin, agent_run_id? | — |
| `approval` | id, subject_type/subject_id, requested_by (actor), request_kind, options[], evidence_refs[], risk_level, status ∈ {pending, approved, rejected, expired}, decided_by?, decided_at?, rationale? | all agent-proposed consequential actions pass through here |
| `evidence` | id, subject_type/subject_id, kind ∈ {link, note, file_ref, message_ref, agent_output}, source_system ∈ {google_drive, google_docs, gmail, manual, onion, llm}, source_ref (URL/id), title, captured_by (actor), captured_at, integration_mode ∈ {linked, synced, writes}, classification | `integration_mode` is set by the integration layer, never by the UI |
| `template` | id, key, version, body, status | Qualification Guide v1 |
| `agent_definition` | code-defined (not a table): id, role, owner_user_id, objective, autonomy_level, scope_policy, tool_allowlist, read/write policy, approval_policy, evidence_policy, prompt_version, output_schema, stop_conditions, eval_suite_id | compiled registry; snapshot stored on each run |
| `agent_run` | id, agent_id, agent_version, trigger ∈ {user, schedule, event}, triggered_by, objective, scope (subject refs), status ∈ {queued, running, awaiting_approval, completed, failed, cancelled, escalated}, model_provider, model_name, prompt_version, token_in/out, cost_estimate, started_at, finished_at, error? | one row per run; immutable after terminal state |
| `agent_output` | id, run_id, schema_id, payload (validated JSON), decision_trace {evidence_considered[], rules_applied[], assumptions[], result, risk, approval_required}, fact_claims[] with evidence_refs, analysis_claims[] | stored only after schema validation |
| `audit_event` | id, occurred_at, actor (typed), action, subject refs, before/after digest, operation_key, reason, run_id? | append-only; no update/delete grants |

## 8. Authority model (Sprint 1 subset)

| Verb | Principal | Consultant | Agent (any) |
|---|---|---|---|
| VIEW | all records | records where owner/assignee or organization is in assigned set | records inside run scope only |
| COMMENT | yes | yes (scoped) | via `agent_output` only |
| CONTRIBUTE (create lead, evidence, action) | yes | yes (scoped) | propose only (A2) → `approval` |
| MANAGE (edit lead, reassign) | yes | own records | no |
| APPROVE (decisions, approvals) | yes | no (pilot) | **never** |
| DELEGATE | yes | no | never |
| ADMINISTER (users, agents, connections) | yes | no | never |

Enforced by a pure policy function `can(actor, verb, record) → allow | deny(reason)` with exhaustive unit tests and negative integration tests.

## 9. Runtime topology (pending DR-02 / DR-05)

```
[Browser: Principal / Consultant]
        │ HTTPS, session cookie (httpOnly, SameSite=Lax)
        ▼
[Onion app server — one process]
   ui (server-rendered + islands) · commands/queries · authorization · audit
   agents runtime → tool gateway → { module query APIs, llm provider adapter }
        │                                  │
        ▼                                  ▼
[PostgreSQL dialect: PGlite (dev/test/early pilot) · hosted PostgreSQL (shared/staging/prod) behind one driver adapter]   [LLM provider API (server-side key)]
        │
[Google Identity (OIDC, hd = practice domain)]   [Google Drive/Docs/Gmail: URL references only in Sprint 1]
```

The public front door is a separate deployable (Later) that can only call `POST /intake` → `lead.status = new`.

## 10. Canonical stack (DR-02 approved with refinement — `decisions/ADR-002-engineering-stack.md`)

TypeScript end-to-end; **Next.js App Router** with Server Components and server-side boundaries by default; React + an Onion-specific design system; **PostgreSQL dialect from day one**; **Drizzle ORM + Drizzle Kit** (schema in TypeScript, generated SQL migrations committed, no out-of-band production schema changes); **PGlite** for development, automated tests and early controlled pilot; hosted **PostgreSQL** for shared/staging/production behind a single driver adapter with no vendor coupling in the domain layer; **Better Auth** with Google Workspace sign-in restricted server-side to the authorized domain/account policy, identity-only scopes; application-controlled authorization with the seven authority verbs; Zod for input/output schemas; Vitest; Playwright + `@axe-core/playwright`; structured JSON logging; a fixture LLM provider for deterministic tests and evals; OpenAI adapter as the first live provider behind the provider-neutral control plane.

Spike evidence for this combination (versions, checks, two gotchas: Better Auth schema must be CLI-generated; PGlite must be in `serverExternalPackages`) is recorded in ADR-002.

## 11. Environment layout

| Environment | Purpose | Data | Credentials |
|---|---|---|---|
| `dev` (local) | engineering | synthetic fixtures; PGlite | personal dev OAuth client, fixture LLM by default |
| `test` (CI) | automated tests, evals, migration checks | synthetic fixtures; PGlite in-memory, plus a PostgreSQL service for migration parity | none (fixture LLM only) |
| `preview` | AXE/RAS review of a branch | synthetic fixtures; PGlite or hosted PostgreSQL | preview OAuth client, fixture LLM or capped live key |
| `pilot` | controlled real use (Commissioned) | real practice data; PGlite (early single-user, with file backups) or hosted PostgreSQL | pilot OAuth client, live LLM key with budget, backups enabled |

No `production` environment exists until launch gates are met.

## 12. What this document deliberately does not decide

- ~~Repository visibility and long-term host (DR-01)~~ — decided: ADR-001
- ~~Final stack and persistence (DR-02)~~ — decided: ADR-002
- Google Cloud project ownership for OIDC (DR-03)
- LLM provider and billing (DR-04)
- Deployment target (DR-05)
- Prototype disposition (DR-06)
- Sprint 1 scope confirmation (DR-07)

---

## 13. AXE ENGINEERING REVIEW (submitted)

```
AXE ENGINEERING REVIEW

Objective: establish the foundation on which the Lead → … → RAS check slice can be Built, Wired and Proven
Current architecture: none (documentation-only branch in a public, org-owned repository)
Proposed architecture: single modular-monolith app under onion/app with identity, authorization, crm, leads,
  opportunities, work, evidence, knowledge, agents (registry/runtime/tool gateway), audit, integrations, ui, platform
Modules affected: all new; zero changes outside /onion
Data changes: new schema (§7); forward-only migrations; no legacy data
Authorization impact: introduces server-side policy engine (§8); APPROVE restricted to principal; agents never approve
Integration impact: Google Workspace OIDC (identity); Google documents referenced by URL only; LLM provider server-side
Failure modes: OIDC misconfiguration (locks out users — mitigated by local dev provider + runbook); LLM provider outage
  (agents degrade to "unavailable", human workflow unaffected); store corruption (scripted backup + restore drill)
Observability: structured logs with run_id/operation_key, health endpoint, error capture, per-agent cost counters
Migration: none (greenfield); prototype treated as design reference only
Rollback: delete deployment + store; nothing outside /onion is touched
Tests: unit (policy, state machines, dedupe, idempotency), integration (commands), isolation, agent evals, injection,
  e2e (Playwright), accessibility (axe), backup/restore drill
Performance: single-digit users; no concern beyond sane indexes and no N+1 in list views
Cost: LLM usage capped per agent/day; one hosting plan; one GCP project (free tier); no other paid services
Evidence: CURSOR_AUDIT.md §1.1; PRD_TRACEABILITY.md
Status: FOUNDER DECISION REQUIRED (DR-01, DR-02) — architecture itself is recommended PROCEED WITH CONDITIONS
  (conditions: private repository or equivalent before application code; stack confirmed; ADRs R-1…R-10 ratified)
```
