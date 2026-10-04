# DECISION_REQUESTS.md
# Onion — Open Cursor Decision Requests (Cycle 1)

**Raised:** 2026-10-03 by Cursor, from the findings in `CURSOR_AUDIT.md`  
**Format:** `CURSOR_CONTINUATION_BRIEF.md` §3  
**Resolution rule:** each request is closed by Uday's recorded decision; the outcome is then encoded as an ADR under `onion/decisions/` and reflected in `ONION_STATE.md`. Until then, dependent work does not start.

| ID | Decision | Blocks | Status (2026-10-03, Founder) |
|---|---|---|---|
| DR-01 | Repository visibility and long-term host | all application code | **APPROVED — option A.** Encoded as `decisions/ADR-001-repository-host.md`; execution via `MIGRATION_RUNBOOK.md`. Repository creation is an account-owner action (A1–A4). |
| DR-02 | Technology stack and pilot persistence | Sprint 0 | **APPROVED — option A with database refinement:** PostgreSQL dialect from day one, PGlite (not SQLite) for dev/test/early pilot, Drizzle ORM + Kit, Better Auth + Google Workspace sign-in, Next.js App Router. Encoded as `decisions/ADR-002-engineering-stack.md` with spike evidence. |
| DR-03 | Google Cloud project for Workspace sign-in | Sprint 0 identity | **Open — implied by DR-02 (Google Workspace sign-in, identity-only scopes) but the project must still be created by Uday under the practice's Workspace.** |
| DR-04 | LLM provider and billing | Sprint 1 agents | **Approved in principle** ("OpenAI may power initial capabilities; roles must not equal a provider"). Account/billing cap and data-handling acceptance still required before Wired. |
| DR-05 | Deployment target for preview/pilot | Wired state | **Open — deferred by Founder until private repository and deployment account exist.** Domain layer must not couple to a managed-Postgres vendor. |
| DR-06 | Disposition of the off-repository prototype and of `DELIVERY-AUDIT.md` claims | documentation truth | **APPROVED (2026-10-04) — option A.** Keep `index.html` / `site.html` only as a **private, non-production design reference** under `reference/prototype-v0/` in the private repository. Not the implementation base; never deployed; its localStorage architecture is not migrated into Onion. `DELIVERY-AUDIT.md` rows remain "reported, superseded". Execution: `MIGRATION_RUNBOOK.md` §B6b (files supplied by Uday). |
| DR-07 | Sprint 1 scope confirmation (internal lead entry only; three capabilities) | Sprint 1 | **APPROVED** as Sprint 1 direction; Lead Qualification A2; Uday is approval authority; explicit deferral list recorded in `ONION_STATE.md` §11/§14. |

---

## DR-01 — Repository visibility and long-term host

**Decision:** Where Onion application source lives from the first line of code onward, and whether it may be public.

**Context:** `Assemble-Teams/CuriousPI` is public and owned by the Assemble Teams GitHub organization. The founder-approved exception covers *source hosting of the bootstrap*. The audit confirms the exception has been honoured for documentation. Application code, prompts, schema and fixtures are the next commits.

**Observed evidence:** `gh repo view` → `visibility: PUBLIC`, description "Assemble Teams Inc - LLM"; PR #1 body: app source withheld because repo is public; `CURSOR_CONTINUATION_BRIEF.md` §24 forbids confidential prompts/data here; PRD principle "private-first".

**Options:**
- **A. New private repository owned by Uday / the practice (recommended).** Migrate `/onion` with history (`git subtree split` or filter-repo). CuriousPI untouched. PR #1 can be closed or merged as a documentation pointer.
- **B. Transfer visibility: make CuriousPI private.** Not acceptable — CuriousPI is an open-source project; this would harm it and still leave org ownership.
- **C. Keep building in `onion-bootstrap` publicly with constraints** (no prompts with business logic, no realistic fixtures, no pilot credentials). Possible for Sprint 0 skeleton only; untenable by Sprint 1 (prompts, qualification guide, schema reveal operating model).
- **D. Private repository inside the Assemble-Teams org.** Resolves visibility, not ownership/access coupling; org admins retain control.

**Tradeoffs:** A costs one migration (small, now) and a new deployment link; C saves nothing but time and creates irreversible exposure; D is a half-measure.

**Recommendation:** A. Execute before any `onion/app` commit.

**Risks:** losing commit history (mitigated by subtree split); confusion about where the canonical docs live (mitigated by a pointer README in CuriousPI's branch and closing PR #1 with a note).

**Reversibility:** A is fully reversible (repo can be moved again). C is irreversible once pushed (public history).

**AXE impact:** none architectural; CI/deploy link follows the repo.

**RAS impact:** closes GATE-01 (standalone repository). RAS should verify no sensitive content was ever pushed to the public branch (current scan: clean).

**Founder decision required:** yes.

---

## DR-02 — Technology stack and pilot persistence

**Decision:** Language/framework for the modular monolith and the pilot persistence store.

**Context:** No stack exists. The PRD requires a data-dense operational UI, server-side authority, Google and LLM integrations, tests, and a cheap later move to PostgreSQL (productization gate). The host repo is Python/ML; Onion should be isolated from it.

**Observed evidence:** `CURSOR_AUDIT.md` §1; Node 22 and Python 3.12 available; no constraints in PRD beyond "modular monolith" and "PostgreSQL later".

**Options:**
- **A. TypeScript end-to-end (recommended):** full-stack React framework (Next.js App Router or equivalent), Drizzle ORM + SQL migrations, ~~embedded SQLite-compatible store for the pilot with the same schema on PostgreSQL~~, Auth.js Google provider, Zod, Vitest, Playwright + axe. *Founder refinement (accepted): the struck claim was wrong — Drizzle schemas are dialect-specific. Approved form: PostgreSQL dialect from day one with PGlite for dev/test/early pilot and hosted PostgreSQL later; Better Auth (with verified `hd` claim enforcement) instead of Auth.js. See ADR-002.*
- **B. Python:** FastAPI + SQLAlchemy/Alembic + Jinja/HTMX or a separate React front end; pytest; Playwright.
- **C. Google Apps Script + Sheets as system of record.** Lowest infrastructure, but no real authorization model, weak audit, poor UI for approvals/agents; fails CURSOR_RULES security principles for shared persistence.

**Tradeoffs:** A: one language for UI, commands, agent runtime and tests; strongest component ecosystem for tables/forms/a11y; clean separation from the Python host. B: fine for agents, weaker for the UI mandate, two languages if React is used, tooling collision risk with root Python project. C: fast start, dead end for authority/audit.

**Persistence sub-decision (as decided):** PGlite — real PostgreSQL embedded in Node — for local development, automated tests and early single-user/controlled pilot; managed PostgreSQL when Onion reaches shared/staging use; provider chosen later, with no vendor coupling in the domain layer. The schema is written once in Drizzle's PostgreSQL dialect and CI runs migrations against both PGlite and a PostgreSQL service for parity.

**Recommendation:** A, with the persistence rule above.

**Risks:** framework churn (mitigate: stay on stable channels, minimal dependencies); single-engineer familiarity (Cursor owns it; ADR + README).

**Reversibility:** medium — switching language after Sprint 1 means rewriting the slice; switching store is cheap by design.

**AXE impact:** sets component library, testing approach and module layout.

**RAS impact:** determines how authorization, audit immutability and backup are evidenced; both A and B can satisfy.

**Founder decision required:** yes.

---

## DR-03 — Google Cloud project for Workspace sign-in

**Decision:** Create a Google Cloud project (OAuth consent screen + OAuth client, internal user type) under the practice's own Google Workspace to enable domain-restricted sign-in.

**Context:** Authentication is a launch gate; Google-first makes Workspace identity the natural provider; `hd` restriction keeps it internal.

**Evidence:** PRD §Google Workspace, launch gates; no existing project.

**Options:** A. Practice-owned GCP project (recommended). B. Reuse any Assemble Teams project (violates boundary). C. Email/password auth (adds secret handling, loses Google-first).

**Tradeoffs:** A is free at this scale; requires Uday to create the project and share client id/secret into the deployment's secret store (never the repo).

**Recommendation:** A, scopes `openid email profile` only; separate OAuth clients per environment.

**Risks:** misconfigured consent screen; mitigated by runbook.

**Reversibility:** high.

**AXE impact:** identity module design. **RAS impact:** GATE-03, GATE-07 (identity part).

**Founder decision required:** yes (new account/service, owned by the practice).

---

## DR-04 — LLM provider and billing

**Decision:** First live LLM provider for the agent runtime and its budget.

**Context:** PRD names ChatGPT; the runtime is provider-abstracted; tests/evals use a fixture provider so Built/Proven do not depend on this decision, but Wired does.

**Options:** A. OpenAI API under a practice-owned account with a hard monthly cap (recommended). B. Another provider. C. Defer live provider; Sprint 1 Proven on fixtures only and Wired later.

**Tradeoffs:** A aligns with PRD; cost is small at pilot volume; data-handling terms of the provider must be acceptable for lead text (no client documents in Sprint 1).

**Recommendation:** A with a cap, zero-data-retention/API data controls reviewed by Privacy/Legal; key held only in deployment secrets.

**Risks:** cost surprise (cap); data exposure (scope limited to lead records; no documents in Sprint 1).

**Reversibility:** high (adapter swap).

**AXE impact:** none structural. **RAS impact:** evidence for AI-authority gates; privacy review required.

**Founder decision required:** yes (paid service; data-handling acceptance).

---

## DR-05 — Deployment target for preview and pilot

**Decision:** Where the app runs for AXE/RAS review (preview) and controlled real use (pilot).

**Context:** No deployment exists. Must support private repository deploys, server-side secrets, persistent storage or managed PostgreSQL, HTTPS, and must be under the practice's own account — not Assemble Teams'.

**Options:** A. A PaaS with persistent volume or managed Postgres and preview deployments per PR (several fit; choose on account ownership, region/data residency and snapshot support). B. A single small VM managed by Uday (more ops burden, full control). C. Defer deployment; local-only (cannot reach Wired).

**Tradeoffs:** A fastest to Wired with least ops; B maximum control; C blocks Wired/Commissioned.

**Recommendation:** A, chosen jointly with DR-02's persistence rule; one project per environment; backups verified by the Sprint 1 restore drill.

**Risks:** vendor data location; mitigated by region choice and Privacy review.

**Reversibility:** high (containerised app, SQL dump).

**AXE impact:** environment layout. **RAS impact:** GATE-02, GATE-05, GATE-11, GATE-12.

**Founder decision required:** yes (new account/service).

---

## DR-06 — Disposition of the off-repository prototype and `DELIVERY-AUDIT.md` claims

**Decision:** How to treat the prototype (`index.html`, `site.html`) described in `README.md` and `DELIVERY-AUDIT.md` but absent from the repository, and the "Built: Yes" rows that describe it.

**Evidence:** `CURSOR_AUDIT.md` §3; all refs searched; PR #1 body confirms deliberate withholding.

**Options:** A. Deliver the prototype files to a private location for Cursor/RAS inspection as a *design reference only*; keep DELIVERY-AUDIT rows marked "reported, unverified". B. Declare the prototype superseded; retire DELIVERY-AUDIT in favour of PRD_TRACEABILITY and keep it as historical record. C. Commit the prototype to the public branch (not recommended: it is the private app, and it would create a second, non-foundational codebase).

**Recommendation:** A if the files still exist and contain UX decisions worth preserving; otherwise B. In both cases the Sprint 1 foundation is server-side and does not migrate the prototype.

**Reversibility:** high. **AXE impact:** design-reference input. **RAS impact:** restores Built/Wired/Proven integrity of the documentation set.

**Founder decision required:** yes.

---

## DR-07 — Sprint 1 scope confirmation

**Decision:** Confirm the slice as scoped in `SPRINT_1_PROPOSAL.md`: internal lead entry (no public intake), three capabilities (Lead Qualification A2, ATLAS Executive Briefing A1, RAS Evidence/Quality A1), Principal-only APPROVE, Google identity wired + documents linked only, no notifications, no engagements/commercial/finance.

**Evidence:** PRD §Core private product areas; continuation brief §19; audit §13 scope-creep risks.

**Options:** A. As proposed (recommended). B. Add public intake (adds abuse controls, privacy notice, public deployable, retention — roughly doubles the slice). C. Add Meeting Intelligence (adds Google scopes and transcript privacy review).

**Tradeoffs:** A proves every primitive with the least surface; B and C pull forward decisions that deserve their own review.

**Recommendation:** A, plus the five Observe-step interview questions in `SPRINT_1_PROPOSAL.md` §13 answered at kickoff.

**Reversibility:** high. **AXE impact:** page inventory fixed. **RAS impact:** acceptance pack fixed (§12).

**Founder decision required:** yes.
