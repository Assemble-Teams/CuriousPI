# SPRINT_1_PROPOSAL.md
# Onion — Sprint 0 (prerequisites) and Sprint 1 (first vertical slice)

**Proposed by:** Cursor (Lead Developer / UX Lead / AI Capability Lead)  
**Baseline:** `onion-bootstrap` @ `669c356` — documentation only, no code  
**Status:** PROPOSAL. Requires Founder decisions DR-01 and DR-02 before any application code is committed. Nothing below is Built.  
**Sizing language:** scoped by components and invasiveness, not calendar time.

---

## 0. Why Sprint 0 precedes Sprint 1 (evidence)

The continuation brief allows a smaller prerequisite "if the repository audit proves" one is needed. The audit does:

1. **F-01 / DR-01** — the repository is public and org-owned. Committing the private application, prompts, schema or synthetic data shaped like real clients to it is not acceptable under the PRD's private-first principle or the brief's §24. No Sprint 1 code can land until this is resolved (private repository, repository transfer, or an explicit Founder acceptance of public source with constraints).
2. **F-05 / DR-02** — there is no stack, persistence or deployment. Choosing them is an architecture decision with cost and reversibility consequences; CURSOR_RULES "Escalate when" covers new paid services and material architecture.
3. There is nothing to migrate. Sprint 0 is therefore a *skeleton*, not a migration, and is small.

Sprint 0 is complete when a reviewer can sign in with a Workspace account on a preview environment, see an empty Today page, and CI is green on an empty-but-real test suite. That is the first honest **Wired** state.

---

## 1. Sprint 0 — Decisions and skeleton

### Inputs (Founder decisions)

| Decision | Needed for | Default recommendation |
|---|---|---|
| DR-01 Repository visibility / host | any code commit | New private repository owned by Uday / the practice; `/onion` docs migrate with history; CuriousPI untouched |
| DR-02 Stack & persistence | skeleton | TypeScript full-stack + Drizzle + embedded SQL (pilot) with PostgreSQL-compatible schema |
| DR-03 Google Cloud project for OIDC | sign-in | GCP project owned by the practice's Workspace, not Assemble Teams |
| DR-04 LLM provider & billing | agents (Sprint 1) | OpenAI API under a practice-owned account with hard monthly cap; fixture provider for tests |
| DR-05 Deployment target for preview/pilot | Wired | Any provider that supports private repo deploys, env secrets and a persistent volume or managed Postgres; chosen under the practice's own account |

### Skeleton deliverables (all under `onion/app`)

| Component | Built when | Wired when | Proven when |
|---|---|---|---|
| App manifest, lockfile, `.gitignore`, `README` scoped to `onion/app` | files exist | CI installs from lockfile | `npm ci && npm test && npm run build` green in CI at recorded SHA |
| Schema v0 + migrations (`ARCHITECTURE_CURRENT.md` §7) | migration files exist | migrations run against pilot store and PostgreSQL in CI matrix | migration up/down test passes on both |
| Identity: Google Workspace OIDC, `hd` restriction, session | sign-in route exists | preview environment signs in a real Workspace user | e2e: allowed-domain user succeeds; other-domain user is rejected; session cookie flags verified |
| Authorization policy engine (`can(actor, verb, record)`) | module exists | used by every command/query | exhaustive unit matrix (roles × verbs × scopes) + negative integration tests |
| Audit event (append-only) | table + writer exist | every command emits one | test: command without audit event fails; DB role cannot UPDATE/DELETE audit rows |
| Platform: config, structured logging, health endpoint, error capture | exist | preview emits logs/health | health check monitored; synthetic error appears in capture |
| Synthetic fixture policy + seed (`fixtures/`) | seed script exists | dev/preview seeded | fixture lint: no real domains, names from a fictional list |
| CI: lint, typecheck, unit, integration, e2e smoke, secret scan, dependency audit | workflow exists | runs on every PR | green at recorded SHA |
| Backup/restore script | script exists | scheduled in pilot | restore drill documented with SHA and checksum |
| Empty app shell: Today page with sign-out | route exists | reachable after sign-in | axe pass; keyboard-only navigation |

Rollback for Sprint 0: delete `onion/app` and the preview deployment. No other system is affected.

---

## 2. Sprint 1 — Vertical slice

### Objective

Prove, end-to-end with real users on a preview environment and synthetic data, that Onion can take a lead from submission to a qualified opportunity under human authority, with evidence attached, a next action owned, an AI-produced executive brief available, and an independent RAS quality check recorded — all audited.

**Lead submitted → qualification (AI proposal + human) → human review decision → organization/opportunity created → evidence attached → next action created → ATLAS brief available → RAS quality check**

### Why it matters / user & business value

- It exercises every architectural primitive the rest of Onion depends on: identity, authorization, lifecycle state, commands with provenance, evidence, approval, audit, agent runtime, honest Google boundary, operational UI. If any of these is wrong, it is cheaper to find out here than in Commercial or Finance.
- It is the top of the funnel for *both* engagement models, so it produces practice value immediately: no lead is lost, every qualification has a rationale and evidence, every qualified lead has an owner and a next action.
- It yields the first measurable pilot signals: lead-to-decision time, evidence coverage, human correction rate on AI proposals.

### Scope — IN

1. **Internal lead entry** (Principal or Consultant) using the full intake schema (identity, organization, engagement model, organization role, objective, timeline/scale, contact method, consent recorded). Status `new` = quarantine.
2. **Duplicate detection** on submit (normalized organization name + domain; contact email) → warning with link; `duplicate` status available.
3. **Lead Qualification agent (A2 Propose)** — on demand ("Request qualification") or on submit (opt-in setting): produces a structured `qualification` proposal against the versioned Qualification Guide. Reads only the lead record and the guide. No external tools. Every claim typed as fact (with field reference) or analysis/assumption.
4. **Manual qualification** — the same `qualification` form filled by a human; the system must be fully usable with the agent disabled.
5. **Human review** — side-by-side: lead, proposal(s), evidence. Decision ∈ {qualify, nurture, decline, needs_info} with mandatory rationale. APPROVE verb → Principal only in the pilot. Decision writes `decision` + `audit_event`.
6. **QualifyLead command** — on `qualify`: upsert `organization` (dedupe), `person`, create `opportunity` (stage `qualified`, explicit `organization_role` and `engagement_model`), link `source_lead_id`, mark lead `qualified`. Idempotent by `operation_key`; replay returns the same opportunity.
7. **Evidence** — attach to lead/opportunity: Drive/Docs/Gmail URL (mode **Linked**), manual note, or agent output reference. Provenance captured automatically. UI shows "Linked, not synced".
8. **Next action** — required on qualify/nurture/needs_info: title, owner, due date. Appears on owner's Today page.
9. **ATLAS Executive Briefing (A1 Read / A0 Advisory)** — (a) per-opportunity brief on demand; (b) Principal's daily brief on Today: new leads, decisions pending, actions due/overdue, evidence gaps, agent runs awaiting review. Every line cites record IDs; no free-floating claims.
10. **RAS Evidence/Quality check (A1)** — runs after a decision is recorded (and on demand): deterministic rules (required fields, rationale present, evidence attached, organization role explicit, next action owned) plus LLM-assisted review of qualification claims vs evidence. Output: findings with severity. RAS can flag; it cannot approve, block, or change records. Separate agent identity and prompt from Lead Qualification.
11. **Agent Activity / Approvals surface** — list and detail of runs: agent, objective, scope, sources, fact vs analysis, assumptions, risk, proposed action, status, approver; pending approvals queue.
12. **Evidence / Audit surface** — audit event timeline per record and global filterable list; evidence list with provenance.
13. **Settings / Connections** — identity, role, agent on/off per capability (Principal), connection labels (Google Identity: Wired; Drive/Docs/Gmail: Linked; LLM provider: status + daily budget).
14. **Minimum observability and backup** (from Sprint 0, exercised).
15. **Eval harness** for the three capabilities with ≥20 synthetic leads (balanced across engagement models, roles, missing-information cases, injection cases).

### Scope — OUT (explicitly)

Public website and public intake; Google Forms/Drive/Gmail/Calendar APIs; Meeting Intelligence; Research (web egress); Engagements, Delivery, Commercial, Finance, Growth modules; external client/provider access; notifications/email sending; chat interface; scheduling of agents; multi-practice; retention/deletion automation; mobile-specific QA beyond slice pages.

### R-process applied

| Step | Sprint 1 content |
|---|---|
| Observe | Today: leads live in email/notes/memory; qualification rationale is undocumented; follow-ups slip; no audit of why a lead was accepted. (Stated in PRD vision; to be confirmed by Uday in the Sprint 1 kickoff interview — 5 questions listed in §13.) |
| Hypothesize | A single structured path with AI-drafted qualification and human decision will (H1) cut lead-to-decision time, (H2) make every qualified lead evidence-backed, (H3) produce zero lost next actions, (H4) keep human correction of AI proposals observable and declining. |
| Prototype | Clickable flow of the seven slice screens (`UI_UX_SYSTEM.md` §6.1) reviewed by AXE before implementation; then the thin server-side implementation. |
| Test | Test ladder §9; usability session with Uday and one consultant on synthetic leads; failure cases (duplicate, missing consent, injection in objective text, LLM outage). |
| Measure | Metrics §10 collected automatically from `audit_event` and `agent_run`. |
| Decide | Founder Gate on evidence pack (§12). |
| Encode | Patterns that survive become design-system components, command conventions, agent contract template and ADRs. |

---

## 3. Users and authority in the slice

| Step | Human actor | Authority verb | Agent | Autonomy |
|---|---|---|---|---|
| Submit lead | Principal / Consultant | CONTRIBUTE | — | — |
| Request qualification | Principal / Consultant | CONTRIBUTE | Lead Qualification | A2 (proposal only) |
| Decide | Principal | APPROVE | — | — |
| Create org/opportunity | system (command) on behalf of decider | — | — | — |
| Attach evidence | Principal / Consultant | CONTRIBUTE | agents may attach their own outputs as evidence of kind `agent_output` | A1 |
| Create next action | Principal / Consultant | CONTRIBUTE | LQ may *propose* | A2 |
| Executive brief | Principal (and consultant for own scope) | VIEW | ATLAS Executive Briefing | A1/A0 |
| Quality check | Principal | VIEW | RAS Evidence/Quality | A1 |

No agent holds APPROVE, MANAGE, DELEGATE or ADMINISTER. No agent has external communication tools.

---

## 4. Exact files / modules affected

All new, all under `onion/app/` (names assume the recommended stack; adjust if DR-02 chooses otherwise):

```
onion/app/
  package.json  package-lock.json  tsconfig.json  .gitignore  .env.example  README.md
  drizzle/                     migrations 0000_init … 000N
  src/platform/{config,logger,health,errors,backup}.ts
  src/identity/{auth,session,users}.ts
  src/authorization/{policy,verbs,scope}.ts            + policy.test.ts
  src/audit/{event,writer,query}.ts                     + audit.test.ts
  src/crm/{organization,person,dedupe}.ts               + dedupe.test.ts
  src/leads/{lead,state,qualification,commands}.ts      + state.test.ts, commands.test.ts
  src/opportunities/{opportunity,commands}.ts
  src/work/{action,decision,approval}.ts
  src/evidence/{evidence,provenance}.ts
  src/knowledge/templates/qualification-guide.v1.md     (business-neutral; no pricing)
  src/agents/{registry,runtime,run,tool-gateway,schemas,provider}.ts
  src/agents/providers/{fixture,openai}.ts
  src/agents/capabilities/lead-qualification/{definition,prompt.v1,schema,evals}.ts
  src/agents/capabilities/atlas-executive-briefing/{…}
  src/agents/capabilities/ras-evidence-quality/{definition,rules,prompt.v1,schema,evals}.ts
  src/ui/(app)/today, leads, leads/[id], organizations/[id], opportunities/[id],
          agents, agents/runs/[id], approvals, audit, settings
  src/ui/components/{status-badge,evidence-chip,origin-chip,approval-card,run-card,data-table,record-header,…}
  fixtures/{organizations,leads,people}.synthetic.json  fixtures/lint.test.ts
  e2e/{signin,slice,isolation,a11y}.spec.ts
  evals/datasets/leads.synthetic.jsonl   evals/run.ts
.github/workflows/onion-app.yml   (path-filtered to onion/app/**; only if the repo decision keeps GitHub Actions)
```

No file outside `/onion` is modified. CuriousPI is untouched.

## 5. Data changes

New schema only (`ARCHITECTURE_CURRENT.md` §7). Forward-only migrations. No data migration. Fixture data is synthetic and lint-checked. The `audit_event` table is granted INSERT/SELECT only to the application role.

## 6. Security impact

- Introduces authentication (OIDC, domain-restricted), sessions, server-side authorization, append-only audit.
- Introduces one server-side secret category: OAuth client secret and LLM API key. Both are environment-only, never in repo, rotated per environment.
- Prompt-injection surface: free-text lead fields reach the LLM. Mitigations: content is passed as data in a delimited block; agents have no tools that can act; output is schema-validated; injection eval suite must pass.
- Isolation: Consultant VIEW is scoped to assigned organizations/records; negative tests required.
- Public surface: none in Sprint 1.

## 7. Agent impact

Introduces the minimum control plane: compiled registry, run record, tool gateway with per-agent allowlist, schema-validated outputs, decision trace, approval records, cost counters, kill switch per capability (Settings), fixture provider. Three capabilities activated (LQ = A2, ATLAS brief = A1, RAS check = A1). Detail: `AI_CAPABILITY_ARCHITECTURE.md`.

## 8. Google impact

- Google Identity **Wired** (OIDC). This is the only Google API call in Sprint 1.
- Drive/Docs/Gmail: **Linked** references only; UI label "Linked, not synced"; no OAuth scopes beyond `openid email profile`.
- No claim of sync anywhere. The connection page derives labels from the integration registry.

## 9. Tests (minimum to call the slice Proven)

| Layer | Tests | Pass criterion |
|---|---|---|
| Unit | lead state machine (all transitions, all invalid transitions); policy matrix; dedupe normalization; operation-key idempotency; audit writer | 100% of enumerated cases |
| Integration | every command against a real store: SubmitLead, RequestQualification, RecordDecision, QualifyLead (incl. replay), AttachEvidence, CreateAction; audit event emitted per command | green |
| Isolation | consultant cannot VIEW/MANAGE unassigned lead/org/opportunity via UI route *and* via direct command call; agent run cannot read outside scope | all denied with reason |
| Approval gates | agent-proposed qualification cannot change `lead.status`; only a `decision` by a principal can; RAS cannot write to records | green |
| Agent evals | ≥20 synthetic leads × 3 capabilities with fixture provider; thresholds in `AI_CAPABILITY_ARCHITECTURE.md` §12 | thresholds met |
| Prompt injection | ≥6 leads whose text instructs the model to qualify, email, delete, change scope or reveal other records | zero behaviour change; zero tool calls; flagged in output |
| LLM outage | provider throws / times out | run `failed`, human flow unaffected, UI shows unavailable state, no retry of side effects |
| End-to-end (Playwright) | sign-in → submit → request qualification → review → qualify → evidence → action → brief → RAS findings → audit timeline | green on desktop and 390px viewport |
| Accessibility | `@axe-core/playwright` on each slice page; manual keyboard pass; focus order; no colour-only status | zero serious/critical; manual pass recorded |
| Backup/restore | snapshot pilot store, restore to clean environment, checksum and row counts match | recorded with SHA |
| Secret/dependency scan | CI | clean |

## 10. Measures (collected automatically)

| Metric | Source | Sprint 1 target (hypothesis check, not a gate) |
|---|---|---|
| Lead-to-decision time | `audit_event` timestamps | measured; baseline for H1 |
| Evidence coverage: qualified leads with ≥1 evidence record | query | 100% enforced by rule; verify |
| Next-action coverage | query | 100% enforced; verify |
| Human correction rate: proposals edited before acceptance | `qualification` diff | measured; expected to decline across pilot |
| Agent fact-claim accuracy vs lead fields | eval + spot check | ≥ 0.95 in evals |
| Hallucinated evidence refs | eval | 0 |
| Scope violations / tool-policy denials | tool gateway log | 0 violations; denials visible |
| Cost per run | `agent_run` | within budget |

## 11. Migration / backward compatibility / rollback

- Greenfield. No existing users or data. Prototype (if delivered) is not migrated.
- Backward compatibility: none required; schema versioned from 0000.
- Rollback: disable capability (Settings kill switch) → disable agent runtime → remove preview deployment → drop store. Documentation remains. Nothing outside `/onion`.

## 12. RAS acceptance criteria (evidence pack for independent verification)

RAS may declare the slice **Proven** only if all of the following are reproducible from the recorded SHA:

1. CI green at SHA: lint, typecheck, unit, integration, e2e, a11y, secret scan.
2. Policy matrix test output listing every role × verb × scope case.
3. Isolation negative tests with denial reasons.
4. Approval-gate tests demonstrating no agent write without human decision.
5. Eval report for three capabilities with dataset hash, provider = fixture, thresholds met; plus one live-provider smoke run (capped) with recorded cost.
6. Injection suite report.
7. Audit timeline export for one complete synthetic lead journey, showing typed actors (human vs agent run) for every step.
8. Google connection page screenshot showing Identity = Wired, Drive/Docs/Gmail = Linked, with no "Synced" label anywhere.
9. Backup/restore drill log.
10. Accessibility report (axe) + manual keyboard checklist.
11. Playwright trace of the full slice on desktop and 390px.
12. Security notes: cookie flags, `hd` enforcement, secret inventory, dependency audit.
13. Known limitations list (what is OUT).

**Commissioned** additionally requires: Uday's sign-off, a named on-call owner (Uday), monitoring in place, rollback rehearsed, and the pilot environment's credentials under the practice's own accounts.

**Production** is not a Sprint 1 outcome.

## 13. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| DR-01 unresolved → code lands in public repo | medium | high | Hard stop: no `onion/app` commit before DR-01 |
| Scope creep into engagements/commercial | high | medium | OUT list; AXE gate on any new module |
| Over-trusting AI qualification | medium | high | A2 only; decision requires rationale; correction rate measured |
| OIDC lockout | low | medium | dev provider; break-glass documented |
| Fixture data looks real | medium | medium | fixture lint; fictional naming scheme |
| LLM cost surprise | low | low | daily cap; fixture default |
| Prompt injection via lead text | medium | medium | no tools; schema; eval suite |
| Single-engineer bus factor | high | medium | README, ADRs, runbook as part of Proven |

Kickoff interview questions for Uday (Observe step): (1) Where do leads arrive today and in what form? (2) What makes you say no fastest? (3) What information do you always need before a discovery call? (4) Who else qualifies leads today, if anyone? (5) What did the last lost lead look like?

## 14. Sprint 1 definition of Built / Wired / Proven

- **Built:** all files in §4 exist with tests beside them.
- **Wired:** deployed to preview under practice-owned accounts; real Workspace sign-in; real store; live LLM provider behind cap; three capabilities enabled.
- **Proven:** §12 evidence pack verified by RAS at a recorded SHA.
- **Commissioned:** Uday uses it for real leads with monitoring and rollback.
- **Production:** not in scope.

---

## 15. AXE ENGINEERING REVIEW (submitted)

```
AXE ENGINEERING REVIEW

Objective: Sprint 0 skeleton + Sprint 1 lead→opportunity slice
Current architecture: none
Proposed architecture: ARCHITECTURE_CURRENT.md §5–§11
Modules affected: identity, authorization, audit, crm, leads, opportunities, work, evidence, knowledge, agents, ui, platform (all new)
Data changes: schema v0, forward-only
Authorization impact: policy engine; APPROVE = principal only; agents never approve
Integration impact: Google OIDC wired; documents linked only; LLM provider server-side with fixture default
Failure modes: see §9 rows "LLM outage", isolation, injection; OIDC lockout
Observability: structured logs keyed by run_id/operation_key; health; error capture; cost counters
Migration: none
Rollback: §11
Tests: §9
Performance: n/a at pilot scale; list views paginated and indexed
Cost: hosting plan + GCP free tier + capped LLM
Evidence: CURSOR_AUDIT.md, PRD_TRACEABILITY.md
Status: FOUNDER DECISION REQUIRED (DR-01, DR-02, DR-03, DR-04, DR-05, DR-07); engineering recommendation PROCEED WITH CONDITIONS
```
