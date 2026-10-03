# PRD_TRACEABILITY.md
# Onion — PRD → Code Traceability Matrix

**Baseline:** `onion/PRD.md` on `onion-bootstrap` @ `669c356debe998845e9d8785360692cc2b2b30b2`  
**Audit date:** 2026-10-03  
**Maintained by:** Cursor (Lead Developer). Verified independently by RAS before any status is promoted.

## How to read this matrix

- **Built** = implementation exists in this repository at the cited code location.
- **Wired** = connected to the intended real workflow/dependency.
- **Proven** = a representative acceptance test passed, with recorded command + SHA.
- **Commissioned** / **Production** are tracked separately in `ONION_STATE.md` once any row reaches Proven.
- **S1** = targeted by the Sprint 1 vertical slice (`SPRINT_1_PROPOSAL.md`). **S0** = Sprint 0 prerequisite. **Later** = sequenced after Sprint 1. **Deferred** = PRD-deferred until pilot evidence.
- A status is never promoted because code exists, a label exists, or a document says so.

**Repository-verified summary at this SHA:** 0 of 82 tracked rows (64 requirements, 16 launch gates, 2 pilot/productization criteria) Built. 0 Wired. 0 Proven. There is no Onion code location to cite. All "Code location" cells therefore read `—`.

---

## A. Vision, principles, users

| ID | Requirement (PRD section) | Code location | Built | Wired | Proven | Evidence | Gap / plan |
|---|---|---|---|---|---|---|---|
| PRD-A1 | Private operating system for the practice; no public SaaS signup in v1 | — | No | No | No | No code | S0 (private app shell, no signup route) |
| PRD-A2 | Two engagement models (Client Transformation; Provider Advisory) share one OS with engagement isolation | — | No | No | No | — | S1 models `organization_role` + `engagement_model` on lead/opportunity; isolation tests |
| PRD-A3 | Principles: private-first, Google-first, evidence-first, pilot-first, human authority | — | No | No | No | — | Encoded as architecture rules (`ARCHITECTURE_CURRENT.md` §5) |
| PRD-A4 | Principal has full authority and final approval unless delegated | — | No | No | No | — | S1 authz policy: APPROVE requires `principal` role |
| PRD-A5 | Internal consultants see assigned work and authorized records only | — | No | No | No | — | S1 record-level scoping + negative tests |
| PRD-A6 | External clients/providers deferred; future access engagement-scoped, never exposing internal assessments/margins/RAS | — | No | No | No | — | Deferred; data model keeps `classification` from day one |

## B. Lifecycle

| ID | Requirement | Code location | Built | Wired | Proven | Evidence | Gap / plan |
|---|---|---|---|---|---|---|---|
| PRD-B1 | Signal → Lead → Qualification states | — | No | No | No | — | S1 (`lead.status` state machine) |
| PRD-B2 | Discovery → Diagnostic → Readiness Assessment → Recommendation | — | No | No | No | — | Later (engagement module) |
| PRD-B3 | Recommendation outcomes: Proceed / Pause / Address readiness gaps / Stop | — | No | No | No | — | Later; enum reserved |
| PRD-B4 | Opportunity state | — | No | No | No | — | S1 (`opportunity.stage`) |
| PRD-B5 | Proposal → NDA/MSA/SOW → Commercial Approval → Signature → Invoice/Payment Terms | — | No | No | No | — | Later (commercial module); no hard-coded terms |
| PRD-B6 | Kickoff → Power Sprint / Delivery → Outcome Review → Acceptance → Closeout | — | No | No | No | — | Later (delivery module) |
| PRD-B7 | Expansion / Referral / Follow-up → Learning | — | No | No | No | — | Later |
| PRD-B8 | Power Sprint as editable 3–4 week model; no hard-coded pricing/percentages/terms | — | No | No | No | — | Later; rule encoded in `CURSOR_RULES.md` and architecture rule R-7 |

## C. Private product areas

| ID | Requirement | Code location | Built | Wired | Proven | Evidence | Gap / plan |
|---|---|---|---|---|---|---|---|
| PRD-C1 | Dashboard: active engagements, pipeline, actions, milestones, risks, approvals, decisions, currency-aware finance snapshot | — | No | No | No | — | S1 ships a minimal **Today** view (actions due, approvals pending, ATLAS brief); full dashboard Later |
| PRD-C2 | Leads & Organizations: org/contact, role/type, source, needs, fit, qualification, next step, follow-up, supporting source | — | No | No | No | — | **S1 core** |
| PRD-C3 | Engagements record | — | No | No | No | — | Later |
| PRD-C4 | Delivery: workstreams, tasks, milestones, meetings, notes, decisions, actions, risks, issues, approvals, deliverables, evidence, closeout | — | No | No | No | — | Later; S1 introduces `action`, `decision`, `approval`, `evidence` primitives reused by Delivery |
| PRD-C5 | Commercial record | — | No | No | No | — | Later |
| PRD-C6 | Finance: proposed/invoiced/collected/outstanding/expenses, currency per record; not accounting software | — | No | No | No | — | Later |
| PRD-C7 | Growth & Signals | — | No | No | No | — | Later; S1 `lead.source`/`signal` fields are the seed |
| PRD-C8 | Knowledge & Templates | — | No | No | No | — | S1 includes one versioned **Qualification Guide** template consumed by the Lead Qualification agent; rest Later |
| PRD-C9 | Evidence / Audit: trace facts to source; distinguish sourced fact, AI analysis, human decision; timestamps/owners/engagement/approval evidence | — | No | No | No | — | **S1 core** |
| PRD-C10 | Agent Activity / Approvals: runs, requested actions, evidence, scopes, permissions, approvals, results, evaluations | — | No | No | No | — | **S1 core** |
| PRD-C11 | Settings / Connections (from CURSOR_RULES surface list) | — | No | No | No | — | S1 minimal: sign-in identity, connection status labels (Linked/Synced/Writes) |

## D. Public website and intake

| ID | Requirement | Code location | Built | Wired | Proven | Evidence | Gap / plan |
|---|---|---|---|---|---|---|---|
| PRD-D1 | Small public front door; explains who/what/approach; Submit Your Project / Start a Conversation | — | No | No | No | — | Later (requires DR-05 deployment + abuse controls) |
| PRD-D2 | Public site must not expose dashboards, assessments, finances, RAS methods, templates, commercial logic | — | No | No | No | — | Architecture rule: separate deployable, no shared session/data path |
| PRD-D3 | Intake captures identity, organization, engagement type, objective, timeline/scale, contact method, consent | — | No | No | No | — | S1 implements the **same lead schema via internal entry**; public channel Later |
| PRD-D4 | Intake enters review/quarantine, never auto-trusted | — | No | No | No | — | S1 `lead.status = new` is quarantine; promotion only by human decision |

## E. Google Workspace

| ID | Requirement | Code location | Built | Wired | Proven | Evidence | Gap / plan |
|---|---|---|---|---|---|---|---|
| PRD-E1 | Drive/Docs for documents and evidence | — | No | No | No | — | S1: **Linked** references only, labelled honestly |
| PRD-E2 | Sheets for lightweight ledgers | — | No | No | No | — | Later (Finance) |
| PRD-E3 | Forms for intake where appropriate | — | No | No | No | — | Later |
| PRD-E4 | Gmail for communications | — | No | No | No | — | S1: Linked (message URL as evidence); no API |
| PRD-E5 | Calendar / Meet for meetings | — | No | No | No | — | Later (Meeting Intelligence) |
| PRD-E6 | Reference native Google records; avoid duplication; never claim sync that is not wired and proven | — | No | No | No | — | Design-system rule: Linked / Synced / Writes labels; Synced requires integration test |
| PRD-E7 | Google identity for sign-in (derived: Google-first + auth gate) | — | No | No | No | — | **S0/S1** Workspace OIDC, domain-restricted (DR-03) |

## F. AI and agent governance

| ID | Requirement | Code location | Built | Wired | Proven | Evidence | Gap / plan |
|---|---|---|---|---|---|---|---|
| PRD-F1 | Authorized AI uses (summaries, briefs, diagnostic prep, action extraction, risk/evidence gaps, research, drafts, decision briefs, outcome comparison, RAS support) | — | No | No | No | — | S1: three of these (lead qualification analysis, executive brief, RAS evidence check) |
| PRD-F2 | Human approval before external comms, commitments, commercial terms, financial changes, permission changes, confidential disclosure, consequential actions | — | No | No | No | — | S1: enforced in tool gateway; agents have no external/write-consequential tools |
| PRD-F3 | Core roles ATLAS / RAS / GROWTH / DELIVERY | — | No | No | No | — | S1 registry declares all four roles; only ATLAS and RAS have active capabilities; Lead Qualification runs under GROWTH |
| PRD-F4 | Specialist cohort: ATLAS Executive Briefing, Meeting Intelligence, Research, RAS Evidence/Quality, Lead Qualification | — | No | No | No | — | S1: Lead Qualification, ATLAS Executive Briefing, RAS Evidence/Quality. Later: Meeting Intelligence, Research |
| PRD-F5 | Autonomy levels A0–A4; no unrestricted autonomy | — | No | No | No | — | S1: enforced per agent in code; LQ = A2, ATLAS brief = A1, RAS check = A1 |
| PRD-F6 | Agent contract fields: identity, owner, objective, scope, tool permissions, read/write, approval policy, evidence policy, prompt/policy version, run ID, stop conditions, output schema, evaluation coverage | — | No | No | No | — | S1: `AgentDefinition` type requires every field at compile time |
| PRD-F7 | Store concise decision traces, not chain-of-thought | — | No | No | No | — | S1 `agent_output.decision_trace` schema |
| PRD-F8 | Prompt-injection policy: retrieved content is data, not authority | — | No | No | No | — | S1 injection eval suite |
| PRD-F9 | Eval coverage per agent (task success, evidence accuracy, source correctness, hallucination, tool selection, scope, approval compliance, duplicates, human correction rate, escalation, latency, cost) | — | No | No | No | — | S1 eval harness with fixture provider; thresholds in `AI_CAPABILITY_ARCHITECTURE.md` |

## G. Data model

| ID | Requirement | Code location | Built | Wired | Proven | Evidence | Gap / plan |
|---|---|---|---|---|---|---|---|
| PRD-G1 | Core graph Practice → Organization → Person → Opportunity → Engagement → Workstream → Meeting → Decision → Action → Risk → Approval → Evidence → Outcome → Commercial → Financial Event → Closeout/Learning | — | No | No | No | — | S1 implements Practice, User, Organization, Person, Lead, Qualification, Opportunity, Decision, Action, Approval, Evidence, AgentRun, AgentOutput, AuditEvent. Rest Later |
| PRD-G2 | Organizations may play multiple roles; engagement role explicit | — | No | No | No | — | S1: role on lead/opportunity, not on organization |
| PRD-G3 | Opaque stable IDs, lifecycle states, UTC timestamps, provenance (CURSOR_RULES) | — | No | No | No | — | S1 schema conventions |
| PRD-G4 | Idempotency and duplicate prevention for side effects | — | No | No | No | — | S1 operation keys; org/lead dedupe |
| PRD-G5 | Versioned schema changes | — | No | No | No | — | S0 migration tooling |

## H. Security / privacy

| ID | Requirement | Code location | Built | Wired | Proven | Evidence | Gap / plan |
|---|---|---|---|---|---|---|---|
| PRD-H1 | Private by default; least privilege | — | No | No | No | — | S1 |
| PRD-H2 | Client/provider isolation | — | No | No | No | — | S1 organization-scoped access + isolation tests (engagement-level Later) |
| PRD-H3 | No plaintext secrets | — (none present) | n/a | n/a | n/a | Scan clean (`CURSOR_AUDIT.md` §1.1 #6–7) | S0 env policy; secret scan in CI |
| PRD-H4 | Explicit lifecycle states | — | No | No | No | — | S1 |
| PRD-H5 | Source provenance | — | No | No | No | — | S1 evidence model |
| PRD-H6 | Auditable consequential actions | — | No | No | No | — | S1 append-only `audit_event` |
| PRD-H7 | Secure backups; retention/deletion rules | — | No | No | No | — | S1 backup drill; retention/deletion Later |
| PRD-H8 | Abuse protection for public endpoints | — | No | No | No | — | Later (with public intake) |
| PRD-H9 | Environment separation; production data never copied into dev | — | No | No | No | — | S0 env layout; synthetic fixtures only |

## I. UX

| ID | Requirement | Code location | Built | Wired | Proven | Evidence | Gap / plan |
|---|---|---|---|---|---|---|---|
| PRD-I1 | Calm professional operational interface; dark-navy-first; accessible light mode; restrained blue/teal | — | No | No | No | — | Tokens defined in `UI_UX_SYSTEM.md`; S1 implements |
| PRD-I2 | Clear hierarchy, tables, status labels | — | No | No | No | — | Patterns in `UI_UX_SYSTEM.md` §8–9 |
| PRD-I3 | Responsive desktop/mobile | — | No | No | No | — | S1 slice pages; full QA Later |
| PRD-I4 | No giant marketing hero inside private app | — | No | No | No | — | Design rule |
| PRD-I5 | Accessibility baseline (launch gate) | — | No | No | No | — | S1 axe + keyboard pass on slice pages |

## J. RAS standard and launch gates (union of PRD §Launch gates and ONION_STATE §13)

| ID | Gate | Code location | Built | Wired | Proven | Evidence | Gap / plan |
|---|---|---|---|---|---|---|---|
| GATE-01 | Approved repository / deployment boundary (standalone repository) | — | No | No | No | Hosting exception approved for source only | **DR-01** |
| GATE-02 | Standalone deployment | — | No | No | No | — | DR-05 |
| GATE-03 | Authentication | — | No | No | No | — | S0/S1 |
| GATE-04 | Authorization | — | No | No | No | — | S1 |
| GATE-05 | Secure shared persistence | — | No | No | No | — | DR-02 → S0 |
| GATE-06 | Client/provider isolation | — | No | No | No | — | S1 partial (org scope) |
| GATE-07 | Real Google integration as claimed | — | No | No | No | — | S1 identity only; labels enforce truthfulness |
| GATE-08 | End-to-end intake | — | No | No | No | — | Later |
| GATE-09 | Privacy retention / deletion | — | No | No | No | — | Later |
| GATE-10 | Abuse controls | — | No | No | No | — | Later |
| GATE-11 | Tested backup / restore | — | No | No | No | — | S1 drill |
| GATE-12 | Operational alerts | — | No | No | No | — | S1 minimum (health + error capture); full Later |
| GATE-13 | Commercial lifecycle | — | No | No | No | — | Later |
| GATE-14 | Onboarding / offboarding | — | No | No | No | — | Later |
| GATE-15 | Browser / mobile QA | — | No | No | No | — | S1 slice pages only |
| GATE-16 | Accessibility baseline | — | No | No | No | — | S1 slice pages |

## K. Pilot success and productization gate

| ID | Requirement | Status | Note |
|---|---|---|---|
| PRD-K1 | Pilot success criteria after 3–10 engagements (context not lost; meetings → decisions/actions; risks early; evidence-backed recommendations; traceable commitments; measurable outcomes; closeout learning; reduced Google duplication; AI saves time without violating authority; friction reveals next automation) | Not measurable yet | Sprint 1 instruments the first three measurable signals: lead-to-decision time, evidence coverage per qualified lead, human correction rate on AI proposals |
| PRD-K2 | Productization gate: external collaboration, multi-tenant, PostgreSQL canonical state, capability-based authz, APIs/connectors, benchmarks, SaaS packaging only after evidence | Respected | DR-02 keeps Postgres migration cheap without adopting it prematurely |

---

## Coverage summary

| Group | Requirements | Built | Wired | Proven | Targeted by S0/S1 |
|---|---:|---:|---:|---:|---:|
| A Vision/users | 6 | 0 | 0 | 0 | 5 |
| B Lifecycle | 8 | 0 | 0 | 0 | 2 |
| C Product areas | 11 | 0 | 0 | 0 | 5 (3 core, 2 minimal) |
| D Public/intake | 4 | 0 | 0 | 0 | 2 (schema + quarantine, internal channel) |
| E Google | 7 | 0 | 0 | 0 | 4 (identity wired; 3 linked-only) |
| F AI governance | 9 | 0 | 0 | 0 | 9 (3 capabilities) |
| G Data model | 5 | 0 | 0 | 0 | 5 |
| H Security | 9 | 0 | 0 | 0 | 7 |
| I UX | 5 | 0 | 0 | 0 | 5 |
| J Launch gates | 16 | 0 | 0 | 0 | 8 (partial) |
| K Pilot/productization | 2 | — | — | — | instrumentation only |
| **Total** | **82 rows (64 requirements + 16 gates + 2 pilot criteria)** | **0** | **0** | **0** | — |

Promotion rule: a row moves to **Built** only when a code path is cited; to **Wired** only when the real dependency (database, Google identity, LLM provider, UI route) is connected in the pilot environment; to **Proven** only when the acceptance test ID, command, commit SHA and result are recorded in this file and reproducible by RAS.
