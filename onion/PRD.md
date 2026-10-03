# Onion — Canonical Product Requirements Document

**Status:** Canonical working PRD  
**Owner:** Uday Teki  
**Stage:** Private pilot / pre-production  
**Initial validation:** 3–10 real consulting engagements

## Product vision

Onion is the private operating system for Uday’s standalone consulting practice. It should keep opportunities, diagnostics, commercial commitments, delivery, evidence, risks, finances, outcomes and organizational learning from becoming scattered across email, meetings, documents, spreadsheets and memory.

The system supports two engagement models:

1. **Client Transformation** — organizations seeking AI/adoption, product engineering, operational/process improvement, business systems, GCC strategy, dedicated engineering or managed-service capability.
2. **Technology/GCC Provider Advisory** — providers improving positioning, offering design, GTM, pipeline, sales, engineering, DevSecOps, talent, delivery, economics, enterprise readiness and customer success.

The two models share one internal operating system but must preserve appropriate confidentiality and engagement isolation.

## Strategic principles

- private-first;
- Google-first where practical;
- evidence-first;
- pilot-first;
- human authority for consequential decisions;
- build software only where it improves the consulting operating model;
- productize only after repeated real-world evidence.

No public SaaS signup in v1.

## Users

**Uday / Principal:** full authority and final approval for consequential actions unless explicitly delegated.

**Internal consultants/employees:** assigned work and authorized records only.

**External clients/providers:** deferred in the initial pilot. Future access must be engagement-scoped and must never expose internal assessments, other engagements, margins, private methodology, RAS analysis or unrelated records.

## Lifecycle

Signal → Lead → Qualification → Discovery → Diagnostic → Readiness/Opportunity Assessment → Recommendation → Opportunity → Proposal → NDA/MSA/SOW → Commercial Approval → Signature → Invoice/Payment Terms → Kickoff → Focused/Power Sprint → Implementation/Provider Delivery → Outcome Review → Acceptance → Closeout → Expansion/Referral/Follow-up → Learning.

Recommendations may be: Proceed, Pause, Address readiness gaps, or Stop.

## Power Sprint

Support a proposed 3–4 week Power Sprint as an editable engagement model. Do not hard-code unapproved pricing, percentages, commissions, revenue requirements, partnership economics or payment terms.

## Core private product areas

### Dashboard
Active engagements, pipeline, actions, milestones, risks, approvals, decisions and a currency-aware lightweight finance snapshot.

### Leads & Organizations
Organization/contact, role/type, source, needs, fit, qualification, next step, follow-up and supporting source.

### Engagements
Objective, scope/exclusions, participants, engagement type/playbook, stage, dates, success criteria, outcome measures, recommendation, source documents, risks, approvals, commercial state and closeout state.

### Delivery
Workstreams, tasks, milestones, meetings, notes, decisions, actions, risks, issues, approvals, deliverables, evidence and closeout.

### Commercial
Proposal status/date, proposed scope, NDA/MSA/SOW, commercial approval, signature, commercial start, payment terms, invoice/collection state, referral/commission information where applicable and engagement economics.

### Finance
Proposed, invoiced, collected, outstanding and expense amounts, currency, payment status, invoice reference and dates. Onion is not accounting software.

### Growth & Signals
Referrals, relationships, market signals, campaigns, inbound/outbound activity, opportunities and follow-ups.

### Knowledge & Templates
Qualification guides, diagnostics, provider/GCC assessments, readiness assessments, evidence requests, recommendations, Power Sprint plans, meeting briefs, decision/risk records, proposal/kickoff/closeout checklists, outcome scorecards and lessons learned.

### Evidence / Audit
Trace material facts to source, distinguish sourced fact from AI analysis and human decision, preserve timestamps/owners/engagement and approval evidence.

### Agent Activity / Approvals
Agent runs, requested actions, evidence, scopes, permissions, approvals, results and evaluations.

## Public website

The public site is a small front door, not the internal OS. It explains who the practice helps, what problems it works on, the approach, and a Submit Your Project / Start a Conversation intake. It must not expose internal dashboards, provider/client assessments, finances, RAS methods, private templates or internal commercial logic.

Public intake should minimally capture identity, organization, engagement type, objective/problem, timeline/scale, contact method and consent. Intake must enter review/quarantine rather than automatically becoming a trusted engagement.

## Google Workspace

Google Workspace is the preferred low-cost working environment:
Drive/Docs for documents/evidence, Sheets for lightweight structured ledgers, Forms for intake where appropriate, Gmail for communications, Calendar for schedule and Meet for meetings.

Onion should reference native Google records and avoid unnecessary duplication. Never claim synchronization that is not wired and proven.

## ChatGPT / AI

Authorized AI uses include summarization, meeting briefs, diagnostic preparation, action extraction, risk/evidence-gap identification, research, draft follow-ups, decision briefs, outcome comparison and RAS support.

Human approval is required before external communication, commitments, commercial terms, financial changes, permission changes, confidential disclosure and other consequential actions.

Core business AI roles:
- ATLAS — Chief of Staff / Intelligence
- RAS — Risk, Assurance & Standards
- GROWTH — Revenue & Relationships
- DELIVERY — Engagement Operations

Initial specialist cohort:
- ATLAS Executive Briefing
- Meeting Intelligence
- Research
- RAS Evidence / Quality
- Lead Qualification

## Agent governance

Use autonomy levels A0 Advisory, A1 Read, A2 Propose, A3 low-risk reversible Internal Operator, A4 Approval-Gated Executor. No unrestricted autonomy in the pilot.

Every agent requires identity, human owner, objective, scope, tool permissions, read/write permissions, approval policy, evidence policy, prompt/policy version, run ID, stop conditions, output schema and evaluation coverage.

## Data model

Core graph:

Practice → Organization → Person/Relationship → Opportunity → Engagement → Project/Workstream → Meeting → Decision → Action → Risk → Approval → Evidence → Outcome → Commercial Record → Financial Event → Closeout/Learning.

Organizations may play multiple roles; engagement role must be explicit.

## Security/privacy

Private by default. Require least privilege, client/provider isolation, no plaintext secrets, explicit lifecycle states, source provenance, auditable consequential actions, secure backups, retention/deletion rules, abuse protection for public endpoints and environment separation.

Production data must not be copied casually into development.

## UX

Calm professional operational interface; dark-navy-first with accessible light mode, restrained blue/teal accents, clear hierarchy/tables/status labels, responsive desktop/mobile behavior. No giant marketing hero inside the private app.

## RAS standard

Every capability reports:
- **Built** — implementation exists;
- **Wired** — connected to intended real workflow/dependency;
- **Proven** — representative acceptance test passed.

## Launch gates

Do not describe Onion as production-ready until evidence exists for:
- approved repository/deployment boundary;
- authentication and authorization;
- secure shared persistence;
- client/provider isolation;
- real Google integration as claimed;
- end-to-end intake;
- privacy retention/deletion;
- abuse controls;
- tested backup/restore;
- operational alerts;
- commercial lifecycle;
- onboarding/offboarding;
- browser/mobile QA;
- accessibility baseline.

## Pilot success

After 3–10 engagements, Onion should show that important context is not lost; meetings reliably produce decisions/actions; risks surface early; recommendations are evidence-backed; commercial commitments are traceable; delivery/outcomes are measurable; closeout captures learning; Google duplication is reduced; AI saves meaningful time without violating authority; and repeated friction reveals the next capability worth automating.

## Productization gate

Only after real engagement evidence should Onion consider scoped external collaboration, stronger multi-tenant architecture, PostgreSQL canonical application state, capability-based authorization, APIs/connectors, benchmark data, developer integrations or SaaS packaging.

## Core success principle

Onion exists to make the consulting practice more capable, evidence-driven, disciplined, scalable and learnable. The first objective is excellent engagements and a repeatable advisory system—not maximizing features or agent count.
