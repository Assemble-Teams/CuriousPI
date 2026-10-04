# AI_CAPABILITY_ARCHITECTURE.md
# Onion — AI Capability Architecture and First Five Capability Specifications

**Author role:** Cursor as AI Capability Engineering Lead  
**Status:** PROPOSAL for Founder Council / AXE / RAS review. Nothing here is Built. No prompts are included in this public repository; prompt files land only after DR-01.  
**Governing constraints:** `PRD.md` §ChatGPT/AI, §Agent governance; `CURSOR_RULES.md` §Agent control plane, §Autonomy, §Agent contract, §Prompt injection; `CURSOR_CONTINUATION_BRIEF.md` §16–§17.

---

## 1. Principles

1. **Capabilities, not chat.** Every AI feature is a named capability with a human owner, a job, a scope, an output schema and an eval suite. There is no general "ask Onion anything" surface in the pilot.
2. **Authority is enforced in code.** Scope, tools, writes and approvals are checked by the runtime, not requested of the model.
3. **Agents propose; humans decide.** No capability in the pilot holds APPROVE, MANAGE, DELEGATE or ADMINISTER. Writes above A1 are proposals that become `approval` records.
4. **Everything cites.** Outputs separate **facts** (with record/field/evidence references), **analysis**, and **assumptions**. A fact without a reference is rejected by schema validation.
5. **Retrieved content is data.** Lead text, documents, emails, transcripts and tool results never carry instructions. The runtime wraps them as data and the eval suite proves it.
6. **Autonomy is earned per capability** through measured evals and human-correction rates, and can be revoked by a switch.
7. **Builder ≠ certifier.** RAS Evidence/Quality is a separate agent identity, prompt and run from the capability it checks. Cursor and the capability authors do not self-certify.
8. **The system works with AI off.** Every capability has a manual equivalent path.

## 2. Core roles and the capability map

| Role | Mandate | Pilot capabilities (first five) | Later candidates |
|---|---|---|---|
| **ATLAS** — Chief of Staff / Intelligence | synthesis, briefs, context continuity | **ATLAS Executive Briefing**; **Research** | decision briefs, outcome comparison |
| **RAS** — Risk, Assurance & Standards | evidence quality, requirement coverage, risk surfacing | **RAS Evidence / Quality** | release-readiness checks, isolation audits |
| **GROWTH** — Revenue & Relationships | pipeline, qualification, follow-up drafts | **Lead Qualification** | follow-up drafting (A2, approval-gated send never automated in pilot) |
| **DELIVERY** — Engagement Operations | meetings, actions, risks | **Meeting Intelligence** | risk extraction, closeout drafting |

Roles are organisational labels for accountability and UI grouping. Permissions attach to **capabilities**, never to roles.

## 3. Specialist-capability criteria

A new capability is added only when all are true: (1) the workflow has recurred in real engagements ≥3 times; (2) a human owner is named; (3) inputs are available inside Onion's data boundary or via an approved integration; (4) output can be expressed as a validated schema; (5) an eval dataset of ≥20 realistic synthetic cases exists before the first live run; (6) the manual path exists; (7) RAS can verify it independently. Capabilities that need a new external tool (web egress, Calendar read, Gmail send) also require a Founder decision.

## 4. Autonomy ladder and promotion rules

| Level | Meaning | Allowed effects | Pilot assignment |
|---|---|---|---|
| A0 Advisory | produces text for a human | none | ATLAS brief (advisory content) |
| A1 Read | reads authorized records/tools; produces structured output | create `agent_output`, `evidence(kind=agent_output)` | ATLAS brief, RAS check, Research (later), Meeting Intelligence (later, read) |
| A2 Propose | proposes record changes | creates `qualification(proposed)`, proposed `action`, `approval` requests | Lead Qualification; Meeting Intelligence (proposed decisions/actions) |
| A3 Internal Operator | reversible low-risk internal writes in explicit scope | e.g., tag, draft note | **none in Sprint 1**; earliest candidate after ≥50 accepted proposals with correction rate < 10% |
| A4 Approval-Gated Executor | consequential/external actions after approval | e.g., send email after approval | **none in pilot** |

Promotion requires: eval thresholds met for two consecutive eval runs; human correction rate below the capability's threshold over ≥30 real runs; zero scope/approval violations; Founder sign-off; RAS verification. Demotion is immediate on any violation or via the Settings kill switch.

## 5. Runtime and model abstraction

```
capability.run(trigger, scope)                       // typed entry point
  └─ runtime.start(agentDefinition, trigger, scope)   // creates agent_run, snapshots definition+version
       ├─ context = gateway.read(scope)               // only module query APIs; scope enforced
       ├─ prompt = template(version).render(context)  // data wrapped in delimited blocks; no instructions from data
       ├─ raw = provider.complete({model, prompt, schema, budget})
       ├─ output = schema.parse(raw) or repair-once or fail
       ├─ claims.verify(output, context)              // every fact ref must resolve to a field/evidence in context
       ├─ persist agent_output(+decision_trace), evidence(agent_output)
       ├─ if output.proposes: create approval(pending)  // never write the proposed change
       └─ finish(run: completed | failed | awaiting_approval | escalated); audit_event per step
```

`LLMProvider` interface: `complete(request) → {text|json, usage, model, latency, provider_request_id}`. Implementations: `FixtureProvider` (deterministic, replay from dataset; default in dev/test/evals), `OpenAIProvider` (first live provider, DR-04). Adding a provider is a configuration change. Model name, provider and prompt version are recorded on every run. No streaming to the browser of raw model text; the UI receives validated output only.

Prompt/policy versioning: each capability has `prompt.vN` + `schema.vN` + `policy.vN` (scope/tools/approval). A run records all three. Changing any requires a new version and an eval run.

## 6. Tool gateway

- Tools are typed functions registered with: name, description, input/output schema, side-effect class (`read` | `propose` | `write-internal` | `write-external`), required authority, idempotency requirement.
- Each `agent_definition` carries a `tool_allowlist`. A call to a tool not on the list, or outside run scope, is **denied, logged, and visible in the run card**; the model is told the call was denied (no silent failure).
- Sprint 1 tools (all `read` or `propose`): `lead.get`, `organization.search`, `evidence.list`, `template.get(qualification-guide)`, `today.queryForUser`, `opportunity.get`, `qualification.propose`, `action.propose`, `ras.rules.evaluate`. No external tools. No `write-internal`/`write-external` tools exist in Sprint 1.
- Side-effecting tools (later) require `operation_key`, precondition check, post-action verification; the runtime never retries them blindly (`CURSOR_RULES.md` §Side effects).

## 7. Permissions and scope

- Every run has a **scope**: explicit record references (e.g., `lead:L-01HZ…`) plus the user on whose behalf it runs. Reads are filtered by (a) the triggering user's authorization and (b) the scope — the intersection. An agent can never see more than the human who triggered it.
- Data classification is honoured: `restricted` fields (future: margins, internal assessments) are excluded from context unless the capability policy includes them *and* the user may view them.
- Agents are actors in the authorization model with their own identity (`agent:lead-qualification@v1`), never a service account with blanket rights.

## 8. Approvals

- Proposals create `approval` records with: requester (agent run), subject, request kind, options, evidence refs, risk level (from capability policy + output), consequence preview (generated by executing the target command in dry-run mode).
- Approvers: Principal in the pilot. Approval executes the command under the **approver's** authority with the agent run cited as origin; the audit event records both.
- Expiry: approvals expire (default 14 days) and cannot be executed afterwards without re-proposal.
- No approval may be granted by an agent, by a scheduled job, or by bulk action.

## 9. Run lifecycle

`queued → running → (awaiting_approval) → completed | failed | cancelled | escalated`

- `escalated` = the capability's policy says a human must look (ambiguity, suspected injection, scope conflict, low confidence). Escalation creates an `action` for the owner with the reason.
- Runs are immutable after a terminal state. Re-running creates a new run linked by `supersedes_run_id`.
- Timeouts: per capability (default 60s); on timeout the run is `failed`, nothing is written, the UI shows a specific message, and the manual path is offered.
- Stop conditions: budget exceeded, schema failure after one repair, scope denial, injection flag at high confidence, provider error.

## 10. Evidence and provenance

- `agent_output.fact_claims[] = {text, refs: [{type: field|evidence|record, id, path}]}`; `analysis_claims[]`; `assumptions[]`; `risks[] = {text, severity}`; `proposed_action?`.
- `decision_trace = {evidence_considered[], rules_applied[], assumptions[], result, risk, approval_required}` — concise; no chain-of-thought stored or displayed.
- Every output is itself an `evidence` record (`kind=agent_output`, `source_system=llm`, `integration_mode=n/a`) attached to its subject, so later human decisions can cite it.
- Claim verification is mechanical: every `ref` must resolve inside the run's context snapshot, else the run fails with `unresolved_reference` — the primary hallucination guard.

## 11. Observability and cost controls

- Structured log per step with `run_id`, `agent_id@version`, `user_id`, `operation_key`, latency, tokens, cost estimate, tool calls with allow/deny, outcome.
- Per-capability counters: runs/day, failures, denials, escalations, cost/day, median latency, human correction rate.
- Budgets: per-capability daily token/cost cap and per-run cap; exceeding → runs refuse with a visible message; Principal can raise caps in Settings (audited).
- Alerts (minimum, Sprint 1): provider error rate > 20% over 15 min; any scope/tool denial; any `unresolved_reference` in live runs; budget at 80%.
- Eval results are stored with dataset hash, prompt/schema/policy versions, provider, and commit SHA.

## 12. Evaluation harness

- Location: `onion/app/evals/`; runs in CI with `FixtureProvider`; a capped live smoke run is executed manually before Wired and recorded.
- Dataset: synthetic, obviously fictional organisations and people; balanced across engagement model, organization role, completeness, duplicates, injection, and "should escalate" cases. ≥20 cases per capability in Sprint 1, growing with every human correction (corrections are converted into new cases after review).
- Metrics (from `CURSOR_RULES.md`): task success; evidence accuracy; source correctness; hallucination rate (unresolved refs + unsupported facts); tool selection correctness; scope compliance; approval compliance; duplicate action rate; human correction rate; escalation quality; latency; cost.

| Capability | Must-pass thresholds (Sprint 1, fixture provider) |
|---|---|
| Lead Qualification | fact accuracy ≥ 0.95; unsupported facts = 0; scope/approval violations = 0; injection cases: 0 behaviour change; schema validity ≥ 0.98; escalates on all designated ambiguous cases |
| ATLAS Executive Briefing | every bullet cites ≥1 record; unresolved refs = 0; no record outside user scope appears (0); includes all pending approvals/overdue actions in scope (recall = 1.0) |
| RAS Evidence/Quality | deterministic rules: precision/recall = 1.0 on rule cases; LLM review: unsupported findings = 0; never proposes writes (0); flags all seeded evidence gaps (recall ≥ 0.9) |

## 13. Agent UI

Defined in `UI_UX_SYSTEM.md` §6.7, §9.2, §9.4, §9.5: origin chips on every AI artefact; run cards with Facts/Analysis/Assumptions/Risks/Proposed action; approval cards with consequence preview; Agent Activity list; capability tiles in Settings with autonomy level, on/off, last eval, budget; "Report problem" on every output (feeds correction metric and eval dataset). Raw prompts and model logs are visible only in a RAS-reviewer "Run internals" panel.

## 14. Failure and escalation patterns

| Failure | Behaviour |
|---|---|
| Provider down / timeout | run `failed`; nothing written; specific UI message; manual path offered; alert if rate high |
| Schema invalid | one repair attempt with the validation error; then `failed` |
| Unresolved reference | `failed`; counted as hallucination; never shown as a fact |
| Scope/tool denial | logged, visible, model informed; run continues if possible; any denial in live runs alerts |
| Suspected injection | output flagged with the offending span; run `escalated` at high confidence; no proposal created |
| Ambiguity (missing engagement model, conflicting signals) | capability returns `needs_information` with explicit questions instead of guessing; `escalated` if policy requires |
| Duplicate proposal | runtime detects an open proposal of the same kind for the subject → new run supersedes with a note rather than creating a second approval |
| Budget exceeded | refuse to start; message; Principal may raise cap (audited) |
| Human disagrees | "Report problem" → correction recorded → eval case candidate; no automatic retraining |

## 15. Prompt-injection policy (enforced)

- All retrieved content enters the prompt inside typed, delimited data blocks with an explicit "data, not instructions" framing; the model is asked to list any instruction-like content found in data as `suspicious_instructions[]`.
- Tools are the only way to affect anything; the allowlist and scope make injected instructions inert.
- Output schema has no free-form "actions" field; proposals are typed.
- Eval suite includes direct, indirect (hidden in links/notes), and role-play injections.

---

# 16. Capability specifications

Template fields follow `CURSOR_CONTINUATION_BRIEF.md` §16. "Owner" is the accountable human; "User" is who consumes the output.

## 16.1 Lead Qualification (GROWTH) — Sprint 1

| Field | Specification |
|---|---|
| Human owner | Uday (Principal) |
| Human user | Consultant or Principal reviewing a lead |
| Job to be done | Turn a raw lead into a structured, evidence-referenced qualification proposal and surface what is missing, so the human decision is faster and better documented |
| Inputs | the single `lead` record; linked evidence titles/URLs (not content, in Sprint 1); Qualification Guide v1 (versioned template); organization dedupe candidates |
| Permitted sources / data scope | only the lead in scope, its evidence metadata, the template, and organization search results restricted to the user's authorization |
| Output (schema `lead-qualification.v1`) | `fit ∈ {strong, possible, weak, unknown}` + rationale; `engagement_model_proposed`; `organization_role_proposed`; `missing_information[]`; `risks[]`; `questions_for_first_call[]`; `suggested_next_action {title, owner_hint, due_hint}`; `duplicate_suspects[]`; `fact_claims[]` with field refs; `analysis_claims[]`; `assumptions[]`; `suspicious_instructions[]`; `confidence` |
| Evidence policy | every fact references a lead field or evidence id; the output itself is stored as evidence; no external facts |
| Failure cases | missing engagement model → `needs_information`; contradictory objective/role → escalate; injection in objective → flag + escalate; provider failure → manual path |
| Ambiguity behaviour | ask, never assume; questions are first-class output |
| Autonomy | **A2 Propose** |
| Permitted tools | `lead.get`, `evidence.list`, `template.get`, `organization.search`, `qualification.propose`, `action.propose` |
| Forbidden actions | changing lead status; creating organizations/opportunities; contacting anyone; reading other leads; pricing or commercial suggestions |
| Approval gates | proposal becomes `qualification(proposed)` + optional `approval`; only a Principal `decision` changes state |
| Evaluation dataset | ≥20 synthetic leads: both engagement models, client/provider/partner roles, complete/incomplete, duplicates, 6 injection cases, 3 ambiguous |
| Success metrics | fact accuracy; schema validity; escalation correctness; human correction rate (target trend ↓); lead-to-decision time (H1) |
| UI surface | Lead record › Qualification panel (side-by-side with human qualification); Approvals queue (if proposal routed); Agent Activity |
| Audit events | `agent_run.started/finished`, `tool.call.allowed/denied`, `qualification.proposed`, `approval.requested`, `evidence.created(agent_output)`, `run.escalated` |
| Cost/rate | ≤ 1 run per lead per trigger; daily cap configurable |
| Human override | Principal can reject/supersede; capability kill switch |

```
AI CAPABILITY REVIEW — Lead Qualification
Capability: Lead Qualification v1 · Human owner: Uday · User: Consultant/Principal · Objective: structured qualification proposal
Agent: agent:lead-qualification@v1 (role GROWTH) · Autonomy level: A2
Data scope: single lead + evidence metadata + template + org search (user-authorized) · Tools: §16.1
Writes: proposals only · Approval gates: Principal decision
Evidence policy: field/evidence refs mandatory · Known failure modes: §16.1
Eval coverage: §12 thresholds · Cost/rate: capped · Human override: reject/supersede/kill switch
RAS concerns: anchoring of human decisions; injection via free text — both measured
Status: PROCEED WITH CONDITIONS (DR-01, DR-04; eval dataset before first live run)
```

## 16.2 ATLAS Executive Briefing (ATLAS) — Sprint 1

| Field | Specification |
|---|---|
| Human owner | Uday |
| Human user | Principal (daily/portfolio brief); Consultant (own-scope brief); anyone opening an opportunity (record brief) |
| Job to be done | Before a decision or a call, know in under two minutes: what changed, what needs me, what is at risk, what evidence is missing — with links |
| Inputs | records within the user's scope: leads, opportunities, decisions, actions, approvals, evidence, recent agent runs, RAS findings |
| Permitted sources / data scope | Onion records only (no documents' content, no email content in Sprint 1); scope = user authorization ∩ (portfolio or single record) |
| Output (schema `executive-brief.v1`) | sections in fixed order: `needs_you[]`, `changes_since_last_brief[]`, `due_and_overdue[]`, `risks[]`, `evidence_gaps[]`, `agent_activity[]`, each item `{text, refs[]}`; `generated_at`; `scope_summary`; `assumptions[]` |
| Evidence policy | every item cites ≥1 record id; items without refs are dropped by validation; brief is stored as evidence of kind `agent_output` |
| Failure cases | empty scope → short honest brief ("nothing pending"); provider failure → "brief unavailable", lists built deterministically from queries still shown |
| Ambiguity behaviour | none to resolve — descriptive only; no recommendations in v1 beyond flagging |
| Autonomy | **A1 Read** (content advisory A0) |
| Permitted tools | `today.queryForUser`, `opportunity.get`, `lead.get`, `evidence.list`, `approval.listPending`, `action.listDue`, `run.listRecent` |
| Forbidden actions | any proposal or write; reading outside user scope; recommending commercial terms |
| Approval gates | none (no writes) |
| Evaluation dataset | ≥20 synthetic portfolio snapshots incl. empty, overloaded, scope-restricted consultant, stale data |
| Success metrics | recall of pending approvals/overdue actions = 1.0; unresolved refs = 0; out-of-scope leakage = 0; user "useful" rating; time-to-first-action after reading |
| UI surface | Today › Executive brief (collapsed, with timestamp and "sources"); Opportunity › AI › Brief; Agent Activity |
| Audit events | `agent_run.started/finished`, `tool.call.*`, `evidence.created(agent_output)`; brief *views* are not audited (not consequential) |
| Cost/rate | on demand + at most one scheduled morning brief per user (scheduling itself is later) |
| Human override | collapse/disable per user; kill switch |

```
AI CAPABILITY REVIEW — ATLAS Executive Briefing
Agent: agent:atlas-executive-briefing@v1 · Autonomy: A1 (advisory content) · Writes: none · Approval gates: n/a
Data scope: user-authorized Onion records · Tools: read-only queries
Known failure modes: stale or incomplete brief; mitigated by deterministic query sections shown regardless of LLM
Eval coverage: recall/leakage/refs · RAS concerns: brief mistaken for truth → every line cites; "AI" origin chip
Status: PROCEED WITH CONDITIONS (DR-01, DR-04)
```

## 16.3 Research (ATLAS) — after Sprint 1 (requires web egress decision)

| Field | Specification |
|---|---|
| Human owner | Uday |
| Human user | Consultant/Principal preparing discovery, diagnostic or provider assessment |
| Job to be done | Assemble a sourced background pack on an organization, market or technology question with every statement linked to a retrievable source and dated |
| Inputs | a research question + subject record; optional seed URLs |
| Permitted sources | allow-listed web domains via a server-side fetch tool with egress logging; Onion records in scope; **no** Google Drive content until Drive read is Wired/Proven |
| Output (schema `research-pack.v1`) | `question`, `findings[] {text, source_url, retrieved_at, quote_span}`, `uncertainties[]`, `conflicting_sources[]`, `suggested_questions[]`; stored as evidence with each source as a `link` evidence record |
| Evidence policy | no finding without URL + retrieved_at; quotes preserved; opinions labelled analysis |
| Failure cases | paywalled/blocked sources → listed as unavailable; low-quality sources → flagged; injection in fetched pages → data only |
| Ambiguity | returns clarifying questions when the subject is ambiguous (e.g., homonymous companies) |
| Autonomy | A1 Read |
| Tools | `web.fetch(allowlist)`, `web.search(provider TBD)`, `evidence.create(link)` (A1 write of evidence only), record reads |
| Forbidden | contacting anyone; scraping personal data beyond business contact context; reading other engagements |
| Approval gates | none for reading; adding a domain to the allowlist is a Principal setting |
| Eval dataset | ≥20 questions with gold sources; injection pages; conflicting-source cases |
| Success metrics | source correctness; citation completeness; staleness; user usefulness |
| UI surface | Opportunity/Engagement › AI › Research; Evidence list |
| Audit events | run events; `egress.fetch(domain, bytes)`; `evidence.created` |
| Decision required | web egress + search provider = new external capability → Founder decision before build |

## 16.4 RAS Evidence / Quality (RAS) — Sprint 1

| Field | Specification |
|---|---|
| Human owner | Uday (as RAS accountable human); designed to be handed to an independent reviewer |
| Human user | Principal; RAS reviewer |
| Job to be done | Independently check that a record and the decisions/AI outputs attached to it meet Onion's evidence and workflow standards, and surface gaps with severity — without being able to change anything |
| Inputs | subject record (lead/opportunity) with its decisions, qualifications, evidence, actions, approvals and recent agent outputs |
| Permitted sources / data scope | subject in scope + its graph; rules library; **never** the prompt or internals of the capability being checked (independence) |
| Output (schema `ras-findings.v1`) | `rule_results[] {rule_id, pass|fail, refs[]}` (deterministic); `review_findings[] {text, severity ∈ {blocker, high, medium, low, verified}, refs[]}` (LLM-assisted); `evidence_gaps[]`; `unsupported_claims[]` (claims in other outputs whose refs don't support them); `summary` |
| Deterministic rule set v1 | decision has rationale; qualified lead has ≥1 evidence; organization role explicit; engagement model explicit; next action exists with owner and due date; consent recorded; no duplicate open opportunity for same organization+model; every AI fact claim resolves; approval exists for any proposal executed |
| Evidence policy | findings cite records; "verified" findings also cite |
| Failure cases | provider failure → deterministic rules still run and are shown; LLM section marked unavailable |
| Ambiguity | reports uncertainty as a finding, does not resolve it |
| Autonomy | **A1 Read** |
| Tools | record reads, `ras.rules.evaluate`, `evidence.create(agent_output)` |
| Forbidden | changing any record; approving/blocking anything; proposing actions (it may *recommend* in text); running on itself |
| Approval gates | none (no writes); findings may be "accepted with note" by the Principal (a human `decision`) |
| Eval dataset | ≥20 synthetic records with seeded gaps and seeded unsupported claims |
| Success metrics | rule precision/recall = 1.0; seeded-gap recall ≥ 0.9; unsupported findings = 0; time-to-triage |
| UI surface | Lead/Opportunity › AI › Quality; Agent Activity; Today › Needs you (blocker findings) |
| Audit events | run events; `ras.findings.recorded`; `ras.finding.accepted_with_note(decision)` |
| Independence rule | separate agent id, prompt, policy and run; cannot share a run or context cache with Lead Qualification or ATLAS |

```
AI CAPABILITY REVIEW — RAS Evidence / Quality
Agent: agent:ras-evidence-quality@v1 · Autonomy: A1 · Writes: findings as evidence only · Approval gates: n/a
RAS concerns: must not become a rubber stamp — deterministic rules are visible and reproducible; LLM findings are advisory
Status: PROCEED WITH CONDITIONS (DR-01, DR-04; rule set v1 ratified by Founder Council)
```

## 16.5 Meeting Intelligence (DELIVERY) — after Sprint 1 (requires Calendar/Meet/transcript decision)

| Field | Specification |
|---|---|
| Human owner | Uday |
| Human user | Consultant/Principal before and after meetings |
| Job to be done | Before: a brief with attendees, open decisions, actions, risks and questions. After: proposed decisions, actions (owner/due), risks and evidence gaps extracted from notes/transcript, for human confirmation |
| Inputs | meeting record (later: Calendar event via **Synced** read), attendees ↔ people, engagement/opportunity context, notes or transcript (Google Doc/Meet transcript via Drive read when Wired/Proven; until then, pasted notes) |
| Permitted sources | records in scope; the specific meeting document(s) explicitly attached as evidence; nothing else in Drive |
| Output (schema `meeting-intelligence.v1`) | pre: `brief` (as ATLAS format); post: `proposed_decisions[]`, `proposed_actions[] {title, owner_hint, due_hint, refs}`, `risks[]`, `open_questions[]`, `evidence_gaps[]`, `quotes[] {span, speaker?}`; all with refs to document spans |
| Evidence policy | every proposed item cites a span in the attached document; the document is Linked/Synced evidence |
| Failure cases | no transcript → pre-brief only; poor transcript → low-confidence flag; speaker ambiguity → owner left blank, never guessed |
| Ambiguity | owners/dates not inferred beyond text; questions listed |
| Autonomy | A1 (pre-brief) / **A2 Propose** (post-meeting proposals) |
| Tools | record reads; `document.read(attached only)`; `decision.propose`, `action.propose`, `risk.propose` |
| Forbidden | sending summaries externally; editing documents; reading unattached documents or calendars |
| Approval gates | each proposed decision/action is confirmed by a human (Consultant for own actions; Principal for decisions) |
| Eval dataset | ≥20 synthetic transcripts/notes with gold decisions/actions, injection lines, ambiguous ownership |
| Success metrics | action/decision recall ≥ 0.9, precision ≥ 0.9; owner-guess errors = 0; human correction rate |
| UI surface | Meeting record › AI; Delivery; Today (pre-brief) |
| Audit events | run events; `decision.proposed`, `action.proposed`, `risk.proposed`; `document.read(doc_id)` |
| Decisions required | Calendar read and Drive read scopes (Google OAuth scopes beyond identity) → Founder decision; transcript handling and retention → Privacy review |

---

## 17. Sequencing

| Order | Capability | Gate |
|---|---|---|
| Sprint 1 | Lead Qualification, ATLAS Executive Briefing, RAS Evidence/Quality | DR-01, DR-02, DR-04 |
| Next | Meeting Intelligence | Google Calendar/Drive read scopes decision; Privacy review for transcripts |
| Next | Research | web egress + search provider decision; allowlist governance |
| Later | A3 promotion for any capability | ≥50 accepted proposals, correction < 10%, zero violations, Founder + RAS |

## 18. What is explicitly not built

General chat; agent-to-agent delegation; scheduled autonomous runs (beyond a single morning brief, later); any external communication tool; any write to Google; agents approving, deleting, or changing permissions; fine-tuning on practice data; storing chain-of-thought.
