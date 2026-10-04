# UI_UX_SYSTEM.md
# Onion — UI / UX System and Whole-Journey Design

**Author role:** Cursor as UI / Experience Design Lead and Creative Product Design Lead  
**Status:** PROPOSAL for AXE Experience Review. Nothing here is Built. Visual identity is original to Onion and deliberately distinct from Assemble Teams / GameChangers / CuriousPI.  
**Baseline mandate:** `PRD.md` §UX, `CURSOR_CONTINUATION_BRIEF.md` §14–§17.

---

## 1. Design stance

Onion is an **operations console for a consulting practice**, not a dashboard product. The person using it is either deciding, preparing, recording, or checking. Every screen must answer four questions in reading order:

1. **What is this?** (type, title, who it concerns)
2. **What state is it in, and who owns it?**
3. **What happens next, by whom, by when?**
4. **What is the evidence, and who/what produced it?**

Everything else is secondary and collapsible.

Non-negotiables:

- Decisions and approvals are the most important objects in the UI. They get the best real estate and the clearest affordances.
- Human and AI origin is always distinguishable by **text label + icon + colour**, never colour alone.
- Nothing is labelled "synced", "live", "automated" or "done" unless the underlying capability is Wired and Proven. Labels are derived from the system, not typed into copy.
- No hero sections, no decorative illustrations, no vanity metrics, no motion that does not teach state.
- The system is fully usable with every AI capability switched off.

## 2. Personas and critical jobs

| Persona | Context | Critical jobs | Device | What they must never experience |
|---|---|---|---|---|
| **Principal (Uday)** | interrupt-driven; portfolio-wide authority; sometimes on phone between meetings | see what needs *him* (approvals, decisions, overdue, risks); decide with rationale in under a minute; trust that nothing consequential happened without him; get a brief before a call | desktop + mobile | hidden AI writes; approval without consequence shown; a queue that mixes "FYI" with "needs you" |
| **Consultant** | desk-bound, keyboard-heavy; owns assigned leads/records | enter a lead well; prepare qualification; attach evidence; keep next actions current; prepare for meetings | desktop | seeing records outside their assignment; ambiguity about who owns a next action |
| **RAS reviewer** (role; may be Uday or an independent later) | periodic verification | trace any fact to source; see who approved what; reproduce an agent run's inputs/outputs; export an audit timeline | desktop | missing provenance; unlabelled AI content |
| **Agent (as visible actor)** | ATLAS / RAS / GROWTH / DELIVERY capabilities | appear as a named, versioned actor; propose; cite; never impersonate a person | — | being rendered as a human avatar or as anonymous "system" |
| External client/provider | **deferred** | — | — | — |

## 3. Navigation model

**App shell:** left rail (collapsible to icons), top bar, content, optional right context rail on record pages.

Top bar: global search / command palette (`⌘K`/`Ctrl+K`), **Needs you** counter (pending approvals + decisions + overdue actions, Principal only sees approvals), environment badge (`PREVIEW` / `PILOT`, never hidden), user menu.

Left rail groups (only surfaces that exist are rendered — no "coming soon" entries):

| Group | Surface | Sprint 1 | Notes |
|---|---|---|---|
| — | **Today** | yes | personal operating view (not a dashboard) |
| Pipeline | **Leads & Organizations** → Leads · Organizations · Opportunities | yes | opportunities live here until Engagements exists |
| Engagements | Engagements · Delivery | later | — |
| Commercial | Commercial · Finance | later | — |
| Growth | Growth & Signals | later | — |
| System | **Knowledge & Templates** | read-only | Qualification Guide v1 |
| System | **Evidence / Audit** | yes | evidence list + audit timeline |
| System | **Agent Activity / Approvals** | yes | runs, pending approvals, capability status |
| System | **Settings / Connections** | yes | identity, roles, capability switches, connection labels |

Breadcrumb: `Leads & Organizations › Leads › Acme Fictional Ltd — AI adoption inquiry`.

Record page anatomy (used by every entity):

```
┌ Record header ─────────────────────────────────────────────────────────────┐
│ LEAD · L-01HZX…  [new]  Owner: Priya K.   Client · Client Transformation    │
│ Acme Fictional Ltd — AI adoption inquiry             [Request qualification]│
├ Next strip ───────────────────────────────────────────────────────────────┤
│ ▶ Next: Schedule discovery call · Owner Priya K. · Due Thu 09 Oct   (edit)   │
├ Tabs: Overview | Evidence (3) | Activity | AI (2) ───────────┬ Context ─────┤
│ …                                                             │ Decisions    │
│                                                               │ Approvals    │
│                                                               │ Evidence gaps│
│                                                               │ Related      │
└───────────────────────────────────────────────────────────────┴──────────────┘
```

One primary action per state. Overflow menu for the rest. The Next strip is always present; absence of a next action is a visible warning, not empty space.

## 4. Page inventory (Sprint 1 pages in bold)

| Page | Purpose | Primary action | Key content | States |
|---|---|---|---|---|
| **Sign in** | Workspace sign-in | Continue with Google Workspace | practice name, environment badge, domain restriction notice | error: domain not allowed |
| **Today** | "what needs me" | varies per row | sections in fixed order: Needs you (approvals, decisions) · Due/overdue actions · New leads · Executive brief (ATLAS, collapsed by default with timestamp + "sources") · Agent runs awaiting review | empty per section; brief unavailable |
| **Leads list** | triage | New lead | table: title · organization · state · engagement model · role · owner · next action due · age; filters by state/owner; saved views "Mine", "Needs decision" | empty; filtered-empty |
| **New lead** | capture | Save as new | single-column grouped form (identity, organization, engagement, objective, timeline/scale, contact, consent, source); duplicate warning inline | validation; duplicate found |
| **Lead record** | review & decide | state-dependent: Request qualification → Record decision | Overview (intake fields), Qualification panel (human / AI proposals side by side), Evidence, Activity, AI; decision drawer | awaiting proposal; proposal failed; decision recorded |
| **Decision drawer** | record decision | Record decision | options (Qualify / Nurture / Needs info / Decline), mandatory rationale, evidence checklist, consequence preview ("Will create Organization + Opportunity"), next action fields | validation; consequence preview |
| **Organizations list / record** | dedupe & context | New organization | legal/display name, domain, roles played (per opportunity), people, opportunities, evidence | — |
| **Opportunity record** | continue | Add next action | stage, role, engagement model, source lead (link), evidence, actions, brief | — |
| **Approvals queue** | Principal decides | Approve / Reject / Needs info | approval cards (§9.4) sorted by risk then age | empty = "Nothing needs your approval" |
| **Agent Activity** | transparency | — | runs table: agent · version · objective · subject · status · started · duration · cost · outcome; capability status tiles (on/off, autonomy, last eval) | empty; capability disabled |
| **Agent run detail** | inspect | Report problem | run card (§9.5): objective, scope, sources, facts vs analysis vs assumptions, risk, proposed action, approval state, trace, cost | failed; awaiting approval |
| **Evidence list** | find source | Add evidence | evidence chips with provenance; filter by subject/source | — |
| **Audit timeline** | verify | Export | per-record and global timeline; typed actors; operation keys; before/after digest | — |
| **Knowledge: Qualification Guide** | reference | — | versioned read-only document; "used by Lead Qualification v1" | — |
| **Settings / Connections** | control | toggle capability (Principal) | identity; role; capabilities with autonomy level + on/off + last eval; connections with Linked / Synced / Writes labels and maturity (Built/Wired/Proven); LLM budget | — |

## 5. Design tokens

### 5.1 Colour — dark (default)

| Token | Value | Use |
|---|---|---|
| `bg.canvas` | `#0B1220` | app background |
| `bg.surface` | `#111A2B` | panels, tables |
| `bg.raised` | `#18233A` | menus, drawers, hover rows |
| `border.subtle` | `#223050` | dividers, table lines |
| `border.strong` | `#34456B` | inputs, focused containers |
| `text.primary` | `#E6EBF2` | body |
| `text.secondary` | `#A6B1C5` | labels, metadata |
| `text.muted` | `#718099` | placeholders, timestamps |
| `accent.blue` | `#5B93FF` | interactive: links, primary buttons, focus |
| `accent.teal` | `#2FB8A6` | AI-origin marker, secondary emphasis |
| `state.neutral` | `#8B97AD` | new, draft, duplicate |
| `state.info` | `#5B93FF` | in review, running |
| `state.success` | `#3EBE7E` | qualified, approved, proven |
| `state.warning` | `#E3A93B` | needs info, pending, overdue soon |
| `state.danger` | `#E8604E` | declined, rejected, failed, overdue |
| `focus.ring` | `#9DBDFF` 2px outside | all focusable elements |

### 5.2 Colour — light (accessible, user-selectable, follows OS by default)

| Token | Value |
|---|---|
| `bg.canvas` `#F5F7FB` · `bg.surface` `#FFFFFF` · `bg.raised` `#FFFFFF` + shadow · `border.subtle` `#DCE3EE` · `border.strong` `#B9C5D8` · `text.primary` `#0F172A` · `text.secondary` `#475569` · `text.muted` `#6B7A90` · `accent.blue` `#2563EB` · `accent.teal` `#0F8A7D` · `state.success` `#1E8E5A` · `state.warning` `#B7791F` · `state.danger` `#C8402F` |

Contrast rule: text ≥ 4.5:1, UI components and state indicators ≥ 3:1, verified per theme in CI (Playwright + axe). Status is always colour + text + (where space) icon.

### 5.3 Typography

- UI family: `Inter`, fallback `system-ui, -apple-system, Segoe UI, Roboto, sans-serif`; self-hosted, no third-party font CDN.
- Mono family: `JetBrains Mono`, fallback `ui-monospace, SFMono-Regular, Menlo, monospace` — IDs, operation keys, timestamps, amounts, run metadata.
- Scale (px / line-height): `12/16` meta · `13/18` table · `14/20` body-compact · `15/22` body · `17/24` section title · `20/28` page title · `24/32` record title. No display sizes inside the app.
- `font-variant-numeric: tabular-nums` on every numeric column, time and amount.
- Weight: 400 body, 500 labels/links, 600 titles. Nothing bolder.

### 5.4 Spacing, density, shape, elevation, motion

- 4px base grid; scale 4 · 8 · 12 · 16 · 24 · 32 · 48.
- Density: default row height 36px (compact), "comfortable" 44px toggle persisted per user. Touch targets ≥ 44px on pointer: coarse.
- Radius: 4px controls · 6px panels/cards · full for badges only.
- Elevation: borders first; one shadow level for menus/drawers/dialogs.
- Motion: 120–160ms ease-out for state transitions, drawer open, row expand. No parallax, no looping animation, no skeleton shimmer longer than 1.2s. `prefers-reduced-motion` disables all non-essential transitions.
- Icons: single outline set (e.g., Lucide), 16px in tables, 20px in headers; never decorative-only.

## 6. End-to-end journeys

Each journey is designed as a sequence of **step → screen → human → AI → data written → evidence → exit criterion**. Journeys 6.1, 6.6 and 6.7 are implemented in Sprint 1; the others set direction so Sprint 1 components are built to extend, not to be replaced.

### 6.1 Lead → qualification → opportunity (Sprint 1)

| # | Step | Screen | Human | AI | Data written | Evidence | Exit |
|---|---|---|---|---|---|---|---|
| 1 | Signal arrives (email, referral, event, later: web form) | — | notices | — | — | original message URL (Gmail link) | decides to enter it |
| 2 | Submit lead | New lead | Consultant/Principal fills form; consent recorded; duplicate warning shown | — | `lead(new)`, `audit_event` | Gmail/Doc link attached at entry (optional) | lead exists in quarantine |
| 3 | Qualification | Lead record › Qualification panel | clicks *Request qualification* **or** fills manual qualification | Lead Qualification (A2) proposes fit, model, missing info, risks, questions, next action; cites fields | `agent_run`, `agent_output`, `qualification(proposed)` | agent output becomes `evidence(kind=agent_output)` | proposal visible, labelled AI, with "sources" |
| 4 | Human review | Lead record › Decision drawer | Principal reads lead + proposal(s) + evidence; edits/accepts; chooses decision; writes rationale; sets next action; sees consequence preview | — | `decision`, `qualification(accepted/rejected)`, `action` | rationale + evidence checklist | decision recorded |
| 5 | Organization/opportunity created | Opportunity record (redirect) | sees new opportunity with role + model explicit | — | `organization` (upsert), `person`, `opportunity(qualified)`, `lead(qualified)`, `audit_event` | source lead linked | opportunity owned, next action present |
| 6 | Evidence attached | Opportunity › Evidence | adds Drive/Docs/Gmail links or notes | — | `evidence(linked)` | provenance auto-captured | ≥1 evidence |
| 7 | ATLAS brief | Opportunity › AI › Brief, and Today | reads brief before call | ATLAS Executive Briefing (A1) summarises state, open questions, risks, evidence gaps, citing record IDs | `agent_run`, `agent_output` | cited records | brief inspected |
| 8 | RAS check | Lead/Opportunity › AI › Quality, and Agent Activity | reviews findings; fixes gaps or accepts with note | RAS Evidence/Quality (A1) checks rules + claims vs evidence; severity | `agent_run`, `agent_output(findings)` | findings reference records | findings triaged |

Friction removed by design: no separate "CRM" entry — organization and person are created *by the decision*; the rationale is captured at the moment of deciding; next action is mandatory so nothing exits the flow unowned; every AI output is attached to the record it concerns, not to a chat.

Failure paths designed: duplicate lead (merge or mark duplicate), missing consent (cannot submit), AI unavailable (manual qualification path identical), injection in objective text (proposal flags suspicious instructions; no behaviour change), consultant attempts decision (button absent; command denied; audit of denial).

### 6.2 Discovery → diagnostic → recommendation (later)

Screens extend the opportunity record with stages. Diagnostic work products are Google Docs **Linked** as evidence; the recommendation (Proceed / Pause / Address readiness gaps / Stop) is a `decision` with mandatory evidence checklist — the same drawer pattern as 6.1 step 4. ATLAS brief pattern reused for "pre-diagnostic brief". Meeting Intelligence feeds decisions/actions into the same `decision` and `action` primitives.

### 6.3 Proposal → contract → kickoff (later)

Commercial state is a vertical stepper on the opportunity/engagement record: Proposal · NDA/MSA/SOW · Commercial approval · Signature · Invoice/payment terms · Kickoff. Every step is an `approval` card (§9.4) because each is consequential. Terms are shown as data with "approved by / on"; the UI has no defaults for prices or percentages. Documents are Linked Google Docs.

### 6.4 Engagement → delivery → outcome → closeout (later)

Delivery page = workstreams × milestones board with risks and decisions pinned; meetings produce decisions/actions via Meeting Intelligence into the same primitives; outcome scorecard compares success criteria (set at kickoff) with evidence; closeout is a checklist whose items are evidence-backed; learning is a template-driven note that ATLAS can draft for human edit.

### 6.5 Consultant daily workflow (Sprint 1 partial via Today)

Morning: Today shows my overdue/due actions, my leads needing work, meetings today (later, Calendar), ATLAS brief for my scope. During the day: record page work; evidence capture as links. End of day: Today "unowned / stale" section nudges. Keyboard: `⌘K` → "new lead", "go to …", "record decision".

### 6.6 Founder portfolio / approval workflow (Sprint 1 partial)

Today for the Principal leads with **Needs you** (approvals, decisions) ordered by risk then age, each with consequence preview and one-tap open; then risks (later), then commercial decisions (later), then ATLAS portfolio brief, then AI runs awaiting review. Approvals are full-screen on mobile with rationale required; no swipe-to-approve.

### 6.7 Agent journey (Sprint 1)

| # | Step | What the user sees |
|---|---|---|
| 1 | Trigger (user click / event) | run appears in Agent Activity as `queued` with trigger and who triggered |
| 2 | Identity | agent name, role (GROWTH/ATLAS/RAS), version, autonomy badge (A1/A2) |
| 3 | Objective | one sentence, as given by the system, not paraphrased by the model |
| 4 | Authorized context | list of records the run may read (scope); anything outside is impossible, not merely discouraged |
| 5 | Tool use | tool calls listed with allow/deny result; denials are visible |
| 6 | Evidence | sources used, as evidence chips |
| 7 | Output | Facts (each with field/evidence ref) · Analysis · Assumptions · Risks · Proposed action |
| 8 | Approval | if a proposal needs a decision, an approval card links here; status shown |
| 9 | Result | what changed (if a human approved) with links to created records |
| 10 | Audit | every step above is an `audit_event`; "Export run" produces the trace |
| 11 | Exit | status terminal; cost; "Report problem" feeds human-correction metric |

## 7. Table pattern

- Dense by default; sticky header; first column is the record title (link); state badge in column 2; owner as initials + name text; dates show relative ("in 2d") with absolute on hover/focus and in a `title`/`aria-label`.
- Sorting on one column at a time; filters as chips above the table; saved views per user.
- Row click opens record; `Enter` opens; `Space` selects when selection is enabled. Bulk actions only for non-consequential operations (assign owner, tag). Never bulk approve.
- Pagination at 50 rows; no infinite scroll.
- Column set is fixed per table (no user column pickers in Sprint 1) to protect scanability.
- Mobile: table collapses to a list of compact cards: title, state, owner, next due.

## 8. Form pattern

- Single column, grouped with section headings; max width 640px; labels above inputs; help text below; required marked by text "(required)" not asterisk alone.
- Inline validation on blur; summary of errors at top on submit, each linking to its field; errors associated via `aria-describedby`.
- Consent is a checkbox with the exact consent text and a timestamp recorded on submit.
- Duplicate detection runs on blur of organization/email and shows a non-blocking warning with a link.
- Save state is explicit ("Saved 12:04:31") and unsaved-changes navigation guard is on.
- No auto-save on consequential forms (decision drawer); drafts are allowed for lead entry.

## 9. Status, loading, empty, error, approval, evidence and agent patterns

### 9.1 Status badge
`[●  Qualified]` — dot + text; colour from state tokens; icon optional. Lifecycle states have fixed vocabulary: Lead `new · in review · needs info · qualified · nurture · declined · duplicate`; Run `queued · running · awaiting approval · completed · failed · cancelled · escalated`; Approval `pending · approved · rejected · expired`.

### 9.2 Origin chip
`[👤 Human · Priya K.]` or `[◆ AI · Lead Qualification v1 · run 01HZ…]`. Teal + diamond icon + the word "AI" for agents; neutral + person icon for humans. Appears on every qualification, note, evidence item, action and brief.

### 9.3 Integration label
`Linked` (URL only; tooltip "Onion stores a link. Content is not read or synced."), `Synced` (shows "last synced <time>"; only rendered when the integration registry reports a passing integration test), `Writes` (approval-gated). Connections page also shows maturity: `Built` / `Wired` / `Proven`.

### 9.4 Approval card
```
┌ APPROVAL · pending · risk: medium · requested 2h ago ────────────────┐
│ Qualify lead "Acme Fictional Ltd — AI adoption inquiry"               │
│ Requested by  [◆ AI · Lead Qualification v1 · run 01HZ…]              │
│ Subject       Lead L-01HZX… → would create Organization + Opportunity │
│ Evidence      [Gmail · intro thread] [Doc · brief] [AI output]        │
│ Why           fit: strong; model: Client Transformation; gaps: budget │
│ Consequence   Creates Opportunity (stage Qualified), marks lead       │
│               qualified, assigns next action to Priya K.              │
│ Rationale (required) [______________________________]                 │
│ [Approve]  [Needs info]  [Reject]                 Audit: A-01HZ…       │
└───────────────────────────────────────────────────────────────────────┘
```
Rules: consequence text is generated from the command's dry-run, not written by hand; approve requires rationale; the approver's name and timestamp are shown after decision; the card never hides the requester's identity.

### 9.5 Agent run card
Header: agent · version · autonomy · status · started · duration · cost. Sections in fixed order: Objective · Scope · Sources · **Facts** (each `claim — source: lead.objective` or evidence chip) · **Analysis** · **Assumptions** · **Risks** · **Proposed action** · Approval · Trace (evidence considered, rules applied, result, risk, approval required) · Problems reported. "Inspect source" opens the record or external link. No raw prompts or model logs are shown to end users; they are available to the RAS reviewer role in a separate "Run internals" panel.

### 9.6 Loading
Skeleton rows matching final layout for lists; inline spinner only inside the triggering button; agent runs show a live status line ("Running · 6s · reading 3 records") not a progress bar guess. Anything beyond 10s shows "still running; you can leave this page".

### 9.7 Empty
One sentence stating what the list is + one primary action. No illustrations. Filtered-empty says which filters are active and offers "Clear filters".

### 9.8 Error
States what failed, what was preserved (e.g., "Your lead draft is saved"), what to do, and an error ID. For agent failures: "Lead Qualification v1 could not run (provider timeout). Nothing was changed. You can qualify manually." Never a generic "Something went wrong".

## 10. Mobile rules

- Must work on 390×844: Today, Approvals, Lead/Opportunity record (read), Decision drawer (full-screen), Agent run detail (read).
- Left rail becomes a bottom sheet; right context rail becomes a "Context" tab.
- No horizontal scrolling for primary content; tables → cards.
- Consequential buttons sit above the keyboard, never under a sticky bar; no swipe gestures for approve/reject.
- Minimum 44px targets; 16px inputs (prevents zoom); safe-area insets respected.

## 11. Accessibility rules (WCAG 2.2 AA baseline, launch gate)

- Full keyboard operability; logical focus order; visible 2px focus ring; skip-to-content link; `Esc` closes drawers and returns focus to the trigger.
- Landmarks (`header`, `nav`, `main`, `aside`); one `h1` per page; headings in order.
- Tables with real `<th scope>`; sortable headers announce state; row links have accessible names including title and state.
- Status never conveyed by colour alone; icons have text or `aria-label`.
- Live regions for async results (agent run completion, save state) at `polite`; errors at `assertive`.
- Forms: labels, descriptions, error association, no placeholder-as-label; consent checkbox text is the legal text.
- Motion respects `prefers-reduced-motion`; no content flashes.
- Target size ≥ 24×24 CSS px everywhere, 44px on touch.
- Automated axe in CI per page per theme; manual keyboard + screen-reader pass recorded for Proven.

## 12. Known UX risks and mitigations

| Risk | Mitigation |
|---|---|
| AI proposal anchors the human decision | proposal is collapsed until the human opens it *or* the human can choose "decide first, compare after"; correction rate measured |
| Approval fatigue turns approvals into rubber stamps | only consequential actions create approvals; consequence preview; risk-sorted queue; no bulk approve |
| Today degrades into a vanity dashboard | fixed section order; sections are queues with actions, not charts; no counters without a list behind them |
| Teal/blue proximity reduces human/AI distinction for colour-blind users | icon + word "AI" always; tested with deuteranopia simulation |
| Density hides priority | one primary action per screen; Needs-you first; typography scale, not colour, carries hierarchy |
| Terminology drift (lead / opportunity / engagement) | glossary in Knowledge; state vocabulary fixed in code and UI |
| Label truthfulness regresses (someone writes "synced" in copy) | labels rendered from integration registry; copy lint forbids the words synced/live/automatic in UI strings |
| Mobile accidental approval | full-screen approval, rationale required, confirm step |

## 13. Before / after (reported prototype → proposed)

| Aspect | Before (reported, not observed in repo) | After (proposed) |
|---|---|---|
| Structure | single `index.html`, tab per surface | app shell with record pages; journeys, not tabs |
| Identity | none; owner-operated | Workspace sign-in; typed actors; roles |
| Data | browser localStorage; JSON import/export | server-side store; audit; backups |
| Human/AI | no AI | origin chips; run cards; approvals |
| Google | manual links | Linked labels enforced; identity wired |
| Status | status labels | fixed state vocabulary + maturity labels |
| Mobile | "mobile CSS" | explicit mobile rules per page |
| Accessibility | unverified | axe + manual pass as Proven criterion |

## 14. Encode step — what becomes the design system after Sprint 1 evidence

Components proven in the slice are promoted to the shared library: `RecordHeader`, `NextStrip`, `StatusBadge`, `OriginChip`, `IntegrationLabel`, `EvidenceChip`, `ApprovalCard`, `RunCard`, `DataTable`, `DecisionDrawer`, `FormSection`, `EmptyState`, `ErrorState`. Anything not used by the slice is not built in advance.

---

## 15. AXE EXPERIENCE REVIEW (submitted)

```
AXE EXPERIENCE REVIEW

User: Principal; Consultant; RAS reviewer
Job: take a lead to an owned, evidence-backed opportunity with a recorded human decision and an inspectable AI contribution
Current friction: no system; leads and rationale live in email/notes/memory; no audit; no next-action discipline (to be confirmed in kickoff interview)
Proposed change: app shell + record-page anatomy + seven Sprint 1 pages + approval/run/evidence patterns (§3–§9)
Workflow impact: organization/opportunity creation is a consequence of the decision, not separate data entry; next action mandatory; AI attached to records, not chat
Information hierarchy: What › State/Owner › Next › Evidence; one primary action per state; Needs-you first on Today
Mobile behavior: §10
Accessibility: §11; axe + manual pass required for Proven
Failure/error behavior: §9.8 and journey 6.1 failure paths
AI implications: origin chips, run cards, consequence preview, correction reporting; usable with AI off
Security/privacy implications: consultant scoping visible in nav and lists; environment badge always visible; no external data
Evidence: none yet — prototype review session and usability test are part of Sprint 1 Test step
Tradeoffs: fixed columns and fixed Today order reduce flexibility in exchange for scanability; no bulk approve adds clicks for safety
Status: PROCEED WITH CONDITIONS — clickable prototype reviewed by AXE before implementation; kickoff interview answers incorporated; tokens contrast-verified in both themes
```
