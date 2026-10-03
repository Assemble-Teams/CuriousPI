# CURSOR_AUDIT.md
# Onion — Repository Engineering Audit (Cycle 1)

**Audit date:** 2026-10-03 (UTC)  
**Auditor role:** Cursor — Lead Developer / Governance & QA lens (evidence provider; not RAS certifier)  
**Repository:** `Assemble-Teams/CuriousPI` (GitHub, **PUBLIC**)  
**Audited branch:** `onion-bootstrap`  
**Audited commit:** `669c356debe998845e9d8785360692cc2b2b30b2`  
**Working tree at audit start:** clean  
**Onion namespace:** `/onion`  
**Status of this document:** Evidence for Founder Council, AXE and RAS review. Nothing here is self-certified as RAS-approved.

---

## 0. Executive summary

1. **The `/onion` namespace contains documentation only.** Nine Markdown files (1,981 lines). There is no application source, configuration, dependency manifest, test, build, or deployment artifact for Onion anywhere in this repository — on any branch or in any unreachable object.
2. **The repository's own delivery claims do not match the repository.** `onion/DELIVERY-AUDIT.md` marks twelve deliverables as *Built: Yes* and `onion/README.md` instructs the reader to run `index.html` and `site.html`. Those files do not exist in any ref. The draft PR body for `onion-bootstrap` states the app source was deliberately withheld because the repository is public. So the prototype described in those documents exists — if at all — outside this repository and **cannot be audited, verified, or counted as Built from here.**
3. **Nothing Onion-related is Built, Wired, Proven, Commissioned, or Production in this repository.** Every PRD requirement traces to "no code location" (see `PRD_TRACEABILITY.md`).
4. **No runtime coupling to Assemble Teams / GameChangers / CuriousPI exists** — because no Onion runtime exists. The *organizational* coupling is real and is the controlling risk: Onion's source lives in a public repository owned by the Assemble Teams GitHub organization, whose description reads "Assemble Teams Inc - LLM". Any code, prompt, synthetic-but-realistic data, or deployment connected to this repository inherits that visibility and that ownership.
5. **The first engineering prerequisite is not a feature. It is a hosting/visibility decision and a stack decision.** Both are Founder decisions. Both are reversible today and expensive to reverse after the first commit of application code. They are presented as structured decision requests in `DECISION_REQUESTS.md` (DR-01, DR-02).
6. **Recommended next slice stands** — Lead → qualification → human review → organization/opportunity → evidence → next action → ATLAS brief → RAS check — but it must be preceded by a small **Sprint 0** (decisions + skeleton) that is described in `SPRINT_1_PROPOSAL.md`. The slice is sized so that one engineer can land it as a single modular monolith with a complete test and eval harness.

**Core test answer (required by `CURSOR_KICKOFF_PROMPT.md`):**

> Does the current system reliably help the consulting practice acquire, diagnose, contract, deliver, measure, close, and learn from engagements while preserving confidentiality, human authority, and evidence?

**No.** There is no system in this repository to do any of those things. What exists is a well-formed and internally consistent governance and product specification. That is valuable — it means the next engineering cycle starts with unusually clear acceptance logic — but it is not an operating system for a consulting practice. Any statement to the contrary would be optimism, not evidence.

---

## 1. Repository identity

| Item | Observed value | Evidence |
|---|---|---|
| Repository | `Assemble-Teams/CuriousPI` | `git remote -v`, `gh repo view` |
| Owner | GitHub organization `Assemble-Teams` | `gh repo view --json ... ` → description "Assemble Teams Inc - LLM " |
| Visibility | **PUBLIC** (`isPrivate: false`) | `gh repo view --json visibility,isPrivate` |
| Default branch | `main` (CuriousPI STEM language-model project) | `gh repo view --json defaultBranchRef` |
| Onion branch | `onion-bootstrap` | `git ls-remote --heads origin` |
| Onion commit | `669c356debe998845e9d8785360692cc2b2b30b2` | `git rev-parse HEAD` |
| Merge base with `main` | `f319bff85aff76bcf5adddfb91c3203f8fac8c56` | `git merge-base main onion-bootstrap` |
| Diff vs `main` outside `/onion` | **none** (`git diff --stat main onion-bootstrap -- . ':!onion'` → empty) | shell |
| Working tree | clean | `git status` |
| Open PRs | #1 "Bootstrap Onion consulting OS" — `onion-bootstrap` → `main`, DRAFT, author `udayteki` | `gh pr list --state all` |
| Detected framework (Onion) | **none** | file inventory |
| Runtime (Onion) | **none** | — |
| Package manager (Onion) | **none** (no `package.json`, `pyproject.toml`, `requirements.txt`, lockfile under `/onion`) | `git ls-tree -r --name-only onion-bootstrap` |
| Deployment configuration (Onion) | **none** (no `vercel.json`, Dockerfile, workflow, `.github/`) | `ls -la .github` → does not exist |
| Environment/config files (Onion) | **none** (`.env*` absent; root `.gitignore` is a generic Python template that already excludes `.env`) | — |
| Test commands (Onion) | **none** | — |
| Build commands (Onion) | **none** | — |
| Host project | CuriousPI: Python 3.9+ ML project (`src/`, `training/`, `data/`, `configs/`, `curiouspi.ipynb`, root `requirements.txt`) | file inventory |
| Host CI | none (`.github/` absent) | — |
| Toolchain available in this audit environment | Node v22.14.0, npm 10.9.7, Python 3.12.3 | `node --version` etc. |

### 1.1 Verification commands run (exact-SHA evidence)

Environment: Cursor cloud agent VM, Linux 6.12, repository at `/workspace`, commit `669c356`, 2026-10-03T20:4xZ.

| # | Command | Result |
|---|---|---|
| 1 | `git fetch origin onion-bootstrap && git checkout onion-bootstrap` | OK; tracking branch created |
| 2 | `git ls-tree -r --name-only onion-bootstrap` | 27 paths; 9 under `onion/`, all `.md` |
| 3 | `for r in $(git for-each-ref --format='%(refname)'); do git ls-tree -r --name-only $r \| grep -iE '\.(html\|js\|ts\|tsx\|css\|json)$'; done` | **no matches on any ref** |
| 4 | `git fsck --lost-found` | no dangling blobs/commits containing Onion code |
| 5 | `git diff --stat main onion-bootstrap -- . ':!onion'` | empty — `/onion` is the only delta |
| 6 | `rg -n -i '(api[_-]?key\|secret\|token\|password\|sk-…\|hf_…\|AKIA…\|BEGIN (RSA\|PRIVATE))' onion` | only policy text ("no plaintext secrets", "design tokens"); **no credentials** |
| 7 | Same scan across host project and `curiouspi.ipynb` | no key-shaped tokens found |
| 8 | `rg -n -i 'assemble\|gamechangers\|onehorn\|vercel\|supabase\|firebase' onion` | 25 hits, **all documentary** (boundary statements, historical names, decision records). No runtime references possible. |
| 9 | `gh pr view 1` | Draft PR body confirms: "The private Onion app source and any sensitive operating data should not be added until repository visibility / long-term hosting is explicitly reviewed." |
| 10 | `wc -l onion/*.md` | 1,981 lines total |
| 11 | `python3 -m http.server` against `onion/index.html` | **not attempted** — file does not exist; README "Run locally" instructions cannot succeed |
| 12 | Install / lint / unit / integration / build / dev-server / browser / mobile / CRUD / negative-path / export tests | **not applicable** — there is nothing to install, lint, build, run or render |

Known limitation of this audit: it cannot observe the off-repository prototype described in `DELIVERY-AUDIT.md`. All statements about that prototype are reported, not verified.

---

## 2. Inventory of the current application

Per `CURSOR_KICKOFF_PROMPT.md` §2, each surface is classified from code, not labels.

| Surface / concern | In repository? | Reported elsewhere? | Classification |
|---|---|---|---|
| Public website routes | No | README: `site.html` | **Not present** |
| Private application routes | No | README: `index.html` | **Not present** |
| Dashboard | No | DELIVERY-AUDIT: Built | **Not present** |
| Leads / Organizations | No | DELIVERY-AUDIT: Built | **Not present** |
| Engagements | No | DELIVERY-AUDIT: Built | **Not present** |
| Delivery | No | DELIVERY-AUDIT: Built | **Not present** |
| Commercial workflow | No | not claimed | **Not present** |
| Finance | No | DELIVERY-AUDIT: Built | **Not present** |
| Growth / Signals | No | DELIVERY-AUDIT: Built | **Not present** |
| Knowledge / Templates | No | DELIVERY-AUDIT: Built | **Not present** |
| Evidence / Audit | No | not claimed | **Not present** |
| Settings / Connections | No | not claimed | **Not present** |
| Agent / AI surfaces | No | DELIVERY-AUDIT: "No automatic API calls" | **Not present** |
| Persistence / storage | No | README: browser localStorage | **Not present** |
| Authentication | No | DELIVERY-AUDIT: No | **Not present** |
| Authorization | No | DELIVERY-AUDIT: No | **Not present** |
| Integrations | No | — | **Not present** |
| Google Workspace connections | No | DELIVERY-AUDIT: "manual source links" | **Not present** |
| Export / backup | No | DELIVERY-AUDIT: JSON backup/CSV export | **Not present** |
| Intake | No | DELIVERY-AUDIT: non-transmitting prototype | **Not present** |
| Notifications | No | — | **Not present** |
| Logging / observability | No | — | **Not present** |
| Tests | No | DELIVERY-AUDIT: "static + HTTP checks" | **Not present** |

---

## 3. Documentation vs repository reality (controlling finding)

| Document | Claim | Repository fact | Consequence |
|---|---|---|---|
| `onion/README.md` §Surfaces, §Run locally | `index.html` and `site.html` exist; run with `python3 -m http.server 4173` | Neither file exists on any ref | Instructions fail; README misdescribes the branch |
| `onion/DELIVERY-AUDIT.md` rows 1–14 | Built: Yes; Wired: Local/Browser/Manual; Proven: Static/HTTP/Partial | No corresponding artifact in repository | Rows are **unverifiable**; cannot be carried forward as evidence |
| `onion/DELIVERY-AUDIT.md` "Acceptance checks completed" | "Static content checks pass", "App route responds over local HTTP", "No Assemble Teams or GameChangers references in the private app file" | No app file in repository to check | Checks cannot be reproduced |
| `onion/REPOSITORY_DECISION.md` §Status | "Built: bootstrap branch and files when committed" | True for documentation | Consistent |
| PR #1 body | "Built: governance, PRD, state, audit and Cursor bootstrap documentation are present … Wired: not yet. Proven: not yet." | Consistent with repository | **This is the accurate statement of state** |

Resolution applied in this cycle (documentation only, no product change):

- `DELIVERY-AUDIT.md` receives a dated audit notice at the top stating that rows describing the prototype are *reported by a prior session* and *not verifiable in this repository*, and that `PRD_TRACEABILITY.md` is the repository-verified matrix.
- `README.md` receives a correction note on the "Surfaces" and "Run locally" sections.
- `ONION_STATE.md` receives a short "repository truth" section with the audit SHA.
- A Founder decision request (DR-06) asks whether the off-repo prototype should be (a) delivered to a private location for audit, (b) treated as design reference only, or (c) considered superseded.

---

## 4. Standalone-boundary audit

Required by `CURSOR_KICKOFF_PROMPT.md` §4. Differentiating legitimate documentation from active coupling:

| Coupling vector | Finding | Severity |
|---|---|---|
| Runtime code dependency on Assemble Teams / GameChangers | None possible — no Onion runtime | 🟢 Verified (vacuously) |
| Deployment projects | None — no deployment config anywhere in repo | 🟢 Verified |
| Analytics | None | 🟢 Verified |
| Secrets | None found in `/onion` or host project (see §1.1 #6–7) | 🟢 Verified |
| Branding | `/onion` docs reference the names only to prohibit their reuse | 🟢 Verified |
| Customer / product data | None | 🟢 Verified |
| **GitHub organization ownership** | Repository is owned by `Assemble-Teams` org. Org owners/admins — not Uday alone — control visibility, branch protection, collaborator access, deploy keys, Actions secrets and app installations. Any future GitHub-connected deployment (Vercel, Cloud Run, etc.) would be authorized against that org. | 🟠 High (organizational coupling, not runtime) |
| **Public visibility** | Every byte pushed to any branch is world-readable, including PR bodies, commit messages and branch names. Prompts, schema, synthetic data shaped like real clients, and the private app itself would all be public. | 🔴 Blocker for application code |
| Branch topology | `/onion` is the only delta from `main`; CuriousPI `main` untouched | 🟢 Verified |
| Historical name "OneHorn" | Mentioned only as a historical name with an explicit "must not be linked" rule | 🟢 Verified |
| Host-project hygiene (out of Onion scope, informational) | `Kimi-K2.6` is a gitlink (`160000 commit 2b2b88e…`) with no `.gitmodules`; `test_kimi.py` imports a package not in the repo. Not Onion's concern; noted so RAS does not mistake it for Onion coupling. | 🔵 Informational |

Conclusion: the founder-approved hosting exception is honoured at the file level. It is not yet safe for application source. See DR-01.

---

## 5. Security and authority findings

Everything below is "absent", which is expected for a documentation-only branch; the value is in recording the required baseline so Sprint 1 can be measured against it.

| Area | State | Required before pilot (per PRD §Security / launch gates) | Severity now |
|---|---|---|---|
| Authentication | Absent | Google Workspace OIDC restricted to the practice's domain; session hardening | 🔴 (gate) |
| Authorization | Absent | Server-side, deterministic; authority verbs VIEW/COMMENT/CONTRIBUTE/MANAGE/APPROVE/DELEGATE/ADMINISTER; tested negative paths | 🔴 (gate) |
| Role boundaries | Absent | Principal, Consultant; external roles deferred | 🔴 (gate) |
| Client/provider isolation | Absent | Engagement/organization-scoped access; isolation tests | 🔴 (gate) |
| Internal/external visibility | Absent | Public front door physically separated from private app; no shared session, no shared data path | 🟠 |
| Secret handling | No secrets present (good) | Server-side env only; no secrets in repo; documented rotation | 🟢 now / 🟠 at Sprint 1 |
| Browser-exposed credentials | None (no browser app) | Google/LLM credentials never shipped to the client | 🟢 now |
| Public endpoint controls | None | Rate limiting, CAPTCHA-or-equivalent, honeypot, size limits, quarantine; **public intake is out of Sprint 1 scope** | 🟠 (deferred) |
| Input / output validation | Absent | Schema validation on every write; agent output schema validation | 🟠 |
| Audit logging | Absent | Append-only `audit_event` for every consequential human and agent action | 🔴 (gate) |
| Destructive operations | None exist | Soft delete + approval + audit; no hard delete in pilot | 🟡 |
| Approval controls | Absent | Approval records with requester (human/agent), approver, options, evidence, outcome | 🔴 (gate) |
| Environment separation | Absent | dev / preview / pilot with separate credentials and databases; synthetic data only in dev | 🟠 |
| Sensitive test data | None (good) | Synthetic fixtures must be obviously fictional (no real company names, no real people) | 🟢 now |
| Backup / restore | Absent | Scripted, tested restore drill with evidence | 🟠 (gate) |

---

## 6. Data-integrity findings

| Item | State | Note |
|---|---|---|
| Source(s) of truth | None in code. PRD designates Google Drive/Docs/Sheets as document source of truth and Onion as the structured operating record. | Must be encoded explicitly per entity (which system is canonical for which field). |
| Persistence mechanism | None | DR-02 |
| Entity identifiers | None | Rule: opaque, stable, sortable (ULID/UUIDv7); never expose sequential ints |
| Organization / contact model | None | PRD: organizations may play multiple roles; engagement role must be explicit — model `organization_role` per engagement/opportunity, not on the organization |
| Engagement / project model | None | Deferred past Sprint 1 (opportunity is the Sprint 1 terminal entity) |
| Commercial model | None | Deferred; must never hard-code terms |
| Finance model | None | Deferred; currency per record, no cross-currency totals |
| Evidence / provenance model | None | Sprint 1 core: `evidence` with `source_type`, `source_ref`, `captured_by`, `captured_at`, `classification`, `origin` (human/agent) |
| Agent run model | None | Sprint 1 core: `agent_run`, `agent_output`, `approval`, `audit_event` |
| Lifecycle / state handling | None | Explicit enums + transition tables tested as state machines |
| Duplicate prevention | None | Organization dedupe on normalized name + domain; lead dedupe on email/domain + 30-day window |
| Idempotency | None | Operation keys on every side-effecting command |
| Migrations / versioning | None | ORM-managed, forward-only, reviewed |
| Backup / export model | None | Scripted DB snapshot + JSON export; restore drill is a RAS gate |
| Hidden business logic | None found | Keep it that way: pricing/percentages/terms are data with approval, never constants |

---

## 7. AI / agent-boundary findings

| Item | State |
|---|---|
| Existing AI calls | None |
| Agent identities | None in code (five named in PRD) |
| Prompts | None in repo (correct for a public repo; prompts with business logic must not land here until DR-01 resolves) |
| Tool permissions / write capabilities | None |
| Approval gates | None |
| Evidence handling | None |
| Model / provider coupling | None — opportunity to design a provider abstraction before the first call |
| Run IDs / audit trail / retry / output schemas / evals | None |

Gap vs desired Agent Control Plane: 100%. Design for the control plane is in `AI_CAPABILITY_ARCHITECTURE.md`; the minimum subset is in `SPRINT_1_PROPOSAL.md`. The first LLM provider (PRD names ChatGPT/OpenAI) is a paid external service → DR-04.

---

## 8. Integration findings — Google Workspace

Per `CURSOR_KICKOFF_PROMPT.md` §8 classification scale: not present / linked-manual / mocked-prototype / implemented-not-wired / wired / proven.

| Service | Classification in repository | Sprint 1 intent | Notes |
|---|---|---|---|
| Google Identity (Workspace sign-in) | **Not present** | Wire (OIDC, domain-restricted) | Smallest honest Google integration; also satisfies the auth gate. Requires a GCP project owned by the practice — DR-03 |
| Drive | Not present | Linked/manual (URL references labelled "linked, not synced") | No Drive API in Sprint 1 |
| Docs | Not present | Linked/manual | — |
| Sheets | Not present | Not in Sprint 1 | Ledger use case arrives with Finance |
| Forms | Not present | Not in Sprint 1 | Candidate public-intake channel later |
| Gmail | Not present | Linked/manual (message URL as evidence) | No read/send API |
| Calendar | Not present | Not in Sprint 1 | Needed by Meeting Intelligence |
| Meet | Not present | Not in Sprint 1 | Needed by Meeting Intelligence |

UI truthfulness rule (encode in design system): every Google reference carries one of exactly three labels — **Linked** (URL only), **Synced** (API-read, with last-sync timestamp), **Writes** (API-write, approval-gated). Nothing may display "Synced" unless an integration test proves it.

---

## 9. UX / accessibility findings

- No UI exists to evaluate. The reported prototype (two static HTML files, browser localStorage) is not in the repository.
- The PRD and continuation brief already contain an unusually precise UX mandate (operational, dark-navy-first, accessible light mode, restrained blue/teal, no hero sections, no AI-magic language). `UI_UX_SYSTEM.md` turns that mandate into tokens, patterns, page inventory and seven journeys so that Sprint 1 screens are designed from the journey inward rather than page by page.
- Accessibility baseline is a launch gate with no current evidence. Sprint 1 must ship with automated axe checks and a manual keyboard pass as part of Proven.

---

## 10. Test findings

None exist. Minimum Sprint 1 test ladder (detail in `SPRINT_1_PROPOSAL.md` §9):

unit (state machines, authz policy, dedupe, idempotency) → integration (command handlers against a real database) → isolation (negative authz paths) → agent evals (fixture provider, ≥20 synthetic leads) → prompt-injection suite → end-to-end (Playwright, whole slice) → accessibility (axe-core on each slice page) → backup/restore drill.

---

## 11. Operational / deployment findings

- No deployment target exists or is approved. `DELIVERY-AUDIT.md` mentions Vercel; nothing is configured.
- A deployment target is a Founder decision because (a) it is a new service/account, (b) it fixes where pilot data physically lives, (c) it determines whether the GitHub org coupling (§4) becomes a runtime coupling. DR-05.
- No observability, alerting, backup, or runbook exists. All are launch gates; Sprint 1 proposes minimum structured logging, health endpoint, error capture and a scripted backup — enough for a *controlled pilot*, not Production.

---

## 12. Technical debt (documentation debt, since there is no code)

| Debt | Impact | Proposed disposition |
|---|---|---|
| `DELIVERY-AUDIT.md` describes an artifact not in the repo | Misleads reviewers; breaks the Built/Wired/Proven discipline the project is founded on | Annotated this cycle; superseded by `PRD_TRACEABILITY.md`; final disposition via DR-06 |
| `README.md` run instructions are false for this branch | Same | Annotated this cycle |
| Three partially overlapping "first assignment" documents (`CURSOR_RULES.md` §First assignment, `CURSOR_KICKOFF_PROMPT.md`, `CURSOR_CONTINUATION_BRIEF.md` §18) | Low; they agree in substance | Leave as-is; continuation brief is the superset |
| Launch-gate lists differ slightly between `PRD.md` (14 items) and `ONION_STATE.md` (16 items, adds "standalone repository", "standalone deployment") | Could cause gate drift | Traceability matrix uses the union (16) |

---

## 13. Scope-creep risks observed in the baseline documents

1. **Eleven private surfaces + public site + five agents** is the full product. Sprint 1 must touch only Leads & Organizations, Evidence/Audit, Agent Activity/Approvals, a minimal Today view, and Settings/Connections (sign-in only).
2. **Meeting Intelligence** and **Research** each require a new external capability (Calendar/Meet/transcript access; web egress). They are specified in `AI_CAPABILITY_ARCHITECTURE.md` but explicitly sequenced after Sprint 1.
3. **Public intake** is tempting to include because the slice begins with "Lead submitted". It is excluded from Sprint 1: it brings abuse controls, privacy notice, retention and a public deployment with it. Sprint 1 proves the same lead record via internal entry; the intake channel is pluggable later.
4. **"Chat with Onion"** is explicitly not built first.
5. **PostgreSQL as canonical state** is listed by the PRD under the *productization gate*; Sprint 1 must not pre-empt that with heavyweight infrastructure, but must not paint itself into localStorage either. DR-02 recommends a server-side SQL store with a Postgres-compatible schema layer so the later move is a configuration change rather than a rewrite.

---

## 14. Findings register

| ID | Severity | Finding | Where | Disposition |
|---|---|---|---|---|
| F-01 | 🔴 Blocker | Application source for a private operating system cannot be committed to a public repository; this blocks all Sprint 1 code | Repo visibility | **DR-01** Founder decision |
| F-02 | 🔴 Blocker | No authentication, authorization, persistence, audit or approval mechanism exists (launch gates 3–6, 12) | `/onion` | Sprint 1 foundation |
| F-03 | 🔴 Blocker (documentation) | `DELIVERY-AUDIT.md` and `README.md` assert Built artifacts that are not in the repository | `onion/DELIVERY-AUDIT.md`, `onion/README.md` | Annotated; DR-06 |
| F-04 | 🟠 High | Organizational coupling: repo owned by Assemble-Teams org; org admins control access, visibility, deploy integrations | GitHub | DR-01 options include transfer / new private repo |
| F-05 | 🟠 High | No technology stack, persistence, or deployment target chosen; each is a new-service or architecture decision requiring Founder sign-off | — | DR-02, DR-03, DR-04, DR-05 |
| F-06 | 🟠 High | No Google integration exists; PRD and launch gates expect "real Google integration as claimed" | — | Sprint 1: Workspace OIDC + linked references only |
| F-07 | 🟠 High | No agent control plane; the first LLM call must not precede registry, run, approval, audit and eval primitives | — | Sprint 1 minimum control plane |
| F-08 | 🟡 Medium | Launch-gate lists differ between PRD (14) and ONION_STATE (16) | docs | Traceability uses union; Council to confirm |
| F-09 | 🟡 Medium | No synthetic dataset or fixture policy exists; risk of realistic-looking test data leaking into a public repo | — | Fixture policy in Sprint 1; obviously fictional names |
| F-10 | 🔵 Improvement | Host-repo hygiene (`Kimi-K2.6` gitlink without `.gitmodules`) | root | Out of Onion scope; informational only |
| F-11 | 🟢 Verified | No secrets, credentials, client/provider data or financial records in `/onion` or host project | repo | — |
| F-12 | 🟢 Verified | `/onion` is the only delta from `main`; CuriousPI untouched | repo | — |
| F-13 | 🟢 Verified | Governance documents are internally consistent on authority, autonomy levels, Built/Wired/Proven and Google truthfulness | `/onion` | — |

---

## 15. Recommended next slice, proposed files, acceptance tests, affected RAS gates

- **Next slice:** Sprint 0 (decisions + skeleton) then Sprint 1 vertical slice. Full detail: `SPRINT_1_PROPOSAL.md`.
- **Proposed files:** listed in `SPRINT_1_PROPOSAL.md` §6 (all under `/onion/app`, isolated from host project).
- **Acceptance tests:** `SPRINT_1_PROPOSAL.md` §9 and §12.
- **Affected RAS gates:** authentication; authorization; secure shared persistence; client/provider isolation (partial — organization scoping); real Google integration as claimed (identity only); accessibility baseline (slice pages); tested backup/restore (drill); operational alerts (minimum). Not affected: end-to-end public intake; privacy retention/deletion; abuse controls; commercial lifecycle; onboarding/offboarding; browser/mobile QA beyond slice pages.

---

## 16. Standing status report

**STATUS** — Audit complete. Repository is documentation-only. No Onion code exists. Sprint 1 blocked on two Founder decisions (visibility/hosting; stack/persistence) and three enabling approvals (GCP project for Workspace sign-in; LLM provider; deployment target).

**COUNCIL FINDINGS** — Product intent is clear and sufficient to design from. Two baseline documents overstate delivery state and have been annotated. The Council should confirm the union launch-gate list (16) and the disposition of the off-repo prototype.

**AXE FINDINGS** — No architecture or experience to review yet. Target architecture (modular monolith, server-side SQL, Workspace OIDC, provider-abstracted LLM, tool gateway with enforced allowlists) and the journey-first UI system are submitted for AXE review in `ARCHITECTURE_CURRENT.md` §5, `UI_UX_SYSTEM.md`, `AI_CAPABILITY_ARCHITECTURE.md`.

**BUILT / DESIGNED** — Built: nothing. Designed (this cycle): audit, traceability, current/target architecture, Sprint 0/1 proposal, UI/UX system, AI capability architecture, decision requests.

**WIRED** — Nothing.

**PROVEN** — Nothing.

**NOT VERIFIED** — The off-repository prototype described in `DELIVERY-AUDIT.md`; every "Built: Yes" row therein.

**RAS FINDINGS** — RAS has not reviewed this cycle. Cursor provides F-01…F-13 as evidence. Cursor does not self-certify.

**RISKS** — (1) committing private application code to a public org-owned repo; (2) building the first LLM call before the control plane exists; (3) scope expanding from one slice to eleven surfaces; (4) encoding unapproved commercial logic in code.

**AUTHORITY / APPROVAL STATE** — No merge to `main`. This cycle's changes are documentation under `/onion` on branch `cursor/onion-audit-foundation-f6d7`, proposed into `onion-bootstrap` via draft PR. Founder decisions DR-01…DR-07 open.

**OPEN ITEMS** — DR-01 … DR-07 in `DECISION_REQUESTS.md`.

**NEXT TARGET** — On DR-01 + DR-02: Sprint 0 skeleton (auth, schema, authz policy, audit event, CI, synthetic fixtures) → Sprint 1 slice.
