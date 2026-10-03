# CURSOR_KICKOFF_PROMPT.md
# Onion — First Cursor Assignment

Read `@CURSOR_RULES.md` completely before doing anything else.

You are joining this project as the **engineering audit and implementation partner** for Onion.

Do not begin by redesigning the application.

Do not move Onion code into any other Assemble Teams or GameChangers repository. Founder-approved exception: this bootstrap is intentionally staged in `Assemble-Teams/CuriousPI` under the `/onion` namespace and a dedicated branch.

Do not add a public SaaS signup flow.

Do not invent integrations, business data, customer results, pricing, partnerships, or production-readiness claims.

Your first responsibility is to establish the **actual engineering state** of the codebase and compare it against the approved Onion PRD and `CURSOR_RULES.md`.

## Assignment

Perform a repository-wide engineering audit and create `CURSOR_AUDIT.md`.

### 1. Establish repository identity

Report:
- repository name
- repository owner
- branch
- current commit SHA
- clean/dirty working-tree state
- detected framework
- runtime
- package manager
- deployment configuration
- environment/config files
- test commands
- build commands

This repository is an explicit founder-approved hosting exception: `Assemble-Teams/CuriousPI`. Do not flag ownership alone as a blocker. Instead, audit and flag any runtime, data, branding, deployment, analytics, secrets, or product coupling between Onion and Assemble Teams/GameChangers. Onion must remain namespaced under `/onion` unless an approved migration changes that.

### 2. Inventory the current application

Inspect and map:
- public website routes
- private application routes
- dashboard
- leads/organizations
- engagements
- delivery
- commercial workflow
- finance
- growth/signals
- templates/knowledge
- evidence/audit
- settings/connections
- agent/AI surfaces
- persistence/storage
- authentication
- authorization
- integrations
- Google Workspace connections
- export/backup
- intake
- notifications
- logging/observability
- tests

Do not assume a feature exists because a label, button, TODO, interface mock, or README mentions it.

### 3. Build a PRD traceability matrix

For each meaningful requirement, record:

| Requirement | Code location | Built | Wired | Proven | Evidence | Gap |
|---|---|---|---|---|---|---|

Use:
- Built = implementation exists
- Wired = connected to intended real dependency/workflow
- Proven = passed a representative acceptance test

Never mark a feature complete merely because code exists.

### 4. Audit the standalone boundary

Search executable code, configuration, environment references, package metadata, deployment config, public assets, and documentation for accidental dependencies on:
- Assemble Teams
- GameChangers
- their repositories
- their deployment projects
- their analytics
- their secrets
- their branding
- their customer/product data

Differentiate legitimate historical documentation from active runtime coupling.

### 5. Audit security and authority

Assess:
- authentication
- authorization
- role boundaries
- client/provider data isolation
- internal/external visibility
- secret handling
- browser-exposed credentials
- public endpoint controls
- input validation
- output validation
- rate limiting / abuse protection
- audit logging
- destructive operations
- approval controls
- environment separation
- sensitive test data
- backup/restore readiness

Treat authorization as a backend/runtime requirement, not a UI feature.

### 6. Audit data architecture

Report:
- source(s) of truth
- persistence mechanism
- entity identifiers
- organization/contact model
- engagement/project model
- commercial model
- finance model
- evidence/provenance model
- agent run model
- lifecycle/state handling
- duplicate prevention
- idempotency
- migrations/versioning
- backup/export model

Flag any hidden business logic, ambiguous ownership, or duplicated state.

### 7. Audit the consulting lifecycle

Trace whether the code can represent:

Lead
→ Qualification
→ Discovery
→ Diagnostic
→ Assessment
→ Recommendation
→ Opportunity
→ Proposal
→ NDA/MSA/SOW
→ Commercial Approval
→ Signature
→ Payment Terms
→ Kickoff
→ Power Sprint / Delivery
→ Outcomes
→ Acceptance
→ Closeout
→ Expansion / Follow-up
→ Learning

Identify missing states, invalid transitions, or areas represented only cosmetically.

### 8. Audit Google Workspace integration

For each of Drive, Docs, Sheets, Forms, Gmail, Calendar, and Meet classify:
- not present
- linked/manual
- mocked/prototype
- implemented but not wired
- wired
- proven

Verify that the UI never claims live sync where none exists.

### 9. Audit AI/agent architecture

Identify:
- existing AI calls
- agent identities
- prompts
- tool permissions
- write capabilities
- approval gates
- evidence handling
- model/provider coupling
- run IDs
- audit trail
- retry behavior
- output schemas
- evaluation tests

Compare findings against the desired Agent Control Plane:
- agent registry
- human owner
- role/objective
- scope
- allowed tools
- read/write permissions
- approval rules
- prompt/policy version
- run state
- evidence
- audit events
- evaluation result

Do not build dozens of agents.

The first intended agent cohort is:
- ATLAS Executive Briefing
- Meeting Intelligence
- Research
- RAS Evidence/Quality
- Lead Qualification

Recommend implementation only after auditing the current foundation.

### 10. Run verification

Where safe and available:
- install dependencies
- run lint/static analysis
- run unit tests
- run integration tests
- run build
- start development server
- inspect browser rendering
- inspect mobile layout
- inspect console errors
- test navigation
- test important CRUD flows using synthetic data
- test negative paths
- test exports/backups

Do not use real client/provider data in development verification.

Record exact commands and results.

### 11. Produce findings by severity

Use:
- 🔴 Blocker
- 🟠 High
- 🟡 Medium
- 🔵 Low / improvement
- 🟢 Verified

Prioritize business and security gaps over visual polish.

### 12. Recommend only the next engineering slice

After the audit, propose **Sprint 1**.

Sprint 1 must be the smallest coherent slice that materially advances the system toward a secure standalone pilot.

Do not propose an enormous rewrite.

For the proposed Sprint 1 provide:
- objective
- why it matters
- user/business value
- exact files/modules affected
- data changes
- security impact
- agent impact
- Google impact
- tests
- migration/backward compatibility
- rollback
- risks
- RAS acceptance criteria

## Output files

Create:
1. `CURSOR_AUDIT.md`
2. `PRD_TRACEABILITY.md`
3. `ARCHITECTURE_CURRENT.md`
4. `SPRINT_1_PROPOSAL.md`

If the repository already contains equivalent canonical files, update them carefully instead of creating conflicting duplicates.

## Stop Condition

After producing the audit and Sprint 1 proposal:

**STOP.**

Do not begin a broad architecture migration until Uday / Founder Council has reviewed the findings.

You may fix a trivial defect that prevents the audit itself from running, but document that change explicitly.

## Core test

At the end of your audit, answer:

> Does the current system reliably help the consulting practice acquire, diagnose, contract, deliver, measure, close, and learn from engagements while preserving confidentiality, human authority, and evidence?

Answer with evidence, not optimism.
