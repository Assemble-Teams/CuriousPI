# Onion Session Handoff

**Date:** 2026-10-03 (second session)  
**Session objective:** Encode Founder decisions DR-01 A and DR-02 A (with database refinement); prepare and test the repository migration; de-risk the approved stack; set up Sprint 0 for execution in the private repository.  
**Repository:** `Assemble-Teams/CuriousPI` (public)  
**Branch:** `cursor/onion-audit-foundation-f6d7`  
**Commit:** see PR #2 for this session's commits (base of session: `a9e2130`)

## What changed
Documentation only under `/onion`. No application code committed (per ADR-001, Sprint 0 is built only in the private repository). No file outside `/onion` touched.

## Files changed
- Added: `decisions/ADR-001-repository-host.md`, `decisions/ADR-002-engineering-stack.md`, `MIGRATION_RUNBOOK.md`, `SPRINT_0_EVIDENCE_PACK.md`, this handoff
- Updated: `DECISION_REQUESTS.md` (statuses), `ARCHITECTURE_CURRENT.md` (§9–§12 canonical stack, environments), `SPRINT_1_PROPOSAL.md` (§1 Sprint 0 = Founder's 15 items with maturity criteria), `ONION_STATE.md` (§14 decisions), `README.md` (document map)

## Tests / verification run
- Access check: `gh auth status` → GitHub App installation token (`ghs_…`) scoped to CuriousPI; `gh api user` → 403 "Resource not accessible by integration". The agent cannot create or push to the new repository.
- Migration dry run: `git subtree split --prefix=onion HEAD -b tmp/onion-history` → 15 commits, Onion-only paths, authorship preserved (Uday T / Cursor Agent), tree `74bfdd13…` identical to `HEAD:onion`. Temporary branch deleted locally; nothing pushed.
- Stack spike in `/tmp/onion-spike` (outside repo, not committed): Next.js 16.3.8 App Router Server Component + Drizzle 0.45.3 + Drizzle Kit 0.31.11 (postgresql dialect) + PGlite 0.5.8 (PostgreSQL 18.3) + Better Auth 1.7.7 (Google only, drizzle adapter `pg`). Results: migration generated and applied; append-only trigger blocks UPDATE/DELETE; duplicate `operation_key` rejected; Better Auth initialises and creates a user via adapter; `next build`/`next start` render `/` reading PGlite with count incrementing across requests. Gotchas: Better Auth schema must be CLI-generated; `serverExternalPackages: ["@electric-sql/pglite"]` required; Drizzle driver errors live in `error.cause`.

## Built
Nothing (software). Decision records, runbook, evidence template.

## Wired
Nothing.

## Proven
Nothing (spike is de-risking evidence, not a Proven artifact).

## Known gaps / failures
Google sign-in flow not exercised (no OAuth client yet); PGlite on-disk persistence and backup/restore not exercised; deployment target undecided.

## Decisions made
By Founder: DR-01 A; DR-02 A with DB refinement; Sprint 0 and Sprint 1 direction; explicit deferral list. By Cursor: none of business consequence. Engineering detail choices (CLI-generated auth schema, driver adapter seam, repository-root layout in the Onion-only repo) recorded in ADR-002 and Sprint 0 §1 for AXE review.

## Human approvals still required
Create the private repository and install Cursor (runbook A1–A4); DR-03 GCP project; DR-04 account and cap; DR-05 deployment target; DR-06 prototype; RAS boundary verification (runbook C) before any confidential material.

## RAS findings
None yet for this session. RAS verification of the private-repository boundary is the gate before Sprint 0 handles anything beyond synthetic data.

## Architecture / data-model changes
Canonical stack fixed (ADR-002). Data model v0 unchanged. Repository layout in the new repo: application at root, documents under `docs/`.

## Agent / permissions changes
None. Control plane remains provider-neutral by decision.

## Google integration changes
Decision: identity-only scopes; scopes added capability-by-capability with approval.

## Commercial workflow changes
None.

## Next recommended action
Uday executes `MIGRATION_RUNBOOK.md` §A (create private `onion`, protections, install Cursor, start a session there with the §A4 prompt). That session executes §B, RAS executes §C, then Sprint 0 begins with `SPRINT_0_EVIDENCE_PACK.md` as the return format.

## Deferred / explicitly not done
No repository created (not possible/appropriate from this agent); PR #1 not closed and `onion-bootstrap` not deleted (sequenced after boundary verification per DR-01); no Sprint 0 code in CuriousPI; no accounts created.
