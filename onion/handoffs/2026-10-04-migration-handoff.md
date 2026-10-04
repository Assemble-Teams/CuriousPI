# Onion Session Handoff

**Date:** 2026-10-04  
**Session objective:** Execute `MIGRATION_RUNBOOK.md` §B from CuriousPI as instructed; verify; begin Sprint 0 after RAS §C.  
**Repository:** `Assemble-Teams/CuriousPI` (public)  
**Branch:** `cursor/onion-audit-foundation-f6d7`  
**Commit:** see PR #2 (session base `111bea9`)

## What changed
Section B was **not executed** and Sprint 0 was **not started**, because this session cannot reach the private repository. Runbook §B0 and §B6b added; this handoff recorded. Nothing else changed. No file outside `/onion` touched.

## Evidence of the access constraint (exact commands)
- `gh api /installation/repositories --jq '.total_count, [.repositories[].full_name]'` → `1`, `["Assemble-Teams/CuriousPI"]`
- `gh repo view udayteki/onion` → "Could not resolve to a Repository" (GitHub returns the same for private repositories the token cannot see; the owner was not confirmed from here)
- `git ls-remote https://github.com/udayteki/onion.git` → "Repository not found"
- `gh api user` → 403 "Resource not accessible by integration" (installation token, not a user token)

Conclusion: the GitHub App installation for the Assemble-Teams organization mints tokens for that installation only. A session started from the `onion` repository (after installing the Cursor GitHub App there) will hold a token for that installation and can execute §B and Sprint 0.

## Migration dry run (reference values for RAS §C4)
At source tip `111bea9b19b967f23251f10ef7baa9af8c2e9205`: `git subtree split --prefix=onion` → extracted tip `17bfecce5e9eb82292f78e45b0782c93e15f988e`, root `b632bfcb289248b25c2cabd17ecf260ff666c53d`, 19 commits (9 Uday T, 10 Cursor Agent), tree `b514dddc38bf494b70f336a2e741dbad7b6d9e5e` = `111bea9:onion`, zero non-Onion paths. Temporary branch deleted; nothing pushed. Later commits on the source branch advance tip/count/tree; the root commit SHA is invariant.

## Built / Wired / Proven
Unchanged: documentation Built; nothing Wired; nothing Proven. Commissioned / Production: n/a.

## Decisions made
None. The Founder's instruction to keep the old prototype under `reference/prototype-v0/` as a non-production design reference is recorded in runbook §B6b (closes DR-06 as option A once the files are supplied).

## Human approvals / actions still required
1. Install the Cursor GitHub App on the private `onion` repository (runbook A3).
2. Start a Cursor session **in that repository** with the prompt in §Next.
3. Supply the prototype files to that session if they are to be kept (B6b).
4. RAS executes §C before any confidential material; then A5 (close PR #1 with the neutral note, delete `onion-bootstrap`).

## Next recommended action — prompt for the new-repository session
> Read `MIGRATION_RUNBOOK.md`, `decisions/ADR-001-repository-host.md` and `decisions/ADR-002-engineering-stack.md` from `https://github.com/Assemble-Teams/CuriousPI` branch `cursor/onion-audit-foundation-f6d7` (path `onion/`). Execute runbook §B1–B9 into this repository. Report §C results for RAS. Then execute Sprint 0 items 1–15 as specified in `SPRINT_1_PROPOSAL.md` §1 and return `SPRINT_0_EVIDENCE_PACK.md` at an exact SHA. No public intake, no generic chat, identity-only Google scopes, synthetic data only. Do not start Sprint 1.

## Deferred / explicitly not done
Section B push, Sprint 0, PR #1 closure, branch deletion, prototype placement — all require the new-repository session or Uday.
