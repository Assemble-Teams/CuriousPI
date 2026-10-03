# MIGRATION_RUNBOOK.md
# Onion — Migration from `Assemble-Teams/CuriousPI` to the private `onion` repository

**Authority:** ADR-001 (DR-01 A, approved 2026-10-03)  
**Executor:** Uday for account-owner steps (A); Cursor session in the new repository for engineering steps (B); RAS for verification (C).  
**Principle:** nothing confidential enters CuriousPI at any step; the private repository URL is never written into CuriousPI.

## A. Account-owner steps (Uday)

| # | Step | Why |
|---|---|---|
| A1 | Create a **private** GitHub repository named `onion` under your personal account or a dedicated practice account/organization — not under `Assemble-Teams`. Leave it empty (no README/licence/gitignore) so history can be pushed cleanly. | ADR-001 req. 1–2 |
| A2 | Enable: branch protection on `main` (PR required, no force-push), secret scanning + push protection, Dependabot alerts. Disable: forking, public visibility changes by others. | Boundary hygiene |
| A3 | Install the Cursor GitHub App on the new repository (and only grant it that repository if using selected-repository mode). | The CuriousPI agent token cannot reach the new repository |
| A4 | Start a Cursor cloud agent session in `onion` with the prompt: "Execute `MIGRATION_RUNBOOK.md` section B from `https://github.com/Assemble-Teams/CuriousPI` branch `cursor/onion-audit-foundation-f6d7`." | Hands off to section B |
| A5 | After C passes: close CuriousPI PR #1 with the neutral note in §D; delete branch `onion-bootstrap` (PR #2 closes automatically with its base); delete branch `cursor/onion-audit-foundation-f6d7`. | ADR-001 req. 7 |
| A6 | Do not merge anything Onion-related into CuriousPI `main`. | ADR-001 req. 9 |

## B. Engineering steps (Cursor, inside the empty `onion` repository)

Source of truth for the migration is the tip of `cursor/onion-audit-foundation-f6d7` (which contains all of `onion-bootstrap` plus audit-cycle commits). If Uday merges PR #2 into `onion-bootstrap` first, use `onion-bootstrap` instead; the result is identical.

```bash
# B1. Fetch the public source (read-only; CuriousPI is public)
git init onion && cd onion
git remote add curiouspi https://github.com/Assemble-Teams/CuriousPI.git
git fetch curiouspi cursor/onion-audit-foundation-f6d7
SRC=curiouspi/cursor/onion-audit-foundation-f6d7

# B2. Extract Onion-only history (authorship and messages preserved)
git subtree split --prefix=onion "$SRC" -b main
git checkout main

# B3. Verify: extracted tree == source onion/ tree; no CuriousPI paths; commit count
test "$(git rev-parse HEAD^{tree})" = "$(git rev-parse "$SRC:onion")" && echo TREE_OK
git log --name-only --format= main | sort -u          # must list only Onion files
git rev-list --count main                             # expected: 15 + commits made after this runbook was written
git log --format='%h %an %s' main | tail -15          # authorship: Uday T (bootstrap), Cursor Agent (audit cycle)

# B4. Remove the CuriousPI remote so nothing can be pushed back by accident
git remote remove curiouspi

# B5. Push to the private origin (URL supplied by the session, never recorded in CuriousPI)
git remote add origin <private-origin-url>
git push -u origin main
```

Tested locally on 2026-10-03 against base `a9e2130`: `git subtree split --prefix=onion` produced 15 commits, tree `74bfdd133dcd03a2bcd3da941482d2fe80cbdf6b` identical to `onion/`, no non-Onion paths. Commits made after that will add to the count; the tree-equality test (B3) is the authoritative check.

Post-push in the new repository (first commits on `main`, by PR):

- B6. Add a root `README.md` stating that Onion is the only product in the repository and pointing to the document map; add `.gitignore` (Node), `.github/workflows` with secret scan and dependency audit (Sprint 0 item 10–11), `CODEOWNERS` (Uday).
- B7. Record in `ONION_STATE.md` §14: migration date, source SHA, destination first SHA.
- B8. Update `REPOSITORY_DECISION.md` status to "superseded by ADR-001; migrated".
- B9. Begin Sprint 0 only after section C passes.

## C. RAS boundary verification (independent; required before any confidential material)

| # | Check | Evidence to record |
|---|---|---|
| C1 | Repository is private; owner is not `Assemble-Teams`; forking disabled | screenshot or `gh repo view --json isPrivate,owner` output |
| C2 | Collaborators and installed apps: only Uday, designated consultants, Cursor app | settings export |
| C3 | Branch protection and push protection active on `main` | settings export |
| C4 | Tree equality (B3) reproduced by RAS | command output with SHAs |
| C5 | No CuriousPI files, no secrets, no client/provider/financial data in history | `git log --name-only`, secret scan report |
| C6 | No private URL present in CuriousPI (`rg -i 'github.com/.*onion' ` on CuriousPI branches) | scan output |
| C7 | PR #1 closed with neutral note; `onion-bootstrap` and `cursor/onion-audit-foundation-f6d7` deleted | `gh pr view 1`, `git ls-remote --heads` |
| C8 | Statement acknowledged: previously public commits remain public history; nothing confidential was among them | RAS sign-off line |

Only after C1–C8: real client/provider data, confidential templates, prompts with business logic, and pilot credentials may be introduced — each still subject to its own gates.

## D. Neutral closing note for CuriousPI PR #1

> Onion development has moved to a private repository. Closing without merge; CuriousPI `main` is unaffected and the `onion-bootstrap` branch will be deleted.

No URL, no description of Onion's purpose. The material already published on this branch is classified by the Founder as previously disclosed (ADR-001 req. 6); the note does not need to address it.

## E. What this runbook deliberately does not do

- It does not rewrite or purge CuriousPI history (nothing confidential exists there; purging public history would be theatre).
- It does not migrate any CuriousPI code, data, licence or branding.
- It does not create deployment, Google Cloud, LLM or database accounts — those are DR-03/DR-04/DR-05 and happen under the practice's own accounts after section C.
