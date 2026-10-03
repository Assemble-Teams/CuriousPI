# ADR-001 — Onion repository host and visibility

**Status:** Accepted (Founder decision DR-01 option A, 2026-10-03)  
**Decider:** Uday Teki  
**Supersedes:** `REPOSITORY_DECISION.md` (2026-10-03 bootstrap exception) once migration is verified

## Decision

Onion moves out of `Assemble-Teams/CuriousPI` into its own **private GitHub repository, outside the Assemble Teams organization**, preferred name `onion`, under Uday's personal GitHub account or a dedicated consulting-practice account/organization.

## Requirements (verbatim intent)

1. Private from inception.
2. Onion is the only product in that repository.
3. CuriousPI code is not migrated.
4. Only Onion source, documentation, architecture decisions and relevant Onion history migrate.
5. Preserve useful authorship/decision history where practical.
6. Migration does **not** make previously public commits confidential: everything pushed to `onion-bootstrap` and `cursor/onion-audit-foundation-f6d7` (PRD, rules, briefs, audit, architecture, UI/UX system, AI capability architecture, decision records) is, and remains, public history. **Founder classification: this material is treated as previously disclosed.** No secrets were committed (scans clean at `669c356` and at every cycle-1 commit), so there is **no credential-rotation incident**. Nothing confidential may be pushed to CuriousPI from now on.
7. After the private repository is verified: close CuriousPI PR #1 with the neutral note *"Onion development has moved to a private repository."*; delete the public `onion-bootstrap` branch (PR #2 closes with its base).
8. Do not publish the private repository URL in CuriousPI.
9. Do not merge Onion into CuriousPI `main`.
10. **RAS must verify the private-repository boundary before any real client/provider data or confidential operating material is added.**

## Context

See `CURSOR_AUDIT.md` §4 and F-01/F-04: public visibility blocks application code; organizational ownership by Assemble Teams couples access control and deploy integrations.

## Consequences

- No `onion/app` code is committed to CuriousPI. Sprint 0 begins only in the private repository.
- The Cursor cloud agent for CuriousPI runs under a GitHub App installation token scoped to CuriousPI; it cannot create or push to the new repository. Repository creation, Cursor app installation and the first agent session in the new repository are account-owner actions.
- Migration procedure and verification: `MIGRATION_RUNBOOK.md`. History is extracted with `git subtree split --prefix=onion`; tested locally on 2026-10-03: 15 Onion-only commits, authorship preserved, extracted tree identical to `onion/` (`74bfdd13…` at base `a9e2130`).
- `REPOSITORY_DECISION.md` remains as the historical record of the bootstrap exception.

## Reversibility

High. A repository can be moved again; history is preserved by the split.
