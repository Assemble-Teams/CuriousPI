# SPRINT_0_EVIDENCE_PACK.md
# Onion — Sprint 0 AXE / RAS Evidence Pack (template; filled in the private repository)

**Rule:** every row cites repository, branch, commit SHA, environment, command, timestamp, result and known limitations (`CURSOR_CONTINUATION_BRIEF.md` §13). Cursor fills; AXE reviews architecture/experience; RAS reproduces independently. No row is promoted on Cursor's word alone.

## Header

| Field | Value |
|---|---|
| Repository | _private `onion` repository (URL not recorded in CuriousPI)_ |
| Branch / PR | |
| Commit SHA | |
| Environment(s) | dev (PGlite) · CI (PGlite + PostgreSQL service) · preview |
| Node / package versions | from lockfile |
| Date (UTC) | |
| Filled by | Cursor |
| Reviewed by | AXE: · RAS: |

## Maturity summary

| Item | Built | Wired | Proven | Notes |
|---|---|---|---|---|
| 1 Next.js skeleton | | | | |
| 2 Workspace-restricted sign-in | | | | Wired requires practice-owned OAuth client (DR-03) |
| 3 Schema v0 | | | | |
| 4 Drizzle migrations | | | | |
| 5 PGlite dev/test | | | | |
| 6 Authorization policy engine | | | | |
| 7 Authorization tests | | | | |
| 8 Append-only audit | | | | |
| 9 Synthetic fixtures | | | | |
| 10 CI | | | | |
| 11 Secret scanning | | | | |
| 12 Backup/export | | | | |
| 13 `/today` shell | | | | |
| 14 Error/logging | | | | |
| 15 Evidence pack | | | | this document |

**Commissioned:** not applicable to Sprint 0. **Production:** no.

## Evidence rows

| # | Claim | Command / artifact | SHA | Result | Limitation |
|---|---|---|---|---|---|
| E1 | Build succeeds | `npm ci && npm run build` | | | |
| E2 | Lint + typecheck clean | `npm run lint && npm run typecheck` | | | |
| E3 | Unit + integration green | `npm test` | | | |
| E4 | Migration parity | CI job: migrate on PGlite and PostgreSQL; `drizzle-kit check` | | | |
| E5 | Authorization matrix | test report listing every role × verb × scope cell | | | |
| E6 | No bypass | static check: every command handler calls policy | | | |
| E7 | Audit immutability | tests: UPDATE/DELETE raise; duplicate operation_key rejected | | | |
| E8 | Sign-in restriction | e2e: allowed domain ok; other domain rejected + audited; cookie flags | | | Wired only with practice OAuth client |
| E9 | Fixture lint | `npm run fixtures:lint` | | | |
| E10 | Secret scanning | push-protection test + CI scanner report | | | |
| E11 | Backup/restore drill | scripts + checksum/row-count comparison | | | |
| E12 | `/today` accessibility | axe report (dark + light), keyboard checklist, 390px screenshot | | | |
| E13 | Logging/health | log sample with request_id; `/health` response | | | |
| E14 | Dependency audit | `npm audit --omit=dev` summary | | | |
| E15 | Boundary | `MIGRATION_RUNBOOK.md` §C sign-off reference | | | |

## AXE review

```
AXE ENGINEERING REVIEW — Sprint 0
Objective:
Current architecture:
Proposed architecture:
Modules affected:
Data changes:
Authorization impact:
Integration impact:
Failure modes:
Observability:
Migration:
Rollback:
Tests:
Performance:
Cost:
Evidence: E1–E15
Status: PROCEED | PROCEED WITH CONDITIONS | HOLD | FOUNDER DECISION REQUIRED
```

```
AXE EXPERIENCE REVIEW — /today shell
User: Job: Current friction: Proposed change: Workflow impact: Information hierarchy:
Mobile behavior: Accessibility: Failure/error behavior: AI implications: Security/privacy implications:
Evidence: E12  Tradeoffs:  Status:
```

## RAS independent verification

| Check | Reproduced by RAS (y/n) | SHA | Notes / dissent |
|---|---|---|---|
| E1–E15 | | | |
| Red Team: sign-in bypass, authz bypass via direct command, audit tamper, fixture realism | | | |
| Security/Trust: secrets inventory, cookie flags, dependency audit | | | |
| Reliability/SRE: restore drill, health, log coverage | | | |
| UX/Accessibility: axe + manual pass | | | |
| Privacy/Legal/Finance: n/a in Sprint 0 (no real data, no paid LLM) | | | |

Unresolved findings and dissent are preserved here, not forced to green.

## Founder Gate

Decision: PROMOTE TO SPRINT 1 | HOLD | REVISE — by Uday, date, note.
