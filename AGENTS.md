# AGENTS.md — Onion / Cursor Cloud Instructions

## Repository context

This repository contains two distinct efforts:

1. Existing CuriousPI Python/ML project on the repository root/default branch.
2. Onion consulting operating system under `/onion` on branch `onion-bootstrap`.

When assigned to Onion, work only within the approved Onion scope unless the task explicitly requires repository-level configuration.

Do not install the root CuriousPI ML dependencies merely to work on Onion.
Do not modify CuriousPI model/training code as part of Onion tasks.
Do not move Onion into CuriousPI source folders.
Do not add secrets or customer/client/provider data to this public repository.

## Cursor Cloud specific instructions

Before Onion work:

```bash
git branch --show-current
ls onion
```

Expected branch: `onion-bootstrap` or an Onion child branch based on it.

Read in order:

1. `onion/PRD.md`
2. `onion/CURSOR_RULES.md`
3. `onion/CURSOR_CONTINUATION_BRIEF.md`
4. `onion/ONION_STATE.md`
5. `onion/CURSOR_KICKOFF_PROMPT.md`
6. `onion/DELIVERY-AUDIT.md`
7. `onion/REPOSITORY_DECISION.md`
8. `onion/CURSOR_CLOUD_SETUP.md`

Follow Founder Council, AXE, RAS, Authority Control, the R-process, and Built/Wired/Proven/Commissioned/Production maturity rules.

If the cloud agent reports that no usable Build exists, do not treat the default image as proof of a valid engineering environment. Continue only with low-risk repository analysis/documentation and report environment verification as NOT PROVEN until a successful active Build exists.
