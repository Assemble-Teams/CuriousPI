# Onion Repository Decision

**Decision date:** 2026-10-03  
**Decision owner:** Uday Teki

## Decision

Use `Assemble-Teams/CuriousPI` as the founder-approved bootstrap source host for Onion, but stage Onion on a dedicated branch and under the `/onion` namespace.

## Reason

The repository already exists and the founder explicitly selected it for the Onion effort. It is not an empty repository: its `main` branch currently contains the CuriousPI open-source STEM language-model project.

## Guardrails

- Do not overwrite or repurpose CuriousPI `main` without a separate explicit decision.
- Keep Onion source under `/onion`.
- Do not reuse CuriousPI/Assemble Teams/GameChangers branding or data in Onion.
- Do not share client/provider records, analytics, secrets, or production credentials across products.
- Deployment/runtime must be separately reviewed before production use.
- Cursor and RAS must audit repository coupling before merge.

## Status

**Built:** bootstrap branch and files when committed.  
**Wired:** not yet.  
**Proven:** not yet.
