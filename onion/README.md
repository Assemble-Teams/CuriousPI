# Onion — Private Consulting OS Pilot

Onion is the private operating-system prototype for Uday Teki's consulting practice. By explicit founder decision on 2026-10-03, the bootstrap source is being staged inside `Assemble-Teams/CuriousPI` under the `/onion` namespace on a dedicated branch. This repository-hosting exception does not authorize reuse of Assemble Teams/GameChangers branding, product data, customer data, analytics, secrets, or runtime infrastructure.

## Surfaces
- `index.html` — private owner-operated consulting OS prototype.
- `site.html` — public consulting website/intake preview. The form is deliberately non-transmitting until the production intake backend/privacy controls are wired.

## Current capabilities
Dashboard, client/provider leads, engagement playbooks, delivery records, risks/decisions/approvals, lightweight finance with per-record currency, growth signals, reusable templates, JSON backup/import, CSV finance export, Google source links, light/dark appearance, mobile layout.

## Important pilot boundaries
- Browser-local storage is not a secure multi-user database.
- No public signup.
- No automatic Google synchronization yet.
- No automatic ChatGPT/API transmission yet.
- External collaborator/client/provider access remains deferred until authenticated engagement-scoped permissions are implemented.
- Consequential communications, commitments, and financial changes remain human-approved.

## Run locally
`python3 -m http.server 4173`
Then open `http://localhost:4173/index.html` and `http://localhost:4173/site.html`.
