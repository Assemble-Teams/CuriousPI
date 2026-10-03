# Onion — Delivery Audit

Status model: **Built ≠ Wired ≠ Proven**.

| Deliverable | Built | Wired | Proven | Notes / next gate |
|---|---:|---:|---:|---|
| Standalone consulting boundary | Yes | Yes | Yes | No Assemble Teams/GameChangers branding in private app runtime. Public preview includes an explicit non-affiliation notice only. |
| Responsive private workspace | Yes | Local | Partial | Static + HTTP checks pass. Browser screenshot automation unavailable in current container; manual browser QA remains. |
| Dashboard | Yes | Local data | Static | Active engagements, pipeline, upcoming actions, risks, finance summary. |
| Leads & organizations | Yes | Local data | Static | Client vs technology/GCC provider distinction. |
| Engagements | Yes | Local data | Static | Objective, scope, type, playbook, status, dates, outcomes, recommendation, Google source link. |
| Delivery | Yes | Local data | Static | Tasks, milestones, notes, decisions, risks, approvals, closeout actions; editable status. |
| Finance | Yes | Local data | Static | Proposed/collected/expenses/invoice/payment; per-record currency; CSV export. Not accounting software. |
| Growth & signals | Yes | Local data | Static | Referrals, sales activity, campaigns, market signals, follow-up. |
| Knowledge & templates | Yes | Local data | Static | Qualification, diagnostics, provider/GCC assessment, recommendation, sprint, meeting, closeout. |
| Backup / export | Yes | Browser | Static | JSON backup/import and finance CSV export. |
| Light / dark appearance | Yes | Local | Static | Accessible light/dark palettes; mobile CSS. |
| Public consulting front door | Yes | Preview only | HTTP | Separate `site.html`; no internal data exposure. |
| Project submission | Yes | Prototype only | Static | Explicitly non-transmitting until backend/privacy controls are wired. |
| Google Workspace source links | Yes | Manual | Static | Drive/Docs/Sheets remain source of truth; no fake synchronization. |
| Google automated intake | No | No | No | Next production integration gate. |
| ChatGPT automatic assistance | No | No | No | Current app makes no automatic API calls; human approval policy is documented. |
| Authentication | No | No | No | Required before collaborators or clients/providers receive access. |
| Engagement-scoped permissions | No | No | No | Required before external collaboration. |
| Secure shared database | No | No | No | Browser localStorage is pilot-only. |
| Production privacy / retention / deletion | No | No | No | Required before real public intake. |
| Rate limiting / abuse controls | No | No | No | Required at deployment layer before public submissions are enabled. |
| Browser/device/accessibility QA | Partial | — | No | Static/HTTP checks passed; full browser QA remains. |
| Source repository | Exception approved | Branch/path scoped | Pending | Founder explicitly approved `Assemble-Teams/CuriousPI` as the bootstrap engineering host on 2026-10-03. Keep Onion under a dedicated branch and `/onion` namespace; runtime/data/branding remain isolated. |
| Vercel preview deployment | No | — | — | Requires a standalone repo/project or deployable standalone Vercel project context. |
| External client/provider portal | Deferred | — | — | Revisit after 3–5 proven engagements. |
| SaaS/public signup | Deferred | — | — | Revisit only after 3–10 project evidence and productization decision. |

## Acceptance checks completed
- Static content checks pass.
- App route responds over local HTTP.
- Public website route responds over local HTTP.
- No Assemble Teams or GameChangers references in the private app file.
- Financial totals do not mix currencies.

## Immediate next deployment sequence
1. Review the Onion bootstrap branch/PR in `Assemble-Teams/CuriousPI`.
2. Run Cursor repository audit against the `/onion` namespace and repository-level coupling risks.
3. Decide whether Onion remains namespaced here or moves to a dedicated repository after audit.
4. Create/link a deployment project that is operationally isolated from Assemble Teams/GameChangers products.
5. Deploy a preview (not production).
6. Run desktop/mobile/keyboard/accessibility browser QA.
7. Wire Google intake behind explicit consent/privacy controls.
8. Prove one end-to-end intake before enabling real submissions.
