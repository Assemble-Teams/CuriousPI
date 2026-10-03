# Cursor Cloud Environment Setup for Onion

## Why this exists

`Assemble-Teams/CuriousPI` has an existing CuriousPI Python/ML project on `main`.
Onion is being developed on branch `onion-bootstrap` under `/onion`.

Cursor Cloud Builds prepare repositories from their default branch before checking out a feature branch. Therefore the environment must **not** blindly install the root `requirements.txt` as the Onion setup. Those packages belong to CuriousPI and are unrelated to Onion.

## Recommended saved Cloud Agent environment

Create a Cursor Cloud Agent environment for repository:

`Assemble-Teams/CuriousPI`

Use a lightweight, idempotent install command:

```bash
set -e
printf 'Preparing shared CuriousPI/Onion cloud environment\n'
git --version
python3 --version
if command -v node >/dev/null 2>&1; then node --version; fi
if command -v npm >/dev/null 2>&1; then npm --version; fi

# Onion dependencies are installed only when the requested branch contains them.
if [ -f onion/package-lock.json ]; then
  npm --prefix onion ci
elif [ -f onion/package.json ]; then
  npm --prefix onion install
fi

printf 'Environment preparation complete\n'
```

This command intentionally does **not** run root `pip install -r requirements.txt`.

## Starting an Onion agent

Start the agent against branch:

`onion-bootstrap`

Then verify:

```bash
git branch --show-current
pwd
ls onion
```

The agent must read:

- `onion/PRD.md`
- `onion/CURSOR_RULES.md`
- `onion/CURSOR_CONTINUATION_BRIEF.md`
- `onion/ONION_STATE.md`
- `onion/CURSOR_KICKOFF_PROMPT.md`
- `onion/DELIVERY-AUDIT.md`
- `onion/REPOSITORY_DECISION.md`

before changing product code.

## Build expectations

A usable Cursor Cloud Build should be marked successful before relying on cloud agents for autonomous verification.

If Cursor says it started from the default image because no usable Build exists:

1. Open Cursor Dashboard → Cloud Agents → Environments.
2. Open the environment for `Assemble-Teams/CuriousPI`.
3. Use **New Setup Run** or **Update with Agent**.
4. Apply the install command above.
5. Trigger a Build.
6. Inspect Build logs.
7. Make sure the Build reaches **Success** and becomes active.
8. Start a **new** cloud agent on `onion-bootstrap`.

Do not rely on an already-running agent to pick up a newly configured environment.

## When Onion gains a real application stack

Once Onion has its own `package.json`, tests, dev server and deployment configuration, update the environment to install those dependencies and add the appropriate startup/terminal command.

Do not add secrets to the public repository. Use Cursor's Cloud Agent Secrets settings.

## RAS status rule

- Environment configuration written = **Built**
- Successful active Cursor Build = **Wired**
- New Onion agent successfully runs repository checks/tests from that Build = **Proven**
